"""
Main FastAPI application — single server for chat, auth, tools, and sessions.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.api.routes import auth, models, admin, chat, sessions, tools, jobs, rag_upload_async, cluster

_START_TIME = time.time()

app = FastAPI(
    title="LLM API",
    description="OpenAI-compatible LLM API with native tool calling via vLLM",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _session_artifact_paths(sessions_dir: Path, session_id: str) -> list[Path]:
    return [
        sessions_dir / f"{session_id}.jsonl",
        sessions_dir / f"{session_id}.recent.json",
        sessions_dir / f"{session_id}.json",
        sessions_dir / f"{session_id}.lock",
    ]


def _session_artifacts_exist(sessions_dir: Path, session_id: str) -> bool:
    return any(path.exists() for path in _session_artifact_paths(sessions_dir, session_id)[:-1])


def _session_id_from_artifact(path: Path) -> str:
    if path.name.endswith(".recent.json"):
        return path.name[:-len(".recent.json")]
    return path.stem


def _job_artifact_paths(jobs_dir: Path, job_id: str) -> list[Path]:
    return [
        jobs_dir / f"{job_id}.json",
        jobs_dir / f"{job_id}.lock",
        jobs_dir / f"{job_id}.output.txt",
        jobs_dir / f"{job_id}.events.jsonl",
    ]


def _cleanup_old_sessions():
    """Delete session artifacts older than SESSION_CLEANUP_DAYS."""
    if config.SESSION_CLEANUP_DAYS <= 0:
        return

    sessions_dir = config.SESSIONS_DIR
    if not sessions_dir.exists():
        return

    cutoff = datetime.now() - timedelta(days=config.SESSION_CLEANUP_DAYS)
    removed = 0
    session_ids = {
        _session_id_from_artifact(path)
        for path in sessions_dir.iterdir()
        if path.is_file() and not path.name.endswith(".lock")
    }
    for session_id in session_ids:
        try:
            artifacts = [
                path for path in _session_artifact_paths(sessions_dir, session_id)
                if path.exists() and not path.name.endswith(".lock")
            ]
            if not artifacts:
                continue
            updated = max(datetime.fromtimestamp(path.stat().st_mtime) for path in artifacts)
            if updated < cutoff:
                for path in _session_artifact_paths(sessions_dir, session_id):
                    if path.exists():
                        path.unlink()
                removed += 1
        except Exception:
            pass
    if removed:
        print(f"[Startup] Cleaned up {removed} old session(s)")


def _cleanup_old_jobs():
    """Delete job JSON files older than JOBS_CLEANUP_DAYS."""
    if config.JOBS_CLEANUP_DAYS <= 0:
        return

    jobs_dir = config.JOBS_DIR
    if not jobs_dir.exists():
        return
    cutoff = datetime.now() - timedelta(days=config.JOBS_CLEANUP_DAYS)
    removed = 0
    for f in jobs_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(data.get("created_at", ""))
            if created < cutoff:
                for path in _job_artifact_paths(jobs_dir, f.stem):
                    if path.exists():
                        path.unlink()
                removed += 1
        except Exception:
            pass
    if removed:
        print(f"[Startup] Cleaned up {removed} old job file(s)")


def _cleanup_old_tool_results():
    """Delete tool result directories that are orphaned or older than TOOL_RESULTS_CLEANUP_DAYS."""
    if config.TOOL_RESULTS_CLEANUP_DAYS <= 0:
        return

    tool_results_dir = config.TOOL_RESULTS_DIR
    if not tool_results_dir.exists():
        return
    sessions_dir = config.SESSIONS_DIR
    cutoff = datetime.now() - timedelta(days=config.TOOL_RESULTS_CLEANUP_DAYS)
    removed = 0
    for d in tool_results_dir.iterdir():
        if not d.is_dir():
            continue
        too_old = datetime.fromtimestamp(d.stat().st_mtime) < cutoff
        if _session_artifacts_exist(sessions_dir, d.name) and not too_old:
            continue
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception:
            pass
    if removed:
        print(f"[Startup] Cleaned up {removed} orphaned/old tool result dir(s)")


def _cleanup_old_scratch():
    """Delete scratch directories older than SCRATCH_CLEANUP_DAYS or with no session file."""
    if config.SCRATCH_CLEANUP_DAYS <= 0:
        return

    scratch_dir = config.SCRATCH_DIR
    if not scratch_dir.exists():
        return
    sessions_dir = config.SESSIONS_DIR
    cutoff = datetime.now() - timedelta(days=config.SCRATCH_CLEANUP_DAYS)
    removed = 0
    for d in scratch_dir.iterdir():
        if not d.is_dir():
            continue
        too_old = datetime.fromtimestamp(d.stat().st_mtime) < cutoff
        if _session_artifacts_exist(sessions_dir, d.name) and not too_old:
            continue
        try:
            shutil.rmtree(d)
            removed += 1
        except Exception:
            pass
    if removed:
        print(f"[Startup] Cleaned up {removed} orphaned/old scratch dir(s)")


def _cleanup_old_llm_generated():
    """Delete files in LLM_GENERATED_DIR older than LLM_FILE_RETENTION_DAYS; remove empty dirs."""
    if config.LLM_FILE_RETENTION_DAYS <= 0:
        return
    llm_dir = config.LLM_GENERATED_DIR
    if not llm_dir.exists():
        return
    cutoff = datetime.now() - timedelta(days=config.LLM_FILE_RETENTION_DAYS)
    removed_files = removed_dirs = 0
    for entry in sorted(llm_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            if entry == llm_dir:
                continue
            if entry.is_file() or entry.is_symlink():
                if datetime.fromtimestamp(entry.stat().st_mtime) < cutoff:
                    entry.unlink()
                    removed_files += 1
            elif entry.is_dir() and not any(entry.iterdir()):
                entry.rmdir()
                removed_dirs += 1
        except Exception:
            pass
    if removed_files or removed_dirs:
        print(f"[Cleanup] llm_generated: removed {removed_files} file(s), {removed_dirs} empty dir(s)")


def _cleanup_old_cluster_tasks():
    """Delete cluster task records (+ events + artifacts) past retention.

    Terminal tasks (completed/failed/cancelled) age off completed_at; tasks
    that never reached a terminal state (e.g. no node ever matched to lease
    them) age off created_at instead, so the queue can't grow unbounded.
    """
    if config.CLUSTER_TASKS_CLEANUP_DAYS <= 0:
        return
    tasks_dir = config.CLUSTER_DIR / "tasks"
    if not tasks_dir.exists():
        return
    artifacts_dir = config.CLUSTER_DIR / "artifacts"
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.CLUSTER_TASKS_CLEANUP_DAYS)
    removed = 0
    for path in tasks_dir.glob("*.json"):
        task_id = path.stem
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = task.get("status")
        ts = task.get("completed_at") if status in {"completed", "failed", "cancelled"} else task.get("created_at")
        try:
            when = datetime.fromisoformat(ts) if ts else None
        except ValueError:
            when = None
        if when is None or when >= cutoff:
            continue
        for suffix in (".json", ".lock", ".events.jsonl"):
            p = tasks_dir / f"{task_id}{suffix}"
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        artifact_dir = artifacts_dir / task_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
        removed += 1
    if removed:
        print(f"[Startup] Cleaned up {removed} old cluster task(s)")


def _cleanup_stale_turn_locks():
    """Delete per-session *.turn.lock files left behind by idle/deleted sessions.

    _cleanup_old_sessions never touches lock files, and on Linux filelock
    leaves the file in place after release, so these accumulate one per
    session forever. Only a lock (and its session's history files) untouched
    for a day is considered, and only if a non-blocking acquire proves no
    turn holds it right now.
    """
    sessions_dir = config.SESSIONS_DIR
    if not sessions_dir.exists():
        return
    from filelock import FileLock, Timeout as FileLockTimeout

    cutoff = time.time() - 24 * 60 * 60
    removed = 0
    for path in sessions_dir.glob("*.turn.lock"):
        try:
            session_id = path.name[:-len(".turn.lock")]
            related = [path] + _session_artifact_paths(sessions_dir, session_id)[:-1]
            if any(p.exists() and p.stat().st_mtime >= cutoff for p in related):
                continue
            lock = FileLock(str(path))
            try:
                lock.acquire(timeout=0)
            except FileLockTimeout:
                continue  # a turn holds it right now
            try:
                if sys.platform != "win32":
                    # POSIX: unlink while still holding the flock.
                    path.unlink(missing_ok=True)
            finally:
                lock.release(force=True)
            # Windows: the file can't be removed while open (filelock usually
            # deletes it on release already).
            path.unlink(missing_ok=True)
            removed += 1
        except Exception:
            pass
    if removed:
        print(f"[Startup] Cleaned up {removed} stale session turn lock(s)")


def _cleanup_stale_cluster_files():
    """Drop orphaned cluster index markers and old quarantined *.corrupt-* files.

    Index markers (index/queued/{key}__{task_id}, index/leased/{task_id}) are
    hints only; one whose task record no longer exists (retention cleanup,
    quarantine) is dead weight. Markers whose task still exists are never
    touched here — the store's own lock-holding paths own those. Quarantined
    records (renamed *.corrupt-<ts>) are kept for CLUSTER_TASKS_CLEANUP_DAYS
    for forensics, then removed.
    """
    if config.CLUSTER_TASKS_CLEANUP_DAYS <= 0:
        return
    cluster_dir = config.CLUSTER_DIR
    if not cluster_dir.exists():
        return
    tasks_dir = cluster_dir / "tasks"
    cutoff = time.time() - config.CLUSTER_TASKS_CLEANUP_DAYS * 24 * 60 * 60
    markers = corrupt = 0

    def _old(p: Path) -> bool:
        return p.stat().st_mtime < cutoff

    for sub in ("queued", "leased"):
        index_dir = cluster_dir / "index" / sub
        if not index_dir.is_dir():
            continue
        for marker in index_dir.iterdir():
            try:
                if not marker.is_file() or marker.name.startswith("."):
                    continue
                task_id = marker.name.partition("__")[2] if sub == "queued" else marker.name
                if not task_id or (tasks_dir / f"{task_id}.json").exists():
                    continue
                if _old(marker):
                    marker.unlink()
                    markers += 1
            except Exception:
                pass

    # Quarantined task records (tasks/) and nodes.json (cluster root).
    for d in (cluster_dir, tasks_dir):
        if not d.is_dir():
            continue
        for path in d.glob("*.corrupt-*"):
            try:
                # os.replace keeps the original mtime, so the quarantine
                # stamp in the name (UTC) is the better age when present.
                stamp = path.name.rpartition(".corrupt-")[2].split("-", 1)[0]
                try:
                    quarantined = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(
                        tzinfo=timezone.utc
                    ).timestamp()
                except ValueError:
                    quarantined = 0.0
                if path.is_file() and max(quarantined, path.stat().st_mtime) < cutoff:
                    path.unlink()
                    corrupt += 1
            except Exception:
                pass
    if markers or corrupt:
        print(f"[Startup] Cleaned up {markers} orphaned cluster index marker(s), "
              f"{corrupt} old quarantined file(s)")


def _rotate_log_if_stale(log_path: Path, label: str):
    """Rotate a .log file if its ctime is older than LOG_ROTATION_DAYS."""
    if not log_path.exists():
        return
    cutoff = datetime.now() - timedelta(days=config.LOG_ROTATION_DAYS)
    created = datetime.fromtimestamp(log_path.stat().st_ctime)
    if created >= cutoff:
        return
    try:
        bak = log_path.with_name(log_path.name + ".bak")
        if bak.exists():
            bak.unlink()
        log_path.rename(bak)
        log_path.touch()
        print(f"[Startup] Rotated {label} (was {(datetime.now() - created).days} days old)")
    except Exception:
        pass


def _cleanup_old_logs():
    _rotate_log_if_stale(config.PROMPTS_LOG_PATH, "prompts.log")
    agent_path = Path(getattr(config, "AGENT_LOG_PATH", config.PROMPTS_LOG_PATH))
    if agent_path.resolve() != config.PROMPTS_LOG_PATH.resolve():
        _rotate_log_if_stale(agent_path, str(agent_path.name))


@app.on_event("startup")
async def startup_event():
    from backend.utils.stop_signal import clear_stop
    clear_stop()

    _cleanup_old_sessions()
    _cleanup_old_jobs()
    _cleanup_old_tool_results()
    _cleanup_old_scratch()
    _cleanup_old_llm_generated()
    _cleanup_old_cluster_tasks()
    _cleanup_stale_cluster_files()
    _cleanup_stale_turn_locks()
    _cleanup_old_logs()

    try:
        p = config.PROMPTS_LOG_PATH.resolve()
        ap = Path(getattr(config, "AGENT_LOG_PATH", config.PROMPTS_LOG_PATH)).resolve()
        if ap == p:
            print(f"[Startup] Full log (LLM + agent + tools): {p}")
        else:
            print(f"[Startup] Agent-only log: {ap}")
            print(f"[Startup] LLM + tools log: {p}")
    except Exception:
        pass

    from backend.core.llm_backend import llm_backend
    available = await llm_backend.is_available()
    active_host = getattr(llm_backend.backend, "host", config.VLLM_HOST)
    if available:
        print(f"[Startup] vLLM backend available at {active_host}")
    else:
        print(f"[Startup] WARNING: vLLM backend NOT available at {config.VLLM_HOST}")

    if available and int(getattr(config, "MODEL_CONTEXT_WINDOW", 0) or 0) <= 0:
        detected = await llm_backend.get_context_window(config.VLLM_MODEL)
        if detected > 0:
            config.MODEL_CONTEXT_WINDOW = detected
            print(f"[Startup] Auto-detected MODEL_CONTEXT_WINDOW={detected} from vLLM "
                  f"(set MODEL_CONTEXT_WINDOW explicitly to override)")
        else:
            print("[Startup] Could not auto-detect context window from vLLM — "
                  "proactive compaction stays off. Set MODEL_CONTEXT_WINDOW to enable it.")

    async def _periodic_llm_cleanup():
        while True:
            await asyncio.sleep(24 * 60 * 60)
            try:
                _cleanup_old_llm_generated()
            except Exception as e:
                print(f"[Cleanup] llm_generated background error: {e}")

    asyncio.create_task(_periodic_llm_cleanup())

    if "rag" in config.AVAILABLE_TOOLS and getattr(config, "RAG_PRELOAD_MODELS", False):
        from tools.rag import preload_models
        preload_models()
    elif "rag" in config.AVAILABLE_TOOLS:
        reason = "Windows default" if sys.platform == "win32" else "config override"
        print(f"[Startup] Skipping RAG preload (RAG_PRELOAD_MODELS=False, {reason})")


@app.on_event("shutdown")
async def shutdown_event():
    from backend.core.llm_backend import _backend
    await _backend.close()
    print("[Shutdown] HTTP connection pool closed")


@app.get("/")
def root():
    from backend.core.llm_backend import llm_backend
    return {
        "status": "online",
        "service": "LLM API",
        "version": "2.0.0",
        "cluster": {
            "enabled": config.CLUSTER_ENABLED,
            "role": config.CLUSTER_ROLE,
            "node_name": config.NODE_NAME,
            "advertised_url": config.ADVERTISED_LLM_API_URL,
        },
        "backend": {
            "type": "vllm",
            "host": getattr(llm_backend.backend, "host", config.VLLM_HOST),
        },
    }


async def _get_health_data() -> dict:
    from backend.core.llm_backend import llm_backend
    vllm_ok = await llm_backend.is_available()
    active_host = getattr(llm_backend.backend, "host", config.VLLM_HOST)
    disk = shutil.disk_usage(".")
    opencode_enabled = config.PYTHON_EXECUTOR_MODE == "opencode"
    return {
        "status": "ok" if vllm_ok else "degraded",
        "uptime_s": round(time.time() - _START_TIME),
        "vllm": {
            "available": vllm_ok,
            "host": active_host,
        },
        "opencode": {
            "enabled": opencode_enabled,
            "host": f"{config.OPENCODE_SERVER_HOST}:{config.OPENCODE_SERVER_PORT}" if opencode_enabled else None,
        },
        "disk": {
            "free_gb": round(disk.free / 1e9, 1),
            "used_gb": round(disk.used / 1e9, 1),
            "total_gb": round(disk.total / 1e9, 1),
        },
        "config": {
            "cluster_role": config.CLUSTER_ROLE,
            "node_name": config.NODE_NAME,
            "agent_max_iterations": config.AGENT_MAX_ITERATIONS,
            "available_tools": config.AVAILABLE_TOOLS,
            "python_executor_mode": config.PYTHON_EXECUTOR_MODE,
        },
    }


@app.get("/health")
async def health():
    return await _get_health_data()


@app.get("/api/health")
async def api_health():
    return await _get_health_data()


# Routes
app.include_router(auth.router)      # /api/auth/*
app.include_router(models.router)    # /v1/models
app.include_router(admin.router)     # /api/admin/*
app.include_router(chat.router)      # /v1/chat/completions
app.include_router(sessions.router)  # /api/chat/sessions, /api/chat/history
app.include_router(tools.router)           # /api/tools/*
app.include_router(jobs.router)            # /api/jobs/*
app.include_router(rag_upload_async.router)  # /api/rag/upload/stream (SSE progress)
app.include_router(cluster.router)         # /api/cluster/*


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": {"message": str(exc), "type": "internal_error"}},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level=config.LOG_LEVEL.lower())
