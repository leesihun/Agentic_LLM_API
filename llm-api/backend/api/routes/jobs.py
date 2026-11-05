"""
Background job queue endpoints.

Allows submitting long-running agent tasks as background jobs.
Clients get a job_id immediately and poll for status/output.

POST   /api/jobs              Submit job → {job_id, session_id, status}
GET    /api/jobs              List user's jobs
GET    /api/jobs/{id}         Get job status + full output
GET    /api/jobs/{id}/stream  SSE stream of job output
DELETE /api/jobs/{id}         Cancel a running job
"""
import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from sse_starlette.sse import EventSourceResponse

from filelock import Timeout as FileLockTimeout

from backend.core.database import db, conversation_store, SessionBusyError, TurnLease
from backend.core.job_store import job_store
from backend.core.llm_backend import TextEvent, ToolStatusEvent
from backend.utils.auth import get_optional_user
from backend.utils.file_handler import save_uploaded_files
from backend.utils.stop_signal import StopInferenceError, clear_session_stop
from backend.api.routes.sessions import stop_enforced_events
import config

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

# How long a submission waits for a running turn on the same session to
# release its lease before answering 409 "session busy" (same as chat).
_LEASE_WAIT_SECONDS = 2.0

# Module-level dict of running asyncio tasks for cancellation
_running_tasks: Dict[str, asyncio.Task] = {}

# Per-job asyncio.Event for SSE stream wake-up — replaces 0.2s polling
_job_signals: Dict[str, asyncio.Event] = {}


def _get_job_signal(job_id: str) -> asyncio.Event:
    if job_id not in _job_signals:
        _job_signals[job_id] = asyncio.Event()
    return _job_signals[job_id]


def _signal_job(job_id: str):
    """Wake up any SSE stream subscribers waiting on this job."""
    event = _job_signals.get(job_id)
    if event:
        event.set()


# ============================================================================
# Background runner
# ============================================================================

def _persist_assistant(session_id: str, text: str) -> None:
    """Append the job's assistant output to the session (sync; best effort)."""
    if not text.strip():
        return
    try:
        conversation_store.append_messages(
            session_id, [{"role": "assistant", "content": text}]
        )
        db.increment_session_message_count(session_id, 1)
    except Exception as exc:
        print(f"[JOBS] WARNING: failed to persist assistant output for session {session_id}: {exc}")


async def _run_job(
    job_id: str,
    username: str,
    session_id: str,
    messages: List[Dict[str, Any]],
    file_metadata: List[Dict[str, Any]],
    model: str,
    temperature: Optional[float],
    lease: TurnLease,
    workspace_dir: Optional[str] = None,
):
    """Run the agent loop in the background, streaming output to the job file.

    Owns *lease* (the session's turn lease, acquired at submission) and
    releases it when the run ends however it ends, so a chat turn or manual
    /compact on the same session can't interleave with this run.
    """
    from backend.agent import AgentLoop

    assistant_message = ""
    persisted = False
    try:
        job_store.update_status(job_id, "running")
        _signal_job(job_id)

        agent = AgentLoop(
            model=model,
            temperature=temperature,
            session_id=session_id,
            username=username,
            workspace_dir=workspace_dir,
        )

        # Enforces a session stop even while the loop is blocked in a long
        # tool or LLM call (cancels the run after a short grace period).
        agent_events = stop_enforced_events(
            agent.run_stream(messages, file_metadata), session_id
        )
        try:
            async for event in agent_events:
                if isinstance(event, TextEvent):
                    assistant_message += event.content
                    job_store.append_chunk(job_id, event.content)
                    _signal_job(job_id)
                elif isinstance(event, ToolStatusEvent):
                    job_store.append_tool_event(
                        job_id,
                        tool_name=event.tool_name,
                        status=event.status,
                        duration=getattr(event, "duration", 0.0),
                        activity=getattr(event, "activity", ""),
                        user_name=getattr(event, "user_name", ""),
                    )
                    _signal_job(job_id)
        finally:
            await agent_events.aclose()

        persisted = True
        await asyncio.to_thread(_persist_assistant, session_id, assistant_message)

        job_store.update_status(job_id, "completed")
        _signal_job(job_id)

    except asyncio.CancelledError:
        job_store.update_status(job_id, "cancelled")
        _signal_job(job_id)
    except StopInferenceError:
        job_store.update_status(job_id, "cancelled", error="stopped by user request")
        _signal_job(job_id)
    except Exception as e:
        job_store.update_status(job_id, "failed", error=str(e))
        _signal_job(job_id)
    finally:
        try:
            # Keep partial output of an interrupted/failed run in history
            # (synchronous: a to_thread scheduled during cancellation may
            # never run).
            if not persisted:
                _persist_assistant(session_id, assistant_message)
        finally:
            lease.release()
            _running_tasks.pop(job_id, None)
            _job_signals.pop(job_id, None)


# ============================================================================
# Endpoints
# ============================================================================

def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content) if content else ""


def _build_storage_messages(messages_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "role": msg["role"],
            "content": _extract_text_from_content(msg.get("content")),
        }
        for msg in messages_data
    ]


async def _parse_job_request(request: Request) -> dict:
    """Parse job submission from either JSON or multipart form."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        messages_raw = form.get("messages", "[]")
        try:
            messages_data = json.loads(messages_raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid messages JSON")
        model = form.get("model") or None
        temperature = form.get("temperature")
        session_id = form.get("session_id") or None
        files = [v for v in form.getlist("files") if isinstance(v, UploadFile) and v.filename]
    else:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        messages_data = body.get("messages", [])
        model = body.get("model") or None
        temperature = body.get("temperature")
        session_id = body.get("session_id") or None
        files = []
    if not isinstance(messages_data, list) or not all(
        isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages_data
    ):
        raise HTTPException(
            status_code=400, detail="messages must be a list of objects with a 'role'"
        )
    if temperature is not None and temperature != "":
        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="temperature must be a number")
    else:
        temperature = None
    return dict(
        messages_data=messages_data,
        model=model,
        temperature=temperature,
        session_id=session_id,
        files=files,
    )


@router.post("")
async def submit_job(
    request: Request,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Submit a background agent job. Returns immediately with a job_id.
    The agent runs asynchronously; poll GET /api/jobs/{job_id} for status.
    """
    parsed = await _parse_job_request(request)
    messages_data = parsed["messages_data"]
    username = current_user["username"] if current_user else "guest"
    model_name = parsed["model"] or config.VLLM_MODEL
    # Pass None through to AgentLoop when unspecified so its per-model
    # temperature override (e.g. GLM -> 1.0) applies; the job record keeps
    # the configured default for display.
    requested_temp = parsed["temperature"]
    temp = requested_temp if requested_temp is not None else config.DEFAULT_TEMPERATURE
    session_id = parsed["session_id"]
    files: List[UploadFile] = parsed["files"]
    new_history_messages = _build_storage_messages(messages_data)

    lease: Optional[TurnLease] = None
    # Set once the background task owns the lease (it releases it).
    lease_handed_off = False
    try:
        # Session handling
        existing_session = False
        workspace_dir: Optional[str] = None
        if session_id:
            session = db.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            existing_session = True
            workspace_dir = session.get("workspace_dir") or None
        else:
            session_id = str(uuid.uuid4())
            db.create_session(session_id, username)
            # Auto-title
            for msg in messages_data:
                if msg.get("role") == "user" and msg.get("content"):
                    text = _extract_text_from_content(msg["content"]).strip().replace("\n", " ")
                    db.update_session_title(session_id, text[:60] + ("…" if len(text) > 60 else ""))
                    break

        # Same per-session turn lease as /v1/chat/completions: load -> run ->
        # persist must not interleave with a chat turn, another job, or a
        # manual /compact. Held until the background run finishes.
        lease = await conversation_store.acquire_turn_lease(
            session_id, timeout=_LEASE_WAIT_SECONDS
        )
        # A stop marker written before this run began targeted an earlier run.
        clear_session_stop(session_id)

        history: List[Dict[str, Any]] = []
        if existing_session:
            history = await asyncio.to_thread(
                conversation_store.load_recent_conversation, session_id
            ) or []
        agent_messages = list(history)
        agent_messages.extend(new_history_messages)

        if new_history_messages:
            await asyncio.to_thread(
                conversation_store.append_messages,
                session_id,
                new_history_messages,
            )
            await asyncio.to_thread(
                db.increment_session_message_count,
                session_id,
                len(new_history_messages),
            )

        # File uploads
        file_metadata: List[Dict[str, Any]] = []
        if files:
            from backend.utils.file_handler import extract_file_metadata
            from pathlib import Path
            file_paths = save_uploaded_files(files, username, session_id)
            for fp in file_paths:
                path = Path(fp)
                try:
                    meta = extract_file_metadata(fp)
                    file_metadata.append({"name": path.name, "path": fp, "size": path.stat().st_size, **meta})
                except Exception as e:
                    file_metadata.append({"name": path.name, "path": fp, "error": str(e)})

        job_id = str(uuid.uuid4())
        job_store.create(job_id, username, session_id, model_name, temp)

        task = asyncio.create_task(
            _run_job(
                job_id, username, session_id, agent_messages, file_metadata,
                model_name, requested_temp, lease, workspace_dir,
            )
        )
        lease_handed_off = True
        held = lease

        def _on_task_done(t: asyncio.Task) -> None:
            # Backstop: a task cancelled before its first step never enters
            # _run_job's body or finally; release() is idempotent.
            held.release()
            if t.cancelled():
                try:
                    job_store.update_status(job_id, "cancelled")
                except Exception as exc:
                    print(f"[JOBS] WARNING: could not mark job {job_id} cancelled: {exc}")
                _signal_job(job_id)
                _running_tasks.pop(job_id, None)
                _job_signals.pop(job_id, None)

        task.add_done_callback(_on_task_done)
        _running_tasks[job_id] = task
    except SessionBusyError:
        raise HTTPException(status_code=409, detail="session busy")
    except FileLockTimeout:
        raise HTTPException(status_code=503, detail="session storage busy")
    finally:
        if lease is not None and not lease_handed_off:
            lease.release()

    return JSONResponse(
        status_code=202,
        content={
            "job_id": job_id,
            "session_id": session_id,
            "status": "pending",
        },
    )


@router.get("")
def list_jobs(current_user: Optional[dict] = Depends(get_optional_user)):
    """List all background jobs for the current user."""
    username = current_user["username"] if current_user else "guest"
    jobs = job_store.list_jobs(username)
    return {"jobs": jobs}


@router.get("/{job_id}")
def get_job(job_id: str, current_user: Optional[dict] = Depends(get_optional_user)):
    """Get the full status and output of a job."""
    job = job_store.load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    username = current_user["username"] if current_user else "guest"
    if job["username"] != username:
        raise HTTPException(status_code=403, detail="Access denied")

    output = job_store.read_output(job_id)
    return {
        "job_id": job["job_id"],
        "session_id": job["session_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "output": output,
        "tool_events": job_store.load_tool_events(job_id),
        "error": job.get("error"),
    }


@router.get("/{job_id}/stream")
async def stream_job(job_id: str, current_user: Optional[dict] = Depends(get_optional_user)):
    """SSE stream of job output. Sends chunks as they arrive, then closes when job finishes."""
    job = job_store.load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    username = current_user["username"] if current_user else "guest"
    if job["username"] != username:
        raise HTTPException(status_code=403, detail="Access denied")

    async def generator():
        last_offset = 0
        signal = _get_job_signal(job_id)
        while True:
            current = job_store.load(job_id)
            if current is None:
                break

            output_update = job_store.read_output_since(job_id, last_offset)
            if output_update["content"]:
                yield {"data": json.dumps({"content": output_update["content"]})}
            last_offset = output_update["next_offset"]

            status = current.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                yield {"data": json.dumps({
                    "done": True,
                    "status": status,
                    "error": current.get("error"),
                })}
                break

            # Wait for a signal from _run_job (max 2s safety-net for race conditions).
            # No shield: a shielded wait() left pending on every timeout piles up
            # tasks on an Event nobody sets once _run_job has dropped it.
            signal.clear()
            try:
                await asyncio.wait_for(signal.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    async def guarded():
        try:
            async for item in generator():
                yield item
        finally:
            # Streaming a job that already finished (or outlived _run_job's own
            # cleanup) re-created its Event; drop it once no runner remains.
            if job_id not in _running_tasks:
                _job_signals.pop(job_id, None)

    return EventSourceResponse(guarded())


@router.delete("/{job_id}")
def cancel_job(job_id: str, current_user: Optional[dict] = Depends(get_optional_user)):
    """Cancel a running or pending job."""
    job = job_store.load(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    username = current_user["username"] if current_user else "guest"
    if job["username"] != username:
        raise HTTPException(status_code=403, detail="Access denied")

    if job["status"] in ("completed", "failed", "cancelled"):
        return {"job_id": job_id, "status": job["status"], "message": "Job already finished."}

    # Cancel asyncio task if still running
    task = _running_tasks.get(job_id)
    if task and not task.done():
        task.cancel()
    else:
        # Task may have already finished; mark as cancelled in store
        job_store.update_status(job_id, "cancelled")

    return {"job_id": job_id, "status": "cancelled"}
