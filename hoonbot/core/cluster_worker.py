"""Slave worker loop for master-owned cluster tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

import config
from core import llm_api
from core.cluster_http import (
    cluster_timeout,
    master_url,
    post_json,
    post_json_retry,
    status_code_of,
)
from core.context import build_llm_context

logger = logging.getLogger(__name__)

_TASK_EXECUTION_TEMPLATE = config.read_prompt("cluster/task_execution.txt")

# Keep in sync with MAX_ARTIFACTS_PER_TASK / MAX_ARTIFACT_BYTES in
# llm-api/backend/core/cluster_store.py — the master rejects excess uploads.
_ARTIFACT_LIMIT = 10
_ARTIFACT_MAX_BYTES = 50 * 1024 * 1024

# Extra room on top of the transcript budget for the delegated instruction
# itself (the master caps the attached transcript at
# CLUSTER_DELEGATION_TRANSCRIPT_TOKENS; this re-applies the cap locally so an
# older master or a pasted wall of text can't force a giant cold prefill).
_PROMPT_HEADROOM_CHARS = 16_000

# Attempts for the final /complete post (transport errors / 5xx are retried
# with exponential backoff; 4xx are final).
_COMPLETE_ATTEMPTS = 5

# Consecutive 404 lease-renew responses after which the task is treated as gone
# and local execution is aborted. A 409 (lease held by another node / stale
# lease epoch) is authoritative and aborts immediately.
_LEASE_LOST_AFTER = 2

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _delegation_prompt_cap() -> int:
    try:
        tokens = int(getattr(config, "CLUSTER_DELEGATION_TRANSCRIPT_TOKENS", 32000))
    except (TypeError, ValueError):
        tokens = 32000
    return max(4_000, tokens * 4) + _PROMPT_HEADROOM_CHARS


def _head_tail(text: str, limit: int, head_ratio: float = 0.4) -> str:
    """Keep the head and tail of *text* within *limit* chars, with a marker."""
    if len(text) <= limit:
        return text
    head = int(limit * head_ratio)
    tail = max(0, limit - head)
    omitted = len(text) - head - tail
    tail_text = text[-tail:] if tail else ""
    return f"{text[:head]}\n\n[... {omitted:,} chars omitted ...]\n\n{tail_text}"


def _node_payload() -> dict[str, Any]:
    disk = shutil.disk_usage(".")
    return {
        "node_name": config.NODE_NAME,
        "role": config.CLUSTER_ROLE,
        "ip": config.NODE_IP,
        "api_url": getattr(config, "LLM_API_URL", ""),
        "capabilities": config.NODE_CAPABILITIES,
        "tags": config.NODE_TAGS,
        "prompt_profile": getattr(config, "PROMPT_PROFILE", "slave"),
        "heartbeat_profile": getattr(config, "HEARTBEAT_PROFILE", "slave"),
        "skills_profile": getattr(config, "SKILLS_PROFILE", "slave"),
        "model": config.LLM_MODEL,
        "disk_free_gb": round(disk.free / 1e9, 1),
    }


async def _register(client: httpx.AsyncClient) -> None:
    await post_json(client, "/api/cluster/register", _node_payload())
    logger.info("[Cluster] Registered node '%s' with master %s", config.NODE_NAME, config.CLUSTER_MASTER_API_URL)


async def _heartbeat(client: httpx.AsyncClient) -> None:
    payload = _node_payload()
    payload["heartbeat_at_monotonic"] = time.monotonic()
    await post_json(client, "/api/cluster/heartbeat", payload)


async def _lease(client: httpx.AsyncClient) -> dict[str, Any] | None:
    result = await post_json(client, "/api/cluster/tasks/lease", {
        "node_name": config.NODE_NAME,
        "capabilities": config.NODE_CAPABILITIES,
        "tags": config.NODE_TAGS,
    })
    return result.get("task")


async def _task_event(client: httpx.AsyncClient, task_id: str, message: str, event_type: str = "event") -> None:
    await post_json(client, f"/api/cluster/tasks/{task_id}/events", {
        "node_name": config.NODE_NAME,
        "type": event_type,
        "message": message,
    })


async def _safe_event(client: httpx.AsyncClient, task_id: str, message: str, event_type: str = "event") -> None:
    """_task_event that never raises (except CancelledError) — progress events
    are best-effort and must not fail the task."""
    try:
        await _task_event(client, task_id, message, event_type)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("[Cluster] Could not post %s event for %s: %s", event_type, task_id, exc)


def _lease_renew_interval(task: dict[str, Any]) -> float:
    """Renew at a third of the lease duration (floor 10s), so one failed renew
    still leaves two more chances before the lease expires. Deriving the
    interval from the leased task keeps the slave correct even if the master
    changes CLUSTER_TASK_LEASE_SECONDS; falls back to 200s if the timestamps
    are absent."""
    try:
        leased = datetime.fromisoformat(task["leased_at"])
        expires = datetime.fromisoformat(task["lease_expires_at"])
        duration = (expires - leased).total_seconds()
        if duration > 0:
            return max(10.0, duration / 3)
    except (KeyError, TypeError, ValueError):
        pass
    return 200.0


def _lease_fields(task: dict[str, Any]) -> dict[str, Any]:
    """node_name plus the lease_epoch fencing token handed out at lease time
    (when the master supplied one), so a renew/complete from a stale copy of
    this task — e.g. after the lease expired and this same node re-leased it —
    is rejected with 409 instead of being accepted twice."""
    fields: dict[str, Any] = {"node_name": config.NODE_NAME}
    epoch = task.get("lease_epoch")
    if epoch is not None and epoch != "":
        fields["lease_epoch"] = epoch
    return fields


def _stop_task(state: dict[str, Any], reason: str) -> None:
    """Mark the task as no longer ours and cancel the running LLM work. The
    executor then returns without reporting anything to the master."""
    state["lease_lost"] = True
    state["stop_reason"] = reason
    work = state.get("work")
    if work is not None and not work.done():
        work.cancel()


async def _renew_lease_loop(client: httpx.AsyncClient, task: dict[str, Any], state: dict[str, Any]) -> None:
    """Keep a running task's lease fresh so the master never re-queues it (which
    would duplicate execution) while it runs — however long that takes.

    Renews every lease/3 seconds. Failures are logged at WARNING and retried
    sooner than the normal cadence. The task is treated as lost — execution
    is cancelled via ``_stop_task`` and nothing is reported — when the master
    answers 409 (lease held by another node / stale epoch), repeatedly
    answers 404 (task removed), or returns the task in a terminal status
    (e.g. ``cancelled`` by an admin)."""
    task_id = task["task_id"]
    interval = _lease_renew_interval(task)
    delay = interval
    not_found = 0
    while True:
        await asyncio.sleep(delay)
        try:
            data = await post_json(client, f"/api/cluster/tasks/{task_id}/renew", _lease_fields(task))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status = status_code_of(exc)
            if status == 409:
                logger.warning(
                    "[Cluster] Lease renew for %s rejected (HTTP 409): %s - task lost, "
                    "aborting local execution to avoid duplicate work",
                    task_id, exc,
                )
                _stop_task(state, "lease lost to another node")
                return
            if status == 404:
                not_found += 1
                logger.warning(
                    "[Cluster] Lease renew for %s rejected (HTTP 404, %d/%d): %s",
                    task_id, not_found, _LEASE_LOST_AFTER, exc,
                )
                if not_found >= _LEASE_LOST_AFTER:
                    logger.warning(
                        "[Cluster] Task %s no longer exists on the master - aborting local execution",
                        task_id,
                    )
                    _stop_task(state, "task removed on master")
                    return
            else:
                logger.warning(
                    "[Cluster] Lease renew failed for %s: %s: %s", task_id, type(exc).__name__, exc,
                )
            # Retry sooner than the normal cadence so a transient blip doesn't
            # eat a whole third of the lease.
            delay = min(interval, 15.0)
            continue

        not_found = 0
        delay = interval
        renewed = data.get("task") if isinstance(data, dict) else None
        status = renewed.get("status") if isinstance(renewed, dict) else None
        if status in _TERMINAL_STATUSES:
            # The master answers 200 with the terminal task when it was
            # finished without us (an admin cancel completes with no
            # node_name): stop working and report nothing.
            logger.warning(
                "[Cluster] Task %s is %s on the master - aborting local execution",
                task_id, status,
            )
            _stop_task(state, f"task {status} on master")
            return


async def _complete(
    client: httpx.AsyncClient,
    task: dict[str, Any],
    result: str | None = None,
    error: str | None = None,
) -> bool:
    """Report the task outcome to the master, retrying transient failures
    (the master treats an identical retried report as idempotent).

    Returns True when the master accepted it. Never raises (except
    CancelledError): a late failure here must never be turned into a second,
    contradicting "failed" completion of a task that actually succeeded. A
    409 means the task is no longer ours (re-leased elsewhere, or already
    finished/cancelled): it is not retried and nothing else is reported."""
    task_id = task["task_id"]
    try:
        await post_json_retry(
            client,
            f"/api/cluster/tasks/{task_id}/complete",
            {
                **_lease_fields(task),
                "status": "failed" if error else "completed",
                "result": result,
                "error": error,
            },
            max_attempts=_COMPLETE_ATTEMPTS,
            base_delay=2.0,
            label=f"cluster complete {task_id}",
        )
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if status_code_of(exc) == 409:
            logger.warning(
                "[Cluster] Master rejected completion of %s (HTTP 409): task lost "
                "(re-leased elsewhere or already finished) - dropping the result",
                task_id,
            )
        else:
            logger.error(
                "[Cluster] Could not report completion of %s (%s): %s: %s",
                task_id, "failed" if error else "completed", type(exc).__name__, exc,
            )
        return False


def _outbox_dir(task_id: str) -> Path:
    return Path(config.DATA_DIR) / "cluster_outbox" / task_id


async def _upload_artifacts(client: httpx.AsyncClient, task_id: str) -> int:
    """Upload every file the agent left in the task outbox to the master.

    Never raises (except CancelledError) — artifact trouble is reported as a
    task event and must not fail an otherwise successful task."""
    outbox = _outbox_dir(task_id)
    try:
        files = sorted(p for p in outbox.rglob("*") if p.is_file()) if outbox.is_dir() else []
    except OSError as exc:
        logger.warning("[Cluster] Could not scan outbox %s: %s", outbox, exc)
        return 0
    if len(files) > _ARTIFACT_LIMIT:
        await _safe_event(client, task_id, f"Outbox has {len(files)} files; uploading first {_ARTIFACT_LIMIT}")
        files = files[:_ARTIFACT_LIMIT]

    headers = {"x-cluster-token": config.CLUSTER_TOKEN} if config.CLUSTER_TOKEN else {}
    uploaded = 0
    for path in files:
        try:
            if path.stat().st_size > _ARTIFACT_MAX_BYTES:
                await _safe_event(client, task_id, f"Skipped oversize artifact {path.name}")
                continue
            with open(path, "rb") as f:
                resp = await client.post(
                    master_url(f"/api/cluster/tasks/{task_id}/artifacts"),
                    headers=headers,
                    data={"node_name": config.NODE_NAME},
                    files={"file": (path.name, f)},
                )
            resp.raise_for_status()
            uploaded += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[Cluster] Artifact upload failed for %s: %s", path, exc)
            await _safe_event(client, task_id, f"Artifact upload failed: {path.name} ({exc})")
    return uploaded


def _build_messages(task_id: str, prompt: str, outbox: Path) -> list[dict[str, str]]:
    """Task text FIRST, then the (long) node context.

    Long tool-heavy runs get auto-compacted inside llm-api; with the bulky
    context first, the task instruction sat at the end of the oldest message
    and could be summarized away. Leading with the task (and repeating a
    one-line reminder at the end) keeps it intact. The context is rebuilt per
    call so a retry sees memory.md after any condensing."""
    task_prompt = _TASK_EXECUTION_TEMPLATE.format(
        task_id=task_id,
        node_name=config.NODE_NAME,
        prompt=prompt,
        outbox_dir=str(outbox.resolve()),
    )
    context = build_llm_context()
    content = (
        f"{task_prompt}\n\n---\n\n"
        "# Node context (background: this node's operating prompt and memory)\n\n"
        f"{context}\n\n---\n\n"
        f"Reminder: execute cluster task {task_id} exactly as stated at the top of this message."
    )
    return [{"role": "user", "content": content}]


async def _await_stream(stream: llm_api.StreamChat) -> str:
    return await stream


async def _enforce_memory_cap(reason: str, force: bool = False) -> None:
    try:
        from core import memory_guard  # lazy: provided by the hoonbot core package
    except Exception as exc:
        logger.debug("[Cluster] memory_guard unavailable: %s", exc)
        return
    try:
        await memory_guard.enforce_memory_cap(reason, force=force)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("[Cluster] memory_guard.enforce_memory_cap(%s) failed: %s", reason, exc)


def _no_text_result(stream: llm_api.StreamChat) -> str:
    diag = stream.diagnostics or {}
    finish = stream.finish_reason or diag.get("finish_reason") or "unknown"
    return (
        f"(no text output; finish_reason={finish}. "
        "Check the attached outbox artifacts, if any.)"
    )


def _diag_summary(stream: llm_api.StreamChat) -> str:
    diag = stream.diagnostics or {}
    finish = stream.finish_reason or diag.get("finish_reason") or "unknown"
    parts = [f"finish_reason={finish}"]
    for key in ("iterations", "tool_calls", "fallback_used"):
        if key in diag:
            parts.append(f"{key}={diag[key]}")
    return ", ".join(parts)


def _outbox_has_files(outbox: Path) -> bool:
    try:
        return outbox.is_dir() and any(p.is_file() for p in outbox.rglob("*"))
    except OSError:
        return False


def _being_cancelled() -> bool:
    """True when the current task itself has a pending cancellation (py3.11+)."""
    current = asyncio.current_task()
    cancelling = getattr(current, "cancelling", None)
    if current is None or cancelling is None:
        return False
    try:
        return bool(cancelling())
    except Exception:
        return False


async def _execute_task(client: httpx.AsyncClient, task: dict[str, Any]) -> None:
    task_id = task["task_id"]
    raw_prompt = task.get("prompt") or ""
    prompt = _head_tail(raw_prompt, _delegation_prompt_cap())
    if len(prompt) < len(raw_prompt):
        logger.info("[Cluster] Task %s prompt capped from %d to %d chars", task_id, len(raw_prompt), len(prompt))
    await _safe_event(client, task_id, "Starting local LLM execution", "started")

    if not config.LLM_API_KEY:
        await _complete(client, task, error="LLM_API_KEY is not configured on slave")
        return
    if not config.LLM_MODEL:
        await _complete(client, task, error="LLM_MODEL is not configured on slave")
        return

    outbox = _outbox_dir(task_id)
    outbox.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    # lease_lost: set by the renew loop (_stop_task) when the task is no longer
    # ours — re-leased elsewhere (409), removed, or cancelled on the master.
    state: dict[str, Any] = {"lease_lost": False, "stop_reason": "", "work": None}
    renew = asyncio.create_task(_renew_lease_loop(client, task, state), name=f"cluster-renew-{task_id}")

    result: str | None = None
    error: str | None = None
    try:
        session_id = f"cluster_{task_id}"
        for attempt in range(2):
            if state["lease_lost"]:
                break  # lost between attempts (e.g. during memory condensing)
            payload = {
                "model": config.LLM_MODEL,
                "messages": json.dumps(_build_messages(task_id, prompt, outbox)),
                "session_id": session_id,
            }
            # Streaming: SSE keepalive pings + the idle watchdog replace the old
            # wall-clock cap, so long agentic runs are never cut off mid-way.
            stream = llm_api.stream_chat(payload, headers, kind="cluster")
            work = asyncio.create_task(_await_stream(stream), name=f"cluster-llm-{task_id}")
            state["work"] = work
            try:
                text = await work
            except asyncio.CancelledError:
                # Our own work task was cancelled by the renew loop (not this
                # runner by shutdown): abandon quietly, report nothing.
                if state["lease_lost"] and not _being_cancelled():
                    break
                raise
            except llm_api.LLMAPIError as exc:
                if attempt == 0 and not exc.received_output and llm_api.is_context_overflow_error(exc):
                    logger.warning(
                        "[Cluster] Task %s hit context overflow - condensing memory and retrying in a fresh session",
                        task_id,
                    )
                    await _enforce_memory_cap("overflow", force=True)
                    session_id = f"cluster_{task_id}_r1"
                    continue
                partial = (stream.text or "").strip()
                error = f"LLMAPIError: {exc}"
                if partial:
                    error += "\n\nPartial output before the failure:\n" + _head_tail(partial, 6000, head_ratio=0.3)
                break
            text = (text or "").strip()
            diag = stream.diagnostics or {}
            finish = str(stream.finish_reason or diag.get("finish_reason") or "")
            if finish in ("interrupted", "error"):
                # Stop request / agent failure: the stream still ends with a
                # final chunk + [DONE], so StreamChat did not raise. Never
                # report this as a completed task.
                error = f"LLM turn ended without a final answer ({_diag_summary(stream)})"
                if text:
                    error += "\n\nPartial output before the failure:\n" + _head_tail(text, 6000, head_ratio=0.3)
            elif text:
                result = text
            elif _outbox_has_files(outbox):
                # No text, but the agent left artifacts — they are the result.
                result = _no_text_result(stream)
            else:
                error = f"LLM turn produced no text output and no artifacts ({_diag_summary(stream)})"
            break
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error("[Cluster] Task %s failed: %s", task_id, exc, exc_info=True)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        renew.cancel()
        try:
            await renew
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    if state["lease_lost"]:
        # The task is no longer ours: whatever finished locally (result,
        # error, or nothing) is dropped — reporting it would only be rejected
        # (409) or, worse, contradict the outcome recorded by the new holder.
        logger.warning(
            "[Cluster] Task %s abandoned (%s); local %s not reported",
            task_id, state.get("stop_reason") or "lease lost",
            "result" if result is not None else ("error" if error else "run"),
        )
        shutil.rmtree(outbox, ignore_errors=True)
        return

    try:
        if result is not None:
            # Ship outbox files before completing so the relay sees them on
            # delivery. Upload/complete trouble never downgrades the result.
            uploaded = await _upload_artifacts(client, task_id)
            if await _complete(client, task, result=result):
                logger.info("[Cluster] Completed task %s (%s chars, %s artifact(s))", task_id, len(result), uploaded)
        else:
            logger.error("[Cluster] Task %s failed: %s", task_id, (error or "unknown failure")[:500])
            await _complete(client, task, error=error or "unknown failure")
    finally:
        shutil.rmtree(outbox, ignore_errors=True)


async def run_slave_worker_loop() -> None:
    if not config.CLUSTER_ENABLED:
        logger.info("[Cluster] Disabled by config")
        return
    if config.CLUSTER_ROLE != "slave":
        logger.info("[Cluster] Worker loop skipped on role=%s", config.CLUSTER_ROLE)
        return

    try:
        max_conn = max(20, 2 * int(config.CLUSTER_SLAVE_MAX_CONCURRENT_TASKS) + 5)
    except (TypeError, ValueError):
        max_conn = 20
    async with httpx.AsyncClient(
        timeout=cluster_timeout(),
        trust_env=False,
        limits=httpx.Limits(max_connections=max_conn, max_keepalive_connections=10),
    ) as client:
        while True:
            try:
                await _register(client)
                break
            except Exception as exc:
                logger.warning("[Cluster] Register failed: %s", exc)
                await asyncio.sleep(config.CLUSTER_SLAVE_POLL_INTERVAL_SECONDS)

        running: set[asyncio.Task[None]] = set()

        def _on_task_done(done: asyncio.Task[None]) -> None:
            running.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.error("[Cluster] Task runner crashed: %s", done.exception(), exc_info=done.exception())

        try:
            while True:
                try:
                    await _heartbeat(client)
                    # Fill free execution slots, then sleep; running tasks
                    # proceed in the background so the loop keeps polling.
                    while len(running) < config.CLUSTER_SLAVE_MAX_CONCURRENT_TASKS:
                        task = await _lease(client)
                        if not task:
                            break
                        runner = asyncio.create_task(
                            _execute_task(client, task),
                            name=f"cluster-task-{task['task_id']}",
                        )
                        running.add(runner)
                        runner.add_done_callback(_on_task_done)
                    await asyncio.sleep(config.CLUSTER_SLAVE_POLL_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[Cluster] Worker loop error: %s", exc)
                    await asyncio.sleep(config.CLUSTER_SLAVE_POLL_INTERVAL_SECONDS)
        finally:
            for runner in running:
                runner.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
