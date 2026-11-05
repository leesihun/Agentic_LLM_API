"""
Session management endpoints
/api/chat/sessions - List or search user sessions
/api/chat/sessions/{session_id} - Rename a session
/api/chat/sessions/{session_id}/compact - Summarize old history in-place
/api/chat/history/{session_id} - Get conversation history
"""
import asyncio
import time

import anyio
from fastapi import APIRouter, HTTPException, Depends, Request
from filelock import Timeout as FileLockTimeout
from pydantic import BaseModel
from typing import Any, AsyncIterator, Optional, Tuple

from backend.models.schemas import SessionsListResponse, SessionInfo, ChatHistoryResponse, ChatMessage
from backend.core.database import db, conversation_store, SessionBusyError
from backend.utils.auth import get_current_user, get_optional_user
from backend.utils.stop_signal import StopInferenceError, is_stop_requested, clear_session_stop

router = APIRouter(prefix="/api/chat", tags=["sessions"])

# How long a manual compaction waits for a running turn to release the
# session lease before answering 409 (same budget as the chat route).
_LEASE_WAIT_SECONDS = 2.0

# A manual compaction polls the stop marker and the client connection this
# often while its summarizer call runs (see compact_session).
_COMPACT_POLL_SECONDS = 1.0

# Stop enforcement (see stop_enforced_events). The marker is polled this often;
# a stop the agent loop hasn't honoured within the grace period is enforced by
# cancelling the run; cancellation cleanup (process-tree kills, tool-task
# reaping) gets at most _STOP_CLEANUP_SECONDS before the stop is reported.
_STOP_POLL_SECONDS = 0.5
_STOP_GRACE_SECONDS = 2.0
_STOP_CLEANUP_SECONDS = 30.0


async def _reap_task(task: "asyncio.Future", timeout: float) -> None:
    """Wait (bounded) for a cancelled *task* to finish its cleanup.

    Shielded from anyio cancellation: on an SSE client disconnect Starlette /
    sse_starlette cancel the response's anyio scope, which re-cancels every
    await inside it (level-triggered), so an unshielded wait returns at once
    and the caller would release the session lease while the agent is still
    killing process trees / closing the vLLM stream. Never raises.
    """
    try:
        with anyio.CancelScope(shield=True):
            await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        # Native task.cancel() of the caller still lands here; the caller's
        # original exception propagates and the task finishes on its own.
        pass


async def stop_enforced_events(
    events: AsyncIterator[Any],
    session_id: Optional[str],
) -> AsyncIterator[Any]:
    """Relay *events* (an AgentLoop.run_stream generator) while enforcing stops.

    The agent loop only checks the stop marker at iteration boundaries, so a
    serial shell_exec / code_exec (up to its 3600s / 1800s hard cap) or a long
    LLM generation would ignore POST /sessions/{id}/stop (and data/STOP) until
    it finished. Here the generator runs in its own task; if a stop for
    *session_id* stays unobserved for _STOP_GRACE_SECONDS, that task is
    cancelled — which kills the running tool's process tree and closes the
    vLLM stream — the session's own marker is consumed, and
    StopInferenceError is raised exactly as if the loop had stopped itself.

    Any other outcome of the run (normal end, the loop's own
    StopInferenceError, errors) is passed through unchanged. Callers should
    aclose() this generator when they stop iterating early.
    """
    queue: asyncio.Queue = asyncio.Queue()
    end = object()

    async def _produce() -> None:
        try:
            async for event in events:
                queue.put_nowait(event)
        finally:
            queue.put_nowait(end)
            try:
                await events.aclose()
            except Exception:
                pass

    producer = asyncio.create_task(_produce())
    # Never leave "Task exception was never retrieved" behind on teardown paths.
    producer.add_done_callback(lambda t: t.cancelled() or t.exception())
    getter: Optional[asyncio.Future] = None
    stop_seen_at: Optional[float] = None
    next_check = 0.0
    reaped = False
    try:
        while True:
            if getter is None:
                getter = asyncio.ensure_future(queue.get())
            await asyncio.wait({getter}, timeout=_STOP_POLL_SECONDS)
            if getter.done():
                event = getter.result()
                getter = None
                if event is end:
                    await producer  # re-raises the run's own exception, if any
                    return
                yield event
            now = time.monotonic()
            if now < next_check:
                continue
            next_check = now + _STOP_POLL_SECONDS
            if not is_stop_requested(session_id):  # global data/STOP too
                stop_seen_at = None
                continue
            if stop_seen_at is None:
                stop_seen_at = now
                continue
            if now - stop_seen_at < _STOP_GRACE_SECONDS:
                continue
            # The loop is blocked somewhere it can't see the marker.
            print(
                f"[STOP] session {session_id}: stop not honoured within "
                f"{_STOP_GRACE_SECONDS:.0f}s (agent blocked in a tool or LLM call); "
                f"cancelling the run"
            )
            producer.cancel()
            reaped = True
            await _reap_task(producer, _STOP_CLEANUP_SECONDS)
            is_stop_requested(session_id, consume=True)
            raise StopInferenceError(
                f"Inference stopped - session {session_id} stop requested"
            )
    finally:
        if getter is not None and not getter.done():
            getter.cancel()
        if not producer.done():
            producer.cancel()
            if not reaped:
                await _reap_task(producer, _STOP_CLEANUP_SECONDS)


class RenameSessionRequest(BaseModel):
    title: str


@router.get("/sessions", response_model=SessionsListResponse)
def list_sessions(
    q: Optional[str] = None,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    List sessions for the current user. Pass ?q=keyword to search by title or session ID.
    """
    username = current_user["username"] if current_user else "guest"

    if q and q.strip():
        sessions = db.search_sessions(username, q.strip())
    else:
        sessions = db.list_user_sessions(username)

    session_infos = [
        SessionInfo(
            session_id=session["id"],
            title=session.get("title"),
            created_at=session["created_at"],
            message_count=session["message_count"],
        )
        for session in sessions
        if not session["id"].startswith("hb_")
    ]

    return SessionsListResponse(sessions=session_infos)


@router.patch("/sessions/{session_id}", response_model=SessionInfo)
def rename_session(
    session_id: str,
    body: RenameSessionRequest,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Rename a session by setting its title."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user:
        if session["username"] != current_user["username"] and session["username"] != "guest":
            raise HTTPException(status_code=403, detail="Access denied")

    title = body.title.strip()[:120]  # cap at 120 chars
    db.update_session_title(session_id, title)

    return SessionInfo(
        session_id=session_id,
        title=title,
        created_at=session["created_at"],
        message_count=session["message_count"],
    )


async def _await_stoppable(
    coro,
    request: Request,
    session_id: str,
) -> Tuple[Optional[str], Any]:
    """Run *coro* in its own task, polling for a stop or a client disconnect.

    Returns (None, result) when it finishes, or ("stopped" | "disconnected",
    None) after cancelling it because POST /sessions/{id}/stop (or data/STOP)
    was observed or the HTTP client went away. Exceptions from *coro*
    propagate. The task is always finished or cancelled on return, so the
    caller can release the session lease right after.
    """
    task = asyncio.ensure_future(coro)
    # Never leave "Task exception was never retrieved" behind on teardown paths.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    outcome: Optional[str] = None
    try:
        while True:
            await asyncio.wait({task}, timeout=_COMPACT_POLL_SECONDS)
            if task.done():
                return None, task.result()
            # consume=True: only this session's own marker is removed.
            if is_stop_requested(session_id, consume=True):
                outcome = "stopped"
                break
            try:
                gone = await request.is_disconnected()
            except Exception:
                gone = False
            if gone:
                outcome = "disconnected"
                break
        print(f"[COMPACT] session {session_id}: {outcome} — cancelling the summarizer")
        return outcome, None
    finally:
        if not task.done():
            task.cancel()
            await _reap_task(task, _STOP_CLEANUP_SECONDS)


@router.post("/sessions/{session_id}/compact")
async def compact_session(
    session_id: str,
    request: Request,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Summarize the older half of a session's conversation history in place.

    The session_id is preserved — the conversation continues with the same
    context but a reduced message count. Useful when approaching context limits.

    Runs under the same per-session turn lease as /v1/chat/completions and
    /api/jobs, so a turn can never append while the history is being
    rewritten (and a compaction never starts in the middle of a turn). 409
    when busy.

    Stoppable: while the summarizer LLM call runs, POST /sessions/{id}/stop
    (or data/STOP) or the client disconnecting cancels it within about
    _COMPACT_POLL_SECONDS; the history is left untouched and the lease is
    released. A stop answers {"success": false, "stopped": true, ...}.
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        lease = await conversation_store.acquire_turn_lease(session_id, timeout=_LEASE_WAIT_SECONDS)
    except SessionBusyError:
        raise HTTPException(status_code=409, detail="session busy")

    try:
        # Holding the lease means no turn is running: a session marker written
        # before now targeted an earlier run (same rule as chat/jobs).
        clear_session_stop(session_id)
        try:
            messages = await asyncio.to_thread(conversation_store.load_conversation, session_id) or []
        except FileLockTimeout:
            raise HTTPException(status_code=503, detail="session storage busy")
        original_count = len(messages)

        if original_count < 4:
            return {
                "success": False,
                "message": "Not enough messages to compact",
                "original_count": original_count,
                "new_count": original_count,
            }

        def _stopped_reply(outcome: str) -> dict:
            return {
                "success": False,
                "stopped": True,
                "message": (
                    "Compaction stopped" if outcome == "stopped"
                    else "Compaction abandoned (client disconnected)"
                ),
                "original_count": original_count,
                "new_count": original_count,
            }

        # data/STOP (global) is not cleared above; don't start a summarizer
        # call that the first poll would cancel anyway.
        if is_stop_requested(session_id, consume=True):
            return _stopped_reply("stopped")

        from backend.agent import UnifiedAgent
        agent = UnifiedAgent(
            session_id=session_id,
            username=session.get("username", "guest"),
        )
        # _summarize_and_compact_msgs only mutates `messages` after the
        # summarizer returns, so a cancelled run leaves it (and the stored
        # history, which is not saved) untouched.
        outcome, compacted = await _await_stoppable(
            agent._summarize_and_compact_msgs(messages), request, session_id,
        )
        if outcome is not None:
            return _stopped_reply(outcome)

        if not compacted:
            return {
                "success": False,
                "message": "Nothing left to compact",
                "original_count": original_count,
                "new_count": original_count,
            }

        try:
            await asyncio.to_thread(conversation_store.save_conversation, session_id, messages)
        except FileLockTimeout:
            raise HTTPException(status_code=503, detail="session storage busy")
        return {
            "success": True,
            "message": f"Compacted {original_count} → {len(messages)} messages",
            "original_count": original_count,
            "new_count": len(messages),
        }
    finally:
        lease.release()


@router.post("/sessions/{session_id}/stop")
async def stop_session(
    session_id: str,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Halt the in-flight agent loop for this session.

    The loop honours the marker at its next checkpoint; if it is blocked in a
    long tool (serial shell_exec / code_exec) or LLM call, the chat/jobs route
    running it cancels the run after a short grace period
    (stop_enforced_events), killing the tool's process tree.

    A running manual POST /sessions/{id}/compact is stopped the same way: its
    summarizer call is cancelled and the history is left unchanged.

    Scoped to one session (and its subagents, which observe the parent's
    marker) — other sessions keep running. A marker written while nothing is
    running can no longer kill the NEXT run: chat turns, /jobs runs and manual
    compactions all clear stale markers once they hold the session lease.
    `running` reports whether anything currently holds the session's turn
    lease — a chat turn, a background /jobs run (held from submission until
    the run ends) or a manual compaction. Best effort: it is a snapshot that
    can change right after it is taken.
    """
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user:
        if session["username"] != current_user["username"] and session["username"] != "guest":
            raise HTTPException(status_code=403, detail="Access denied")

    running = conversation_store.is_turn_active(session_id)

    from backend.utils.stop_signal import request_session_stop
    request_session_stop(session_id)

    return {
        "success": True,
        "running": running,
        "message": f"Stop requested for session {session_id}",
    }


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
def get_history(
    session_id: str,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """Get conversation history for a specific session."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if current_user:
        if session["username"] != current_user["username"] and session["username"] != "guest":
            raise HTTPException(status_code=403, detail="Access denied")

    try:
        messages = conversation_store.load_conversation(session_id)
    except FileLockTimeout:
        raise HTTPException(status_code=503, detail="session storage busy")
    if messages is None:
        messages = []

    # Skip records that don't fit the schema (e.g. an unknown role) instead
    # of failing the whole history view.
    chat_messages = []
    for msg in messages:
        try:
            chat_messages.append(ChatMessage(**msg))
        except Exception:
            continue
    return ChatHistoryResponse(messages=chat_messages)
