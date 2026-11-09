"""
Stop signal utility for halting in-flight LLM inference.

Usage:
  - Create data/STOP file to signal ALL running agent loops to stop
  - Delete data/STOP file (or restart servers) to resume normal operation
  - Use stop_inference.py script at project root for convenience
  - Per-session: data/sessions/{session_id}.stop halts only that session's
    agent loop, without affecting anyone else.

Per-session marker lifecycle:
  - The chat route clears a stale marker at the START of every turn
    (`clear_session_stop`), so a stop aimed at an earlier (already finished)
    run never kills the next turn.
  - The session's own loop consumes the marker when it observes it
    (`check_stop`), so the next turn starts clean even without the reset.
  - Subagent (child) sessions are named `{parent}__sub_{uuid8}` (legacy:
    `{parent}::{call_id}`). A child observes its parent's marker WITHOUT
    consuming it — every sibling and the parent itself must still see it.
"""
import threading
import time
from typing import Dict, List, Optional

import config


# Child-session separators, newest first. "__sub_" is filename-safe on
# Windows; "::" is the legacy form (kept so old ids still resolve).
_CHILD_SEPARATORS = ("__sub_", "::")


# In-memory record of when each session last had a stop requested (monotonic
# seconds). The chat route uses it to tell "stopped on request" (e.g. a newer
# message preempted the turn) from a plain client disconnect when it writes the
# "[interrupted]" marker into history. Process-local (SERVER_WORKERS=1) and
# purely informational: the on-disk marker stays the source of truth for
# actually halting a run.
_STOP_TIMES: Dict[str, float] = {}
_STOP_TIMES_LOCK = threading.Lock()
_STOP_TIMES_MAX = 512
_STOP_TIMES_TTL_SECONDS = 3600.0


class StopInferenceError(Exception):
    """Raised when a stop signal is detected during inference."""
    pass


def _global_stop_requested() -> bool:
    try:
        return config.STOP_FILE.exists()
    except OSError:
        return False


def _session_stop_path(session_id: str):
    return config.SESSIONS_DIR / f"{session_id}.stop"


def _ancestor_session_ids(session_id: str) -> List[str]:
    """Return the parent chain of a child session id (nearest parent first).

    `root__sub_aaaa__sub_bbbb` -> [`root__sub_aaaa`, `root`].
    """
    ancestors: List[str] = []
    current = session_id
    while True:
        cut = -1
        for sep in _CHILD_SEPARATORS:
            idx = current.rfind(sep)
            if idx > cut:
                cut = idx
        if cut <= 0:
            break
        current = current[:cut]
        ancestors.append(current)
    return ancestors


def _marker_exists(session_id: str) -> bool:
    try:
        return _session_stop_path(session_id).exists()
    except OSError:
        return False


def _unlink_marker(session_id: str) -> None:
    try:
        _session_stop_path(session_id).unlink()
    except OSError:
        pass


def is_stop_requested(session_id: Optional[str] = None, consume: bool = False) -> bool:
    """Return True if a stop is requested.

    With no session_id: only the global data/STOP file is checked (legacy
    behaviour used by admin routes and scripts/stop_inference.py).

    With a session_id: True if the global file exists, the session's own
    marker exists, or any ancestor session's marker exists (children stop
    together with their parent). With consume=True only the session's OWN
    marker is removed — ancestor markers are never consumed here.
    """
    if _global_stop_requested():
        return True
    if not session_id:
        return False
    if _marker_exists(session_id):
        if consume:
            _unlink_marker(session_id)
        return True
    for parent_id in _ancestor_session_ids(session_id):
        if _marker_exists(parent_id):
            return True
    return False


def check_stop(session_id: str = None):
    """Raise StopInferenceError if a global or per-session stop signal is active.

    The session's own marker is consumed (self-clearing). A marker on a parent
    session stops this (child) loop too but is left in place for the parent
    and sibling subagents to observe.
    """
    if _global_stop_requested():
        raise StopInferenceError("Inference stopped — data/STOP file detected")
    if not session_id:
        return
    if _marker_exists(session_id):
        _unlink_marker(session_id)
        raise StopInferenceError(f"Inference stopped — session {session_id} stop requested")
    for parent_id in _ancestor_session_ids(session_id):
        if _marker_exists(parent_id):
            raise StopInferenceError(
                f"Inference stopped — parent session {parent_id} stop requested"
            )


def request_stop():
    """Create the stop file to signal running inference to halt."""
    config.STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.STOP_FILE.touch()
    print("[STOP] Stop file created — inference will halt at next checkpoint")


def clear_stop():
    """Remove the stop file to allow inference to proceed."""
    try:
        if config.STOP_FILE.exists():
            config.STOP_FILE.unlink()
            print("[STOP] Stop file removed")
    except OSError as exc:
        print(f"[STOP] Could not remove stop file: {exc}")


def request_session_stop(session_id: str) -> None:
    """Signal a single session's agent loop (and its subagents) to halt at the next checkpoint."""
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _session_stop_path(session_id).touch()
    _note_stop_time(session_id)
    print(f"[STOP] Session stop requested: {session_id}")


def _note_stop_time(session_id: str) -> None:
    now = time.monotonic()
    with _STOP_TIMES_LOCK:
        _STOP_TIMES[session_id] = now
        if len(_STOP_TIMES) > _STOP_TIMES_MAX:
            cutoff = now - _STOP_TIMES_TTL_SECONDS
            for sid in [k for k, v in _STOP_TIMES.items() if v < cutoff]:
                _STOP_TIMES.pop(sid, None)
            # Still too many (burst of distinct sessions): drop the oldest.
            while len(_STOP_TIMES) > _STOP_TIMES_MAX:
                oldest = min(_STOP_TIMES, key=_STOP_TIMES.get)
                _STOP_TIMES.pop(oldest, None)


def session_stop_requested_since(session_id: str, since_monotonic: float) -> bool:
    """True if a stop was requested for *session_id* at/after `since_monotonic`
    (a `time.monotonic()` reading). Never consumes anything."""
    if not session_id:
        return False
    with _STOP_TIMES_LOCK:
        stamp = _STOP_TIMES.get(session_id)
    return stamp is not None and stamp >= since_monotonic


def clear_session_stop(session_id: str) -> bool:
    """Remove a (stale) per-session stop marker. Returns True if one existed.

    Called by the chat route at the start of each turn: any marker written
    before this turn began was aimed at an earlier run.
    """
    if not session_id or not _marker_exists(session_id):
        return False
    _unlink_marker(session_id)
    print(f"[STOP] Cleared stale stop marker for session {session_id}")
    return True


def is_session_stop_requested(session_id: str) -> bool:
    """True if this session's own marker (or an ancestor's) exists. Never consumes."""
    if not session_id:
        return False
    if _marker_exists(session_id):
        return True
    return any(_marker_exists(p) for p in _ancestor_session_ids(session_id))
