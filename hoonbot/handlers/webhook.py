"""
Webhook handler — receives events from Messenger and processes them.

Supports: new_message, message_edited, message_deleted.
Features: streaming responses, reply threading, file/image handling,
          session lifecycle, structured logging with timing.

Rapid-input model (per room, see RoomState):
  - Every accepted message goes into the room's FIFO queue; one worker task
    per room drains it. A debounce window (DEBOUNCE_SECONDS) gathers a burst,
    then the worker takes ALL queued chat messages (<= BATCH_MAX_MESSAGES) as
    ONE turn. Multiple messages are sent to the LLM as labelled <message>
    blocks so each one is answered separately instead of being mixed up.
  - Messages that arrive while a turn runs wait for the NEXT turn (⏳ reaction
    while waiting). FOLLOWUP_POLICY="interrupt" restores the old cancel+merge
    behaviour for same-sender follow-ups only.
  - @stop / @clear run immediately, outside the queue. @compact and /goal are
    handled in order inside the queue, as is a catch-up-replayed @clear (so it
    cannot jump ahead of replayed messages sent before it).
  - Catch-up replay (process_message) and external webhooks go through the
    same queue machinery; external webhooks use their own session key.
  - Queued and in-flight Messenger msg_ids are persisted per room in
    data/room_queue.json (debounced atomic writes). shutdown_room_workers()
    stops the workers on shutdown and flushes that file; on the next start
    hoonbot.py's catch-up re-enqueues every persisted id that was not
    answered (enqueue_catchup_message(..., accepted=True)).
"""
import asyncio
import json
import logging
import os
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request

import config
from core import llm_api
from core import messenger
from core.cluster_client import try_submit_from_message
from core.context import build_llm_context, build_per_turn_context

logger = logging.getLogger(__name__)

_MEMORY_FLUSH_HINT = config.read_prompt("webhook/memory_flush_hint.txt")
router = APIRouter()


# ---------------------------------------------------------------------------
# Cross-partition contracts (resolved defensively so this module keeps working
# while core/llm_api.py, core/messenger.py and core/memory_guard.py evolve)
# ---------------------------------------------------------------------------

_LLMAPIErrorBase = getattr(llm_api, "LLMAPIError", None)
if _LLMAPIErrorBase is None:
    class LLMAPIError(Exception):
        """Local fallback matching core.llm_api.LLMAPIError (C8)."""

        def __init__(self, message: str, status_code: int | None = None,
                     session_id: str | None = None, received_output: bool = False):
            super().__init__(message)
            self.message = message
            self.status_code = status_code
            self.session_id = session_id
            self.received_output = received_output
else:
    LLMAPIError = _LLMAPIErrorBase


def _make_llm_error(message: str, status_code: int | None = None,
                    session_id: str | None = None, received_output: bool = False) -> Exception:
    try:
        return LLMAPIError(message, status_code=status_code, session_id=session_id,
                           received_output=received_output)
    except TypeError:
        return LLMAPIError(message)


def _llm_client() -> httpx.AsyncClient:
    """Shared pooled LLM API client (C8 "interactive" kind when available)."""
    try:
        return llm_api.get_client("interactive")
    except TypeError:
        return llm_api.get_client()


_memory_guard_mod = None
_memory_guard_checked_at = 0.0


def _get_memory_guard():
    """Lazily import core.memory_guard (C10); register the condensed callback once."""
    global _memory_guard_mod, _memory_guard_checked_at
    if _memory_guard_mod is not None:
        return _memory_guard_mod
    now = time.monotonic()
    if _memory_guard_checked_at and now - _memory_guard_checked_at < 60:
        return None
    _memory_guard_checked_at = now
    try:
        from core import memory_guard as mod  # noqa: WPS433 — optional module
    except Exception:
        return None
    _memory_guard_mod = mod
    register = getattr(mod, "register_on_condensed", None)
    if callable(register):
        try:
            register(_on_memory_condensed)
        except Exception as exc:
            logger.warning(f"[MemoryGuard] Could not register condensed callback: {exc}")
    return mod


def _memory_max_chars() -> int:
    mod = _memory_guard_mod
    return int(getattr(mod, "MEMORY_MAX_CHARS", None) or getattr(config, "MEMORY_MAX_CHARS", 12000))


def _memory_soft_chars() -> int:
    mod = _memory_guard_mod
    return int(getattr(mod, "MEMORY_SOFT_CHARS", None) or getattr(config, "MEMORY_SOFT_CHARS", 8000))


def _memory_size() -> int | None:
    mod = _get_memory_guard()
    if mod is not None and callable(getattr(mod, "memory_size", None)):
        try:
            return int(mod.memory_size())
        except Exception:
            pass
    try:
        with open(os.path.join(config.DATA_DIR, "memory.md"), "r", encoding="utf-8") as f:
            return len(f.read())
    except FileNotFoundError:
        return 0
    except Exception:
        return None


def _memory_fingerprint() -> str | None:
    mod = _get_memory_guard()
    if mod is not None and callable(getattr(mod, "memory_fingerprint", None)):
        try:
            return str(mod.memory_fingerprint())
        except Exception:
            return None
    return None


async def _enforce_memory_cap(reason: str, force: bool = False) -> bool:
    mod = _get_memory_guard()
    fn = getattr(mod, "enforce_memory_cap", None) if mod is not None else None
    if not callable(fn):
        return False
    try:
        return bool(await fn(reason, force=force))
    except Exception as exc:  # contract says it never raises; be safe anyway
        logger.warning(f"[MemoryGuard] enforce_memory_cap({reason!r}) failed: {exc}")
        return False


_memory_cap_task: asyncio.Task | None = None


def _maybe_condense_memory() -> None:
    """Post-turn check: kick off a background condensation when over the soft cap."""
    global _memory_cap_task
    size = _memory_size()
    if size is None or _get_memory_guard() is None:
        return
    if size <= _memory_soft_chars():
        return
    if _memory_cap_task is not None and not _memory_cap_task.done():
        return
    logger.info(f"[MemoryGuard] memory.md is {size} chars (> soft {_memory_soft_chars()}) — condensing in background")
    _memory_cap_task = asyncio.create_task(_enforce_memory_cap("post-turn"))


# ---------------------------------------------------------------------------
# Fire-and-forget helpers (typing indicator / reactions must never block/raise)
# ---------------------------------------------------------------------------

_background_tasks: set = set()


def _spawn(coro) -> asyncio.Task:
    """create_task that keeps a strong ref and swallows/logs exceptions."""
    async def _runner():
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug(f"[Webhook] background task failed: {exc}")

    task = asyncio.create_task(_runner())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _retrieve_exception(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()  # mark retrieved — no "never retrieved" warning


async def _run_shielded(coro):
    """Run *coro* as its own (strongly referenced) task and await it under
    asyncio.shield: cancelling the caller does not cancel the work, which
    keeps running to completion in the background."""
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_retrieve_exception)
    return await asyncio.shield(task)


def _fire_typing(room_id: int, status_text: str | None = None) -> None:
    if room_id is None:
        return
    _spawn(messenger.send_typing(room_id, status_text=status_text))


def _fire_stop_typing(room_id: int) -> None:
    if room_id is None:
        return
    _spawn(messenger.stop_typing(room_id))


_WAITING_REACTION = "⏳"


async def _add_waiting_reaction(room_id: int, item: "_QueuedMessage") -> None:
    fn = getattr(messenger, "add_reaction", None)
    if not callable(fn) or item.msg_id is None:
        return
    try:
        ok = await fn(room_id, item.msg_id, _WAITING_REACTION)
        item.reacted = ok is not False
    except Exception:
        item.reacted = False


async def _remove_waiting_reaction(room_id: int, item: "_QueuedMessage") -> None:
    # Wait for an in-flight add so the remove can't race ahead of it.
    if item.reaction_task is not None and not item.reaction_task.done():
        try:
            await item.reaction_task
        except Exception:
            pass
    if not item.reacted:
        return
    fn = getattr(messenger, "remove_reaction", None)
    if not callable(fn) or item.msg_id is None:
        return
    try:
        await fn(room_id, item.msg_id, _WAITING_REACTION)
    except Exception:
        pass
    item.reacted = False


# ---------------------------------------------------------------------------
# Persistent state: room sessions + goal mode
# ---------------------------------------------------------------------------

# Per-room message counter (resets on new session); keyed by session key.
_room_msg_count: dict = {}
# Message count at which the memory-flush hint was last injected (soft-cap path).
_room_last_hint: dict = {}

# Per-room persistent session data:
#   {session_key: {"session_id", "created_at", "memory_fp", "memory_chars",
#                  "memory_refresh" (optional: memory.md condensed since the
#                  session's first-turn copy — re-send it on the next turn)}}
# session_key is the Messenger room id (int) or "ext_<source>" for external
# webhooks.
_SESSIONS_FILE = os.path.join(config.DATA_DIR, "room_sessions.json")
_room_sessions: dict = {}

# Bumped whenever memory.md is condensed. A session created by a turn that was
# in flight across a condensation carries the pre-condense memory copy, so it is
# saved flagged for a memory refresh (see _capture_session).
_sessions_epoch = 0

# Per-room sticky goal-mode flag: {room_id: True}. Toggled by the /goal directive
# and forwarded to the LLM API as mode="goal" on every turn while set.
_GOAL_MODE_FILE = os.path.join(config.DATA_DIR, "room_goal_mode.json")
_room_goal_mode: dict[int, bool] = {}

# Leading `/goal` or `@goal` directive; group(1) is the remaining task text
# (or "off"/"stop"/... to disable). Anchored at the start so the token can be
# stripped and the rest processed as the task.
_GOAL_DIRECTIVE_RE = re.compile(r"^\s*[/@]goal\b[ \t]*(.*)$", re.IGNORECASE | re.DOTALL)
_GOAL_OFF_WORDS = {"off", "stop", "disable", "end", "끄기", "종료"}


def _session_key_from_json(k: str):
    try:
        return int(k)
    except (TypeError, ValueError):
        return str(k)


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON via tmp file + os.replace (retrying briefly on Windows locks)."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                # Windows: target briefly locked by AV/indexer/another reader.
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_room_sessions() -> None:
    global _room_sessions
    try:
        with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            _room_sessions = {}
            for k, v in raw.items():
                key = _session_key_from_json(k)
                if isinstance(v, str):
                    # Migrate old format: {room_id: session_id_string}
                    _room_sessions[key] = {"session_id": v, "created_at": datetime.now(timezone.utc).isoformat()}
                elif isinstance(v, dict):
                    _room_sessions[key] = v
        logger.info(f"[Sessions] Loaded {len(_room_sessions)} room session(s) from disk")
    except FileNotFoundError:
        _room_sessions = {}
    except Exception as e:
        logger.warning(f"[Sessions] Could not load room sessions: {e}")
        _room_sessions = {}


def _save_room_sessions() -> None:
    try:
        _atomic_write_json(_SESSIONS_FILE, {str(k): v for k, v in _room_sessions.items()})
    except Exception as e:
        logger.warning(f"[Sessions] Could not save room sessions: {e}")


def _peek_session_id(key) -> str | None:
    """Raw lookup — no expiry/staleness side effects."""
    entry = _room_sessions.get(key)
    if not entry:
        return None
    return entry.get("session_id") or None


def _session_stale_reason(entry: dict) -> str | None:
    """Why a cached session must not be continued (memory snapshot checks).

    Only sessions created before memory snapshots were tracked are dropped:
    their memory.md copy predates the injection cap and may be arbitrarily
    large. Every newer session got a capped view (core.context, C11), so its
    pinned copy can never be oversized even when memory_chars (the raw file
    size at creation) was — a condensation since then is handled by a memory
    refresh on the next turn (see _on_memory_condensed), not by a reset.
    """
    if _get_memory_guard() is None:
        return None
    if "memory_fp" not in entry:
        return "legacy session without memory snapshot"
    return None


def _get_session_id(room_id) -> str | None:
    """Get session_id for a room (session key), respecting max age and memory snapshot."""
    entry = _room_sessions.get(room_id)
    if not entry:
        return None

    session_id = entry.get("session_id")
    if not session_id:
        return None

    # Check session age
    if config.SESSION_MAX_AGE_DAYS > 0:
        try:
            created = datetime.fromisoformat(entry["created_at"])
            age_days = (datetime.now(timezone.utc) - created).days
            if age_days >= config.SESSION_MAX_AGE_DAYS:
                logger.info(f"[Sessions] Room {room_id} session expired ({age_days}d old), starting fresh")
                _drop_session(room_id)
                return None
        except (KeyError, ValueError, TypeError):
            pass

    stale = _session_stale_reason(entry)
    if stale:
        logger.info(f"[Sessions] Room {room_id} session {session_id} dropped: {stale}")
        _drop_session(room_id)
        return None

    return session_id


def _set_session_id(room_id, session_id: str, memory_fp: str | None = None,
                    memory_chars: int | None = None, memory_refresh: bool = False) -> None:
    entry = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "memory_fp": memory_fp,
        "memory_chars": memory_chars,
    }
    if memory_refresh:
        entry["memory_refresh"] = True
    _room_sessions[room_id] = entry
    _save_room_sessions()


def _drop_session(key) -> None:
    _room_sessions.pop(key, None)
    _room_msg_count.pop(key, None)
    _room_last_hint.pop(key, None)
    _save_room_sessions()


def _on_memory_condensed(*_args, **_kwargs) -> None:
    """memory_guard callback: memory.md was rewritten.

    The memory snapshot is sent only in a session's FIRST-turn system message,
    and llm-api persists that message in the session history, so every cached
    session still holds the pre-condense copy. That copy is capped (C11), so it
    cannot overflow the context; it is only stale. Rather than dropping every
    room's conversation, flag each session so its next turn carries the
    current (condensed) memory.md — see _memory_refresh_block.
    """
    global _sessions_epoch
    _sessions_epoch += 1
    count = 0
    for entry in _room_sessions.values():
        if isinstance(entry, dict) and entry.get("session_id"):
            entry["memory_refresh"] = True
            count += 1
    if count:
        _save_room_sessions()
    logger.info(f"[Sessions] memory.md condensed — {count} cached room session(s) flagged for a memory refresh")


def _read_memory_view() -> str:
    """Current memory.md as injected into prompts (capped at MEMORY_MAX_CHARS)."""
    try:
        from core import context as _context
        view = getattr(_context, "memory_view", None)
        if callable(view):
            return view()
    except Exception:
        pass
    try:
        with open(os.path.join(config.DATA_DIR, "memory.md"), "r", encoding="utf-8", errors="replace") as f:
            return f.read()[:_memory_max_chars()]
    except Exception:
        return ""


def _memory_refresh_block() -> str:
    """Per-turn notice replacing the stale first-turn memory copy after a condense."""
    memory = _read_memory_view().strip()
    size = _memory_size()
    size_txt = f"{size:,} chars" if isinstance(size, int) else "size unknown"
    body = memory if memory else "(memory.md is empty)"
    return (
        "<system-reminder>\n"
        "memory.md was condensed since this conversation started. The \"Current Memory\" "
        "copy in the first system message is STALE — do not rely on it and never write it "
        f"back to memory.md. Current memory.md ({size_txt}):\n\n"
        f"{body}\n"
        "</system-reminder>"
    )


# ---------------------------------------------------------------------------
# Goal-mode persistence (sticky per-room; toggled by the /goal directive)
# ---------------------------------------------------------------------------

def _load_room_goal_mode() -> None:
    global _room_goal_mode
    try:
        with open(_GOAL_MODE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            _room_goal_mode = {_session_key_from_json(k): bool(v) for k, v in raw.items() if v}
        logger.info(f"[GoalMode] Loaded {len(_room_goal_mode)} room(s) in goal mode from disk")
    except FileNotFoundError:
        _room_goal_mode = {}
    except Exception as e:
        logger.warning(f"[GoalMode] Could not load goal-mode state: {e}")
        _room_goal_mode = {}


def _save_room_goal_mode() -> None:
    try:
        _atomic_write_json(_GOAL_MODE_FILE, {str(k): True for k, v in _room_goal_mode.items() if v})
    except Exception as e:
        logger.warning(f"[GoalMode] Could not save goal-mode state: {e}")


# Load on module import
_load_room_sessions()
_load_room_goal_mode()
_get_memory_guard()


# ---------------------------------------------------------------------------
# Per-room queue state
# ---------------------------------------------------------------------------

_seq_counter = 0


def _next_seq() -> int:
    global _seq_counter
    _seq_counter += 1
    return _seq_counter


@dataclass
class _QueuedMessage:
    msg_id: int | None
    content: str
    sender: str
    kind: str = "chat"                  # chat | compact | goal | clear (replayed @clear only)
    files: list = field(default_factory=list)
    reply_to_data: dict | None = None
    at: str = ""                        # HH:MM:SS (local)
    source: str = "messenger"           # messenger | catchup | external
    seq: int = field(default_factory=_next_seq)
    reacted: bool = False
    reaction_task: asyncio.Task | None = None
    # Names this message's attachments were uploaded to the LLM API under
    # (unique within the turn; set by _process_turn).
    upload_names: list = field(default_factory=list)


class _TurnContext:
    """Book-keeping for one running turn (one batch)."""

    def __init__(self, batch: list, generation: int, epoch: int):
        self.batch = batch
        self.generation = generation
        self.epoch = epoch
        self.cancel_reason: str | None = None     # "stop" | "interrupt" | "shutdown"
        # True once the final answer is being written — too late to interrupt
        # (re-queueing would answer the same batch twice).
        self.finalized = False
        # True while a cluster delegation is being submitted — not
        # interruptible (a re-queue could submit the same task twice).
        self.delegating = False
        self.session_id: str | None = None
        self.memory_fp: str | None = None
        self.memory_chars: int | None = None

    @property
    def senders(self) -> set:
        return {m.sender for m in self.batch}


class RoomState:
    """Per-room FIFO queue + single worker task.

    key       — session key ("room id" for Messenger rooms, "ext_<source>" for
                external webhooks)
    room_id   — Messenger room the replies are posted to
    task      — the room's worker task (drains the queue)
    turn_task — the currently running turn (one batch); @stop cancels this
    generation— bumped by @clear so an in-flight turn never re-saves a session
    """

    def __init__(self, key, room_id: int, external: bool = False):
        self.key = key
        self.room_id = room_id
        self.external = external
        self.queue: deque = deque()
        self.task: asyncio.Task | None = None
        self.turn_task: asyncio.Task | None = None
        self.turn_ctx: _TurnContext | None = None
        # The batch taken off the queue whose answer is not settled yet. Kept
        # (and persisted) when the turn is cut short by shutdown, so the next
        # start replays it; cleared once the turn ends any other way.
        self.inflight_ctx: _TurnContext | None = None
        self.generation = 0
        self.last_arrival = 0.0
        self.last_accepted: tuple | None = None   # (sender, monotonic) for continuation window
        self.queue_full_notified = False

    @property
    def pending(self) -> int:
        return len(self.queue)

    @property
    def busy(self) -> bool:
        return self.turn_task is not None and not self.turn_task.done()


_rooms: dict = {}

# LRU of processed/queued message ids so a message is never handled twice
# (webhook redelivery, catch-up replay racing the live webhook, ...).
_SEEN_MAX = 2000
_seen_msg_ids: "OrderedDict[tuple, None]" = OrderedDict()

# Group-room continuation: an unmentioned attachment from the same sender
# shortly after an accepted (mentioned) message is treated as part of it.
_CONTINUATION_WINDOW_SECONDS = 15.0

_FOLLOWUP_NOTICE_QUEUE_FULL = "대기 중인 메시지가 너무 많아 이 메시지는 처리하지 못했어요. 잠시 후 다시 보내주세요."


def _get_state(key, room_id: int | None = None, external: bool = False) -> RoomState:
    state = _rooms.get(key)
    if state is None:
        state = RoomState(key, room_id if room_id is not None else key, external=external)
        _rooms[key] = state
    elif room_id is not None:
        state.room_id = room_id
    return state


def _mark_seen(room_id, msg_id) -> bool:
    """Return True if (room, msg_id) was already seen; record it otherwise."""
    if msg_id is None:
        return False
    # Normalise types: the live webhook and the catch-up replay may disagree
    # on int vs str ids, which would defeat the dedupe.
    key = (str(room_id), str(msg_id))
    if key in _seen_msg_ids:
        _seen_msg_ids.move_to_end(key)
        return True
    _seen_msg_ids[key] = None
    while len(_seen_msg_ids) > _SEEN_MAX:
        _seen_msg_ids.popitem(last=False)
    return False


# ---------------------------------------------------------------------------
# Queue persistence (data/room_queue.json) + graceful shutdown
# ---------------------------------------------------------------------------
#
# File shape: {"version": 1, "saved_at": iso, "rooms": {"<room_id>": [msg_id, ...]}}
# Each room lists the msg_ids of its in-flight batch (not yet answered) and of
# every message still waiting in its queue, in order. External-webhook items
# carry no msg_id and are not persisted. Ids loaded from a previous run stay in
# the file until catch-up re-enqueues (or discards) them.

_QUEUE_FILE = os.path.join(config.DATA_DIR, "room_queue.json")
_QUEUE_SAVE_DELAY_SECONDS = 0.5

# room_id -> [msg_id, ...] restored from the previous run, not yet replayed.
_persisted_pending: dict = {}
_queue_save_handle: asyncio.TimerHandle | None = None
_queue_last_written: str | None = None
# Set by shutdown_room_workers(): no new batches start; arrivals are only queued
# (and persisted) for the next run.
_shutting_down = False


def _norm_msg_id(msg_id):
    try:
        return int(msg_id)
    except (TypeError, ValueError):
        return msg_id


def _queue_snapshot() -> dict:
    rooms: dict = {}

    def _add(room_id, msg_id) -> None:
        if msg_id is None or room_id is None:
            return
        ids = rooms.setdefault(str(room_id), [])
        mid = _norm_msg_id(msg_id)
        if mid not in ids:
            ids.append(mid)

    for room_id, ids in _persisted_pending.items():
        for mid in ids:
            _add(room_id, mid)
    for state in list(_rooms.values()):
        if state.external:
            continue
        inflight = state.inflight_ctx.batch if state.inflight_ctx is not None else []
        for m in list(inflight) + list(state.queue):
            _add(state.room_id, m.msg_id)
    return {k: v for k, v in rooms.items() if v}


def _save_queue_now() -> None:
    """Write room_queue.json now (skipped when nothing changed)."""
    global _queue_save_handle, _queue_last_written
    if _queue_save_handle is not None:
        _queue_save_handle.cancel()
        _queue_save_handle = None
    snapshot = _queue_snapshot()
    fingerprint = json.dumps(snapshot, sort_keys=True)
    if fingerprint == _queue_last_written:
        return
    try:
        _atomic_write_json(_QUEUE_FILE, {
            "version": 1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "rooms": snapshot,
        })
        _queue_last_written = fingerprint
    except Exception as e:
        logger.warning(f"[Queue] Could not save queue state: {e}")


def _schedule_queue_save() -> None:
    """Coalesce queue-state writes: at most one per _QUEUE_SAVE_DELAY_SECONDS."""
    global _queue_save_handle
    if _queue_save_handle is not None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _save_queue_now()
        return
    _queue_save_handle = loop.call_later(_QUEUE_SAVE_DELAY_SECONDS, _save_queue_now)


def _load_room_queue() -> None:
    global _persisted_pending, _queue_last_written
    try:
        with open(_QUEUE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        _persisted_pending = {}
        return
    except Exception as e:
        logger.warning(f"[Queue] Could not load queue state: {e}")
        _persisted_pending = {}
        return
    rooms = raw.get("rooms") if isinstance(raw, dict) else None
    pending: dict = {}
    for k, ids in (rooms or {}).items():
        if not isinstance(ids, list):
            continue
        clean = []
        for mid in ids:
            mid = _norm_msg_id(mid)
            if mid is not None and mid not in clean:
                clean.append(mid)
        if clean:
            pending[_session_key_from_json(k)] = clean
    _persisted_pending = pending
    _queue_last_written = json.dumps({str(k): v for k, v in pending.items()}, sort_keys=True)
    total = sum(len(v) for v in pending.values())
    if total:
        logger.info(f"[Queue] {total} unanswered message(s) in {len(pending)} room(s) persisted by the previous run")


def persisted_queue() -> dict:
    """{room_id: [msg_id, ...]} left queued / in flight by the previous run and
    not replayed yet (used by hoonbot.py's catch-up)."""
    return {k: list(v) for k, v in _persisted_pending.items()}


def forget_persisted(room_id, msg_id) -> None:
    """Drop one restored id (replayed, deleted, or no longer replayable)."""
    for key in list(_persisted_pending):
        if str(key) != str(room_id):
            continue
        ids = [m for m in _persisted_pending[key] if str(m) != str(msg_id)]
        if len(ids) == len(_persisted_pending[key]):
            return
        if ids:
            _persisted_pending[key] = ids
        else:
            _persisted_pending.pop(key, None)
        _schedule_queue_save()
        return


async def shutdown_room_workers(timeout: float = 15.0) -> None:
    """Graceful shutdown of the per-room queues.

    Stops new batches from starting, lets a turn that is already writing its
    final answer (or submitting a cluster delegation) finish, cancels every
    other running turn (its draft is settled with a restart marker), and
    flushes data/room_queue.json so the next start replays the unanswered
    messages. Never raises.
    """
    global _shutting_down
    _shutting_down = True
    try:
        workers = []
        turns = []
        for state in list(_rooms.values()):
            task = state.task
            if task is None or task.done():
                continue
            workers.append(task)
            # The worker only cancels its turn and exits; the turn's shielded
            # _finalize_cancelled is spawned later (after the httpx stream's
            # async-with exit awaits), so wait on the turn itself too.
            if state.turn_task is not None and not state.turn_task.done():
                turns.append(state.turn_task)
            ctx = state.turn_ctx
            if ctx is not None and (ctx.finalized or ctx.delegating):
                continue  # finishes on its own; the worker then stops (see _room_worker)
            if ctx is not None and ctx.cancel_reason is None:
                ctx.cancel_reason = "shutdown"
            task.cancel()
        deadline = time.monotonic() + max(0.0, timeout)
        if workers:
            _, pending = await asyncio.wait(workers, timeout=max(0.1, deadline - time.monotonic()))
            for task in pending:
                for state in list(_rooms.values()):
                    ctx = state.turn_ctx
                    if state.task is task and ctx is not None and ctx.cancel_reason is None:
                        ctx.cancel_reason = "shutdown"
                task.cancel()
            if pending:
                await asyncio.wait(pending, timeout=2.0)
        live_turns = [t for t in turns if not t.done()]
        if live_turns:
            await asyncio.wait(live_turns, timeout=max(0.5, deadline - time.monotonic()))
        # Shielded final writes / stop markers run as background tasks.
        background = [t for t in list(_background_tasks) if not t.done()]
        if background:
            await asyncio.wait(background, timeout=max(0.5, deadline - time.monotonic()))
        stats = _queue_snapshot()
        total = sum(len(v) for v in stats.values())
        if total:
            logger.info(f"[Queue] Shutdown: {total} unanswered message(s) saved for replay on next start")
    except Exception as exc:
        logger.warning(f"[Queue] Shutdown of room workers failed: {exc}")
    finally:
        _save_queue_now()


_load_room_queue()


def _followup_policy() -> str:
    return str(getattr(config, "FOLLOWUP_POLICY", "queue") or "queue").strip().lower()


def _batch_max() -> int:
    return max(1, int(getattr(config, "BATCH_MAX_MESSAGES", 20) or 20))


def _queue_max() -> int:
    return max(1, int(getattr(config, "ROOM_QUEUE_MAX", 50) or 50))


def _enqueue(state: RoomState, item: _QueuedMessage) -> bool:
    if len(state.queue) >= _queue_max():
        logger.warning(f"[Queue] Room {state.key}: queue full ({len(state.queue)}), dropping message {item.msg_id}")
        if not state.queue_full_notified and not state.external:
            state.queue_full_notified = True
            _spawn(messenger.send_message(state.room_id, _FOLLOWUP_NOTICE_QUEUE_FULL, reply_to_id=item.msg_id))
        return False
    state.queue_full_notified = False

    busy = state.busy
    ctx = state.turn_ctx
    if (
        busy
        and ctx is not None
        and not state.external
        and item.kind == "chat"
        and _followup_policy() == "interrupt"
        and ctx.cancel_reason is None
        and not ctx.finalized
        and not ctx.delegating
        and all(m.kind == "chat" for m in ctx.batch)
        and ctx.senders == {item.sender}
    ):
        # Old behaviour, opt-in: a same-sender follow-up cancels the running
        # turn; the worker re-queues the interrupted batch ahead of this one so
        # both are answered together in the next turn.
        logger.info(f"[Queue] Room {state.key}: same-sender follow-up interrupts running turn")
        ctx.cancel_reason = "interrupt"
        state.turn_task.cancel()
        sid = ctx.session_id or _peek_session_id(state.key)
        if sid:
            _spawn(_post_session_action(sid, "stop", timeout=10.0))
        busy = False

    state.queue.append(item)
    state.last_arrival = time.monotonic()
    if _shutting_down:
        # No new turns during shutdown — persist it for the next run instead.
        logger.info(f"[Queue] Room {state.key}: message {item.msg_id} kept for the next run (shutting down)")
        if not state.external:
            _save_queue_now()
        return True
    if busy and item.msg_id is not None and not state.external:
        item.reaction_task = _spawn(_add_waiting_reaction(state.room_id, item))
        logger.info(f"[Queue] Room {state.key}: message {item.msg_id} queued for next turn (pending={state.pending})")
    if not state.external:
        _schedule_queue_save()
    _ensure_worker(state)
    return True


def _ensure_worker(state: RoomState) -> None:
    if _shutting_down:
        return
    if state.task is None or state.task.done():
        state.task = asyncio.create_task(_room_worker(state))


def _take_batch(state: RoomState) -> list:
    """Pop the next batch: a lone directive item, or consecutive chat items."""
    first = state.queue.popleft()
    if first.kind != "chat":
        return [first]
    batch = [first]
    limit = _batch_max()
    while state.queue and len(batch) < limit and state.queue[0].kind == "chat":
        batch.append(state.queue.popleft())
    return batch


async def _room_worker(state: RoomState) -> None:
    me = asyncio.current_task()
    try:
        while True:
            # Debounce: wait until the burst has been quiet for DEBOUNCE_SECONDS.
            delay = state.last_arrival + float(config.DEBOUNCE_SECONDS) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
                continue
            if not state.queue or _shutting_down:
                break

            batch = _take_batch(state)
            for m in batch:
                if m.reacted or m.reaction_task is not None:
                    _spawn(_remove_waiting_reaction(state.room_id, m))

            ctx = _TurnContext(batch, state.generation, _sessions_epoch)
            state.inflight_ctx = ctx
            turn = asyncio.create_task(_run_batch(state, ctx))
            state.turn_task = turn
            state.turn_ctx = ctx
            try:
                # asyncio.wait never raises when the *turn* is cancelled (@stop);
                # only a cancellation of the worker itself propagates here.
                await asyncio.wait({turn})
            except asyncio.CancelledError:
                turn.cancel()
                # Worker cancelled (shutdown): the batch stays persisted for
                # replay unless its answer was already being written.
                if ctx.finalized and state.inflight_ctx is ctx:
                    state.inflight_ctx = None
                raise
            finally:
                if state.turn_task is turn:
                    state.turn_task = None
                    state.turn_ctx = None

            if not turn.cancelled() and turn.exception() is not None:
                logger.error(f"[Queue] Room {state.key}: turn crashed: {turn.exception()!r}")
            if ctx.cancel_reason == "interrupt" and not ctx.finalized:
                state.queue.extendleft(reversed([m for m in ctx.batch if m.kind == "chat"]))
            if state.inflight_ctx is ctx and not (ctx.cancel_reason == "shutdown" and not ctx.finalized):
                state.inflight_ctx = None
            if not state.external:
                _schedule_queue_save()
    finally:
        if state.task is me:
            state.task = None


# ---------------------------------------------------------------------------
# Directive classification
# ---------------------------------------------------------------------------

_LEADING_TOKEN_RE = re.compile(r"^([@/][A-Za-z_]+)\b")


def _strip_mention(content: str) -> str:
    bot = config.MESSENGER_BOT_NAME or ""
    if not bot:
        return (content or "").strip()
    return re.sub(rf"@{re.escape(bot)}(?![\w-])", "", content or "", flags=re.IGNORECASE).strip()


def _classify(content: str) -> tuple:
    """Return (kind, cleaned_content). kind: stop | clear | compact | goal | chat.

    Mentions are stripped the same way in every room (home included) and the
    directive is the LEADING token only — "@stop" in the middle of a sentence
    is ordinary text.
    """
    stripped = _strip_mention(content)
    lead = stripped.lstrip(" \t\r\n,:;")
    m = _LEADING_TOKEN_RE.match(lead)
    tok = m.group(1).lower() if m else ""
    if tok == "@stop":
        return "stop", lead
    if tok == "@clear":
        return "clear", lead
    if tok == "@compact":
        return "compact", lead
    if _GOAL_DIRECTIVE_RE.match(lead):
        return "goal", lead
    return "chat", stripped


def _format_clock(created_at) -> str:
    """HH:MM:SS (local) from Messenger's createdAt, falling back to now."""
    dt = None
    try:
        if isinstance(created_at, (int, float)):
            ts = created_at / 1000.0 if created_at > 1e11 else float(created_at)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(created_at, str) and created_at.strip():
            s = created_at.strip().replace("Z", "+00:00")
            if " " in s and "T" not in s:
                s = s.replace(" ", "T", 1)
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)  # sqlite datetime('now') is UTC
    except Exception:
        dt = None
    if dt is None:
        return datetime.now().strftime("%H:%M:%S")
    return dt.astimezone().strftime("%H:%M:%S")


def _ingest_message(room_id: int, content: str, sender_name: str, msg_id: int | None,
                    file_infos: list | None, reply_to_data: dict | None,
                    source: str = "messenger", created_at=None) -> str:
    """Classify one Messenger message and route it (immediate directive or queue)."""
    if _mark_seen(room_id, msg_id):
        logger.info(f"[Queue] Room {room_id}: duplicate message {msg_id} ignored ({source})")
        return "duplicate"

    files = list(file_infos or [])
    kind, clean = _classify(content or "")
    if kind == "chat" and not clean:
        if files:
            names = ", ".join(fi.get("name", "file") for fi in files)
            clean = f"[Attachments: {names}]"
        else:
            return "empty"

    state = _get_state(room_id, room_id=room_id)
    log_prefix = f"[Room {room_id}] [{sender_name}]"

    if kind == "stop":
        if source == "catchup":
            logger.info(f"{log_prefix} Ignoring replayed @stop")
            return "ignored"
        _spawn(_handle_stop(state, sender_name))
        return "stop"
    if kind == "clear" and source != "catchup" and not state.queue:
        _handle_clear(state, sender_name)
        return "clear"
    # A replayed @clear — or a live one while earlier messages are still
    # waiting in the queue — is queued like any other message, so it runs
    # AFTER the messages sent before it instead of jumping ahead and wiping
    # the session they are about to be answered in (see _run_batch).

    item = _QueuedMessage(
        msg_id=msg_id,
        content=clean,
        sender=sender_name,
        kind=kind,
        files=files,
        reply_to_data=reply_to_data if isinstance(reply_to_data, dict) else None,
        at=_format_clock(created_at),
        source=source,
    )
    logger.info(f"{log_prefix} Queued {kind} message {msg_id} ({source}): {clean[:80]!r}")
    return "queued" if _enqueue(state, item) else "dropped"


def _message_file_infos(data: dict) -> tuple[list, str | None]:
    """Build ``[{"url","name","type"}]`` for a Messenger message (live webhook
    payload or a ``GET /api/messages`` row — same shape).

    Uses ``attachments`` when present, else the legacy single ``fileUrl`` /
    ``fileName`` of an image/file message. Returns ``(file_infos, missing)``
    where *missing* is ``"image"``/``"file"`` when such a message has no
    usable URL at all (the caller decides how to report it), else ``None``.
    """
    file_infos: list = []
    raw_attachments = data.get("attachments") if isinstance(data.get("attachments"), list) else []
    for attachment in raw_attachments:
        if not isinstance(attachment, dict):
            continue
        file_url = str(attachment.get("fileUrl") or "").strip()
        if not file_url:
            continue
        file_infos.append({
            "url": file_url,
            "name": attachment.get("fileName") or file_url.rsplit("/", 1)[-1] or "attachment",
            "type": attachment.get("type") or "file",
        })
    msg_type = data.get("type", "text")
    if not file_infos and msg_type in ("image", "file"):
        file_url = str(data.get("fileUrl") or "").strip()
        if not file_url:
            return [], msg_type
        file_infos.append({
            "url": file_url,
            "name": data.get("fileName") or msg_type,
            "type": msg_type,
        })
    return file_infos, None


def _attachment_label(file_infos: list) -> str:
    """Placeholder content for an attachment-only message."""
    label = "Attachments" if len(file_infos) > 1 else ("Image" if file_infos[0].get("type") == "image" else "File")
    names = ", ".join(str(fi.get("name") or "file") for fi in file_infos)
    return f"[{label}: {names}]"


# ---------------------------------------------------------------------------
# Webhook entry point
# ---------------------------------------------------------------------------

@router.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    event = payload.get("event")
    data = payload.get("data", {}) or {}
    room_id = payload.get("roomId")

    # --- Handle message edits: update a still-queued message in place ---
    if event == "message_edited":
        mid = data.get("messageId") or data.get("id")
        logger.info(f"[Webhook] Message edited in room {room_id}: id={mid}")
        _update_queued(room_id, mid, data.get("content"))
        return {"ok": True}

    # --- Handle message deletes: drop a still-queued message ---
    if event == "message_deleted":
        mid = data.get("messageId") or data.get("id")
        logger.info(f"[Webhook] Message deleted in room {room_id}: id={mid}")
        _remove_queued(room_id, mid)
        if room_id is not None and mid is not None:
            forget_persisted(room_id, mid)
        return {"ok": True}

    if event != "new_message":
        return {"ok": True}

    content = (data.get("content") or "").strip()
    msg_type = data.get("type", "text")
    sender_name = data.get("senderName") or data.get("sender_name", "")
    is_bot = data.get("isBot") or data.get("is_bot", False)
    msg_id = data.get("id")

    # Ignore bot messages
    if sender_name == config.MESSENGER_BOT_NAME or is_bot:
        return {"ok": True}

    # Handle file/image messages — download and forward to LLM API
    file_infos, missing_kind = _message_file_infos(data)
    if missing_kind:
        logger.warning(f"[Webhook] {missing_kind.capitalize()} message {msg_id} has no fileUrl; refusing to process it")
        is_home = room_id == config.MESSENGER_HOME_ROOM_ID
        mention_tag = f"@{config.MESSENGER_BOT_NAME}"
        is_solo = await messenger.is_bot_solo_room(room_id) if room_id is not None else False
        if room_id is not None and (is_home or is_solo or mention_tag.lower() in content.lower()):
            notice = (
                "이미지 파일을 찾을 수 없어 처리할 수 없어요. 다시 업로드해주세요."
                if missing_kind == "image"
                else "파일을 찾을 수 없어 처리할 수 없어요. 다시 업로드해주세요."
            )
            _spawn(messenger.send_message(room_id, notice, reply_to_id=msg_id))
        return {"ok": False, "error": f"{missing_kind.capitalize()} message missing fileUrl"}
    if not file_infos and msg_type != "text":
        return {"ok": True}
    if file_infos and not content:
        content = _attachment_label(file_infos)

    if not content or room_id is None:
        return {"ok": True}

    # Respond unconditionally in:
    #   - the configured home room, OR
    #   - any room where the bot is alone with a single user (2 members total).
    #     This covers 1-on-1 DMs and 2-member group rooms alike. Adding a third
    #     member reverts the room to @mention-only once the cache refreshes.
    # Other rooms still require an explicit @mention — except an unmentioned
    # attachment sent by the same person right after a mentioned message
    # (continuation window), which is treated as part of that request.
    is_home = room_id == config.MESSENGER_HOME_ROOM_ID
    is_solo = await messenger.is_bot_solo_room(room_id)
    mention_tag = f"@{config.MESSENGER_BOT_NAME}"
    if not is_home and not is_solo and mention_tag.lower() not in content.lower():
        state = _rooms.get(room_id)
        last = state.last_accepted if state else None
        is_continuation = bool(
            file_infos
            and last
            and last[0] == sender_name
            and time.monotonic() - last[1] <= _CONTINUATION_WINDOW_SECONDS
        )
        if not is_continuation:
            return {"ok": True}
        logger.info(f"[Webhook] Room {room_id}: unmentioned attachment from {sender_name!r} accepted as continuation")

    # Mark as read (best-effort)
    if msg_id:
        _spawn(messenger.mark_read(room_id, [msg_id]))

    reply_to_data = data.get("replyTo")
    status = _ingest_message(
        room_id, content, sender_name, msg_id, file_infos, reply_to_data,
        source="messenger", created_at=data.get("createdAt") or data.get("created_at"),
    )
    if status in ("queued", "stop", "clear"):
        _get_state(room_id, room_id=room_id).last_accepted = (sender_name, time.monotonic())
    return {"ok": True}


def _find_queued(room_id, msg_id):
    state = _rooms.get(room_id)
    if state is None or msg_id is None:
        return None, None
    for item in state.queue:
        if item.msg_id == msg_id:
            return state, item
    return state, None


def _update_queued(room_id, msg_id, new_content) -> None:
    state, item = _find_queued(room_id, msg_id)
    if item is None or item.kind != "chat" or not isinstance(new_content, str):
        return
    kind, clean = _classify(new_content.strip())
    if kind != "chat" or not clean:
        return
    item.content = clean
    logger.info(f"[Queue] Room {room_id}: queued message {msg_id} updated after edit")


def _remove_queued(room_id, msg_id) -> None:
    state, item = _find_queued(room_id, msg_id)
    if item is None:
        return
    try:
        state.queue.remove(item)
    except ValueError:
        return
    if not state.external:
        _schedule_queue_save()
    logger.info(f"[Queue] Room {room_id}: queued message {msg_id} removed after delete")


# ---------------------------------------------------------------------------
# Backward-compatible entry points
# ---------------------------------------------------------------------------

def _schedule_debounced(room_id: int, content: str, sender_name: str, msg_id: int | None = None,
                        file_infos: list | None = None, reply_to_data: dict | None = None) -> None:
    """Compat shim — the debounce now lives in the per-room queue worker."""
    _ingest_message(room_id, content, sender_name, msg_id, file_infos, reply_to_data)


async def process_message(room_id: int, content: str, sender_name: str, reply_to_id: int | None = None,
                          _is_retry: bool = False, file_infos: list | None = None,
                          reply_to_data: dict | None = None) -> None:
    """Catch-up / missed-message replay entry point (signature kept for callers).

    Routes through the same per-room queue as live webhooks (with msg_id dedupe)
    instead of running a turn directly, so a replayed message can never race or
    duplicate a live one. ``reply_to_id`` is the replayed message's own id.
    """
    _ingest_message(room_id, content or "", sender_name, reply_to_id, file_infos, reply_to_data,
                    source="catchup")


async def enqueue_catchup_message(room_id: int, msg: dict, *, continuation: bool = False,
                                  accepted: bool = False) -> str:
    """Catch-up hook used by hoonbot.py: queue one missed Messenger message.

    Applies the same room gating as live webhooks (home room / solo room /
    explicit @mention, or an attachment continuing the same sender's accepted
    message when *continuation*) and the same msg_id dedupe, so a replayed
    message is never answered twice nor answered where a live one would be
    ignored. Attachments are rebuilt exactly like the live webhook does
    (``attachments`` list, else ``fileUrl``/``fileName``), so attachment-only
    messages replay too.

    *accepted*: the message was already accepted before a restart (it came
    from ``data/room_queue.json``) — gating is skipped and its persisted entry
    is dropped once it has been handed back to the queue.
    """
    if not isinstance(msg, dict):
        return "ignored"
    msg_id = msg.get("id")
    try:
        content = (msg.get("content") or "").strip()
        sender_name = msg.get("senderName") or msg.get("sender_name") or "unknown"
        if msg.get("isDeleted") or msg.get("isBot") or sender_name == config.MESSENGER_BOT_NAME:
            return "ignored"
        file_infos, missing_kind = _message_file_infos(msg)
        if missing_kind:
            logger.warning(f"[Catchup] Room {room_id}: {missing_kind} message {msg_id} has no fileUrl; skipped")
            return "ignored"
        if not file_infos and (msg.get("type") or "text") != "text":
            return "ignored"
        if not file_infos and not content:
            return "ignored"
        if file_infos and not content:
            content = _attachment_label(file_infos)

        if not accepted:
            is_home = room_id == config.MESSENGER_HOME_ROOM_ID
            if not is_home:
                try:
                    is_solo = await messenger.is_bot_solo_room(room_id)
                except Exception:
                    is_solo = False
                mentioned = f"@{config.MESSENGER_BOT_NAME}".lower() in content.lower()
                if not is_solo and not mentioned and not (continuation and file_infos):
                    return "ignored"
        status = _ingest_message(
            room_id, content, sender_name, msg_id, file_infos, msg.get("replyTo"),
            source="catchup", created_at=msg.get("createdAt") or msg.get("created_at"),
        )
        if status in ("queued", "clear"):
            _get_state(room_id, room_id=room_id).last_accepted = (sender_name, time.monotonic())
        return status
    finally:
        if accepted and msg_id is not None:
            forget_persisted(room_id, msg_id)


# ---------------------------------------------------------------------------
# Immediate directives (@stop / @clear) and in-queue directives
# ---------------------------------------------------------------------------

async def _post_session_action(session_id: str, action: str, timeout: float) -> dict | None:
    """POST /api/chat/sessions/{id}/{action} on the shared LLM API client."""
    resp = await _llm_client().post(
        f"{config.LLM_API_URL}/api/chat/sessions/{session_id}/{action}",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return None


async def _stop_state(state: RoomState, log_prefix: str, only_if_active: bool = False) -> int:
    """Drain *state*'s queue, cancel its running turn and ask the LLM API to
    stop its session. Returns the number of dropped queued items.

    only_if_active: skip the server-side stop when nothing was running or
    queued (used for the external-webhook sessions swept by a room @stop).
    """
    dropped = 0
    while state.queue:
        item = state.queue.popleft()
        dropped += 1
        if item.reacted or item.reaction_task is not None:
            _spawn(_remove_waiting_reaction(state.room_id, item))
    if dropped and not state.external:
        _schedule_queue_save()

    turn = state.turn_task
    ctx = state.turn_ctx
    active = turn is not None and not turn.done() and turn is not asyncio.current_task()
    if active:
        if ctx is not None:
            ctx.cancel_reason = "stop"
        turn.cancel()
    if only_if_active and not active:
        return dropped

    session_id = (ctx.session_id if ctx else None) or _peek_session_id(state.key)
    if session_id:
        try:
            await _post_session_action(session_id, "stop", timeout=10.0)
        except Exception as e:
            logger.warning(f"{log_prefix} @stop request to LLM API failed: {e}")
    return dropped


async def _handle_stop(state: RoomState, sender_name: str) -> None:
    """@stop — cancel the room's running turn via its handle (never this task).

    Also stops external-webhook sessions ("ext_<source>") whose replies stream
    into this room: they run in their own queue, but the user can only see
    (and @stop) them here.
    """
    log_prefix = f"[Room {state.key}] [{sender_name}]"
    dropped = await _stop_state(state, log_prefix)
    if not state.external:
        for other in list(_rooms.values()):
            if other is state or not other.external or str(other.room_id) != str(state.room_id):
                continue
            dropped += await _stop_state(other, f"[Room {other.key}] [{sender_name}]", only_if_active=True)
    _fire_stop_typing(state.room_id)
    msg = "중지했습니다."
    if dropped:
        msg += f" (대기 중이던 메시지 {dropped}개도 취소했어요.)"
    try:
        await messenger.send_message(state.room_id, msg)
    except Exception as e:
        logger.warning(f"{log_prefix} Could not send @stop confirmation: {e}")
    logger.info(f"{log_prefix} Stopped by @stop command (dropped {dropped} queued)")


def _handle_clear(state: RoomState, sender_name: str) -> None:
    """@clear — wipe the room session immediately (outside the queue)."""
    state.generation += 1  # an in-flight turn must not re-save its session
    _drop_session(state.key)
    # A reset also exits goal mode.
    if _room_goal_mode.pop(state.key, None):
        _save_room_goal_mode()
    _spawn(messenger.send_message(state.room_id, "컨텍스트가 초기화되었습니다. 새 대화를 시작하세요."))
    logger.info(f"[Room {state.key}] [{sender_name}] Session cleared by @clear command")


async def _handle_compact(state: RoomState, item: _QueuedMessage) -> None:
    room_id = state.room_id
    log_prefix = f"[Room {state.key}] [{item.sender}]"
    session_id = _get_session_id(state.key)
    if not session_id:
        await messenger.send_message(room_id, "압축할 대화가 없습니다. 먼저 대화를 시작하세요.")
        logger.info(f"{log_prefix} @compact ignored — no active session")
        return
    _fire_typing(room_id, "요약 중...")
    try:
        # Summarising a long (up to 1M-token) session is a slow LLM call.
        try:
            data = await _post_session_action(session_id, "compact", timeout=600.0) or {}
        except (asyncio.CancelledError, httpx.TransportError):
            # @stop / client timeout / dropped connection: llm-api does not
            # abort a running compaction and keeps holding the session lease.
            # Make the next chat turns on this session wait it out instead of
            # failing on 409 after the short retry budget.
            _compact_abandoned[session_id] = time.monotonic() + _COMPACT_ABANDON_WAIT_SECONDS
            logger.warning(f"{log_prefix} @compact abandoned client-side — server may still hold session {session_id}")
            raise
        _compact_abandoned.pop(session_id, None)
        if data.get("success"):
            orig = data.get("original_count")
            new = data.get("new_count")
            _room_msg_count[state.key] = 0
            await messenger.send_message(
                room_id,
                f"컨텍스트가 요약되었습니다. ({orig}개 → {new}개 메시지)"
            )
            logger.info(f"{log_prefix} Session compacted {orig} → {new} messages by @compact")
        else:
            await messenger.send_message(room_id, f"압축 실패: {data.get('message', '알 수 없는 오류')}")
            logger.warning(f"{log_prefix} @compact returned failure: {data.get('message')}")
    except Exception as e:
        await messenger.send_message(room_id, f"압축 중 오류 발생: {str(e)[:500]}")
        logger.error(f"{log_prefix} @compact failed: {e}")
    finally:
        _fire_stop_typing(room_id)


async def _run_batch(state: RoomState, ctx: _TurnContext) -> None:
    """Run one queued batch: an in-queue directive or a chat turn."""
    head = ctx.batch[0]
    log_prefix = f"[Room {state.key}] [{head.sender}]"
    try:
        if head.kind == "clear":
            # Replayed (catch-up) @clear, or a live one that arrived while
            # earlier messages were queued — run in queue order. A live @clear
            # with an empty queue is handled immediately in _ingest_message.
            _handle_clear(state, head.sender)
            return

        if head.kind == "compact":
            await _handle_compact(state, head)
            return

        if head.kind == "goal":
            # /goal: toggle sticky goal mode. `/goal off` disables; `/goal <task>`
            # (or a bare `/goal`) enables and, when a task follows, processes it
            # in goal mode.
            goal_match = _GOAL_DIRECTIVE_RE.match(head.content)
            remainder = goal_match.group(1).strip() if goal_match else ""
            if remainder.lower() in _GOAL_OFF_WORDS:
                if _room_goal_mode.pop(state.key, None):
                    _save_room_goal_mode()
                await messenger.send_message(state.room_id, "목표 모드를 껐어요.")
                logger.info(f"{log_prefix} Goal mode disabled by /goal off")
                return
            _room_goal_mode[state.key] = True
            _save_room_goal_mode()
            if not remainder:
                await messenger.send_message(
                    state.room_id,
                    "목표 모드를 켰어요. 완료할 목표를 알려주세요. 목표를 끝까지 추적해서 완수할게요. (끄려면 /goal off)",
                )
                logger.info(f"{log_prefix} Goal mode enabled (no task provided)")
                return
            # A task followed the directive — process it in goal mode.
            head.content = remainder
            head.kind = "chat"
            ctx.batch = [head]
            logger.info(f"{log_prefix} Goal mode enabled with task: {remainder[:80]!r}")

        if not state.external:
            # Cluster delegation directives (@node task, @all-slaves task, ...)
            # are classified per message so they never get mixed into a batch.
            # A handled message leaves ctx.batch immediately, so an "interrupt"
            # follow-up re-queues only messages not yet handled (never
            # re-submits a delegated task). The submission itself is not
            # interruptible: a cancel mid-POST could not tell whether the
            # master already accepted the task.
            for m in list(ctx.batch):
                ctx.delegating = True
                try:
                    delegation = await try_submit_from_message(
                        m.content, m.sender, state.room_id, session_id=_peek_session_id(state.key)
                    )
                except Exception as exc:
                    ctx.batch = [x for x in ctx.batch if x is not m]
                    logger.warning(f"[Room {state.key}] [{m.sender}] Cluster delegation failed: {exc}")
                    await messenger.send_message(state.room_id, f"Cluster delegation failed: {exc}", reply_to_id=m.msg_id)
                    continue
                finally:
                    ctx.delegating = False
                if delegation:
                    ctx.batch = [x for x in ctx.batch if x is not m]
                    await messenger.send_message(state.room_id, delegation["message"], reply_to_id=m.msg_id)
                    logger.info(f"[Room {state.key}] [{m.sender}] Delegated cluster task: {delegation}")
                    continue
            if not ctx.batch:
                return

        await _process_turn(state, ctx)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"{log_prefix} Batch failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Message processing — prompt assembly
# ---------------------------------------------------------------------------

_BATCH_INTRO = "다음은 짧은 시간에 들어온 {n}개의 메시지입니다. 각각을 구분해서 모두 답하세요 (메시지 id/보낸 사람 표시)."
_REPLY_QUOTE_MAX_CHARS = 1000
_REPLY_QUOTE_MAX_CHARS_BOT = 4000


def _reply_quote(reply_to_data: dict | None) -> str:
    if not reply_to_data or reply_to_data.get("isDeleted") or not reply_to_data.get("content"):
        return ""
    reply_sender = reply_to_data.get("senderName", "unknown")
    is_bot_reply = bool(reply_to_data.get("isBot")) or reply_sender == config.MESSENGER_BOT_NAME
    limit = _REPLY_QUOTE_MAX_CHARS_BOT if is_bot_reply else _REPLY_QUOTE_MAX_CHARS
    text = str(reply_to_data["content"])
    if len(text) > limit:
        text = text[:limit] + "…"
    return f"> {reply_sender}: \"{text}\""


def _build_message_header(room_name: str, room_id: int, is_group: bool, sender_name: str, reply_to_data: dict | None) -> str:
    """Build the bracketed context header prepended to every LLM message."""
    room_type = "group" if is_group else "DM"
    header = f"[Room: {room_name} (id:{room_id}, {room_type}) | From: {sender_name}]"
    quote = _reply_quote(reply_to_data)
    if quote:
        header += f"\n{quote}"
    return header


_MESSAGE_TAG_RE = re.compile(r"<(\s*/?\s*message)", re.IGNORECASE)


def _neutralise_tags(text: str) -> str:
    """Stop user text from closing/opening a <message> block."""
    return _MESSAGE_TAG_RE.sub(r"&lt;\1", text or "")


def _attr(value) -> str:
    return (str(value).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _upload_name(raw_name, msg_id, used: set) -> str:
    """File name an attachment is uploaded to the LLM API under: path
    separators neutralised, prefixed with *msg_id* when given, and made
    unique within the turn (``stem_2.ext``) — recorded in *used*."""
    base = re.sub(r"[\\/\x00]", "_", str(raw_name or "").strip()).lstrip(".") or "file"
    if msg_id is not None:
        base = f"{msg_id}_{base}"
    stem, ext = os.path.splitext(base)
    name = base
    n = 2
    while name.lower() in used:
        name = f"{stem}_{n}{ext}"
        n += 1
    used.add(name.lower())
    return name


def _format_message_block(m: _QueuedMessage) -> str:
    attrs = []
    if m.msg_id is not None:
        attrs.append(f'id="{_attr(m.msg_id)}"')
    attrs.append(f'from="{_attr(m.sender)}"')
    if m.at:
        attrs.append(f'at="{_attr(m.at)}"')
    reply_id = (m.reply_to_data or {}).get("id")
    if reply_id is not None:
        attrs.append(f'reply_to="{_attr(reply_id)}"')
    if m.upload_names:
        # Exact uploaded file names, so each file can be tied to its message.
        attrs.append(f'attachments="{_attr(", ".join(m.upload_names))}"')
    quote = _neutralise_tags(_reply_quote(m.reply_to_data))
    text = _neutralise_tags(m.content)
    inner = f"{quote}\n{text}" if quote else text
    return f"<message {' '.join(attrs)}>{inner}</message>"


def _format_batch(batch: list, room_name: str, room_id: int, is_group: bool) -> str:
    if len(batch) == 1:
        m = batch[0]
        header = _build_message_header(room_name, room_id, is_group, m.sender, m.reply_to_data)
        return f"{header}\n{m.content}"
    ordered = sorted(batch, key=lambda m: (m.msg_id is None, m.msg_id or 0, m.seq))
    senders: list = []
    for m in ordered:
        if m.sender not in senders:
            senders.append(m.sender)
    room_type = "group" if is_group else "DM"
    lines = [
        f"[Room: {room_name} (id:{room_id}, {room_type}) | From: {', '.join(senders)}]",
        _BATCH_INTRO.format(n=len(ordered)),
    ]
    lines.extend(_format_message_block(m) for m in ordered)
    return "\n".join(lines)


def _format_flush_hint(mem_chars: int | None) -> str:
    """Fill {memory_chars}/{memory_cap} (and legacy {size}/{cap}) without
    str.format, so a template with other braces or no placeholders is safe."""
    cap = _memory_max_chars()
    chars_txt = f"{mem_chars:,}" if isinstance(mem_chars, int) else "?"
    cap_txt = f"{cap:,}"
    text = _MEMORY_FLUSH_HINT
    had_placeholder = any(p in text for p in ("{memory_chars}", "{memory_cap}", "{size}", "{cap}"))
    text = (text.replace("{memory_chars}", chars_txt).replace("{memory_cap}", cap_txt)
            .replace("{size}", chars_txt).replace("{cap}", cap_txt))
    if not had_placeholder:
        text += f"\n(memory.md: {chars_txt} / {cap_txt} chars)"
    return text


# Minimum turns between two soft-cap-triggered flush hints (threshold hints
# are unaffected) so a stuck-large memory.md doesn't nag every single turn.
_SOFT_HINT_MIN_GAP_TURNS = 5


def _first_msg_id(batch: list) -> int | None:
    ids = [m.msg_id for m in batch if m.msg_id is not None]
    return min(ids) if ids else None


# Explicit phrases only (local fallback when core.llm_api has no helper). A
# false positive here force-condenses memory.md and drops the room session, so
# generic wording like "exceeds the maximum" (file sizes, timeouts) must not match.
_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length|context[ _-]length|context_length_exceeded|context window|"
    r"prompt is too long|too many tokens|maximum model length|max_model_len|"
    r"reduce the length of the messages|context overflow",
    re.IGNORECASE,
)


def _looks_like_overflow(text: str) -> bool:
    helper = getattr(llm_api, "is_context_overflow_error", None)
    if callable(helper):
        try:
            if helper(Exception(text)):
                return True
        except Exception:
            pass
    return bool(_CONTEXT_OVERFLOW_RE.search(text or ""))


def _is_context_overflow(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, _SessionNotFound)):
        return False
    return _looks_like_overflow(f"{getattr(exc, 'message', '') or ''} {exc}")


def _user_error_message(exc: BaseException) -> str:
    if isinstance(exc, httpx.PoolTimeout):
        return "LLM 서버 연결이 모두 사용 중이라 요청을 보내지 못했어요 (동시 요청 과다). 잠시 후 다시 시도해주세요."
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "LLM 서버에 연결할 수 없어요. 잠시 후 다시 시도해주세요."
    if isinstance(exc, httpx.TimeoutException):
        return "LLM 서버 응답 시간이 초과됐어요. 잠시 후 다시 시도해주세요."
    if isinstance(exc, _SessionNotFound):
        return "대화 세션을 찾을 수 없어 새 세션으로도 처리하지 못했어요. 잠시 후 다시 시도해주세요."
    if getattr(exc, "status_code", None) == 409:
        return "이 대화의 이전 요청이 아직 처리 중이에요. 잠시 후 다시 시도하거나 @stop 후 보내주세요."
    detail = str(exc).strip() or type(exc).__name__
    return f"오류: {detail[:500]}"


class _SessionNotFound(Exception):
    """LLM API returned 404 for the room's session."""


# Max time an interactive turn waits for the forced memory condensation after
# a context overflow (memory_guard's lock may be held by a background condense,
# and the LLM condense itself can take up to MEMORY_CONDENSE_TIMEOUT_SECONDS)
# before retrying on a fresh session anyway. Memory injection is capped by
# core.context, so a fresh session fits regardless; the condensation keeps
# running in the background and the fresh session is flagged for a memory
# refresh when it lands (see _on_memory_condensed).
_OVERFLOW_CONDENSE_WAIT_SECONDS = 90.0


async def _process_turn(state: RoomState, ctx: _TurnContext) -> None:
    """Core chat-turn pipeline for one batch, with structured logging and timing."""
    start = time.monotonic()
    room_id = state.room_id
    key = state.key
    batch = ctx.batch
    senders = ", ".join(dict.fromkeys(m.sender for m in batch))
    log_prefix = f"[Room {key}] [{senders}]"
    reply_to_id = None if state.external else _first_msg_id(batch)
    logger.info(f"{log_prefix} Processing batch of {len(batch)}: {batch[0].content[:80]!r}")

    _fire_typing(room_id, _STATUS_GENERATING)

    try:
        if not config.LLM_API_KEY:
            raise ValueError("LLM_API_KEY is not configured. Run: python setup.py")
        if not config.LLM_MODEL:
            raise ValueError("LLM_MODEL is not configured. Run: python setup.py")

        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}

        # Download attached files from Messenger. The LLM API stores uploads by
        # file name, so names are made unique within the turn (two pasted
        # screenshots are both "image.png") and, in a multi-message batch,
        # prefixed with their message id so the model can tell whose they are.
        downloaded_files: list[tuple[str, bytes]] = []
        used_names: set = set()
        for m in batch:
            m.upload_names = []
            for fi in m.files or []:
                result = await messenger.download_file(fi["url"])
                if result:
                    file_bytes, _filename = result
                    name = _upload_name(fi.get("name"), m.msg_id if len(batch) > 1 else None, used_names)
                    m.upload_names.append(name)
                    downloaded_files.append((name, file_bytes))
                    logger.info(f"{log_prefix} Downloaded file: {fi.get('name')} as {name} ({len(file_bytes)} bytes)")
                else:
                    logger.warning(f"{log_prefix} Failed to download: {fi.get('name')}")

        room_info = await messenger.get_room_info(room_id)
        body = _format_batch(batch, room_info.get("name", str(room_id)), room_id, bool(room_info.get("isGroup")))
        ambient = build_per_turn_context(profile=getattr(config, "CLUSTER_ROLE", "flutter"))
        base_user_content = f"{ambient}\n\n{body}"

        overflow_retry_left = True
        not_found_retry_left = True
        while True:
            existing_session_id = _get_session_id(key)
            user_content = base_user_content
            refresh_epoch = None  # set when this turn re-sends a condensed memory.md
            if existing_session_id:
                # Context already in history — don't re-inject PROMPT.md every turn.
                # memory.md was condensed since this session's first-turn copy:
                # re-send the current memory once instead of resetting the
                # conversation.
                if (_room_sessions.get(key) or {}).get("memory_refresh"):
                    refresh_epoch = _sessions_epoch
                    user_content = f"{_memory_refresh_block()}\n\n{user_content}"
                    logger.info(f"{log_prefix} Re-sending condensed memory.md to session {existing_session_id}")
                # Append the memory-flush hint only when due (threshold) or when
                # memory.md is above the soft cap.
                count = _room_msg_count.get(key, 0) + 1
                _room_msg_count[key] = count
                threshold = int(getattr(config, "MEMORY_FLUSH_THRESHOLD", 0) or 0)
                due = threshold > 0 and count % threshold == 0
                mem_chars = _memory_size()
                over_soft = isinstance(mem_chars, int) and mem_chars > _memory_soft_chars()
                last_hint = _room_last_hint.get(key)
                if due or (over_soft and (last_hint is None or count - last_hint >= _SOFT_HINT_MIN_GAP_TURNS)):
                    user_content += f"\n\n{_format_flush_hint(mem_chars)}"
                    _room_last_hint[key] = count
                    logger.info(f"{log_prefix} Memory flush hint injected at message #{count} (memory={mem_chars})")

                messages = [{"role": "user", "content": user_content}]
                llm_data = {
                    "model": config.LLM_MODEL,
                    "messages": json.dumps(messages),
                    "session_id": existing_session_id,
                }
                ctx.session_id = existing_session_id
                logger.info(f"{log_prefix} Continuing session {existing_session_id} (msg #{count})")
            else:
                # First turn of a new session: inject the full hoonbot context
                # (PROMPT.md + session vars + memory) as a system message so it
                # lands in history once and doesn't bloat every subsequent user turn.
                _room_msg_count[key] = 0
                _room_last_hint.pop(key, None)
                # The memory copy below is read now: only a condensation after
                # this point makes the new session's copy stale.
                ctx.epoch = _sessions_epoch
                ctx.memory_fp = _memory_fingerprint()
                ctx.memory_chars = _memory_size()
                context = build_llm_context()
                messages = [
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_content},
                ]
                llm_data = {
                    "model": config.LLM_MODEL,
                    "messages": json.dumps(messages),
                }
                ctx.session_id = None
                logger.info(f"{log_prefix} Starting new session")

            # Sticky goal mode (toggled by /goal) — forwarded as a form field the
            # LLM API parses into AgentLoop(mode="goal").
            if _room_goal_mode.get(key):
                llm_data["mode"] = "goal"
                logger.info(f"{log_prefix} Goal mode active — sending mode=goal")

            try:
                if config.STREAMING_ENABLED:
                    reply = await _process_streaming(
                        state, ctx, llm_data, headers, existing_session_id, log_prefix,
                        reply_to_id, downloaded_files, allow_overflow_retry=overflow_retry_left,
                    )
                else:
                    reply = await _process_sync(
                        state, ctx, llm_data, headers, existing_session_id, log_prefix,
                        reply_to_id, downloaded_files,
                    )
            except _SessionNotFound:
                logger.warning(f"{log_prefix} Session {existing_session_id} not found (404), starting fresh")
                _drop_session(key)
                if not_found_retry_left:
                    not_found_retry_left = False
                    continue
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if overflow_retry_left and _is_context_overflow(exc):
                    overflow_retry_left = False
                    logger.warning(f"{log_prefix} Context overflow ({str(exc)[:200]}) — condensing memory and retrying with a fresh session")
                    _fire_typing(room_id, "메모리 정리 중")
                    try:
                        # The condense is an LLM call (up to ~10 min). Bound how
                        # long this turn waits; the condensation itself keeps
                        # running in the background (shielded) if we stop waiting.
                        await asyncio.wait_for(
                            _run_shielded(_enforce_memory_cap("overflow", force=True)),
                            timeout=_OVERFLOW_CONDENSE_WAIT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"{log_prefix} Memory condensation still running after "
                                       f"{_OVERFLOW_CONDENSE_WAIT_SECONDS:.0f}s — retrying without waiting")
                    _fire_typing(room_id, _STATUS_GENERATING)
                    # Do not keep the failed session (whether pre-existing or
                    # just captured from this failed attempt).
                    _drop_session(key)
                    ctx.session_id = None
                    continue
                raise
            break

        if refresh_epoch is not None:
            # The refreshed memory is now in the session history (llm-api
            # persists the user turn even when the stream was cut short).
            # Keep the flag if another condensation landed meanwhile.
            entry = _room_sessions.get(key)
            if (isinstance(entry, dict) and entry.get("session_id") == existing_session_id
                    and _sessions_epoch == refresh_epoch and entry.pop("memory_refresh", None)):
                _save_room_sessions()

        elapsed = time.monotonic() - start
        logger.info(f"{log_prefix} Completed in {elapsed:.1f}s, reply={len(reply or '')} chars")
        _maybe_condense_memory()

    except asyncio.CancelledError:
        elapsed = time.monotonic() - start
        logger.info(f"{log_prefix} Turn cancelled after {elapsed:.1f}s ({ctx.cancel_reason or 'cancelled'})")
        raise
    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error(f"{log_prefix} Failed after {elapsed:.1f}s: {exc}", exc_info=True)
        try:
            await messenger.send_message(room_id, _user_error_message(exc), reply_to_id=reply_to_id)
        except Exception:
            pass
    finally:
        _fire_stop_typing(room_id)


def _capture_session(state: RoomState, ctx: _TurnContext, session_id: str | None,
                     existing_session_id: str | None, log_prefix: str) -> None:
    """Record the session id as soon as it is known (header / first SSE event).

    Never overwrites a known id with None; never saves after @clear (generation)
    happened mid-turn. A session created across a memory condensation (epoch
    changed) is kept but flagged for a memory refresh on its next turn.
    """
    if not session_id:
        return
    ctx.session_id = session_id
    if session_id == (existing_session_id or ""):
        return
    if state.generation != ctx.generation:
        logger.info(f"{log_prefix} Not saving session {session_id} (cleared mid-turn)")
        return
    if _peek_session_id(state.key) == session_id:
        return
    refresh = _sessions_epoch != ctx.epoch
    _set_session_id(state.key, session_id, memory_fp=ctx.memory_fp, memory_chars=ctx.memory_chars,
                    memory_refresh=refresh)
    logger.info(f"{log_prefix} Session saved as {session_id}"
                + (" (memory condensed mid-turn — flagged for refresh)" if refresh else ""))


async def _save_session_from_response(room_id: int, result: dict, existing_session_id: str | None, log_prefix: str) -> None:
    """Save session_id returned by LLM API (compat helper)."""
    returned_session_id = result.get("x_session_id")
    if returned_session_id and returned_session_id != (existing_session_id or ""):
        _set_session_id(room_id, returned_session_id)
        logger.info(f"{log_prefix} Session saved as {returned_session_id}")


def _diag_tool_count(diagnostics: dict, fallback: int) -> int:
    value = (diagnostics or {}).get("tool_calls")
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    return fallback


def _empty_reply_notice(finish_reason: str | None, tools: int, post_tool: bool = False) -> str:
    fr = finish_reason or "unknown"
    if post_tool:
        return (f"⚠️ 도구 실행 후 최종 답변이 비어 있습니다 (finish_reason={fr}, tools={tools}). "
                "다시 요청하거나 @clear 후 시도해 주세요.")
    return (f"⚠️ 모델이 최종 답변을 생성하지 못했습니다 (finish_reason={fr}, tools={tools}). "
            "다시 요청하거나 @clear 후 시도해 주세요.")


_LENGTH_NOTICE = "⚠️ 응답이 최대 길이에 도달해 잘렸을 수 있습니다 (finish_reason=length)."


def _truncated_notice(reason: str) -> str:
    return f"⚠️ 응답이 중간에 끊겼습니다: {(reason or 'unknown')[:500]}"


# Inline cut notices llm-api itself appends to the content (and persists with
# the turn) on finish_reason length / error. When the body already ends with
# one, hoonbot's own notice would only duplicate it.
_LLM_INLINE_CUT_NOTICES = (
    "[출력이 길이 제한으로 잘렸습니다]",
    "[응답 생성 중 연결 오류로 출력이 중단되었습니다]",
)


def _has_inline_cut_notice(body: str | None) -> bool:
    return bool(body) and body.rstrip().endswith(_LLM_INLINE_CUT_NOTICES)


# Transport errors that are safe to retry when NOTHING was received yet, and
# HTTP statuses retried on the same session (409 = session lease busy).
_PRE_OUTPUT_RETRY_ERRORS = (
    httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
    httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
)
_RETRYABLE_STATUS = {409, 500, 502, 503, 504}
_RETRY_BACKOFF_SECONDS = (1.5, 4.0)

# Non-streaming turns: the response only arrives once the whole agent turn has
# run, and llm-api persists the turn before answering 500/502. So only errors
# that prove the turn never started are retried — connect-phase transport
# errors, a body that could not be fully sent, 409 (lease busy), 503 (storage
# lock / not ready) and 504. Read-phase errors and 500/502 are final: a resend
# would duplicate the user turn and re-run its side-effecting tools.
_SYNC_RETRY_ERRORS = (
    httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.WriteError,
)
_SYNC_RETRYABLE_STATUS = {409, 503, 504}

# Sessions whose @compact was abandoned client-side (@stop, timeout, transport
# error) while llm-api may still be summarising under the session's turn
# lease: session_id -> monotonic deadline. Chat turns on such a session keep
# polling through 409 "session busy" until the lease is released (or the
# deadline passes) instead of giving up after the short retry budget.
_compact_abandoned: dict = {}
# llm-api's non-streaming summariser call can wait up to its first-byte
# timeout (30 min) — keep waiting a little longer than that.
_COMPACT_ABANDON_WAIT_SECONDS = 1900.0
_COMPACT_WAIT_POLL_SECONDS = 5.0


def _compact_wait_active(session_id: str | None) -> bool:
    if not session_id:
        return False
    deadline = _compact_abandoned.get(session_id)
    if deadline is None:
        return False
    if time.monotonic() >= deadline:
        _compact_abandoned.pop(session_id, None)
        return False
    return True


def _retry_delay(status: int, attempt: int, session_id: str | None, retryable: set) -> tuple:
    """(delay, counts_as_attempt) for a retryable HTTP status, or (None, False).

    A 409 on a session with an abandoned @compact polls until the lease frees
    up without consuming the normal retry budget."""
    if status == 409 and _compact_wait_active(session_id):
        return _COMPACT_WAIT_POLL_SECONDS, False
    if status in retryable and attempt < len(_RETRY_BACKOFF_SECONDS):
        return _RETRY_BACKOFF_SECONDS[attempt], True
    return None, False


async def _read_error_body(response: httpx.Response) -> str:
    try:
        raw = await response.aread()
        return raw.decode("utf-8", errors="replace")[:500]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Synchronous (non-streaming) processing
# ---------------------------------------------------------------------------

async def _process_sync(state: RoomState, ctx: _TurnContext, llm_data: dict, headers: dict,
                        existing_session_id: str | None, log_prefix: str, reply_to_id: int | None,
                        downloaded_files: list[tuple[str, bytes]] | None = None) -> str:
    """Send request, wait for full response, send to Messenger."""
    room_id = state.room_id
    files_payload = [("files", (name, data, "application/octet-stream")) for name, data in (downloaded_files or [])]
    # A non-streaming agent turn returns nothing until it is done, so the
    # pooled client's per-read idle timeout (sized for SSE pings) is too short;
    # use the wall-clock cap for non-streaming calls instead.
    try:
        read_timeout = float(getattr(config, "LLM_TIMEOUT_SECONDS", 3600) or 0) or None
    except (TypeError, ValueError):
        read_timeout = 3600.0
    sync_timeout = httpx.Timeout(connect=10.0, read=read_timeout, write=60.0, pool=None)
    attempt = 0
    while True:
        try:
            response = await _llm_client().post(
                f"{config.LLM_API_URL}/v1/chat/completions",
                data=llm_data,
                files=files_payload or None,
                headers=headers,
                timeout=sync_timeout,
            )
        except _SYNC_RETRY_ERRORS as exc:
            if attempt >= len(_RETRY_BACKOFF_SECONDS):
                raise
            delay = _RETRY_BACKOFF_SECONDS[attempt]
            attempt += 1
            logger.warning(f"{log_prefix} LLM API {type(exc).__name__} — retry {attempt} in {delay}s")
            await asyncio.sleep(delay)
            continue

        status = response.status_code
        if status == 404 and existing_session_id:
            raise _SessionNotFound(existing_session_id)
        delay, counted = _retry_delay(status, attempt, existing_session_id, _SYNC_RETRYABLE_STATUS)
        if delay is not None:
            if counted:
                attempt += 1
                logger.warning(f"{log_prefix} LLM API HTTP {status} — retry {attempt} on same session in {delay}s")
            else:
                logger.info(f"{log_prefix} Session busy with an abandoned @compact — polling again in {delay}s")
            await asyncio.sleep(delay)
            continue
        if status != 409:
            _compact_abandoned.pop(existing_session_id or "", None)
        if status >= 400:
            body = response.text[:500] if response.content else ""
            raise _make_llm_error(f"LLM API HTTP {status}: {body}", status_code=status,
                                  session_id=existing_session_id)
        break

    result = response.json()
    _capture_session(state, ctx, result.get("x_session_id"), existing_session_id, log_prefix)
    diagnostics = result.get("x_diagnostics") or {}
    choice = (result.get("choices") or [{}])[0] or {}
    reply = ((choice.get("message") or {}).get("content")) or ""
    finish_reason = choice.get("finish_reason") or diagnostics.get("finish_reason")
    tools = _diag_tool_count(diagnostics, 0)

    ctx.finalized = True
    if not reply.strip():
        reply = _empty_reply_notice(finish_reason, tools)
        logger.warning(f"EMPTY_REPLY room={state.key} session={ctx.session_id} reason={finish_reason} tools={tools}")
    elif finish_reason == "length" and not _has_inline_cut_notice(reply):
        reply = f"{reply.rstrip()}\n\n{_LENGTH_NOTICE}"
    await messenger.send_message(room_id, reply, reply_to_id=reply_to_id)
    return reply


# ---------------------------------------------------------------------------
# Streaming processing — streams text into one live-edited draft bubble,
# surfaces tool activity only through the ephemeral typing indicator.
# ---------------------------------------------------------------------------

# Status shown in the typing indicator when the LLM is thinking/generating
# (i.e. no tool is currently running). Rendered as "{bot}님이 응답 생성 중...".
_STATUS_GENERATING = "응답 생성 중"

# How often to re-fire the typing event so the indicator stays visible during
# long tool calls. Must stay below the Messenger server-side auto-clear (15s)
# and the client-side safety timeout (20s).
_TYPING_REFRESH_INTERVAL_SECONDS = 10

# Divider inserted before the post-tool text when a turn used tools, so the
# final answer stands out from the work narration above it. Styling only —
# a misplaced boundary shifts the marker but never drops content. The client
# renderer only supports bold/italic/code, so a plain rule + bold header is
# the strongest emphasis available inside one bubble.
_ANSWER_DIVIDER = "──────────\n✅ **답변**"

# How often the live draft bubble is updated while tokens stream in. Sends are
# fire-and-forget/coalesced (see _process_streaming's flush pump) so this can
# be low without ever blocking token consumption on a slow Messenger round trip.
# NOTE: each edit re-sends the *full* accumulated text (hoonbot->messenger and
# messenger->all clients), so a lower interval smooths short replies but ~2.5x's
# the transport/re-render cost on long ones — consider an adaptive interval that
# widens as the message grows if long-answer load becomes a problem.
_DRAFT_EDIT_INTERVAL_SECONDS = 0.1

_STOP_MARKER = "⏹️ 중지됨"
_INTERRUPT_MARKER = "⏹️ 중지됨 (이어서 온 메시지와 합쳐 다시 답할게요)"
_SHUTDOWN_MARKER = "⏹️ 재시작으로 중단됨 (다시 시작하면 이어서 답할게요)"

# Draft-bubble creations that definitely failed (never reached Messenger)
# before the turn gives up on a live draft and only posts the final reply.
_DRAFT_CREATE_MAX_MISSES = 3


def _format_tool_status(running_tools: dict[str, str]) -> str:
    """Return the typing-indicator text for the current set of running tools.

    Preserves insertion order so the indicator reads `websearch + code_exec
    사용 중` in the order the tools started. Falls back to the generating
    status when no tool is active.
    """
    if not running_tools:
        return _STATUS_GENERATING
    return " + ".join(running_tools.values()) + " 사용 중"


def _render_tool_log(tool_log: list[dict]) -> str:
    """Render the ordered tool-call log as a compact in-bubble worklog.

    Unlike the transient bottom typing indicator (which only shows tools that
    are running *right now*), this persists every tool call in start order with
    its outcome — ⏳ running, ✅ done (+elapsed), ❌ failed — so the user can see
    what actually ran. The client renderer supports only bold/italic/code, so
    this uses emoji + plain text.
    """
    if not tool_log:
        return ""
    icons = {"running": "⏳", "completed": "✅", "failed": "❌"}
    lines = ["🔧 **작업**"]
    for e in tool_log:
        icon = icons.get(e.get("state"), "•")
        label = e.get("label") or "?"
        if e.get("state") == "completed":
            dur = e.get("duration") or 0.0
            lines.append(f"{icon} {label}" + (f" · {dur:.1f}s" if dur else ""))
        elif e.get("state") == "failed":
            lines.append(f"{icon} {label} · 실패")
        else:
            lines.append(f"{icon} {label} …")
    return "\n".join(lines)


async def _process_streaming(state: RoomState, ctx: _TurnContext, llm_data: dict, headers: dict,
                             existing_session_id: str | None, log_prefix: str, reply_to_id: int | None,
                             downloaded_files: list[tuple[str, bytes]] | None = None,
                             allow_overflow_retry: bool = True) -> str:
    """Stream SSE from LLM API into a single live-edited Messenger bubble.

    UX model:
      - Streamed text is posted as one draft message and edited in place as
        tokens arrive; at the end of the turn it settles into the final reply.
        One bubble per turn. When the turn used tools, the text after the
        last tool event is set off with `_ANSWER_DIVIDER` so the answer
        stands out from the work narration above it.
      - Tool activity is surfaced through the typing indicator at the bottom
        of the chat ("{bot}님이 websearch 사용 중...") plus the in-bubble
        checklist. The draft bubble is never deleted — the client renders
        deleted messages as a permanent "삭제된 메시지입니다." placeholder —
        it is always settled by an edit (answer, notice, or stop marker).
      - A stream that ends without [DONE] or with an error keeps the partial
        draft and appends "⚠️ 응답이 중간에 끊겼습니다: <reason>".
      - An empty final answer is replaced by an informative notice built
        from x_diagnostics (never a bare "완료.").

    Raises _SessionNotFound (404), the context-overflow LLMAPIError (only
    while nothing is visible yet, so the caller can retry on a fresh
    session), and errors raised before any output (connect / HTTP status).
    """
    room_id = state.room_id
    llm_data["stream"] = "true"
    files_payload = [("files", (name, data, "application/octet-stream")) for name, data in (downloaded_files or [])]
    idle_timeout = float(getattr(config, "LLM_STREAM_IDLE_TIMEOUT_SECONDS", 900) or 0) or None

    full_text = ""
    # Character offset in full_text at the last tool event. Text streamed after
    # it gets the answer divider at settle time (styling only, never a split).
    final_split_pos: int = 0
    running_tools: dict[str, str] = {}  # tool_call_id -> tool_name, insertion-ordered
    # Persistent, ordered tool-call log for the in-bubble worklog checklist
    # (see _render_tool_log). Distinct from running_tools, which only feeds the
    # transient bottom typing indicator. tool_log_index maps tool_call_id -> the
    # same dict for O(1) status updates on completed/failed.
    tool_log: list[dict] = []
    tool_log_index: dict[str, dict] = {}
    draft_msg_id: int | None = None   # ID of the live text-draft message in chat
    last_draft_edit: float = 0.0
    first_sent = False

    saw_done = False
    received_any = False
    error_reason: str | None = None
    error_is_overflow = False
    diagnostics: dict = {}
    finish_reason: str | None = None
    finalize_started = False

    async def _refresh_typing() -> None:
        """Periodically re-fire the typing event so the indicator stays visible."""
        while True:
            await asyncio.sleep(_TYPING_REFRESH_INTERVAL_SECONDS)
            _fire_typing(room_id, _format_tool_status(running_tools))

    refresh_task = asyncio.create_task(_refresh_typing())

    # Draft-bubble sends/edits are dispatched through a coalescing background
    # pump instead of being awaited inline in the SSE read loop: a slow
    # Messenger round trip used to stall token consumption for its full
    # duration every time the interval elapsed. The pump keeps at most one
    # HTTP call in flight and always carries the latest text — if new tokens
    # arrive while a send is in flight, they simply replace `pending_text`
    # and the pump picks them up on its next iteration instead of queuing
    # redundant calls.
    pending_text: str | None = None
    flush_task: asyncio.Task | None = None
    # Set when creating the draft bubble failed for good: no further drafts are
    # created this turn — _final_write posts the reply once instead. An
    # ambiguous failure (messenger.MaybeDeliveredError: may have been stored
    # with the response lost) sets it at once unless _recover_draft_id finds
    # the bubble; a definite failure only after _DRAFT_CREATE_MAX_MISSES tries.
    draft_create_failed = False
    # Consecutive draft sends that were definitely NOT delivered (connect-phase
    # failure / 4xx): safe to retry on the next flush, up to a small cap.
    draft_create_misses = 0

    async def _flush_loop() -> None:
        nonlocal draft_msg_id, pending_text, flush_task, draft_create_failed, draft_create_misses
        while pending_text is not None:
            text_to_send = pending_text
            pending_text = None
            try:
                if draft_msg_id is None:
                    if draft_create_failed:
                        continue
                    try:
                        new_id = await messenger.send_message_returning_id(
                            room_id, text_to_send, reply_to_id
                        )
                    except messenger.MaybeDeliveredError:
                        # The bubble may exist with its response lost: adopt it
                        # if it can be found, never post a second draft.
                        new_id = await _recover_draft_id(text_to_send)
                        if new_id is None:
                            draft_create_failed = True
                            logger.warning(f"{log_prefix} Draft send outcome unknown and the bubble was not "
                                           "found — no live draft this turn, the final reply is sent once")
                        draft_msg_id = new_id
                        continue
                    if new_id is None:
                        # Definitely not delivered — a later flush may retry.
                        draft_create_misses += 1
                        if draft_create_misses >= _DRAFT_CREATE_MAX_MISSES:
                            draft_create_failed = True
                            logger.warning(f"{log_prefix} Draft bubble could not be created "
                                           f"({draft_create_misses} attempts) — the final reply is sent once")
                        continue
                    draft_create_misses = 0
                    draft_msg_id = new_id
                else:
                    await messenger.edit_message(draft_msg_id, text_to_send)
            except Exception:
                pass
        flush_task = None

    async def _recover_draft_id(text: str) -> int | None:
        """A failed draft send may still have been stored (response lost after
        the request reached Messenger). Look for it among the room's newest
        messages so the live draft keeps editing that bubble instead of
        leaving it frozen next to a second one."""
        try:
            recent = await messenger.get_room_messages(room_id, limit=5)
        except Exception:
            return None
        want = (text or "").strip()
        for m in reversed(recent or []):
            if not isinstance(m, dict) or not m.get("isBot") or m.get("isDeleted"):
                continue
            if reply_to_id is not None and m.get("replyToId") != reply_to_id:
                continue
            if str(m.get("content") or "").strip() == want and m.get("id") is not None:
                logger.info(f"{log_prefix} Recovered draft bubble id {m['id']} after an ambiguous send failure")
                return m["id"]
        return None

    def _schedule_flush(text: str) -> None:
        nonlocal pending_text, flush_task
        if finalize_started or (draft_create_failed and draft_msg_id is None):
            return
        pending_text = text
        if flush_task is None or flush_task.done():
            flush_task = asyncio.create_task(_flush_loop())

    async def _settle_flush() -> None:
        """Drain any in-flight/queued background flush before an authoritative
        write (final settle, error/cancel marker) so it can't race a later
        write and so draft_msg_id is populated if a send was in flight."""
        nonlocal flush_task
        if flush_task is not None:
            try:
                await flush_task
            except Exception:
                pass
            flush_task = None

    def _compose_draft(body: str) -> str:
        """Prepend the live tool-call checklist above the streamed text so the
        bubble shows what ran (with per-tool state) as well as the answer."""
        block = _render_tool_log(tool_log)
        body = (body or "").strip()
        if block and body:
            return f"{block}\n\n{body}"
        return block or body

    async def _safe_edit(message_id: int, text: str, progress: dict) -> bool:
        try:
            # final=True: authoritative settle — lets messenger split an
            # overlong reply into head edit + follow-up messages (C9).
            # draft stays True (server semantics unchanged from before).
            # *progress* records which follow-up chunks were already posted,
            # so a retry never posts them twice.
            ok = await messenger.edit_message(message_id, text, final=True, room_id=room_id,
                                              progress=progress)
        except Exception:
            return False
        return ok is not False  # legacy edit_message returns None on success

    async def _final_write(text: str) -> None:
        """Authoritative write: edit the draft (retry once), else send.

        Idempotent across the retry and the fallback: once the bubble holds
        the head of the reply, only the still-unposted follow-up chunks are
        sent — the room never sees a chunk (or the whole reply) twice.
        """
        progress: dict = {}
        if draft_msg_id is not None:
            if await _safe_edit(draft_msg_id, text, progress):
                return
            await asyncio.sleep(0.5)
            if await _safe_edit(draft_msg_id, text, progress):
                return
            if progress.get("head_done"):
                pending = [c for c in (progress.get("pending") or []) if c and c.strip()]
                logger.warning(f"{log_prefix} Final reply follow-up failed twice — "
                               f"sending {len(pending)} remaining chunk(s) as new messages")
                for chunk in pending:
                    try:
                        await messenger.send_message(room_id, chunk)
                    except Exception as exc:
                        logger.error(f"{log_prefix} Could not deliver final reply chunk: {exc}")
                        return
                return
            logger.warning(f"{log_prefix} Final draft edit failed twice — sending as a new message")
        try:
            await messenger.send_message(room_id, text, reply_to_id=reply_to_id)
        except Exception as exc:
            logger.error(f"{log_prefix} Could not deliver final reply: {exc}")

    def _visible() -> bool:
        return bool(full_text.strip()) or bool(tool_log) or draft_msg_id is not None

    async def _finalize(reason: str | None) -> str:
        """Settle the draft into the final reply (runs under asyncio.shield)."""
        nonlocal finalize_started
        finalize_started = True
        await _settle_flush()

        worklog_text = full_text[:final_split_pos].strip()
        answer_text = full_text[final_split_pos:].strip()
        body = full_text.strip()
        # When the turn used tools, set the post-tool text off with the answer
        # divider so it stands out from the work narration above it.
        if worklog_text and answer_text:
            body = f"{worklog_text}\n\n{_ANSWER_DIVIDER}\n\n{answer_text}"

        fr = finish_reason or diagnostics.get("finish_reason")
        tools = _diag_tool_count(diagnostics, len(tool_log))
        notices: list[str] = []
        if reason:
            notices.append(_truncated_notice(reason))
            logger.warning(f"{log_prefix} Stream incomplete ({reason[:200]}); kept {len(full_text)} chars")
        elif not body:
            notices.append(_empty_reply_notice(fr, tools))
            logger.warning(f"EMPTY_REPLY room={state.key} session={ctx.session_id} reason={fr} tools={tools}")
        elif tool_log and not answer_text:
            notices.append(_empty_reply_notice(fr, tools, post_tool=True))
            logger.warning(f"EMPTY_REPLY room={state.key} session={ctx.session_id} reason={fr} tools={tools} (post-tool)")
        elif _has_inline_cut_notice(body) and fr in ("length", "error", "interrupted"):
            pass  # llm-api already appended its own inline cut notice
        elif fr == "length":
            notices.append(_LENGTH_NOTICE)
        elif fr in ("error", "interrupted"):
            notices.append(_truncated_notice(f"finish_reason={fr}"))

        text = "\n\n".join(p for p in [body, *notices] if p)
        final_text = _compose_draft(text)
        await _final_write(final_text)
        return final_text

    async def _finalize_cancelled() -> None:
        """Settle the draft with a stop marker — never delete it."""
        nonlocal finalize_started
        finalize_started = True
        await _settle_flush()
        if ctx.cancel_reason == "interrupt":
            marker = _INTERRUPT_MARKER
        elif ctx.cancel_reason == "shutdown":
            marker = _SHUTDOWN_MARKER
        else:
            marker = _STOP_MARKER
        body = full_text.strip()
        if draft_msg_id is None and not body:
            return  # nothing was shown; the @stop handler announces the stop
        text = _compose_draft(f"{body}\n\n{marker}" if body else marker)
        await _final_write(text)

    async def _consume(response: httpx.Response) -> None:
        nonlocal full_text, final_split_pos, last_draft_edit, first_sent
        nonlocal saw_done, received_any, error_reason, error_is_overflow, diagnostics, finish_reason
        lines = response.aiter_lines()
        while True:
            try:
                if idle_timeout:
                    line = await asyncio.wait_for(lines.__anext__(), timeout=idle_timeout)
                else:
                    line = await lines.__anext__()
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                error_reason = f"LLM API로부터 {idle_timeout:.0f}초 동안 데이터가 없습니다 (idle timeout)"
                return
            except httpx.TransportError as exc:
                if not received_any:
                    raise
                error_reason = f"{type(exc).__name__}: {exc}".strip()
                return

            received_any = True
            if not line.startswith("data:"):
                continue  # SSE comments / keepalive pings
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                saw_done = True
                return

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            # Never overwrite a known session id with None.
            _capture_session(state, ctx, event.get("x_session_id"), existing_session_id, log_prefix)
            if isinstance(event.get("x_diagnostics"), dict):
                diagnostics = event["x_diagnostics"]

            if event.get("error"):
                err = event.get("error") or {}
                if isinstance(err, dict):
                    err_type = err.get("type", "stream_error")
                    err_message = str(err.get("message") or "").strip() or json.dumps(err, ensure_ascii=False)
                else:
                    err_type, err_message = "stream_error", str(err)
                error_reason = f"{err_message} ({err_type})"
                error_is_overflow = _looks_like_overflow(f"{err_type} {err_message}")
                return

            if "tool_status" in event:
                ts = event["tool_status"] or {}
                tool_name = ts.get("tool_name", "")
                # Text streamed before a tool event is work narration, not
                # the final answer — remember the boundary for the divider.
                # Autocompact is server housekeeping, not a tool call: it
                # never turns already-streamed text into narration.
                if tool_name != "autocompact":
                    final_split_pos = len(full_text)
                tool_key = ts.get("tool_call_id") or tool_name
                status = ts.get("status", "")
                label = ts.get("user_name") or ts.get("activity") or tool_name
                if status == "started":
                    running_tools[tool_key] = tool_name
                    entry = tool_log_index.get(tool_key)
                    if entry is None:
                        entry = {"label": label, "state": "running", "duration": 0.0}
                        tool_log.append(entry)
                        tool_log_index[tool_key] = entry
                    else:
                        entry["state"] = "running"
                elif status in {"completed", "failed"}:
                    running_tools.pop(tool_key, None)
                    entry = tool_log_index.get(tool_key)
                    if entry is None:
                        entry = {"label": label, "state": status, "duration": ts.get("duration") or 0.0}
                        tool_log.append(entry)
                        tool_log_index[tool_key] = entry
                    else:
                        entry["state"] = status
                        entry["duration"] = ts.get("duration") or entry.get("duration") or 0.0
                _fire_typing(room_id, _format_tool_status(running_tools))
                # Update the in-bubble checklist live — tools often run before
                # any answer text, so this may create the draft bubble too.
                _schedule_flush(_compose_draft(full_text))
                continue

            choices = event.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            text = (choice.get("delta") or {}).get("content") or ""
            if text:
                full_text += text
                stripped = full_text.strip()
                if stripped:
                    now = time.monotonic()
                    if not first_sent:
                        # First visible content — show it immediately, no
                        # interval gate, so the bubble appears the instant
                        # generation starts instead of up to one interval late.
                        first_sent = True
                        last_draft_edit = now
                        _schedule_flush(_compose_draft(full_text))
                    elif now - last_draft_edit >= _DRAFT_EDIT_INTERVAL_SECONDS:
                        last_draft_edit = now
                        _schedule_flush(_compose_draft(full_text))

    try:
        attempt = 0
        while True:
            retry_label = None
            delay = None
            counted = True
            try:
                async with _llm_client().stream(
                    "POST",
                    f"{config.LLM_API_URL}/v1/chat/completions",
                    data=llm_data,
                    files=files_payload or None,
                    headers=headers,
                ) as response:
                    status = response.status_code
                    if status != 409:
                        _compact_abandoned.pop(existing_session_id or "", None)
                    if status >= 400:
                        body = await _read_error_body(response)
                        if status == 404 and existing_session_id:
                            raise _SessionNotFound(existing_session_id)
                        delay, counted = _retry_delay(status, attempt, existing_session_id, _RETRYABLE_STATUS)
                        if delay is not None:
                            retry_label = f"HTTP {status}"
                        else:
                            raise _make_llm_error(f"LLM API HTTP {status}: {body}", status_code=status,
                                                  session_id=existing_session_id)
                    else:
                        # Capture the session id immediately from the header.
                        _capture_session(state, ctx, response.headers.get("x-session-id"),
                                         existing_session_id, log_prefix)
                        await _consume(response)
            except _PRE_OUTPUT_RETRY_ERRORS as exc:
                if received_any or attempt >= len(_RETRY_BACKOFF_SECONDS):
                    raise
                retry_label = type(exc).__name__
                delay, counted = _RETRY_BACKOFF_SECONDS[attempt], True
            if retry_label is None:
                break
            if counted:
                attempt += 1
                logger.warning(f"{log_prefix} LLM API {retry_label} before any output — retry {attempt} on same session in {delay}s")
            else:
                logger.info(f"{log_prefix} Session busy with an abandoned @compact — polling again in {delay}s")
            await asyncio.sleep(delay)

        if error_reason is None and not saw_done:
            error_reason = "stream ended unexpectedly ([DONE] 없이 종료)"

        if error_reason and error_is_overflow and allow_overflow_retry and not _visible():
            await _settle_flush()
            raise _make_llm_error(error_reason, session_id=ctx.session_id, received_output=False)

        # Shield the settle + final write from cancellation so a late @stop
        # can't leave the bubble frozen mid-sentence. Mark the turn finalized
        # synchronously so an "interrupt" follow-up never re-queues a batch
        # whose answer is already being delivered.
        ctx.finalized = True
        return await _run_shielded(_finalize(error_reason))

    except asyncio.CancelledError:
        # Local task cancellation (@stop, interrupt policy, shutdown) — settle
        # the draft with a marker instead of leaving it frozen mid-sentence.
        if not finalize_started:
            try:
                await _run_shielded(_finalize_cancelled())
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        raise
    except _SessionNotFound:
        await _settle_flush()
        raise
    except Exception as exc:
        if finalize_started:
            raise
        # Errors before/while opening the stream. If something is already
        # visible, keep it and append the notice; otherwise let the caller
        # map the error (connect / pool / overflow retry / HTTP).
        if _visible() and not (allow_overflow_retry and _is_context_overflow(exc) and not full_text.strip()):
            ctx.finalized = True
            return await _run_shielded(_finalize(str(exc) or type(exc).__name__))
        await _settle_flush()
        raise
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# External / incoming webhooks
# ---------------------------------------------------------------------------

@router.post("/webhook/incoming/{path:path}")
async def handle_incoming_webhook(path: str, request: Request):
    """Accept POST triggers from external services.

    Runs in its own session ("ext_<source>") and queue, posting replies to the
    home room. No directive parsing, and it never cancels Messenger room turns.
    """
    if config.WEBHOOK_INCOMING_SECRET:
        secret = request.headers.get("x-webhook-secret", "")
        if secret != config.WEBHOOK_INCOMING_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {"payload": payload}

    source = path.strip("/") or "external"
    if "message" in payload:
        content = f"[Webhook from {source}] {payload['message']}"
    else:
        content = f"[Webhook from {source}] {json.dumps(payload, ensure_ascii=False, indent=2)}"

    room_id = config.MESSENGER_HOME_ROOM_ID
    if room_id is None:
        logger.warning(f"[Webhook] Incoming from '{source}' dropped — home room not resolved")
        return {"ok": False, "error": "home room not configured"}
    key = "ext_" + re.sub(r"[^\w.-]", "_", source)[:64]
    logger.info(f"[Webhook] Incoming from '{source}' → room {room_id} (session key {key})")
    state = _get_state(key, room_id=room_id, external=True)
    item = _QueuedMessage(
        msg_id=None,
        content=content,
        sender=f"webhook:{source}",
        kind="chat",
        at=datetime.now().strftime("%H:%M:%S"),
        source="external",
    )
    queued = _enqueue(state, item)
    return {"ok": queued}
