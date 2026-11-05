"""
Hoonbot — entry point.

Startup sequence:
1. Health-check LLM API
2. Schedule a background memory.md size check (core.memory_guard)
3. Register bot with Messenger (get / restore API key)
4. Register webhook subscription (new_message, message_edited, message_deleted)
5. Resolve home room by name (if configured)
6. Catch up on missed messages (via the per-room queue when available)
7. Start heartbeat loop
8. Serve FastAPI on HOONBOT_PORT

Slave nodes run only the cluster worker loop plus a periodic memory size check.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI

import config
from core import messenger
from core.heartbeat import run_heartbeat_loop
from core.llm_api import close_client as close_llm_client
from core.retry import with_retry
from handlers.health import router as health_router
from handlers.webhook import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hoonbot")

_KEY_FILE = os.path.join(os.path.dirname(__file__), "data", ".apikey")
_WEBHOOK_EVENTS = ["new_message", "message_edited", "message_deleted"]
_CATCHUP_MAX_CONCURRENCY = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_saved_key() -> str:
    try:
        with open(_KEY_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _save_key(key: str) -> None:
    os.makedirs(os.path.dirname(_KEY_FILE), exist_ok=True)
    with open(_KEY_FILE, "w") as f:
        f.write(key)


# ---------------------------------------------------------------------------
# Startup steps (each is a self-contained phase)
# ---------------------------------------------------------------------------

async def _probe_llm_url(client: httpx.AsyncClient, url: str) -> str | None:
    """Return url if its /health endpoint returns 200, else None."""
    try:
        resp = await client.get(f"{url}/health", timeout=5)
        return url if resp.status_code == 200 else None
    except Exception:
        return None


async def _autofind_llm_api() -> None:
    """Probe all candidate LLM API URLs concurrently; set config.LLM_API_URL to the first reachable one (in priority order)."""
    if not config.LLM_API_KEY:
        logger.warning("[Health] LLM API key not configured — run setup.py")
        return

    candidates = config.LLM_API_CANDIDATES
    logger.info(f"[Health] Probing LLM API candidates: {candidates}")

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(_probe_llm_url(client, url) for url in candidates))

    reachable = {r for r in results if r}
    for url in candidates:  # honour priority order
        if url in reachable:
            if url != config.LLM_API_URL:
                logger.info(f"[Health] LLM API found at {url} (updating from {config.LLM_API_URL})")
                config.LLM_API_URL = url
            else:
                logger.info(f"[Health] LLM API reachable at {url}")
            return

    logger.warning(f"[Health] LLM API unreachable at all candidates: {candidates}")


async def _register_bot() -> None:
    """Restore saved API key or register a new bot with Messenger."""
    saved_key = _load_saved_key()
    if saved_key:
        messenger.set_api_key(saved_key)
        logger.info("[Messenger] Restored API key from disk")
    else:
        logger.info(f"[Messenger] Base URL: {config.MESSENGER_URL}")
        key = await with_retry(
            messenger.register_bot,
            config.MESSENGER_BOT_NAME,
            max_attempts=config.STARTUP_RETRY_ATTEMPTS,
            base_delay=config.STARTUP_RETRY_DELAY,
            label="Messenger bot registration",
            # Registration mints a new key: a resend after the request reached
            # Messenger would orphan the first one. Retry connect-phase only.
            idempotent=False,
        )
        messenger.set_api_key(key)
        _save_key(key)
        logger.info("[Messenger] Bot registered and key saved")


async def _fetch_bot_identity() -> None:
    """Fetch and cache the bot's user ID for session context injection."""
    bot_info = await messenger.get_bot_info()
    if bot_info:
        config.BOT_USER_ID = bot_info["id"]
        logger.info(f"[Hoonbot] Bot user ID: {config.BOT_USER_ID}")
    else:
        logger.warning("[Hoonbot] Could not fetch bot identity — bot_user_id will be 0 in context")


async def _subscribe_webhooks() -> None:
    """Register webhook subscriptions, re-registering the bot if the key is stale."""
    webhook_url = config.HOONBOT_WEBHOOK_URL
    logger.info(f"[Messenger] Webhook target: {webhook_url}")

    retries = config.STARTUP_RETRY_ATTEMPTS
    delay = config.STARTUP_RETRY_DELAY

    try:
        await with_retry(
            messenger.register_webhook, webhook_url, _WEBHOOK_EVENTS,
            max_attempts=retries, base_delay=delay,
            label="Messenger webhook registration",
        )
        return
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 401:
            raise

    # API key rejected — re-register bot and retry
    logger.warning("[Messenger] API key unauthorized, re-registering bot")
    key = await with_retry(
        messenger.register_bot, config.MESSENGER_BOT_NAME,
        max_attempts=retries, base_delay=delay,
        label="Messenger bot registration",
        idempotent=False,
    )
    messenger.set_api_key(key)
    _save_key(key)
    logger.info("[Messenger] Bot key refreshed and saved")

    await with_retry(
        messenger.register_webhook, webhook_url, _WEBHOOK_EVENTS,
        max_attempts=retries, base_delay=delay,
        label="Messenger webhook registration (after key refresh)",
    )


async def _resolve_home_room() -> None:
    """Resolve home room by name, then fall back to configured numeric ID."""
    fallback_id = config.MESSENGER_HOME_ROOM_ID
    candidates: list[str] = []
    for name in (config.MESSENGER_HOME_ROOM_NAME, "Heartbeat", "heartbeat"):
        name = (name or "").strip()
        if name and name not in candidates:
            candidates.append(name)

    bot_info = await messenger.get_bot_info()
    if not bot_info:
        config.MESSENGER_HOME_ROOM_ID = fallback_id
        logger.warning(
            "[Hoonbot] Could not fetch bot info for home room resolution; "
            f"using configured room id={fallback_id}"
        )
        return

    for name in candidates:
        resolved_id = await messenger.resolve_home_room_by_name(name, bot_info["id"])
        if resolved_id is not None:
            config.MESSENGER_HOME_ROOM_ID = resolved_id
            logger.info(f"[Hoonbot] Home room resolved: '{name}' -> id={resolved_id}")
            return

    config.MESSENGER_HOME_ROOM_ID = fallback_id
    logger.warning(
        "[Hoonbot] Home room name not found "
        f"({', '.join(candidates) or 'no name configured'}); "
        f"using configured room id={fallback_id}"
    )
    return


# The webhook queue marks a message that is waiting for the next turn with the
# bot's ⏳ reaction. Unanswered messages are no longer *detected* through it —
# handlers.webhook persists queued / in-flight msg_ids in data/room_queue.json
# — but a message replayed from that file may still carry the previous run's
# ⏳, which the new queue item does not own, so catch-up clears it.
_WAITING_REACTION = "⏳"


def _is_human(msg: dict) -> bool:
    return msg.get("senderName") != config.MESSENGER_BOT_NAME and not msg.get("isBot")


def _is_candidate(msg) -> bool:
    """A live, human message. Whether it is actually replayable (text or
    attachments, room gating) is decided by enqueue_catchup_message."""
    return isinstance(msg, dict) and _is_human(msg) and not msg.get("isDeleted")


def _bot_reacted_waiting(msg: dict, bot_id) -> bool:
    if bot_id is None:
        return False
    for entry in msg.get("reactions") or []:
        if isinstance(entry, dict) and entry.get("emoji") == _WAITING_REACTION:
            return str(bot_id) in {str(u) for u in (entry.get("userIds") or [])}
    return False


def _id_sort_key(msg: dict):
    try:
        return (0, int(msg.get("id")))
    except (TypeError, ValueError):
        return (1, 0)


async def _catch_up_room(room_id, semaphore: asyncio.Semaphore, bot_id=None,
                         persisted_ids: list | None = None) -> None:
    """Replay one room: every human message after the bot's last post, plus
    the messages the previous run left queued / in flight (*persisted_ids*,
    from data/room_queue.json), in id order."""
    from handlers import webhook as _webhook

    async with semaphore:
        persisted_ids = list(persisted_ids or [])
        messages = await messenger.get_room_messages(room_id, limit=config.CATCHUP_MESSAGE_LIMIT) or []

        # Only THIS bot's own messages mark the "already replied" cut-off
        # (other bots' posts must not hide missed messages).
        last_bot_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("senderName") == config.MESSENGER_BOT_NAME:
                last_bot_idx = i

        window = {str(m.get("id")): m for m in messages if isinstance(m, dict) and m.get("id") is not None}
        missed: dict = {}
        for i, msg in enumerate(messages):
            if i > last_bot_idx and _is_candidate(msg) and msg.get("id") is not None:
                missed[str(msg["id"])] = msg

        # Messages accepted by the previous run but never answered. They may be
        # older than the bot's last post (a later turn, heartbeat or relay
        # result) or outside the history window, so fetch them by id.
        accepted_keys: set = set()
        forget = getattr(_webhook, "forget_persisted", None)
        for mid in persisted_ids:
            key = str(mid)
            msg = window.get(key)
            if msg is None:
                try:
                    msg = await messenger.get_message(room_id, mid)
                except Exception as exc:
                    # Messenger unreachable: keep the id for the next start.
                    logger.warning(f"[CatchUp] Room {room_id}: could not fetch persisted message {mid}: {exc}")
                    continue
            if not _is_candidate(msg):
                logger.info(f"[CatchUp] Room {room_id}: persisted message {mid} is gone or deleted — dropped")
                if callable(forget):
                    forget(room_id, mid)
                continue
            missed[key] = msg
            accepted_keys.add(key)

        if not missed:
            return
        missed_msgs = sorted(missed.values(), key=_id_sort_key)

        # Prefer the per-room queue (handlers.webhook owns it): replay EVERY
        # missed human message, in id order, so nothing sent while the bot
        # was offline is dropped and the queue's msg_id dedupe guarantees a
        # message is never answered twice. Attachments are rebuilt by
        # enqueue_catchup_message exactly like the live webhook does.
        enqueue = getattr(_webhook, "enqueue_catchup_message", None)
        if callable(enqueue):
            logger.info(
                f"[CatchUp] Room {room_id}: queueing {len(missed_msgs)} missed message(s)"
                + (f" ({len(accepted_keys)} left queued by the previous run)" if accepted_keys else "")
            )
            prev: tuple | None = None  # (sender, accepted) of the previous replayed message
            for m in missed_msgs:
                key = str(m.get("id"))
                accepted = key in accepted_keys
                if accepted and _bot_reacted_waiting(m, bot_id):
                    try:
                        await messenger.remove_reaction(room_id, m.get("id"), _WAITING_REACTION)
                    except Exception:
                        pass
                sender = m.get("senderName")
                continuation = bool(prev and prev[0] == sender and prev[1])
                try:
                    result = await enqueue(room_id, m, continuation=continuation, accepted=accepted)
                except Exception as exc:
                    logger.error(f"[CatchUp] Room {room_id}: replay of message {m.get('id')} failed: {exc}")
                    result = "error"
                prev = (sender, result in ("queued", "duplicate", "clear"))
            return

        # Legacy path: answer only the last missed human text message directly.
        texts = [m for m in missed_msgs if m.get("type") == "text" and (m.get("content") or "").strip()]
        if not texts:
            return
        missed_msg = texts[-1]
        content = missed_msg.get("content", "").strip()
        sender = missed_msg.get("senderName", "unknown")
        msg_id = missed_msg.get("id")
        logger.info(f"[CatchUp] Room {room_id}: missed msg from {sender!r}: {content[:50]!r}")
        await _webhook.process_message(room_id, content, sender, msg_id)


async def _catch_up() -> None:
    """Replay human messages the bot missed in each room (offline period, or
    still queued / in flight when the previous run stopped)."""
    from handlers import webhook as _webhook

    try:
        persisted = _webhook.persisted_queue() if hasattr(_webhook, "persisted_queue") else {}
    except Exception as exc:
        logger.warning(f"[CatchUp] Could not read persisted queue state: {exc}")
        persisted = {}

    bot_info = await messenger.get_bot_info()
    if not bot_info:
        logger.warning("[CatchUp] Could not get bot info, skipping (persisted queue kept for the next start)")
        return

    rooms = await messenger.get_rooms(bot_info["id"]) or []
    room_ids: list = []
    seen: set = set()
    for room in rooms:
        rid = room.get("id") if isinstance(room, dict) else None
        if rid is None or str(rid) in seen:
            continue
        seen.add(str(rid))
        room_ids.append(rid)
    # Rooms with persisted messages are replayed even when the room list does
    # not include them (a room the bot left drops its ids on fetch).
    for rid in persisted:
        if str(rid) not in seen:
            seen.add(str(rid))
            room_ids.append(rid)
    pending_by_room = {str(k): v for k, v in persisted.items()}

    logger.info(
        f"[CatchUp] Scanning {len(room_ids)} room(s) for missed messages"
        + (f" ({sum(len(v) for v in persisted.values())} persisted)" if persisted else "")
    )
    if not room_ids:
        return

    semaphore = asyncio.Semaphore(_CATCHUP_MAX_CONCURRENCY)
    results = await asyncio.gather(
        *(
            _catch_up_room(rid, semaphore, bot_info.get("id"), pending_by_room.get(str(rid)))
            for rid in room_ids
        ),
        return_exceptions=True,
    )
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"[CatchUp] Room catch-up failed: {result}", exc_info=result)


# Slaves run no heartbeat loop and no webhook post-turn hook, so they re-check
# memory.md on this period (cheap when under the soft cap: one file read).
_SLAVE_MEMORY_CHECK_SECONDS = 300.0


def _start_memory_guard(repeat_every: float | None = None) -> asyncio.Task:
    """Schedule a memory.md size check in the background: once, or — with
    *repeat_every* (seconds) — at startup and then periodically."""
    from core import memory_guard

    async def _run() -> None:
        try:
            size = memory_guard.memory_size()
            logger.info(
                f"[Hoonbot] memory.md: {size:,} / {memory_guard.MEMORY_MAX_CHARS:,} chars "
                f"(soft cap {memory_guard.MEMORY_SOFT_CHARS:,})"
            )
            await memory_guard.enforce_memory_cap("startup")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # enforce_memory_cap never raises; belt and braces
            logger.warning(f"[Hoonbot] Startup memory guard failed: {exc}")
        if not repeat_every:
            return
        while True:
            await asyncio.sleep(repeat_every)
            try:
                await memory_guard.enforce_memory_cap("periodic")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[Hoonbot] Periodic memory guard failed: {exc}")

    task = asyncio.create_task(_run())
    task.add_done_callback(
        lambda t: None if t.cancelled() or not t.exception()
        else logger.error(f"[Hoonbot] Background task 'memory_guard' died: {t.exception()}")
    )
    return task

# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(config.DATA_DIR, exist_ok=True)

    if getattr(config, "CLUSTER_ROLE", "master") == "slave":
        from core.cluster_worker import run_slave_worker_loop

        worker_task = asyncio.create_task(run_slave_worker_loop())
        # Periodic, not one-shot: delegated tasks may append to memory.md and
        # nothing else on a slave would ever condense it again.
        slave_memory_task = _start_memory_guard(repeat_every=_SLAVE_MEMORY_CHECK_SECONDS)
        logger.info(f"[Hoonbot] Slave node '{config.NODE_NAME}' ready on port {config.HOONBOT_PORT}")
        try:
            yield
        finally:
            worker_task.cancel()
            slave_memory_task.cancel()
            await asyncio.gather(worker_task, slave_memory_task, return_exceptions=True)
            await close_llm_client()
            logger.info("[Hoonbot] Slave shutdown complete")
        return

    await _autofind_llm_api()
    # Condense an oversized memory.md in the background (never blocks startup);
    # scheduled after LLM API discovery so it targets the right URL.
    memory_task = _start_memory_guard()
    try:
        await _register_bot()
        await _fetch_bot_identity()
        await _subscribe_webhooks()
        await _resolve_home_room()
    except BaseException:
        # Startup aborted: don't leave the background condense running.
        memory_task.cancel()
        raise

    logger.info(f"[Hoonbot] Ready on port {config.HOONBOT_PORT}")
    logger.info(
        f"[Hoonbot] Streaming={'on' if config.STREAMING_ENABLED else 'off'}, "
        f"Session max age={config.SESSION_MAX_AGE_DAYS}d, "
        f"Debounce={config.DEBOUNCE_SECONDS}s"
    )

    catchup_task = asyncio.create_task(_catch_up())
    heartbeat_task = asyncio.create_task(run_heartbeat_loop(messenger.send_message_once))

    # Relay finished cluster task results back to the rooms that requested them.
    # No-op unless this node is a master with delegation enabled.
    from core.cluster_relay import run_cluster_relay_loop
    relay_task = asyncio.create_task(run_cluster_relay_loop(messenger.send_message, messenger.send_file))

    def _on_task_done(task: asyncio.Task, name: str = "") -> None:
        if task.cancelled():
            logger.warning(f"[Hoonbot] Background task '{name}' was cancelled")
        elif exc := task.exception():
            logger.error(f"[Hoonbot] Background task '{name}' died: {exc}", exc_info=exc)

    catchup_task.add_done_callback(lambda t: _on_task_done(t, "catch_up"))
    heartbeat_task.add_done_callback(lambda t: _on_task_done(t, "heartbeat"))
    relay_task.add_done_callback(lambda t: _on_task_done(t, "cluster_relay"))

    yield

    # Stop catch-up first (unreplayed persisted ids simply stay on disk), then
    # drain the room queues: running turns are settled with a restart marker
    # and every unanswered msg_id is flushed to data/room_queue.json for the
    # next start's catch-up.
    catchup_task.cancel()
    await asyncio.gather(catchup_task, return_exceptions=True)
    try:
        from handlers import webhook as _webhook
        shutdown_room_workers = getattr(_webhook, "shutdown_room_workers", None)
        if callable(shutdown_room_workers):
            await shutdown_room_workers()
    except Exception as exc:
        logger.warning(f"[Hoonbot] Room worker shutdown failed: {exc}")

    heartbeat_task.cancel()
    relay_task.cancel()
    memory_task.cancel()
    await asyncio.gather(catchup_task, heartbeat_task, relay_task, memory_task, return_exceptions=True)
    await close_llm_client()
    await messenger.close_client()
    logger.info("[Hoonbot] Shutdown complete")


app = FastAPI(title="Hoonbot", lifespan=lifespan)
app.include_router(health_router)
app.include_router(webhook_router)


if __name__ == "__main__":
    uvicorn.run(
        "hoonbot:app",
        host=config.HOONBOT_HOST,
        port=config.HOONBOT_PORT,
        reload=False,
    )
