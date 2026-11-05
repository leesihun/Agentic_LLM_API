"""Async client for the Huni Messenger bot API with persistent connection pool."""
import asyncio
import logging
import os
import time
from collections import OrderedDict
from typing import Optional

import httpx
import config
from core.retry import CONNECT_PHASE_RETRYABLE, with_retry

logger = logging.getLogger(__name__)

# Runtime state — populated during startup
_api_key: str = ""

# Persistent HTTP client — shared across all requests
_client: Optional[httpx.AsyncClient] = None

# Cached bot identity from /api/bots/me
_bot_info_cache: Optional[dict] = None

# Room info cache: room_id -> {"name": str, "isGroup": bool, "memberCount": int}
_room_cache: dict[int, dict] = {}
# The bot only receives message webhooks, not membership/room changes, so the
# room cache is refreshed on this interval. Short enough that adding or removing
# members flips the bot's auto-respond behavior (see is_bot_solo_room) within
# seconds rather than persisting a stale member count.
_room_cache_at: float = 0.0
_ROOM_CACHE_TTL_SECONDS = 30.0

# A *final* edit longer than this is cut at a newline boundary and the rest is
# posted as follow-up messages (see edit_message). Never below the send limit.
EDIT_MAX_CHARS = max(
    int(getattr(config, "MAX_MESSAGE_LENGTH", 2000)),
    int(getattr(config, "MESSENGER_EDIT_MAX_CHARS", 3800)),
)

# message_id -> room_id for bubbles this process created via
# send_message_returning_id, so a final edit can post overflow follow-ups into
# the right room. Bounded LRU.
_message_rooms: "OrderedDict[int, int]" = OrderedDict()
_MESSAGE_ROOMS_MAX = 2000


def _remember_message_room(message_id, room_id: int) -> None:
    try:
        mid = int(message_id)
    except (TypeError, ValueError):
        return
    _message_rooms[mid] = room_id
    _message_rooms.move_to_end(mid)
    while len(_message_rooms) > _MESSAGE_ROOMS_MAX:
        _message_rooms.popitem(last=False)


def _get_client() -> httpx.AsyncClient:
    """Get or create the shared httpx client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            trust_env=False,
            # Many rooms can stream drafts concurrently (plus typing, reactions,
            # heartbeat/relay posts) — 10 connections queued edits behind each other.
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return _client


async def close_client() -> None:
    """Close the shared client. Call during application shutdown."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


def set_api_key(key: str) -> None:
    global _api_key, _bot_info_cache, _room_cache_at
    _api_key = key
    _bot_info_cache = None
    _room_cache.clear()
    _room_cache_at = 0.0
    config.MESSENGER_API_KEY = key


def _headers() -> dict:
    return {"x-api-key": _api_key, "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Bot registration & webhooks
# ---------------------------------------------------------------------------

async def register_bot(name: str) -> str:
    """Register bot with Messenger and return its API key."""
    client = _get_client()
    resp = await client.post(
        f"{config.MESSENGER_URL}/api/bots",
        json={"name": name},
    )
    if resp.status_code == 409:
        raise RuntimeError(
            f'Messenger bot name conflict for "{name}". '
            "A non-bot user already has this name. "
            "Set HOONBOT_BOT_NAME to a unique bot name."
        )
    resp.raise_for_status()
    data = resp.json()
    key = data.get("apiKey") or data.get("key") or data.get("api_key", "")
    bot_id = data.get("bot", {}).get("id") or data.get("id")
    logger.info(f"[Messenger] Bot registered: {name} (id={bot_id})")
    return key


async def register_webhook(url: str, events: list) -> None:
    """Subscribe to Messenger events. Idempotent."""
    client = _get_client()
    resp = await client.get(
        f"{config.MESSENGER_URL}/api/webhooks",
        headers=_headers(),
    )
    if resp.status_code == 200:
        existing = resp.json()
        for wh in existing:
            if wh.get("url") == url:
                logger.info(f"[Messenger] Webhook already registered: {url}")
                return

    resp = await client.post(
        f"{config.MESSENGER_URL}/api/webhooks",
        headers=_headers(),
        json={"url": url, "events": events},
    )
    resp.raise_for_status()
    logger.info(f"[Messenger] Webhook registered: {url} for events={events}")


# ---------------------------------------------------------------------------
# Messages — send, edit, delete
# ---------------------------------------------------------------------------

def _split_message(text: str, limit: int) -> list:
    """Split text into chunks that respect paragraph and line boundaries."""
    if len(text) <= limit:
        return [text]

    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n\n", 0, limit)
        if cut == -1:
            cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = text.rfind(" ", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    return chunks


def _definitely_not_delivered(exc: BaseException) -> bool:
    """True when a failed send provably never created the message: a
    connect-phase error, a 4xx rejection, or a 502/503/504 gateway status.
    Read-phase errors and bare 5xx are ambiguous (the server may have stored
    and broadcast the message before the response was lost)."""
    if isinstance(exc, CONNECT_PHASE_RETRYABLE):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status < 500 or status in (502, 503, 504)
    return False


async def _post_text_message(room_id: int, content: str, reply_to_id: int | None = None) -> dict:
    client = _get_client()
    body = {"roomId": room_id, "content": content, "type": "text"}
    if reply_to_id:
        body["replyToId"] = reply_to_id
    resp = await client.post(
        f"{config.MESSENGER_URL}/api/send-message",
        headers=_headers(),
        json=body,
    )
    resp.raise_for_status()
    return resp.json()


async def send_message_once(room_id: int, content: str, reply_to_id: int | None = None) -> None:
    """Send exactly one message. Used for final heartbeat reports."""
    async def _send():
        await _post_text_message(room_id, content, reply_to_id)

    # Non-idempotent: a resend after the request reached the server would post
    # a duplicate, so only connect-phase errors / 502-504 are retried.
    await with_retry(_send, label="Messenger send once", max_attempts=3, idempotent=False)


async def send_message(room_id: int, content: str, reply_to_id: int | None = None) -> None:
    """Send a message, automatically splitting if it exceeds the character limit."""
    chunks = _split_message(content, config.MAX_MESSAGE_LENGTH)
    for i, chunk in enumerate(chunks):
        async def _send(c=chunk, first=(i == 0)):
            await _post_text_message(room_id, c, reply_to_id if first else None)

        await with_retry(_send, label="Messenger send", max_attempts=3, idempotent=False)


class MaybeDeliveredError(Exception):
    """A send failed in a way that does not prove the message was NOT created
    (read-phase transport error, bare 5xx, unreadable/id-less 2xx response):
    Messenger may already have stored and broadcast it. Callers must not
    blindly resend — look the message up instead (or give up on it)."""


async def send_message_returning_id(
    room_id: int,
    content: str,
    reply_to_id: int | None = None,
) -> int | None:
    """Send a single message and return its ID (for later editing/deletion).

    No auto-splitting — callers that stream into this message should keep the
    initial content short (typically the first token of the streamed reply).

    Failure outcomes are distinguished so a caller never creates a second
    bubble next to one that was actually delivered:
      - returns ``None``: the message was definitely NOT created (connect-phase
        error, 4xx rejection, 502/503/504) — safe to try again later;
      - raises ``MaybeDeliveredError``: it may have been created (the request
        reached Messenger but the response was lost or carried no id).
    """
    async def _send():
        data = await _post_text_message(room_id, content, reply_to_id)
        if not isinstance(data, dict):
            return None
        message = data.get("message")
        return (message.get("id") if isinstance(message, dict) else None) or data.get("id")

    try:
        msg_id = await with_retry(_send, label="Messenger send (id)", max_attempts=3, idempotent=False)
    except Exception as exc:
        if _definitely_not_delivered(exc):
            logger.warning(f"[Messenger] send (id) to room {room_id} failed, not delivered: "
                           f"{type(exc).__name__}: {exc}")
            return None
        logger.warning(f"[Messenger] send (id) to room {room_id} failed, may have been delivered: "
                       f"{type(exc).__name__}: {exc}")
        raise MaybeDeliveredError(f"{type(exc).__name__}: {exc}") from exc
    if msg_id is None:
        logger.warning(f"[Messenger] send (id) to room {room_id}: response carried no message id")
        raise MaybeDeliveredError("send succeeded but the response carried no message id")
    _remember_message_room(msg_id, room_id)
    return msg_id


async def _post_edit(message_id: int, content: str, draft: bool, max_attempts: int) -> bool:
    """POST /api/edit-message with retry. Returns True on success, never raises
    (except CancelledError)."""
    async def _edit():
        client = _get_client()
        resp = await client.post(
            f"{config.MESSENGER_URL}/api/edit-message",
            headers=_headers(),
            json={"messageId": message_id, "content": content, "draft": draft},
        )
        resp.raise_for_status()

    try:
        await with_retry(_edit, label="Messenger edit", max_attempts=max_attempts)
        return True
    except httpx.HTTPStatusError as exc:
        logger.warning(
            f"[Messenger] Edit of message {message_id} rejected "
            f"(HTTP {exc.response.status_code}): {exc.response.text[:200]}"
        )
    except Exception as exc:
        logger.warning(f"[Messenger] Edit of message {message_id} failed: {type(exc).__name__}: {exc}")
    return False


async def edit_message(
    message_id: int,
    content: str,
    draft: bool = True,
    *,
    final: bool | None = None,
    room_id: int | None = None,
    progress: dict | None = None,
) -> bool:
    """Edit a previously sent message. Returns True on success (never raises).

    draft=True (the default — every caller is a live streaming update) tells the
    server this is a streaming draft edit, not a user edit. The server then skips
    the per-edit message_edited webhook dispatch (which otherwise fires ~4×/s
    with a growing payload) and the is_edited flag, so streaming stays cheap. The
    room broadcast still happens so clients see the live text.

    Overlong text: on a *final* edit (``final=True``, or ``draft=False`` when
    *final* is not given) text longer than ``EDIT_MAX_CHARS`` is cut at a
    newline boundary — the bubble keeps the head, and the remainder is posted
    as follow-up messages (each ≤ ``EDIT_MAX_CHARS``) to the message's room
    (*room_id*, or the room remembered when the message was sent via
    ``send_message_returning_id``). Live draft edits are never split — they
    would spawn a new follow-up bubble on every throttled update — so an
    overlong draft is sent whole (Messenger itself imposes no length cap).
    Returns False if the edit or any follow-up failed, so callers can fall
    back to ``send_message``.

    *progress* (optional, caller-owned dict) makes a retried final write
    idempotent: pass the SAME dict on every attempt for the same text and
    follow-up chunks already posted are not posted again. After each call it
    holds ``head_done`` (the bubble now shows the head / whole text) and
    ``pending`` (follow-up chunks still unposted), so a caller falling back to
    ``send_message`` after a failure sends only ``pending`` when ``head_done``
    — never the full text a second time. A different text resets it.
    """
    is_final = (not draft) if final is None else bool(final)
    text = content if isinstance(content, str) else str(content)
    attempts = 3 if is_final else 2

    if progress is not None:
        ident = (len(text), hash(text))
        if progress.get("ident") != ident:
            progress.clear()
            progress.update(ident=ident, head_done=False, next=1, pending=[])

    async def _edit_whole() -> bool:
        ok = await _post_edit(message_id, text, draft, attempts)
        if ok and progress is not None:
            progress.update(head_done=True, pending=[])
        return ok

    if not is_final or len(text) <= EDIT_MAX_CHARS:
        return await _edit_whole()

    target_room = room_id if room_id is not None else _message_rooms.get(message_id)
    if target_room is None:
        # No room to post the overflow into — keep everything in one bubble
        # rather than dropping the tail.
        logger.info(
            f"[Messenger] Final edit of {message_id} is {len(text)} chars but its room "
            "is unknown — sending unsplit"
        )
        return await _edit_whole()

    chunks = _split_message(text, EDIT_MAX_CHARS)
    state = progress if progress is not None else {"head_done": False, "next": 1}
    start = max(1, int(state.get("next") or 1))
    state["pending"] = [c for c in chunks[start:] if c.strip()]
    if not state.get("head_done"):
        if not await _post_edit(message_id, chunks[0], draft, attempts):
            return False
        state["head_done"] = True
    for idx in range(start, len(chunks)):
        chunk = chunks[idx]
        if chunk.strip():
            async def _send(c=chunk):
                await _post_text_message(int(target_room), c)
            try:
                # Non-idempotent like send_message: a plain POST creates a new
                # message every time, so only connect-phase errors / 502-504
                # are retried — never a failure after the server got it.
                await with_retry(_send, label="Messenger edit follow-up", max_attempts=3,
                                 idempotent=False)
            except Exception as exc:
                if not _definitely_not_delivered(exc):
                    # Ambiguous (read-phase error / bare 5xx): Messenger may
                    # already have stored + broadcast the chunk. Treat it as
                    # delivered rather than risk posting it again on the
                    # caller's retry or pending-chunk fallback.
                    logger.warning(
                        f"[Messenger] Follow-up chunk {idx}/{len(chunks) - 1} for message "
                        f"{message_id} may or may not have been posted "
                        f"({type(exc).__name__}: {exc}) — not resending"
                    )
                    state["next"] = idx + 1
                    state["pending"] = [c for c in chunks[idx + 1:] if c.strip()]
                    continue
                logger.error(
                    f"[Messenger] Follow-up chunk {idx}/{len(chunks) - 1} for message "
                    f"{message_id} failed: {type(exc).__name__}: {exc}"
                )
                return False
        state["next"] = idx + 1
        state["pending"] = [c for c in chunks[idx + 1:] if c.strip()]
    return True


async def delete_message(message_id: int) -> None:
    """Soft-delete a previously sent message."""
    try:
        async def _delete():
            client = _get_client()
            resp = await client.post(
                f"{config.MESSENGER_URL}/api/delete-message",
                headers=_headers(),
                json={"messageId": message_id},
            )
            resp.raise_for_status()

        await with_retry(_delete, label="Messenger delete", max_attempts=2)
    except Exception:
        pass  # Best-effort


# ---------------------------------------------------------------------------
# Reactions — POST /api/reactions toggles the bot's own reaction
# ---------------------------------------------------------------------------

async def _toggle_reaction(message_id: int, emoji: str) -> list | None:
    """Toggle the bot's *emoji* reaction; return the message's reaction list
    (``[{emoji, userIds, userNames}]``) or None on failure."""
    async def _toggle():
        client = _get_client()
        resp = await client.post(
            f"{config.MESSENGER_URL}/api/reactions",
            headers=_headers(),
            json={"messageId": message_id, "emoji": emoji},
        )
        resp.raise_for_status()
        return resp.json()

    # Non-idempotent: the endpoint is a toggle, so a resend after the first
    # request reached the server would flip the reaction straight back.
    try:
        data = await with_retry(_toggle, label="Messenger reaction", max_attempts=2, idempotent=False)
    except Exception as exc:
        logger.debug(f"[Messenger] Reaction toggle on {message_id} failed: {type(exc).__name__}: {exc}")
        return None
    reactions = data.get("reactions") if isinstance(data, dict) else None
    return reactions if isinstance(reactions, list) else []


def _bot_has_reaction(reactions: list, emoji: str, bot_id) -> bool | None:
    """Whether the bot's reaction is present; None when it can't be told."""
    if bot_id is None:
        return None
    for entry in reactions:
        if isinstance(entry, dict) and entry.get("emoji") == emoji:
            return bot_id in (entry.get("userIds") or [])
    return False


_REACTION_LOCKS: "OrderedDict[tuple, asyncio.Lock]" = OrderedDict()
_REACTION_LOCKS_CAP = 512


def _reaction_lock(message_id: int, emoji: str) -> asyncio.Lock:
    """Per-(message, emoji) lock: the endpoint is a toggle, so an add and a
    remove racing on the same reaction could each flip it and leave the
    opposite of the last-requested state."""
    key = (message_id, emoji)
    lock = _REACTION_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _REACTION_LOCKS[key] = lock
        while len(_REACTION_LOCKS) > _REACTION_LOCKS_CAP:
            oldest_key, oldest = next(iter(_REACTION_LOCKS.items()))
            if oldest.locked():
                break  # still in use; allow temporary overshoot
            del _REACTION_LOCKS[oldest_key]
    else:
        _REACTION_LOCKS.move_to_end(key)
    return lock


async def _set_reaction(message_id: int, emoji: str, present: bool) -> bool:
    """Drive the toggle endpoint to the desired state (idempotent)."""
    async with _reaction_lock(message_id, emoji):
        return await _set_reaction_locked(message_id, emoji, present)


async def _set_reaction_locked(message_id: int, emoji: str, present: bool) -> bool:
    bot_info = await get_bot_info()
    bot_id = bot_info.get("id") if isinstance(bot_info, dict) else None
    reactions = await _toggle_reaction(message_id, emoji)
    if reactions is None:
        return False
    state = _bot_has_reaction(reactions, emoji, bot_id)
    if state is None or state == present:
        return True
    # The toggle went the wrong way (reaction was already in the desired
    # state) — toggle back.
    reactions = await _toggle_reaction(message_id, emoji)
    if reactions is None:
        return False
    state = _bot_has_reaction(reactions, emoji, bot_id)
    return state is None or state == present


async def add_reaction(room_id: int, message_id: int, emoji: str) -> bool:
    """Ensure the bot has reacted with *emoji* on *message_id*. Best-effort:
    returns True on success, False on failure; never raises."""
    try:
        return await _set_reaction(int(message_id), emoji, True)
    except Exception as exc:
        logger.debug(f"[Messenger] add_reaction({room_id}, {message_id}) failed: {exc}")
        return False


async def remove_reaction(room_id: int, message_id: int, emoji: str) -> bool:
    """Ensure the bot's *emoji* reaction on *message_id* is gone. Best-effort:
    returns True on success, False on failure; never raises."""
    try:
        return await _set_reaction(int(message_id), emoji, False)
    except Exception as exc:
        logger.debug(f"[Messenger] remove_reaction({room_id}, {message_id}) failed: {exc}")
        return False


async def mark_read(room_id: int, message_ids: list) -> None:
    """Mark messages as read (best-effort)."""
    if not message_ids:
        return
    try:
        client = _get_client()
        await client.post(
            f"{config.MESSENGER_URL}/api/mark-read",
            headers=_headers(),
            json={"roomId": room_id, "messageIds": message_ids},
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Typing indicators
# ---------------------------------------------------------------------------

async def send_typing(room_id: int, status_text: str | None = None) -> None:
    """Show a typing indicator. Optional status_text replaces the default verb.

    When omitted, the client renders the native "{name}님이 입력 중...".
    When provided, it renders "{name}님이 {status_text}...".
    """
    body: dict = {"roomId": room_id}
    if status_text:
        body["statusText"] = status_text
    try:
        client = _get_client()
        await client.post(
            f"{config.MESSENGER_URL}/api/typing",
            headers=_headers(),
            json=body,
        )
    except Exception:
        pass  # Best-effort


async def stop_typing(room_id: int) -> None:
    try:
        client = _get_client()
        await client.post(
            f"{config.MESSENGER_URL}/api/stop-typing",
            headers=_headers(),
            json={"roomId": room_id},
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Bot info & room queries
# ---------------------------------------------------------------------------

async def get_bot_info(refresh: bool = False) -> Optional[dict]:
    global _bot_info_cache
    if _bot_info_cache is not None and not refresh:
        return _bot_info_cache

    try:
        client = _get_client()
        resp = await client.get(
            f"{config.MESSENGER_URL}/api/bots/me",
            headers=_headers(),
        )
        if resp.status_code == 200:
            _bot_info_cache = resp.json()
            return _bot_info_cache
    except Exception:
        pass
    return None


async def _reload_room_cache() -> None:
    """Refresh the room cache (name, isGroup, member count) from /api/rooms."""
    global _room_cache_at
    bot_info = await get_bot_info()
    if not bot_info:
        return
    rooms = await get_rooms(bot_info["id"])
    if not rooms:
        return
    fresh: dict[int, dict] = {}
    for room in rooms:
        fresh[room["id"]] = {
            "name": room.get("name") or str(room["id"]),
            "isGroup": bool(room.get("isGroup", False)),
            "memberCount": len(room.get("members") or []),
        }
    _room_cache.clear()
    _room_cache.update(fresh)
    _room_cache_at = time.monotonic()


async def get_room_info(room_id: int) -> dict:
    """Return room metadata (name, isGroup, memberCount) for a given room ID.

    Refreshes the whole cache from /api/rooms when the room is unknown or the
    cache is older than _ROOM_CACHE_TTL_SECONDS, so membership changes are
    picked up promptly. Falls back to a synthetic entry if the room cannot be
    resolved (e.g., API unreachable, or a room the bot isn't a member of).
    """
    stale = (time.monotonic() - _room_cache_at) >= _ROOM_CACHE_TTL_SECONDS
    if room_id not in _room_cache or stale:
        await _reload_room_cache()
    return _room_cache.get(room_id, {"name": str(room_id), "isGroup": False, "memberCount": 0})


async def is_dm_room(room_id: int) -> bool:
    """Return True only when the room is positively confirmed to be a 1-on-1 DM.

    Unknown rooms (cache miss after lookup attempt — e.g., API unreachable, or a
    room the bot isn't a member of) return False. This preserves the existing
    "require @mention in unfamiliar rooms" behavior; the DM bypass only fires
    when we actually verified the room is not a group.
    """
    info = await get_room_info(room_id)
    if room_id not in _room_cache:
        return False
    return not info.get("isGroup", True)


async def is_bot_solo_room(room_id: int) -> bool:
    """True when the room holds exactly the bot and one other user (2 members).

    In such a room the bot answers every message with no @mention required —
    the same behavior as the heartbeat home room and 1:1 DMs. The bot is only
    ever a member of rooms it can see, so a 2-member room is always bot + one
    human. Adding a third member (Messenger 'add members') reverts the room to
    @mention-only automatically once the cache refreshes.
    """
    info = await get_room_info(room_id)
    if room_id not in _room_cache:
        return False
    return info.get("memberCount", 0) == 2


async def get_rooms(bot_user_id: int) -> list:
    try:
        client = _get_client()
        resp = await client.get(
            f"{config.MESSENGER_URL}/api/rooms",
            headers=_headers(),
            params={"userId": bot_user_id},
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.warning(f"[Messenger] get_rooms failed: {exc}")
    return []


async def resolve_home_room_by_name(name: str, bot_user_id: int) -> Optional[int]:
    """
    Find the room whose name matches `name` (case-insensitive) among the bot's
    rooms and return its ID. Returns None if no match is found.
    """
    rooms = await get_rooms(bot_user_id)
    name_lower = name.strip().lower()
    for room in rooms:
        room_name = (room.get("name") or "").strip().lower()
        if room_name == name_lower:
            return room["id"]
    return None


async def download_file(file_url: str) -> tuple[bytes, str] | None:
    """Download a file from Messenger given a relative fileUrl path.

    Returns (file_bytes, filename) or None on failure.
    """
    try:
        client = _get_client()
        full_url = f"{config.MESSENGER_URL}{file_url}"
        resp = await client.get(full_url, headers={"x-api-key": _api_key})
        resp.raise_for_status()
        filename = file_url.rsplit("/", 1)[-1]
        return resp.content, filename
    except Exception as e:
        logger.warning(f"[Messenger] File download failed for {file_url}: {e}")
        return None


async def send_file(room_id: int, file_path: str, caption: str | None = None) -> int:
    """Upload a local file to a room. Returns message_id on success."""
    client = _get_client()
    with open(file_path, "rb") as f:
        data = {"roomId": str(room_id)}
        if caption:
            data["content"] = caption
        resp = await client.post(
            f"{config.MESSENGER_URL}/api/send-file",
            headers={"x-api-key": _api_key},  # no Content-Type — httpx sets multipart boundary
            data=data,
            files={"file": (os.path.basename(file_path), f)},
        )
    resp.raise_for_status()
    result = resp.json()
    msg_id = result.get("message", {}).get("id") or result.get("id")
    if not msg_id:
        raise ValueError("message_id missing in upload response")
    return msg_id


async def send_base64(room_id: int, data_url: str, file_name: str | None = None, caption: str | None = None) -> int:
    """Send a base64 image to a room. data_url must include the full data URL prefix
    (e.g. 'data:image/png;base64,...'). Returns message_id on success."""
    client = _get_client()
    body: dict = {"roomId": room_id, "data": data_url}
    if file_name:
        body["fileName"] = file_name
    if caption:
        body["content"] = caption
    resp = await client.post(
        f"{config.MESSENGER_URL}/api/send-base64",
        headers={"x-api-key": _api_key},
        json=body,  # httpx sets Content-Type: application/json automatically
    )
    resp.raise_for_status()
    result = resp.json()
    msg_id = result.get("message", {}).get("id") or result.get("id")
    if not msg_id:
        raise ValueError("message_id missing in upload response")
    return msg_id


async def get_room_messages(room_id: int, limit: int = 20, *, before: int | None = None) -> list:
    """Newest *limit* messages of a room (oldest first); with *before*, only
    messages whose id is lower than it. Returns [] on any failure."""
    params: dict = {"limit": limit}
    if before is not None:
        params["before"] = before
    try:
        client = _get_client()
        resp = await client.get(
            f"{config.MESSENGER_URL}/api/messages/{room_id}",
            headers=_headers(),
            params=params,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.warning(f"[Messenger] get_room_messages({room_id}) failed: {exc}")
    return []


async def get_message(room_id: int, message_id: int) -> Optional[dict]:
    """Fetch one message of *room_id* by id (``before=id+1, limit=1`` on the
    room history endpoint).

    Returns the message dict, or None when it provably does not exist for the
    bot (no such message in the room, room gone, bot not a member). Raises on
    transport errors and other HTTP failures so a caller can tell "gone" from
    "could not check"."""
    mid = int(message_id)
    client = _get_client()
    resp = await client.get(
        f"{config.MESSENGER_URL}/api/messages/{room_id}",
        headers=_headers(),
        params={"limit": 1, "before": mid + 1},
    )
    if resp.status_code in (403, 404):
        return None
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        for msg in data:
            if isinstance(msg, dict) and str(msg.get("id")) == str(mid):
                return msg
    return None
