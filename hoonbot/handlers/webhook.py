"""
Webhook handler — receives events from Messenger and processes them.

Supports: new_message, message_edited, message_deleted.
Features: streaming responses, reply threading, file/image handling,
          session lifecycle, structured logging with timing.
"""
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request

import config
from core import messenger
from core.cluster_client import try_submit_from_message
from core.context import build_llm_context, build_per_turn_context
from core.llm_api import get_client

logger = logging.getLogger(__name__)

_MEMORY_FLUSH_HINT = config.read_prompt("webhook/memory_flush_hint.txt")
router = APIRouter()

# Per-room debounce state
_room_debounce: dict = {}

# Per-room active processing task (set while process_message is running)
_room_active_task: dict[int, asyncio.Task] = {}

# Content/files of the turn currently being processed, kept so a follow-up that
# interrupts an in-flight turn can be merged with it instead of discarding it.
# Populated right before process_message runs; cleared when it finishes.
_room_inflight: dict[int, dict] = {}

# Per-room message counter (resets on new session)
_room_msg_count: dict[int, int] = {}

# Per-room persistent session data: {room_id: {"session_id": str, "created_at": str}}
_SESSIONS_FILE = os.path.join(config.DATA_DIR, "room_sessions.json")
_room_sessions: dict = {}

# Per-room sticky goal-mode flag: {room_id: True}. Toggled by the /goal directive
# and forwarded to the LLM API as mode="goal" on every turn while set.
_GOAL_MODE_FILE = os.path.join(config.DATA_DIR, "room_goal_mode.json")
_room_goal_mode: dict[int, bool] = {}

# Leading `/goal` or `@goal` directive; group(1) is the remaining task text
# (or "off"/"stop"/... to disable). Anchored at the start so the token can be
# stripped and the rest processed as the task.
_GOAL_DIRECTIVE_RE = re.compile(r"^\s*[/@]goal\b[ \t]*(.*)$", re.IGNORECASE | re.DOTALL)
_GOAL_OFF_WORDS = {"off", "stop", "disable", "end", "끄기", "종료"}


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def _load_room_sessions() -> None:
    global _room_sessions
    try:
        with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            _room_sessions = {}
            for k, v in raw.items():
                if isinstance(v, str):
                    # Migrate old format: {room_id: session_id_string}
                    _room_sessions[int(k)] = {"session_id": v, "created_at": datetime.now(timezone.utc).isoformat()}
                else:
                    _room_sessions[int(k)] = v
        logger.info(f"[Sessions] Loaded {len(_room_sessions)} room session(s) from disk")
    except FileNotFoundError:
        _room_sessions = {}
    except Exception as e:
        logger.warning(f"[Sessions] Could not load room sessions: {e}")
        _room_sessions = {}


def _save_room_sessions() -> None:
    try:
        os.makedirs(os.path.dirname(_SESSIONS_FILE), exist_ok=True)
        with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in _room_sessions.items()}, f)
    except Exception as e:
        logger.warning(f"[Sessions] Could not save room sessions: {e}")


def _get_session_id(room_id: int) -> str | None:
    """Get session_id for a room, respecting max age."""
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
                del _room_sessions[room_id]
                _save_room_sessions()
                return None
        except (KeyError, ValueError):
            pass

    return session_id


def _set_session_id(room_id: int, session_id: str) -> None:
    _room_sessions[room_id] = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_room_sessions()


# ---------------------------------------------------------------------------
# Goal-mode persistence (sticky per-room; toggled by the /goal directive)
# ---------------------------------------------------------------------------

def _load_room_goal_mode() -> None:
    global _room_goal_mode
    try:
        with open(_GOAL_MODE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
            _room_goal_mode = {int(k): bool(v) for k, v in raw.items() if v}
        logger.info(f"[GoalMode] Loaded {len(_room_goal_mode)} room(s) in goal mode from disk")
    except FileNotFoundError:
        _room_goal_mode = {}
    except Exception as e:
        logger.warning(f"[GoalMode] Could not load goal-mode state: {e}")
        _room_goal_mode = {}


def _save_room_goal_mode() -> None:
    try:
        os.makedirs(os.path.dirname(_GOAL_MODE_FILE), exist_ok=True)
        with open(_GOAL_MODE_FILE, "w", encoding="utf-8") as f:
            json.dump({str(k): True for k, v in _room_goal_mode.items() if v}, f)
    except Exception as e:
        logger.warning(f"[GoalMode] Could not save goal-mode state: {e}")


# Load on module import
_load_room_sessions()
_load_room_goal_mode()


# ---------------------------------------------------------------------------
# Webhook entry point
# ---------------------------------------------------------------------------

@router.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    event = payload.get("event")
    data = payload.get("data", {})
    room_id = payload.get("roomId")

    # --- Handle message edits ---
    if event == "message_edited":
        logger.info(f"[Webhook] Message edited in room {room_id}: id={data.get('messageId')}")
        return {"ok": True}

    # --- Handle message deletes ---
    if event == "message_deleted":
        logger.info(f"[Webhook] Message deleted in room {room_id}: id={data.get('messageId')}")
        return {"ok": True}

    if event != "new_message":
        return {"ok": True}

    content = data.get("content", "").strip()
    msg_type = data.get("type", "text")
    sender_name = data.get("senderName") or data.get("sender_name", "")
    is_bot = data.get("isBot") or data.get("is_bot", False)
    msg_id = data.get("id")

    # Ignore bot messages
    if sender_name == config.MESSENGER_BOT_NAME or is_bot:
        return {"ok": True}

    # Handle file/image messages — download and forward to LLM API
    file_info = None
    file_infos = []
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

    if not file_infos and msg_type == "image":
        file_name = data.get("fileName", "image")
        file_url = data.get("fileUrl", "")
        if not file_url:
            logger.warning(f"[Webhook] Image message {msg_id} has no fileUrl; refusing to process as an image")
            is_home = room_id == config.MESSENGER_HOME_ROOM_ID
            mention_tag = f"@{config.MESSENGER_BOT_NAME}"
            is_solo = await messenger.is_bot_solo_room(room_id) if room_id is not None else False
            if room_id is not None and (is_home or is_solo or mention_tag.lower() in content.lower()):
                asyncio.create_task(
                    messenger.send_message(
                        room_id,
                        "이미지 파일을 찾을 수 없어 처리할 수 없어요. 다시 업로드해주세요.",
                        reply_to_id=msg_id,
                    )
                )
            return {"ok": False, "error": "Image message missing fileUrl"}
        file_info = {"url": file_url, "name": file_name}
        if not content:
            content = f"[Image: {file_name}]"
    elif not file_infos and msg_type == "file":
        file_name = data.get("fileName", "file")
        file_url = data.get("fileUrl", "")
        if not file_url:
            logger.warning(f"[Webhook] File message {msg_id} has no fileUrl; refusing to process as a file")
            is_home = room_id == config.MESSENGER_HOME_ROOM_ID
            mention_tag = f"@{config.MESSENGER_BOT_NAME}"
            is_solo = await messenger.is_bot_solo_room(room_id) if room_id is not None else False
            if room_id is not None and (is_home or is_solo or mention_tag.lower() in content.lower()):
                asyncio.create_task(
                    messenger.send_message(
                        room_id,
                        "파일을 찾을 수 없어 처리할 수 없어요. 다시 업로드해주세요.",
                        reply_to_id=msg_id,
                    )
                )
            return {"ok": False, "error": "File message missing fileUrl"}
        file_info = {"url": file_url, "name": file_name}
        if not content:
            content = f"[File: {file_name}]"
    elif not file_infos and msg_type != "text":
        return {"ok": True}

    if file_info:
        file_infos.append(file_info)
    if file_infos and not content:
        label = "Attachments" if len(file_infos) > 1 else ("Image" if file_infos[0].get("type") == "image" else "File")
        names = ", ".join(fi["name"] for fi in file_infos)
        content = f"[{label}: {names}]"

    if not content:
        return {"ok": True}

    # Respond unconditionally in:
    #   - the configured home room, OR
    #   - any room where the bot is alone with a single user (2 members total).
    #     This covers 1-on-1 DMs and 2-member group rooms alike. Adding a third
    #     member reverts the room to @mention-only once the cache refreshes.
    # Other rooms still require an explicit @mention.
    is_home = room_id == config.MESSENGER_HOME_ROOM_ID
    is_solo = await messenger.is_bot_solo_room(room_id)
    mention_tag = f"@{config.MESSENGER_BOT_NAME}"
    if not is_home and not is_solo and mention_tag.lower() not in content.lower():
        return {"ok": True}

    # Strip @mention
    clean_content = content
    if not is_home:
        clean_content = re.sub(
            rf"@{re.escape(config.MESSENGER_BOT_NAME)}", "", content, flags=re.IGNORECASE
        ).strip()
        if not clean_content:
            if file_infos:
                names = ", ".join(fi["name"] for fi in file_infos)
                clean_content = f"[Attachments: {names}]"
            else:
                return {"ok": True}

    # Mark as read (best-effort)
    if msg_id:
        asyncio.create_task(messenger.mark_read(room_id, [msg_id]))

    reply_to_data = data.get("replyTo")
    _schedule_debounced(room_id, clean_content, sender_name, msg_id, file_infos, reply_to_data)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Debounce — combines rapid-fire messages into one
# ---------------------------------------------------------------------------

def _schedule_debounced(room_id: int, content: str, sender_name: str, msg_id: int | None = None, file_infos: list | None = None, reply_to_data: dict | None = None) -> None:
    prior_content = ""
    combined_files: list = []

    entry = _room_debounce.get(room_id)
    if entry and entry["task"] and not entry["task"].done():
        # Follow-up arrived within the debounce window, before the turn began
        # processing — merge with the still-pending content.
        entry["task"].cancel()
        prior_content = entry["content"]
        combined_files = list(entry.get("files") or [])
    else:
        # Follow-up arrived while a turn was already being processed — merge
        # with the interrupted turn's content so the earlier message isn't lost.
        # (llm-api only persists a session at stream end, so a cancelled turn
        # leaves no server-side record to continue from.)
        inflight = _room_inflight.pop(room_id, None)
        if inflight:
            prior_content = inflight["content"]
            combined_files = list(inflight.get("files") or [])

    combined = f"{prior_content}\n{content}" if prior_content else content

    # Cancel any in-flight process_message task for this room
    active = _room_active_task.pop(room_id, None)
    if active and not active.done():
        active.cancel()
    asyncio.create_task(messenger.stop_typing(room_id))

    if file_infos:
        combined_files.extend(file_infos)

    async def _debounce():
        await asyncio.sleep(config.DEBOUNCE_SECONDS)
        final = _room_debounce.pop(room_id, None)
        if final:
            _room_active_task[room_id] = asyncio.current_task()
            # Stash the content now being processed so a follow-up that
            # interrupts this turn can merge with it (see _schedule_debounced).
            _room_inflight[room_id] = {
                "content": final["content"],
                "files": final.get("files") or [],
            }
            try:
                await process_message(
                    room_id, final["content"], final["sender"], final.get("msg_id"),
                    file_infos=final.get("files"), reply_to_data=final.get("reply_to_data"),
                )
            finally:
                _room_active_task.pop(room_id, None)
                _room_inflight.pop(room_id, None)

    _room_debounce[room_id] = {
        "content": combined,
        "sender": sender_name,
        "msg_id": msg_id,
        "files": combined_files,
        "reply_to_data": reply_to_data,
        "task": asyncio.create_task(_debounce()),
    }


# ---------------------------------------------------------------------------
# Message processing — streaming or synchronous
# ---------------------------------------------------------------------------

def _build_message_header(room_name: str, room_id: int, is_group: bool, sender_name: str, reply_to_data: dict | None) -> str:
    """Build the bracketed context header prepended to every LLM message."""
    room_type = "group" if is_group else "DM"
    header = f"[Room: {room_name} (id:{room_id}, {room_type}) | From: {sender_name}]"
    if reply_to_data and not reply_to_data.get("isDeleted") and reply_to_data.get("content"):
        reply_sender = reply_to_data.get("senderName", "unknown")
        reply_content = reply_to_data["content"][:200]
        header += f"\n> {reply_sender}: \"{reply_content}\""
    return header


async def process_message(room_id: int, content: str, sender_name: str, reply_to_id: int | None = None, _is_retry: bool = False, file_infos: list | None = None, reply_to_data: dict | None = None) -> None:
    """Core message processing pipeline with structured logging and timing."""
    start = time.monotonic()
    log_prefix = f"[Room {room_id}] [{sender_name}]"
    logger.info(f"{log_prefix} Processing: {content[:80]!r}")

    # @stop: halt the in-flight response for this room (server-side + local task)
    if "@stop" in content.lower():
        active = _room_active_task.pop(room_id, None)
        if active and not active.done():
            active.cancel()
        session_id = _get_session_id(room_id)
        if session_id:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{config.LLM_API_URL}/api/chat/sessions/{session_id}/stop",
                        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                    )
                    resp.raise_for_status()
            except Exception as e:
                logger.warning(f"{log_prefix} @stop request to LLM API failed: {e}")
        await messenger.stop_typing(room_id)
        await messenger.send_message(room_id, "중지했습니다.")
        logger.info(f"{log_prefix} Stopped by @stop command")
        return

    # @clear: wipe session and return immediately
    if "@clear" in content.lower():
        _room_sessions.pop(room_id, None)
        _room_msg_count.pop(room_id, None)
        _save_room_sessions()
        # A reset also exits goal mode.
        if _room_goal_mode.pop(room_id, None):
            _save_room_goal_mode()
        await messenger.send_message(room_id, "컨텍스트가 초기화되었습니다. 새 대화를 시작하세요.")
        logger.info(f"{log_prefix} Session cleared by @clear command")
        return

    # @compact: summarize old history in-place via LLM API
    if "@compact" in content.lower():
        session_id = _get_session_id(room_id)
        if not session_id:
            await messenger.send_message(room_id, "압축할 대화가 없습니다. 먼저 대화를 시작하세요.")
            logger.info(f"{log_prefix} @compact ignored — no active session")
            return
        await messenger.send_typing(room_id, status_text="요약 중...")
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{config.LLM_API_URL}/api/chat/sessions/{session_id}/compact",
                    headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                )
                resp.raise_for_status()
                data = resp.json()
            if data.get("success"):
                orig = data["original_count"]
                new = data["new_count"]
                _room_msg_count[room_id] = 0
                await messenger.send_message(
                    room_id,
                    f"컨텍스트가 요약되었습니다. ({orig}개 → {new}개 메시지)"
                )
                logger.info(f"{log_prefix} Session compacted {orig} → {new} messages by @compact")
            else:
                await messenger.send_message(room_id, f"압축 실패: {data.get('message', '알 수 없는 오류')}")
                logger.warning(f"{log_prefix} @compact returned failure: {data.get('message')}")
        except Exception as e:
            await messenger.send_message(room_id, f"압축 중 오류 발생: {e}")
            logger.error(f"{log_prefix} @compact failed: {e}")
        return

    # /goal: toggle sticky goal mode. `/goal off` disables; `/goal <task>` (or a
    # bare `/goal`) enables and, when a task follows, processes it in goal mode.
    goal_match = _GOAL_DIRECTIVE_RE.match(content)
    if goal_match:
        remainder = goal_match.group(1).strip()
        if remainder.lower() in _GOAL_OFF_WORDS:
            if _room_goal_mode.pop(room_id, None):
                _save_room_goal_mode()
            await messenger.send_message(room_id, "목표 모드를 껐어요.")
            logger.info(f"{log_prefix} Goal mode disabled by /goal off")
            return
        _room_goal_mode[room_id] = True
        _save_room_goal_mode()
        if not remainder:
            await messenger.send_message(
                room_id,
                "목표 모드를 켰어요. 완료할 목표를 알려주세요. 목표를 끝까지 추적해서 완수할게요. (끄려면 /goal off)",
            )
            logger.info(f"{log_prefix} Goal mode enabled (no task provided)")
            return
        # A task followed the directive — process it in goal mode.
        content = remainder
        logger.info(f"{log_prefix} Goal mode enabled with task: {content[:80]!r}")

    try:
        delegation = await try_submit_from_message(
            content, sender_name, room_id, session_id=_get_session_id(room_id)
        )
        if delegation:
            await messenger.send_message(room_id, delegation["message"], reply_to_id=reply_to_id)
            logger.info(f"{log_prefix} Delegated cluster task: {delegation}")
            return
    except Exception as exc:
        logger.warning(f"{log_prefix} Cluster delegation failed: {exc}")
        await messenger.send_message(room_id, f"Cluster delegation failed: {exc}", reply_to_id=reply_to_id)
        return

    await messenger.send_typing(room_id, status_text=_STATUS_GENERATING)

    try:
        if not config.LLM_API_KEY:
            raise ValueError("LLM_API_KEY is not configured. Run: python setup.py")
        if not config.LLM_MODEL:
            raise ValueError("LLM_MODEL is not configured. Run: python setup.py")

        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
        existing_session_id = _get_session_id(room_id)

        # Download attached files from Messenger
        downloaded_files: list[tuple[str, bytes]] = []
        if file_infos:
            for fi in file_infos:
                result = await messenger.download_file(fi["url"])
                if result:
                    file_bytes, filename = result
                    downloaded_files.append((fi["name"], file_bytes))
                    logger.info(f"{log_prefix} Downloaded file: {fi['name']} ({len(file_bytes)} bytes)")
                else:
                    logger.warning(f"{log_prefix} Failed to download: {fi['name']}")

        room_info = await messenger.get_room_info(room_id)
        msg_header = _build_message_header(
            room_info["name"], room_id, room_info["isGroup"], sender_name, reply_to_data
        )

        ambient = build_per_turn_context(profile=getattr(config, "CLUSTER_ROLE", "flutter"))
        user_content = f"{ambient}\n\n{msg_header}\n{content}"

        if existing_session_id:
            # Context already in history — don't re-inject PROMPT.md every turn.
            # Only append the memory-flush hint when the session is getting long.
            count = _room_msg_count.get(room_id, 0) + 1
            _room_msg_count[room_id] = count

            if config.MEMORY_FLUSH_THRESHOLD > 0 and count % config.MEMORY_FLUSH_THRESHOLD == 0:
                user_content += f"\n\n{_MEMORY_FLUSH_HINT}"
                logger.info(f"{log_prefix} Memory flush hint injected at message #{count}")

            messages = [{"role": "user", "content": user_content}]
            llm_data = {
                "model": config.LLM_MODEL,
                "messages": json.dumps(messages),
                "session_id": existing_session_id,
            }
            logger.info(f"{log_prefix} Continuing session {existing_session_id} (msg #{count})")
        else:
            # First turn of a new session: inject the full hoonbot context
            # (PROMPT.md + session vars + memory) as a system message so it
            # lands in history once and doesn't bloat every subsequent user turn.
            _room_msg_count[room_id] = 0
            context = build_llm_context()
            messages = [
                {"role": "system", "content": context},
                {"role": "user", "content": user_content},
            ]
            llm_data = {
                "model": config.LLM_MODEL,
                "messages": json.dumps(messages),
            }
            logger.info(f"{log_prefix} Starting new session")

        # Sticky goal mode (toggled by /goal) — forwarded as a form field the
        # LLM API parses into AgentLoop(mode="goal").
        if _room_goal_mode.get(room_id):
            llm_data["mode"] = "goal"
            logger.info(f"{log_prefix} Goal mode active — sending mode=goal")

        if config.STREAMING_ENABLED:
            reply = await _process_streaming(room_id, llm_data, headers, existing_session_id, log_prefix, reply_to_id, downloaded_files)
        else:
            reply = await _process_sync(room_id, llm_data, headers, existing_session_id, log_prefix, reply_to_id, downloaded_files)

        if reply is None:
            # Session was deleted (404) — retry once with a fresh session
            if not _is_retry:
                logger.info(f"{log_prefix} Retrying with fresh session after 404")
                await messenger.stop_typing(room_id)
                return await process_message(room_id, content, sender_name, reply_to_id, _is_retry=True, file_infos=file_infos, reply_to_data=reply_to_data)
            logger.warning(f"{log_prefix} Session 404 on retry — giving up")
            return

        elapsed = time.monotonic() - start
        logger.info(f"{log_prefix} Completed in {elapsed:.1f}s, reply={len(reply)} chars")

    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.error(f"{log_prefix} Failed after {elapsed:.1f}s: {exc}", exc_info=True)
        try:
            if "Connect" in type(exc).__name__ or "Timeout" in type(exc).__name__:
                user_msg = "LLM 서버에 연결할 수 없어요. 잠시 후 다시 시도해주세요."
            else:
                user_msg = f"오류: {str(exc)[:100]}"
            await messenger.send_message(room_id, user_msg)
        except Exception:
            pass
    finally:
        await messenger.stop_typing(room_id)


async def _save_session_from_response(room_id: int, result: dict, existing_session_id: str | None, log_prefix: str) -> None:
    """Save session_id returned by LLM API."""
    returned_session_id = result.get("x_session_id")
    if returned_session_id and returned_session_id != (existing_session_id or ""):
        _set_session_id(room_id, returned_session_id)
        logger.info(f"{log_prefix} Session saved as {returned_session_id}")


# ---------------------------------------------------------------------------
# Synchronous (non-streaming) processing
# ---------------------------------------------------------------------------

async def _process_sync(room_id: int, llm_data: dict, headers: dict, existing_session_id: str | None, log_prefix: str, reply_to_id: int | None, downloaded_files: list[tuple[str, bytes]] | None = None) -> str | None:
    """Send request, wait for full response, send to Messenger."""
    files_payload = [("files", (name, data, "application/octet-stream")) for name, data in (downloaded_files or [])]
    client = get_client()
    response = await client.post(
        f"{config.LLM_API_URL}/v1/chat/completions",
        data=llm_data,
        files=files_payload or None,
        headers=headers,
    )

    if response.status_code in (404, 500) and existing_session_id:
        logger.warning(f"{log_prefix} Session {existing_session_id} got {response.status_code}, starting fresh")
        _room_sessions.pop(room_id, None)
        _room_msg_count.pop(room_id, None)
        _save_room_sessions()
        return None

    response.raise_for_status()
    result = response.json()

    await _save_session_from_response(room_id, result, existing_session_id, log_prefix)
    reply = result["choices"][0]["message"]["content"]
    if reply.strip():
        await messenger.send_message(room_id, reply, reply_to_id=reply_to_id)
    else:
        fallback = "완료."
        await messenger.send_message(room_id, fallback, reply_to_id=reply_to_id)
        logger.warning(f"{log_prefix} LLM produced no text output — sent fallback acknowledgment")
        reply = fallback
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
_DRAFT_EDIT_INTERVAL_SECONDS = 0.25

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


async def _process_streaming(room_id: int, llm_data: dict, headers: dict, existing_session_id: str | None, log_prefix: str, reply_to_id: int | None, downloaded_files: list[tuple[str, bytes]] | None = None) -> str | None:
    """Stream SSE from LLM API into a single live-edited Messenger bubble.

    UX model:
      - Streamed text is posted as one draft message and edited in place as
        tokens arrive; at the end of the turn it settles into the final reply.
        One bubble per turn. When the turn used tools, the text after the
        last tool event is set off with `_ANSWER_DIVIDER` so the answer
        stands out from the work narration above it.
      - Tool activity is surfaced only through the typing indicator at the
        bottom of the chat: "{bot}님이 websearch 사용 중..." etc. Multiple
        parallel tools are joined with " + ". No helper bubbles are posted
        and later deleted — the client renders deleted messages as a
        permanent "삭제된 메시지입니다." placeholder, so mid-turn deletions
        must be avoided.
      - While no tool is running, the indicator shows "{bot}님이 응답 생성 중...".
      - A background task re-fires the typing event every
        `_TYPING_REFRESH_INTERVAL_SECONDS` so the indicator survives long
        tool calls (Messenger auto-clears at 15s / client at 20s).
    """
    llm_data["stream"] = "true"
    files_payload = [("files", (name, data, "application/octet-stream")) for name, data in (downloaded_files or [])]

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
    session_id_from_header: str | None = None
    draft_msg_id: int | None = None   # ID of the live text-draft message in chat
    last_draft_edit: float = 0.0
    first_sent = False

    async def _refresh_typing() -> None:
        """Periodically re-fire the typing event so the indicator stays visible."""
        while True:
            await asyncio.sleep(_TYPING_REFRESH_INTERVAL_SECONDS)
            await messenger.send_typing(room_id, status_text=_format_tool_status(running_tools))

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

    async def _flush_loop() -> None:
        nonlocal draft_msg_id, pending_text, flush_task
        while pending_text is not None:
            text_to_send = pending_text
            pending_text = None
            try:
                if draft_msg_id is None:
                    draft_msg_id = await messenger.send_message_returning_id(
                        room_id, text_to_send, reply_to_id
                    )
                else:
                    await messenger.edit_message(draft_msg_id, text_to_send)
            except Exception:
                pass
        flush_task = None

    def _schedule_flush(text: str) -> None:
        nonlocal pending_text, flush_task
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

    try:
        client = get_client()
        async with client.stream(
            "POST",
            f"{config.LLM_API_URL}/v1/chat/completions",
            data=llm_data,
            files=files_payload or None,
            headers=headers,
        ) as response:
            if response.status_code in (404, 500) and existing_session_id:
                logger.warning(f"{log_prefix} Session {existing_session_id} got {response.status_code}, starting fresh")
                _room_sessions.pop(room_id, None)
                _room_msg_count.pop(room_id, None)
                _save_room_sessions()
                return None

            response.raise_for_status()
            session_id_from_header = response.headers.get("x-session-id")

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "error" in event:
                    err = event.get("error") or {}
                    err_type = err.get("type", "stream_error")
                    err_message = str(err.get("message") or "").strip()
                    if not err_message:
                        err_message = json.dumps(err, ensure_ascii=False)
                    raise RuntimeError(f"LLM API stream error ({err_type}): {err_message}")

                if "tool_status" in event:
                    # Text streamed before a tool event is work narration, not
                    # the final answer — remember the boundary for the divider.
                    final_split_pos = len(full_text)
                    ts = event["tool_status"]
                    tool_name = ts.get("tool_name", "")
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
                    await messenger.send_typing(room_id, status_text=_format_tool_status(running_tools))
                    # Update the in-bubble checklist live — tools often run before
                    # any answer text, so this may create the draft bubble too.
                    _schedule_flush(_compose_draft(full_text))
                    continue

                if "x_session_id" in event:
                    session_id_from_header = event["x_session_id"]

                choices = event.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("delta", {}).get("content", "")
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

    except httpx.ReadTimeout:
        logger.warning(f"{log_prefix} Stream read timeout after collecting {len(full_text)} chars")
        await _settle_flush()
        if full_text.strip():
            msg = _compose_draft(full_text.strip() + "\n\n⚠️ (응답이 시간 초과로 잘렸어요)")
            if draft_msg_id is not None:
                await messenger.edit_message(draft_msg_id, msg)
            else:
                await messenger.send_message(room_id, msg, reply_to_id=reply_to_id)
            return full_text
        raise
    except asyncio.CancelledError:
        # Local task cancellation (e.g. @stop, or a new message superseding
        # this one) — settle the draft with a marker instead of leaving it
        # frozen mid-sentence with no indication anything happened.
        await _settle_flush()
        if draft_msg_id is not None:
            try:
                if full_text.strip():
                    await messenger.edit_message(draft_msg_id, _compose_draft(full_text.strip() + "\n\n⏹️ 중지됨"))
                else:
                    await messenger.delete_message(draft_msg_id)
            except Exception:
                pass
        raise
    except Exception:
        # Keep whatever already streamed — settle the draft with a warning
        # instead of deleting it (deletions leave a permanent "삭제된
        # 메시지입니다." placeholder in the client).
        await _settle_flush()
        if draft_msg_id is not None:
            try:
                if full_text.strip():
                    await messenger.edit_message(
                        draft_msg_id,
                        _compose_draft(full_text.strip() + "\n\n⚠️ 오류가 발생해 응답이 중단되었어요."),
                    )
                else:
                    await messenger.delete_message(draft_msg_id)
            except Exception:
                pass
        raise
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass

    # Normal completion — drain any queued/in-flight background flush before
    # the authoritative final write so it can't be overwritten by a stale one.
    await _settle_flush()

    if session_id_from_header:
        result_like = {"x_session_id": session_id_from_header}
        await _save_session_from_response(room_id, result_like, existing_session_id, log_prefix)

    final_text = full_text.strip()

    # When the turn used tools, set the post-tool text off with the answer
    # divider so it stands out from the work narration above it.
    worklog_text = full_text[:final_split_pos].strip()
    answer_text = full_text[final_split_pos:].strip()
    if worklog_text and answer_text:
        final_text = f"{worklog_text}\n\n{_ANSWER_DIVIDER}\n\n{answer_text}"

    # Prepend the structured tool-call checklist above the answer. When tools
    # ran but the model produced no text, this shows the checklist instead of
    # the bare "완료." fallback.
    final_text = _compose_draft(final_text)

    if draft_msg_id is not None:
        # A live draft was already posted — settle it to the final text or clean up
        if final_text:
            await messenger.edit_message(draft_msg_id, final_text)
        else:
            fallback = "완료."
            await messenger.edit_message(draft_msg_id, fallback)
            logger.warning(f"{log_prefix} LLM produced no text output — sent fallback acknowledgment")
            return fallback
    else:
        if final_text:
            await messenger.send_message(room_id, final_text, reply_to_id=reply_to_id)
        else:
            fallback = "완료."
            await messenger.send_message(room_id, fallback, reply_to_id=reply_to_id)
            logger.warning(f"{log_prefix} LLM produced no text output — sent fallback acknowledgment")
            return fallback

    return final_text


# ---------------------------------------------------------------------------
# External / incoming webhooks
# ---------------------------------------------------------------------------

@router.post("/webhook/incoming/{path:path}")
async def handle_incoming_webhook(path: str, request: Request):
    """Accept POST triggers from external services."""
    if config.WEBHOOK_INCOMING_SECRET:
        secret = request.headers.get("x-webhook-secret", "")
        if secret != config.WEBHOOK_INCOMING_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    source = path.strip("/") or "external"
    if "message" in payload:
        content = f"[Webhook from {source}] {payload['message']}"
    else:
        content = f"[Webhook from {source}] {json.dumps(payload, ensure_ascii=False, indent=2)}"

    room_id = config.MESSENGER_HOME_ROOM_ID
    logger.info(f"[Webhook] Incoming from '{source}' → room {room_id}")
    _schedule_debounced(room_id, content, f"webhook:{source}")
    return {"ok": True}
