"""
Chat completions endpoint (OpenAI-compatible with extensions)
/v1/chat/completions

Accepts both:
  - application/json  (standard OpenAI SDK, Jupyter, agents, etc.)
  - multipart/form-data  (file uploads, legacy frontend)

Both streaming and non-streaming go through the AgentLoop,
which uses native tool calling via llama.cpp.
"""
import asyncio
import inspect
import json
import time
import uuid
from typing import Optional, List, Dict, Any, Tuple

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from filelock import Timeout as FileLockTimeout
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from sse_starlette.sse import EventSourceResponse

from backend.models.schemas import (
    ChatMessage,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
)
from backend.core.database import db, conversation_store, SessionBusyError
from backend.core.llm_backend import TextEvent, ToolStatusEvent
from backend.utils.file_handler import save_uploaded_files, extract_file_metadata, is_image_file, encode_image_base64
from backend.utils.auth import get_optional_user
from backend.utils.stop_signal import (
    StopInferenceError,
    clear_session_stop,
    session_stop_requested_since,
)
from backend.api.routes.sessions import stop_enforced_events
from backend.agent import AgentLoop
import config

try:  # partition A contract (C1); tolerate an older backend during rollout
    from backend.core.llm_backend import VllmStreamError
except ImportError:  # pragma: no cover
    class VllmStreamError(Exception):  # type: ignore[no-redef]
        pass

router = APIRouter(prefix="/v1", tags=["chat"])


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: {repr(exc)}"


# ---------------------------------------------------------------------------
# Request parsing — handles both JSON and multipart/form-data
# ---------------------------------------------------------------------------

_TRUE_STRINGS = {"1", "true", "yes", "on"}


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse a bool from JSON (bool/number) or form (string) input."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    return text in _TRUE_STRINGS


def _as_optional_float(value: Any, field: str) -> Optional[float]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value!r}")


def _as_optional_int(value: Any, field: str) -> Optional[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid {field}: {value!r}")


def _reasoning_effort_kwargs(value: Any) -> Optional[Dict[str, Any]]:
    """Map the `reasoning_effort` request field onto chat_template_kwargs (C5).

    "max" | "high" -> {"reasoning_effort": value}; "none" -> non-thinking
    mode ({"enable_thinking": False}); unset -> None (config default).
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in ("none", "off", "false", "disabled"):
        return {"enable_thinking": False}
    if text in ("max", "high"):
        return {"reasoning_effort": text}
    raise HTTPException(
        status_code=400,
        detail=f"Invalid reasoning_effort {value!r}; expected 'max', 'high' or 'none'",
    )


async def _parse_request(request: Request) -> dict:
    """
    Parse /v1/chat/completions from either JSON or multipart form.

    Returns a dict with keys:
      messages_data, model, stream, temperature, max_tokens, session_id,
      workspace, mode, files, response_format, guided_json,
      disable_tools, ephemeral, chat_template_kwargs
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        messages_raw = form.get("messages", "[]")
        model = form.get("model") or None
        stream = _as_bool(form.get("stream"), False)
        temperature = form.get("temperature")
        max_tokens = form.get("max_tokens")
        session_id = form.get("session_id") or None
        workspace = form.get("workspace") or None
        mode = form.get("mode") or None
        response_format = form.get("response_format")
        guided_json = form.get("guided_json")
        disable_tools = _as_bool(form.get("disable_tools"), False)
        ephemeral = _as_bool(form.get("ephemeral"), False)
        reasoning_effort = form.get("reasoning_effort")
        files = [v for v in form.getlist("files") if isinstance(v, UploadFile) and v.filename]
        try:
            messages_data = json.loads(messages_raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid messages JSON")
    else:
        # JSON body (standard OpenAI format)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        messages_data = body.get("messages", [])
        model = body.get("model") or None
        stream = _as_bool(body.get("stream"), False)
        temperature = body.get("temperature")
        max_tokens = body.get("max_tokens")
        session_id = body.get("session_id") or None
        workspace = body.get("workspace") or None
        mode = body.get("mode") or None
        response_format = body.get("response_format")
        guided_json = body.get("guided_json")
        disable_tools = _as_bool(body.get("disable_tools"), False)
        ephemeral = _as_bool(body.get("ephemeral"), False)
        reasoning_effort = body.get("reasoning_effort")
        files = []

    if not isinstance(messages_data, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    # response_format / guided_json may arrive as a JSON-encoded string (form
    # posts) or an object (JSON body). Normalize to a dict or None.
    def _coerce_obj(v):
        if isinstance(v, str):
            try:
                return json.loads(v) if v.strip() else None
            except json.JSONDecodeError:
                return None
        return v or None

    return dict(
        messages_data=messages_data,
        model=model,
        stream=stream,
        temperature=_as_optional_float(temperature, "temperature"),
        max_tokens=_as_optional_int(max_tokens, "max_tokens"),
        session_id=str(session_id) if session_id else None,
        workspace=workspace,
        mode=mode,
        files=files,
        response_format=_coerce_obj(response_format),
        guided_json=_coerce_obj(guided_json),
        disable_tools=disable_tools,
        ephemeral=ephemeral,
        chat_template_kwargs=_reasoning_effort_kwargs(reasoning_effort),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_from_content(content) -> str:
    """Extract plain text from content (str or content array)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(content) if content else ""


def _build_storage_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """Convert incoming chat messages into the persisted text-only history form."""
    return [
        {"role": msg.role, "content": _extract_text_from_content(msg.content) or ""}
        for msg in messages
    ]


def _generate_session_title(messages: list) -> str:
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if role == "user" and content:
            text = _extract_text_from_content(content).strip().replace("\n", " ")
            return text[:60] + ("…" if len(text) > 60 else "")
    return "Untitled"


def _prepare_messages_with_files(
    messages: List[ChatMessage],
    file_paths: List[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from pathlib import Path

    message_dicts = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            message_dicts.append({"role": msg.role, "content": content})
        else:
            message_dicts.append({"role": msg.role, "content": content or ""})

    image_paths: List[str] = []
    non_image_paths: List[str] = []
    file_metadata: List[Dict[str, Any]] = []

    if file_paths:
        for file_path in file_paths:
            if is_image_file(file_path):
                image_paths.append(file_path)
            else:
                non_image_paths.append(file_path)

        for file_path in non_image_paths:
            path = Path(file_path)
            try:
                file_size = path.stat().st_size
                file_type = path.suffix.lstrip('.')
                text_ext = {'txt', 'md', 'json', 'csv', 'py', 'js', 'html', 'xml', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'ts', 'jsx', 'tsx'}
                data_ext = {'csv', 'xlsx', 'xls', 'json'}
                code_ext = {'py', 'js', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'ts', 'jsx', 'tsx', 'html', 'css'}
                category = 'binary'
                if file_type in text_ext:
                    category = 'text'
                if file_type in data_ext:
                    category = 'data'
                if file_type in code_ext:
                    category = 'code'
                rich_metadata = extract_file_metadata(file_path)
                file_metadata.append({
                    "name": path.name, "path": file_path,
                    "size": file_size, "type": file_type, "category": category,
                    **rich_metadata,
                })
            except Exception as e:
                file_metadata.append({"name": path.name, "path": file_path, "error": str(e)})

    if image_paths and getattr(config, "MODEL_SUPPORTS_VISION", True):
        last_user_idx = None
        for i in range(len(message_dicts) - 1, -1, -1):
            if message_dicts[i]["role"] == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            msg = message_dicts[last_user_idx]
            existing_content = msg["content"]
            if isinstance(existing_content, list):
                content_parts = list(existing_content)
            else:
                content_parts = [{"type": "text", "text": existing_content or ""}]
            for img_path in image_paths:
                encoded = encode_image_base64(img_path)
                if encoded:
                    content_parts.append({"type": "image_url", "image_url": encoded})
                    print(f"[VISION] Embedded image: {Path(img_path).name}")
            msg["content"] = content_parts
    elif image_paths:
        # Vision disabled for this model — surface images as metadata instead of
        # embedding image_url parts a text-only model can't decode.
        for img_path in image_paths:
            p = Path(img_path)
            try:
                file_metadata.append({
                    "name": p.name, "path": img_path,
                    "size": p.stat().st_size, "type": p.suffix.lstrip('.'),
                    "category": "image",
                    "note": "image attached but vision is disabled on this model",
                })
            except Exception as e:
                file_metadata.append({"name": p.name, "path": img_path, "error": str(e)})

    return message_dicts, file_metadata


# ---------------------------------------------------------------------------
# Turn bookkeeping
# ---------------------------------------------------------------------------

# How long a new turn waits for the previous turn on the same session to
# release its lease before answering 409 "session busy" (hoonbot retries 409).
# Long enough to cover the previous turn's stop cleanup (process-tree kills,
# partial-turn persistence) when a newer message preempts it right after
# POST /sessions/{id}/stop; ephemeral turns take no lease at all.
_LEASE_WAIT_SECONDS = 8.0
# Non-streaming path: how often to poll for a client disconnect.
_DISCONNECT_POLL_SECONDS = 1.0
# Max distinct tool names listed in the persisted ledger of a turn.
_LEDGER_MAX_TOOLS = 20

_DIAGNOSTIC_DEFAULTS: Dict[str, Any] = {
    "finish_reason": "stop",
    "iterations": 0,
    "truncation_continuations": 0,
    "empty_retries": 0,
    "tool_calls": 0,
    "compactions": 0,
    "fallback_used": False,
    "supervisor_reviews": 0,
    "watchdog_stops": 0,
}

# Limits for the live tool_status fields (K1) sent to clients.
_STATUS_DETAIL_MAX = 120
_STATUS_PROCESSES_MAX = 3
_STATUS_COMMAND_MAX = 100
_STATUS_TAIL_MAX = 160
# Non-tool pseudo "tools" reported through ToolStatusEvent; never part of the ledger.
_NON_TOOL_STATUS_NAMES = frozenset({"autocompact", "supervisor"})


def _validate_session_id(session_id: str) -> None:
    """Session ids become file names under data/sessions — refuse path tricks."""
    if (
        len(session_id) > 200
        or session_id != session_id.strip()
        or ".." in session_id
        or any(ch in session_id for ch in ("/", "\\", "\x00"))
    ):
        raise HTTPException(status_code=400, detail="Invalid session_id")


def _as_float(value: Any, digits: int = 1) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return round(number, digits)


def _trim_status_processes(processes: Any) -> List[Dict[str, Any]]:
    """Normalise the live-process list of a tool_status event (K1).

    <=3 entries of {pid, command<=100, elapsed, idle, tail<=160}; accepts the
    registry's `idle_seconds` spelling too. Never raises.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(processes, (list, tuple)):
        return out
    for proc in list(processes)[:_STATUS_PROCESSES_MAX]:
        if not isinstance(proc, dict):
            continue
        try:
            pid = int(proc.get("pid"))
        except (TypeError, ValueError):
            pid = None
        idle = proc.get("idle", proc.get("idle_seconds"))
        out.append({
            "pid": pid,
            "command": str(proc.get("command") or "")[:_STATUS_COMMAND_MAX],
            "elapsed": _as_float(proc.get("elapsed")),
            "idle": _as_float(idle),
            "tail": str(proc.get("tail") or "")[-_STATUS_TAIL_MAX:],
        })
    return out


def _tool_status_payload(event: ToolStatusEvent) -> Dict[str, Any]:
    """The `tool_status` SSE object. New K1 fields are read with getattr so an
    older ToolStatusEvent (no detail/elapsed/processes) still serialises."""
    payload = {
        "tool_name": event.tool_name,
        "tool_call_id": event.tool_call_id,
        "status": event.status,
        "duration": event.duration,
        "activity": event.activity,
        "user_name": event.user_name,
        "detail": str(getattr(event, "detail", "") or "")[:_STATUS_DETAIL_MAX],
        "elapsed": _as_float(getattr(event, "elapsed", 0.0)),
        "processes": _trim_status_processes(getattr(event, "processes", None)),
    }
    # A child-tool "started" re-emit (activity "sub:<tool>") carries the
    # dataclass defaults, not real measurements; omit them so clients keep the
    # parent agent call's own timer and live process lines.
    if str(event.activity or "").startswith("sub:") and event.status == "started":
        if not payload["elapsed"]:
            payload.pop("elapsed")
        if not payload["processes"]:
            payload.pop("processes")
    return payload


class _TurnRecord:
    """What happened during one turn — drives persistence and diagnostics."""

    def __init__(self) -> None:
        self.assistant_parts: List[str] = []
        self.tools: List[Tuple[str, bool]] = []
        # Tool calls already counted in the ledger (a call must never be
        # counted twice, whatever mix of events the loop emitted for it).
        self._counted_calls: set = set()
        # Calls that started but have not completed/failed yet, in start order
        # (call id -> (tool name, detail)). Whatever is left when a turn is cut
        # off was running at that moment and may have had partial side effects.
        self._running_calls: Dict[str, Tuple[str, str]] = {}
        # Pessimistic defaults: anything that doesn't reach a known end
        # (generator torn down by a client disconnect) counts as interrupted.
        self.outcome = "interrupted"            # complete | interrupted | error
        self.reason = "client disconnected"
        self.finish_reason = "interrupted"      # stop | length | error | interrupted
        self.persisted = False

    @property
    def assistant_text(self) -> str:
        return "".join(self.assistant_parts)

    def add_text(self, text: str) -> None:
        if text:
            self.assistant_parts.append(text)

    def note_tool_status(self, event: ToolStatusEvent) -> None:
        status = str(getattr(event, "status", "") or "").lower()
        name = getattr(event, "tool_name", "") or "tool"
        # Autocompaction is reported as a ToolStatusEvent but is not a tool
        # the model ran — keep it out of the ledger/tool_calls count.
        if name in _NON_TOOL_STATUS_NAMES:
            return
        call_id = str(getattr(event, "tool_call_id", "") or "")
        if status in ("started", "progress"):
            # "progress" is just "still running": never a completion, never a
            # second ledger entry. Remember the call so a cut-off turn can say
            # what was in flight.
            if call_id and call_id not in self._counted_calls and call_id not in self._running_calls:
                detail = str(getattr(event, "detail", "") or "")[:80]
                self._running_calls[call_id] = (name, detail)
            return
        if status in ("completed", "failed"):
            if call_id:
                self._running_calls.pop(call_id, None)
                if call_id in self._counted_calls:
                    return
                self._counted_calls.add(call_id)
            self.tools.append((name, status == "completed"))

    def in_flight_summary(self) -> str:
        """Tool calls still running when the turn ended, as "name (detail), ..."."""
        parts: List[str] = []
        for name, detail in list(self._running_calls.values())[:5]:
            parts.append(f"{name} ({detail})" if detail else name)
        extra = len(self._running_calls) - 5
        if extra > 0:
            parts.append(f"+{extra} more")
        return ", ".join(parts)

    def mark_complete(self, finish_reason: str) -> None:
        """Record a run that ended without raising.

        An agent that ends on its own with finish "interrupted"/"error" (it
        caught the stop/error itself) is still persisted with the
        "[interrupted]" marker, so the next turn knows the reply is partial.
        """
        self.finish_reason = finish_reason
        if finish_reason == "interrupted":
            self.outcome = "interrupted"
            self.reason = "stopped by user request"
        elif finish_reason == "error":
            self.outcome = "error"
            self.reason = "agent reported an error"
        else:
            self.outcome = "complete"
            self.reason = ""

    def tool_ledger(self) -> str:
        """Compact "name ×N (k failed)" summary of the tools run this turn."""
        counts: Dict[str, List[int]] = {}
        for name, ok in self.tools:
            entry = counts.setdefault(name, [0, 0])
            entry[0] += 1
            if not ok:
                entry[1] += 1
        parts: List[str] = []
        for name, (total, failed) in list(counts.items())[:_LEDGER_MAX_TOOLS]:
            part = name + (f" ×{total}" if total > 1 else "")
            if failed:
                part += f" ({failed} failed)"
            parts.append(part)
        extra = len(counts) - _LEDGER_MAX_TOOLS
        if extra > 0:
            parts.append(f"+{extra} more")
        return ", ".join(parts)

    def persisted_assistant_content(self) -> str:
        """Assistant message to store for this turn (never empty).

        A complete turn stores its answer verbatim. An interrupted/errored
        turn stores the partial text plus an "[interrupted]" marker and the
        tool ledger, so the next turn knows the earlier reply was cut off and
        which side effects already happened (instead of believing it answered).
        """
        text = self.assistant_text
        ledger = self.tool_ledger()
        if self.outcome == "complete":
            if text.strip():
                return text
            note = f"[no reply produced: finish_reason={self.finish_reason}"
            if ledger:
                note += f"; tools run this turn: {ledger}"
            return note + "]"
        marker = f"[interrupted] This reply was cut off before completion ({self.reason[:300]})."
        if ledger:
            marker += f" Tools run this turn: {ledger}."
        else:
            marker += " No tools completed this turn."
        in_flight = self.in_flight_summary()
        if in_flight:
            marker += (
                f" Still running when it was cut off (may have partly executed): {in_flight}."
            )
        if text.strip():
            return text.rstrip() + "\n\n" + marker
        return marker


def _agent_finish_reason(agent: Any) -> str:
    """Real finish reason of a completed run (C7), normalised for clients."""
    try:
        finish = getattr(agent, "final_finish_reason", None)
    except Exception:
        finish = None
    if not finish:
        return "stop"
    finish = str(finish)
    if finish in ("tool", "tool_calls"):
        return "stop"
    return finish


def _build_diagnostics(agent: Any, record: _TurnRecord) -> Dict[str, Any]:
    diag: Dict[str, Any] = dict(_DIAGNOSTIC_DEFAULTS)
    try:
        # Never let a misbehaving diagnostics property break the final
        # chunk / error event (the stream must still end with [DONE]).
        agent_diag = getattr(agent, "diagnostics", None) if agent is not None else None
    except Exception as exc:
        print(f"[CHAT] WARNING: agent.diagnostics unavailable: {_format_exception(exc)}")
        agent_diag = None
    if isinstance(agent_diag, dict):
        diag.update(agent_diag)
    if not diag.get("tool_calls"):
        diag["tool_calls"] = len(record.tools)
    diag["finish_reason"] = record.finish_reason
    try:
        # Guarantee JSON-safety whatever the agent put in there.
        return json.loads(json.dumps(diag, default=str))
    except Exception:
        return {"finish_reason": record.finish_reason}


def _error_type(exc: BaseException) -> str:
    if isinstance(exc, VllmStreamError):
        return "upstream_error"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    return "internal_error"


_AGENT_INIT_PARAMS: Optional[set] = None
_AGENT_ACCEPTS_ANY_KWARG = False


def _build_agent(**kwargs: Any) -> AgentLoop:
    """Construct AgentLoop, dropping kwargs its constructor doesn't know.

    Lets this route ship ahead of/behind agent-side changes (e.g. the
    chat_template_kwargs / max_tokens parameters) without a TypeError; a
    dropped non-None value is logged so the gap is visible.
    """
    global _AGENT_INIT_PARAMS, _AGENT_ACCEPTS_ANY_KWARG
    if _AGENT_INIT_PARAMS is None:
        try:
            params = inspect.signature(AgentLoop.__init__).parameters
            _AGENT_ACCEPTS_ANY_KWARG = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            _AGENT_INIT_PARAMS = set(params)
        except (TypeError, ValueError):
            _AGENT_INIT_PARAMS = set()
            _AGENT_ACCEPTS_ANY_KWARG = True
    if not _AGENT_ACCEPTS_ANY_KWARG:
        for key in list(kwargs):
            if key not in _AGENT_INIT_PARAMS:
                value = kwargs.pop(key)
                if value is not None:
                    print(f"[CHAT] WARNING: AgentLoop does not accept '{key}'; ignoring {value!r}")
    return AgentLoop(**kwargs)


def _chunk_payload(chunk: ChatCompletionChunk, public_session_id: Optional[str]) -> Dict[str, Any]:
    """Serialize a chunk; x_session_id is the real id on EVERY chunk (never null)."""
    data = chunk.model_dump()
    if public_session_id:
        data["x_session_id"] = public_session_id
    else:
        data.pop("x_session_id", None)
    return data


def _sse(data: Dict[str, Any]) -> Dict[str, str]:
    return {"data": json.dumps(data, ensure_ascii=False, default=str)}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    current_user: Optional[dict] = Depends(get_optional_user),
):
    lease = None
    # Set once the streaming response owns the lease (released by the
    # generator / background finalizer instead of this function).
    lease_handed_off = False
    try:
        parsed = await _parse_request(request)
        messages_data = parsed["messages_data"]
        # This backend wraps exactly one local vLLM serving one model, so the
        # node's configured VLLM_MODEL is authoritative. Clients (OpenAI SDK,
        # Messenger, hoonbot) routinely send a placeholder like "default" or
        # "gpt-4" in the required `model` field; honoring those would make vLLM
        # 404 with "model '<placeholder>' does not exist". Always use the served
        # model so any client placeholder works.
        model_name = config.VLLM_MODEL
        is_streaming = parsed["stream"]
        # Leave unset (None) when the client didn't specify one, so
        # AgentLoop._resolve_temperature can apply MODEL_TEMPERATURE_OVERRIDES
        # (e.g. GLM -> 1.0) before falling back to DEFAULT_TEMPERATURE. Passing
        # DEFAULT_TEMPERATURE here would count as an explicit value and bypass
        # the per-model override entirely.
        temp = parsed["temperature"]
        session_id = parsed["session_id"]
        workspace = parsed["workspace"]
        mode = parsed["mode"]
        response_format = parsed["response_format"]
        guided_json = parsed["guided_json"]
        ephemeral: bool = parsed["ephemeral"]
        disable_tools: bool = parsed["disable_tools"]
        chat_template_kwargs = parsed["chat_template_kwargs"]
        max_tokens = parsed["max_tokens"]
        files: List[UploadFile] = parsed["files"]
        username = current_user["username"] if current_user else "guest"

        try:
            chat_messages = [ChatMessage(**msg) for msg in messages_data]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid messages: {exc}")
        new_history_messages = _build_storage_messages(chat_messages)

        # Session handling
        if ephemeral:
            # Ephemeral (C5): no history loaded, nothing persisted, no session
            # row/file, no lease. The internal id only scopes scratch/tool
            # artifacts for this one run and is never returned to the client.
            session_id = f"ephemeral_{uuid.uuid4().hex[:12]}"
            history: List[Dict[str, Any]] = []
            effective_workspace = workspace
        else:
            existing_session = False
            if session_id:
                _validate_session_id(session_id)
                session = db.get_session(session_id)
                if not session:
                    if session_id.startswith("hb_"):
                        # Heartbeat sessions are auto-created on first tick of each hour
                        db.create_session(session_id, username)
                        db.update_session_title(session_id, f"Heartbeat {session_id[3:]}")
                    elif session_id.startswith("cluster_"):
                        # Cluster task sessions are auto-created on first use — the
                        # slave worker names them cluster_<task_id> before any
                        # session exists (see hoonbot/core/cluster_worker.py).
                        db.create_session(session_id, username)
                        db.update_session_title(session_id, f"Cluster task {session_id[8:]}")
                    else:
                        raise HTTPException(status_code=404, detail="Session not found")
                else:
                    existing_session = True
            else:
                session_id = str(uuid.uuid4())
                db.create_session(session_id, username)
                title = _generate_session_title(chat_messages)
                db.update_session_title(session_id, title)

            # One turn per session at a time (across workers too): load ->
            # run -> persist must not interleave with another turn or with a
            # manual /compact rewrite.
            lease = await conversation_store.acquire_turn_lease(
                session_id, timeout=_LEASE_WAIT_SECONDS
            )
            # Any stop marker written before this turn began was aimed at an
            # earlier run — don't let it kill this one at iteration 0.
            clear_session_stop(session_id)

            history = []
            if existing_session:
                history = await asyncio.to_thread(
                    conversation_store.load_recent_conversation, session_id
                ) or []

            # Workspace selection: explicit request value (if non-empty) updates the
            # persisted session workspace. Otherwise reuse whatever was stored on
            # the session row. Validation lives in AgentLoop._validate_workspace —
            # we just record whatever the caller asked for.
            if workspace:
                db.update_session_workspace(session_id, workspace)
                effective_workspace = workspace
            else:
                session_row = db.get_session(session_id) or {}
                effective_workspace = session_row.get("workspace_dir")

        public_session_id: Optional[str] = None if ephemeral else session_id

        # File uploads
        file_paths: List[str] = []
        if files:
            file_paths = save_uploaded_files(files, username, session_id)

        llm_messages, file_metadata = _prepare_messages_with_files(chat_messages, file_paths)

        # Build conversation context for the agent from bounded hot history + new request delta.
        agent_messages = list(history)
        agent_messages.extend(llm_messages)

        agent_kwargs: Dict[str, Any] = dict(
            model=model_name,
            temperature=temp,
            session_id=session_id,
            username=username,
            workspace_dir=effective_workspace,
            response_format=response_format,
            guided_json=guided_json,
            mode=mode or "normal",
        )
        if disable_tools:
            # Explicit empty list = no tools at all (AgentLoop distinguishes
            # [] from None, which means config.AVAILABLE_TOOLS).
            agent_kwargs["tools"] = []
        if chat_template_kwargs is not None:
            agent_kwargs["chat_template_kwargs"] = chat_template_kwargs
        if max_tokens is not None:
            agent_kwargs["max_tokens"] = max_tokens
        agent = _build_agent(**agent_kwargs)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_timestamp = int(time.time())
        record = _TurnRecord()
        # Monotonic start of this turn (after the lease): a stop requested at or
        # after it was aimed at this turn.
        turn_started_at = time.monotonic()
        response_headers: Dict[str, str] = (
            {"X-Session-Id": public_session_id} if public_session_id else {}
        )

        # Persist the user turn + the assistant outcome, exactly once. Runs on
        # normal completion AND on interruption (client disconnect, @stop,
        # upstream error) so a long agentic turn is never lost from history.
        # Called synchronously from a finally: a create_task/to_thread
        # scheduled during request/generator teardown may never get to run.
        def _persist_turn() -> None:
            if record.persisted:
                return
            record.persisted = True
            if (
                record.outcome == "interrupted"
                and record.reason == "client disconnected"
                and session_stop_requested_since(session_id, turn_started_at)
            ):
                # The stream was torn down because a stop was requested (e.g. a
                # newer message preempted this turn), not because the client
                # simply vanished - say so in the history marker.
                record.reason = "stopped by user request"
            text = record.assistant_text
            print(
                f"[CHAT] turn end session={session_id} outcome={record.outcome} "
                f"finish={record.finish_reason} chars={len(text)} tools={len(record.tools)}"
            )
            if record.outcome == "complete" and not text.strip():
                print(f"[CHAT] EMPTY_REPLY session={session_id} finish={record.finish_reason}")
            if ephemeral:
                return
            to_store = list(new_history_messages)
            assistant_content = record.persisted_assistant_content()
            if assistant_content.strip():
                to_store.append({"role": "assistant", "content": assistant_content})
            if not to_store:
                return
            try:
                conversation_store.append_messages(session_id, to_store)
                db.increment_session_message_count(session_id, len(to_store))
            except Exception as exc:
                print(f"[CHAT] WARNING: failed to persist turn for session {session_id}: {_format_exception(exc)}")

        def _finish_turn() -> None:
            try:
                _persist_turn()
            finally:
                if lease is not None:
                    lease.release()

        def _final_chunk(finish_reason: str) -> Dict[str, str]:
            chunk = ChatCompletionChunk(
                id=request_id,
                created=created_timestamp,
                model=model_name,
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkDelta(),
                        finish_reason=finish_reason,
                    )
                ],
                x_session_id=public_session_id,
            )
            data = _chunk_payload(chunk, public_session_id)
            data["x_diagnostics"] = _build_diagnostics(agent, record)
            return _sse(data)

        if is_streaming:
            async def generate_stream():
                # Runs the agent in its own task so a stop is enforced even
                # while the loop is blocked in a long tool or LLM call.
                agent_events = stop_enforced_events(
                    agent.run_stream(agent_messages, file_metadata), session_id
                )
                try:
                    # First event: the session id, before any work starts, so a
                    # client can resume this session even if the turn later fails.
                    first: Dict[str, Any] = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_timestamp,
                        "model": model_name,
                        "choices": [],
                    }
                    if public_session_id:
                        first["x_session_id"] = public_session_id
                    yield _sse(first)

                    async for event in agent_events:
                        if isinstance(event, TextEvent):
                            if not event.content:
                                continue
                            record.add_text(event.content)
                            chunk = ChatCompletionChunk(
                                id=request_id,
                                created=created_timestamp,
                                model=model_name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        delta=ChatCompletionChunkDelta(content=event.content)
                                    )
                                ],
                            )
                            # Unnamed event — parsed by standard OpenAI clients
                            yield _sse(_chunk_payload(chunk, public_session_id))
                        elif isinstance(event, ToolStatusEvent):
                            record.note_tool_status(event)
                            tool_data = json.dumps(
                                {"tool_status": _tool_status_payload(event)},
                                default=str,
                            )
                            # Named event — standard clients ignore unknown event types
                            yield {"event": "tool_status", "data": tool_data}

                    record.mark_complete(_agent_finish_reason(agent))
                    yield _final_chunk(record.finish_reason)
                    yield {"data": "[DONE]"}

                except StopInferenceError:
                    record.outcome = "interrupted"
                    record.reason = "stopped by user request"
                    record.finish_reason = "interrupted"
                    yield _final_chunk("interrupted")
                    yield {"data": "[DONE]"}
                except Exception as e:
                    record.outcome = "error"
                    record.reason = f"error: {_format_exception(e)}"
                    record.finish_reason = "error"
                    print(f"[CHAT] ERROR session={session_id}: {_format_exception(e)}")
                    error_data: Dict[str, Any] = {
                        "error": {
                            "message": _format_exception(e),
                            "type": _error_type(e),
                        },
                        "x_diagnostics": _build_diagnostics(agent, record),
                    }
                    upstream_status = getattr(e, "status_code", None)
                    if isinstance(upstream_status, int):
                        error_data["error"]["code"] = upstream_status
                    if public_session_id:
                        error_data["x_session_id"] = public_session_id
                    yield _sse(error_data)
                    yield {"data": "[DONE]"}
                finally:
                    # Fires on normal end, error, and client-disconnect
                    # cancellation (generator aclose) alike. Close the agent
                    # relay first (cancels + reaps a still-running agent).
                    try:
                        await agent_events.aclose()
                    except BaseException:
                        pass
                    finally:
                        _finish_turn()

            stream_gen = generate_stream()

            async def _finalize_stream() -> None:
                # Backstop: runs after the SSE response ends however it ended.
                # Closing a generator suspended at a yield runs its finally
                # (persist + lease release); both steps are idempotent.
                try:
                    await stream_gen.aclose()
                except Exception:
                    pass
                _finish_turn()

            response = EventSourceResponse(
                stream_gen,
                headers=response_headers,
                ping=15,
                background=BackgroundTask(_finalize_stream),
            )
            lease_handed_off = True
            return response

        else:
            async def _consume() -> None:
                agent_events = stop_enforced_events(
                    agent.run_stream(agent_messages, file_metadata), session_id
                )
                try:
                    async for event in agent_events:
                        if isinstance(event, TextEvent):
                            record.add_text(event.content)
                        elif isinstance(event, ToolStatusEvent):
                            record.note_tool_status(event)
                finally:
                    await agent_events.aclose()

            agent_task = asyncio.create_task(_consume())
            client_gone = {"value": False}

            async def _watch_disconnect() -> None:
                # Starlette doesn't cancel a plain endpoint when the client
                # goes away; without this a timed-out caller leaves the agent
                # burning vLLM/tool slots and persisting a turn nobody reads.
                try:
                    while not agent_task.done():
                        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)
                        if agent_task.done():
                            return
                        try:
                            gone = await request.is_disconnected()
                        except Exception:
                            return
                        if gone:
                            client_gone["value"] = True
                            print(f"[CHAT] client disconnected — cancelling session={session_id}")
                            agent_task.cancel()
                            return
                except asyncio.CancelledError:
                    pass

            watcher = asyncio.create_task(_watch_disconnect())
            try:
                await asyncio.wait({agent_task})
            except asyncio.CancelledError:
                # The request handler itself was cancelled (server shutdown):
                # stop the agent too, record the interruption, re-raise.
                agent_task.cancel()
                watcher.cancel()
                record.reason = "request cancelled"
                try:
                    await asyncio.wait({agent_task}, timeout=5)
                except BaseException:
                    pass
                _persist_turn()
                raise
            watcher.cancel()

            if agent_task.cancelled():
                record.reason = "client disconnected" if client_gone["value"] else "run cancelled"
                _persist_turn()
                return JSONResponse(
                    status_code=499,
                    content={"detail": "client disconnected", "x_session_id": public_session_id},
                    headers=response_headers,
                )

            exc = agent_task.exception()
            if exc is None:
                record.mark_complete(_agent_finish_reason(agent))
            elif isinstance(exc, StopInferenceError):
                record.outcome = "interrupted"
                record.reason = "stopped by user request"
                record.finish_reason = "interrupted"
            else:
                record.outcome = "error"
                record.reason = f"error: {_format_exception(exc)}"
                record.finish_reason = "error"
                print(f"[CHAT] ERROR session={session_id}: {_format_exception(exc)}")
                _persist_turn()
                raise HTTPException(
                    status_code=502 if isinstance(exc, VllmStreamError) else 500,
                    detail=_format_exception(exc),
                    headers=response_headers or None,
                )
            _persist_turn()

            body = ChatCompletionResponse(
                id=request_id,
                created=created_timestamp,
                model=model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=record.assistant_text),
                        finish_reason=record.finish_reason,
                    )
                ],
                x_session_id=public_session_id or "",
            ).model_dump()
            body["x_session_id"] = public_session_id
            body["x_diagnostics"] = _build_diagnostics(agent, record)
            return JSONResponse(content=body, headers=response_headers)

    except HTTPException:
        raise
    except SessionBusyError:
        raise HTTPException(status_code=409, detail="session busy")
    except FileLockTimeout:
        raise HTTPException(status_code=503, detail="session storage busy")
    except Exception as e:
        print(f"[CHAT] ERROR (setup): {_format_exception(e)}")
        raise HTTPException(status_code=500, detail=_format_exception(e))
    finally:
        if lease is not None and not lease_handed_off:
            lease.release()
