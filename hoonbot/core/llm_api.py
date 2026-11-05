"""Shared HTTP clients + chat-call helpers for Hoonbot -> LLM API requests.

Three separately pooled clients (``get_client(kind)``) keep background work
(heartbeat, cluster tasks) from starving interactive Messenger turns:

  - ``"interactive"`` — webhook turns (default, so existing callers keep working)
  - ``"heartbeat"``   — heartbeat planner/executor/validator/summarizer
  - ``"cluster"``     — delegated cluster tasks on slaves

``stream_chat`` returns a ``StreamChat`` object that can be either awaited
(-> accumulated answer text, the historical behaviour) or iterated with
``async for`` (-> parsed SSE event dicts, for live consumers). Both raise
``LLMAPIError`` on in-band error events, HTTP errors, a stream that ends
without ``[DONE]``, and an idle stream (no bytes — not even the ~15s SSE
keepalive pings — for ``LLM_STREAM_IDLE_TIMEOUT_SECONDS``).

Retry policy: a request is retried (up to 2 extra attempts with backoff) ONLY
while nothing has been received from it yet and the failure is transport-level,
5xx, or 409 (session busy). Once output has arrived, re-POSTing would re-run
the agent turn (and its non-idempotent tools), so the error is raised instead.
"""
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import httpx

import config

logger = logging.getLogger(__name__)

CLIENT_KINDS = ("interactive", "heartbeat", "cluster")

_clients: dict[str, httpx.AsyncClient] = {}

# Connection went stale between requests (server restarted / keepalive dropped)
# — safe to retry while nothing has been received from the request yet.
STALE_ERRORS = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)

# Transport failures where the request most likely never reached llm-api.
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

# Transport failures that are retryable before any output was received.
_PRE_OUTPUT_RETRYABLE = _CONNECT_ERRORS + STALE_ERRORS + (httpx.ReadTimeout, httpx.WriteTimeout)

# Backoff (seconds) before retry attempt 1, 2. len() == number of extra attempts.
_RETRY_DELAYS = (1.5, 4.0)

# HTTP statuses retryable before any output: 409 = another turn holds the
# session lease (server waits ~2s first), 5xx = transient server failure.
_RETRYABLE_STATUS = {409, 500, 502, 503, 504}

_ERROR_BODY_LIMIT = 500

# Explicit phrases only — a bare "context" substring matches far too much.
_OVERFLOW_PHRASES = (
    "maximum context length",
    "context length",
    "context_length_exceeded",
    "context window",
    "too many tokens",
    "prompt is too long",
    "maximum model length",
    "please reduce the length",
    "reduce the length of the messages",
    "longer than the maximum",
)


class LLMAPIError(Exception):
    """A request to llm-api failed in a way the caller should surface.

    status_code      HTTP status when the failure was an HTTP error (else None).
    session_id       llm-api session id when known (header / first SSE event).
    received_output  True when some text/tool events were already received, i.e.
                     the turn partially ran and must NOT be blindly re-sent.
    error_type       ``error.type`` from an in-band SSE error event, if any.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        session_id: str | None = None,
        received_output: bool = False,
        error_type: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.session_id = session_id
        self.received_output = received_output
        self.error_type = error_type

    @property
    def is_context_overflow(self) -> bool:
        return is_context_overflow_error(self)


def is_context_overflow_error(exc: BaseException) -> bool:
    """True when *exc* looks like a genuine model context-window overflow."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in _OVERFLOW_PHRASES)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def _idle_timeout() -> float:
    try:
        value = float(getattr(config, "LLM_STREAM_IDLE_TIMEOUT_SECONDS", 900))
    except (TypeError, ValueError):
        value = 900.0
    return value if value > 0 else 900.0


def _timeout(read: Optional[float]) -> httpx.Timeout:
    # pool=None: waiting for a free pooled connection is bounded by the pool
    # size, not a timer — a PoolTimeout mid-burst would fail a healthy turn.
    return httpx.Timeout(connect=10.0, read=read, write=60.0, pool=None)


def get_client(kind: str = "interactive") -> httpx.AsyncClient:
    """Return the pooled LLM API client for *kind* (interactive/heartbeat/cluster)."""
    if kind not in CLIENT_KINDS:
        logger.warning(f"[LLM] Unknown client kind {kind!r} — using 'interactive'")
        kind = "interactive"
    client = _clients.get(kind)
    if client is None or client.is_closed:
        try:
            max_conn = int(getattr(config, "LLM_POOL_MAX_CONNECTIONS", 100))
        except (TypeError, ValueError):
            max_conn = 100
        max_conn = max(1, max_conn)
        client = httpx.AsyncClient(
            # The read timeout is a backstop per-socket-read: a healthy SSE
            # stream carries ~15s keepalive pings, so this only fires when
            # llm-api is truly stuck. Streaming calls also run their own idle
            # watchdog; non-streaming callers should pass an explicit timeout.
            timeout=_timeout(_idle_timeout()),
            trust_env=False,
            limits=httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=min(20, max_conn),
                keepalive_expiry=30.0,
            ),
        )
        _clients[kind] = client
    return client


async def close_client() -> None:
    """Close every pooled LLM API client."""
    clients = list(_clients.values())
    _clients.clear()
    for client in clients:
        if not client.is_closed:
            try:
                await client.aclose()
            except Exception as exc:
                logger.debug(f"[LLM] Client close failed: {exc}")


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _as_form_bool(value: bool) -> str:
    return "true" if value else "false"


def _build_form(
    payload: dict,
    *,
    stream: bool,
    disable_tools: bool,
    ephemeral: bool,
    reasoning_effort: Optional[str],
) -> dict:
    """Merge the caller's form payload with the C5 request flags.

    The optional flags are only sent when set, so requests stay byte-identical
    to before for callers that don't use them.
    """
    data = dict(payload)
    data["stream"] = _as_form_bool(stream)
    if disable_tools:
        data["disable_tools"] = "true"
    if ephemeral:
        data["ephemeral"] = "true"
    if reasoning_effort:
        data["reasoning_effort"] = str(reasoning_effort)
    return data


def _status_retryable(status: int, *, idempotent: bool) -> bool:
    if status == 409 or status in (502, 503, 504):
        return True
    # A plain 500 may come after the agent already ran tools; only re-run when
    # the call is tool-less (idempotent).
    return status == 500 and idempotent


async def _error_body(resp: httpx.Response) -> str:
    try:
        raw = await resp.aread()
    except Exception:
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    # FastAPI errors are {"detail": "..."}; prefer the detail text.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            detail = parsed.get("detail") or parsed.get("error")
            if isinstance(detail, dict):
                detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
            if detail:
                text = str(detail)
    except (ValueError, TypeError):
        pass
    return text[:_ERROR_BODY_LIMIT]


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------

async def chat_json(
    payload: dict,
    headers: dict,
    timeout: Optional[float] = None,
    *,
    disable_tools: bool = False,
    ephemeral: bool = False,
    reasoning_effort: Optional[str] = None,
    kind: str = "interactive",
    files: Any = None,
) -> dict:
    """Non-streaming /v1/chat/completions call; returns the parsed JSON body.

    Raises LLMAPIError (with up to 500 chars of the response body) on HTTP
    errors. Connect-level failures, 409 and 502-504 are retried up to twice;
    a 500 or a dropped connection after the request was sent is retried only
    for tool-less (``disable_tools``) calls, which are safe to re-run.
    """
    data = _build_form(
        payload, stream=False, disable_tools=disable_tools,
        ephemeral=ephemeral, reasoning_effort=reasoning_effort,
    )
    kwargs: dict = {"data": data, "headers": headers}
    if files:
        kwargs["files"] = files
    # A non-streaming call receives no bytes until the whole agent turn is
    # done, so the pool's idle-watchdog read timeout (900s) would cut long
    # turns off; default to the wall-clock LLM_TIMEOUT_SECONDS instead.
    if timeout is None:
        timeout = getattr(config, "LLM_TIMEOUT_SECONDS", 3600)
    try:
        kwargs["timeout"] = _timeout(float(timeout))
    except (TypeError, ValueError):
        kwargs["timeout"] = _timeout(3600.0)
    url = f"{config.LLM_API_URL}/v1/chat/completions"
    attempts = len(_RETRY_DELAYS) + 1

    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            resp = await get_client(kind).post(url, **kwargs)
        except _CONNECT_ERRORS as exc:
            if last:
                raise
            logger.warning(f"[LLM] chat connect failure ({type(exc).__name__}) — retry {attempt + 1}")
            await asyncio.sleep(_RETRY_DELAYS[attempt])
            continue
        except (STALE_ERRORS + (httpx.ReadTimeout, httpx.WriteTimeout)) as exc:
            # The request may have been processed; re-running is only safe
            # when the call has no tools.
            if last or not disable_tools:
                raise
            logger.warning(f"[LLM] chat transport failure ({type(exc).__name__}) — retry {attempt + 1}")
            await asyncio.sleep(_RETRY_DELAYS[attempt])
            continue

        if resp.status_code >= 400:
            body = await _error_body(resp)
            if not last and _status_retryable(resp.status_code, idempotent=disable_tools):
                logger.warning(f"[LLM] chat HTTP {resp.status_code} — retry {attempt + 1}: {body[:200]}")
                await asyncio.sleep(_RETRY_DELAYS[attempt])
                continue
            raise LLMAPIError(
                f"LLM API HTTP {resp.status_code}: {body or resp.reason_phrase}",
                status_code=resp.status_code,
                session_id=resp.headers.get("x-session-id"),
            )
        try:
            return resp.json()
        except ValueError as exc:
            raise LLMAPIError(
                f"LLM API returned invalid JSON: {resp.text[:_ERROR_BODY_LIMIT]}",
                status_code=resp.status_code,
            ) from exc

    raise LLMAPIError("LLM API request failed after retries")  # pragma: no cover


async def chat(
    payload: dict,
    headers: dict,
    timeout: Optional[float] = None,
    *,
    disable_tools: bool = False,
    ephemeral: bool = False,
    reasoning_effort: Optional[str] = None,
    kind: str = "interactive",
) -> str:
    """Non-streaming /v1/chat/completions call; returns the response text."""
    body = await chat_json(
        payload, headers, timeout,
        disable_tools=disable_tools, ephemeral=ephemeral,
        reasoning_effort=reasoning_effort, kind=kind,
    )
    if isinstance(body, dict) and body.get("error"):
        err = body.get("error")
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise LLMAPIError(
            f"LLM API error: {str(message)[:_ERROR_BODY_LIMIT]}",
            session_id=body.get("x_session_id"),
        )
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMAPIError(
            f"LLM API response missing choices: {json.dumps(body, ensure_ascii=False)[:_ERROR_BODY_LIMIT]}",
            session_id=body.get("x_session_id") if isinstance(body, dict) else None,
        ) from exc
    return content or ""


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class StreamChat:
    """One streaming /v1/chat/completions call.

    ``await stream_chat(...)``          -> accumulated answer text (str)
    ``async for event in stream_chat(...)`` -> parsed SSE event dicts
    ``async with stream_chat(...) as s: async for event in s:`` -> same, and
    the connection is released promptly if the consumer stops early.

    Events are yielded unchanged (``tool_status``, ``choices`` deltas,
    ``x_session_id``, ``x_diagnostics`` …). When the response carries an
    ``X-Session-Id`` header, the first yielded item is a synthesized
    ``{"x_session_id": <id>, "choices": []}``; consumers must tolerate
    ``choices: []``. The ``[DONE]`` sentinel and error events are not yielded
    (errors raise LLMAPIError).

    After iteration finishes, these attributes describe the outcome:
    ``text``, ``session_id``, ``finish_reason``, ``diagnostics``,
    ``saw_done``, ``received_output``.
    """

    def __init__(
        self,
        payload: dict,
        headers: dict,
        *,
        files: Any = None,
        disable_tools: bool = False,
        ephemeral: bool = False,
        reasoning_effort: Optional[str] = None,
        timeout: Optional[float] = None,
        kind: str = "interactive",
    ):
        self._data = _build_form(
            payload, stream=True, disable_tools=disable_tools,
            ephemeral=ephemeral, reasoning_effort=reasoning_effort,
        )
        self._headers = headers
        self._files = files
        self._kind = kind
        # ``timeout`` for a stream is the idle watchdog (seconds without any
        # bytes, keepalive pings included), not a wall clock.
        try:
            self._idle = float(timeout) if timeout else _idle_timeout()
        except (TypeError, ValueError):
            self._idle = _idle_timeout()
        if self._idle <= 0:
            self._idle = _idle_timeout()
        self._gen: Optional[AsyncIterator[dict]] = None

        self.text: str = ""
        self.session_id: Optional[str] = None
        self.finish_reason: Optional[str] = None
        self.diagnostics: Optional[dict] = None
        self.saw_done: bool = False
        self.received_output: bool = False

    def __aiter__(self) -> AsyncIterator[dict]:
        if self._gen is not None:
            raise RuntimeError("StreamChat can only be consumed once")
        self._gen = self._events()
        return self._gen

    async def aclose(self) -> None:
        """Close the underlying stream early (releases the pooled connection)."""
        gen = self._gen
        if gen is not None:
            try:
                await gen.aclose()  # type: ignore[attr-defined]
            except Exception:
                pass

    async def __aenter__(self) -> "StreamChat":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def __await__(self):
        return self._collect().__await__()

    async def _collect(self) -> str:
        async for _event in self:
            pass
        return self.text

    def _error(self, message: str, **kwargs) -> LLMAPIError:
        kwargs.setdefault("session_id", self.session_id)
        kwargs.setdefault("received_output", self.received_output)
        return LLMAPIError(message, **kwargs)

    def _note_event(self, event: dict) -> None:
        sid = event.get("x_session_id")
        if sid:
            self.session_id = str(sid)  # never overwrite a known id with None
        diag = event.get("x_diagnostics")
        if isinstance(diag, dict):
            self.diagnostics = diag
        if "tool_status" in event:
            self.received_output = True
            ts = event.get("tool_status") or {}
            logger.debug(f"[LLM] Tool {ts.get('tool_name')}: {ts.get('status')}")
        choices = event.get("choices") or []
        if choices and isinstance(choices[0], dict):
            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}
            chunk = delta.get("content") if isinstance(delta, dict) else None
            if chunk:
                self.text += chunk
                self.received_output = True
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if not self.finish_reason and isinstance(self.diagnostics, dict):
            fr = self.diagnostics.get("finish_reason")
            if fr:
                self.finish_reason = str(fr)

    async def _events(self) -> AsyncIterator[dict]:
        url = f"{config.LLM_API_URL}/v1/chat/completions"
        attempts = len(_RETRY_DELAYS) + 1
        # Per-request read timeout sits above the watchdog so the watchdog
        # (with its clearer error) normally fires first.
        req_timeout = _timeout(self._idle + 30.0)

        for attempt in range(attempts):
            last = attempt == attempts - 1
            opened = False  # True once the server accepted the turn (2xx)
            try:
                async with get_client(self._kind).stream(
                    "POST",
                    url,
                    data=self._data,
                    files=self._files or None,
                    headers=self._headers,
                    timeout=req_timeout,
                ) as resp:
                    header_sid = resp.headers.get("x-session-id")
                    if header_sid:
                        self.session_id = header_sid

                    if resp.status_code >= 400:
                        body = await _error_body(resp)
                        if not last and resp.status_code in _RETRYABLE_STATUS:
                            logger.warning(
                                f"[LLM] stream HTTP {resp.status_code} before output — "
                                f"retry {attempt + 1}: {body[:200]}"
                            )
                            await asyncio.sleep(_RETRY_DELAYS[attempt])
                            continue
                        raise self._error(
                            f"LLM API HTTP {resp.status_code}: {body or resp.reason_phrase}",
                            status_code=resp.status_code,
                        )

                    opened = True
                    if header_sid:
                        yield {"x_session_id": header_sid, "choices": []}

                    async for event in self._read_sse(resp):
                        yield event
                    return  # clean [DONE]

            except LLMAPIError:
                raise
            except asyncio.CancelledError:
                raise
            except _PRE_OUTPUT_RETRYABLE as exc:
                # Once the server answered 2xx the agent turn is running on the
                # llm-api side (user message persisted, tools may execute), so
                # re-POSTing would duplicate the turn / hit 409 session-busy.
                if self.received_output or opened:
                    raise self._error(
                        f"LLM API stream interrupted ({type(exc).__name__}: {exc})",
                    ) from exc
                if last:
                    raise
                logger.warning(
                    f"[LLM] stream transport failure before output "
                    f"({type(exc).__name__}) — retry {attempt + 1}"
                )
                await asyncio.sleep(_RETRY_DELAYS[attempt])
                continue

    async def _read_sse(self, resp: httpx.Response) -> AsyncIterator[dict]:
        """Parse SSE lines from *resp*, enforcing the idle watchdog."""
        lines = resp.aiter_lines().__aiter__()
        try:
            while True:
                try:
                    line = await asyncio.wait_for(lines.__anext__(), timeout=self._idle)
                except StopAsyncIteration:
                    break
                except (asyncio.TimeoutError, httpx.ReadTimeout) as exc:
                    raise self._error(
                        f"LLM API stream idle for {int(self._idle)}s (no data, no keepalive) — "
                        "aborted by watchdog",
                    ) from exc

                if not line.startswith("data:"):
                    continue  # SSE comments (": ping"), event:/id: fields, blanks
                raw = line[5:].strip()
                if not raw:
                    continue
                if raw == "[DONE]":
                    self.saw_done = True
                    return
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug(f"[LLM] Skipping undecodable SSE data: {raw[:200]}")
                    continue
                if not isinstance(event, dict):
                    continue

                if "error" in event and not event.get("choices"):
                    sid = event.get("x_session_id")
                    if sid:
                        self.session_id = str(sid)
                    err = event.get("error")
                    if isinstance(err, dict):
                        err_type = err.get("type") or "stream_error"
                        message = str(err.get("message") or "").strip() or json.dumps(err, ensure_ascii=False)
                    else:
                        err_type = "stream_error"
                        message = str(err or "unknown error")
                    status = err.get("code") if isinstance(err, dict) else None
                    raise self._error(
                        f"LLM API stream error ({err_type}): {message[:_ERROR_BODY_LIMIT]}",
                        status_code=status if isinstance(status, int) else None,
                        error_type=err_type,
                    )

                self._note_event(event)
                yield event
        finally:
            try:
                await lines.aclose()
            except Exception:
                pass

        # Iterator exhausted without the [DONE] sentinel: the connection was
        # closed mid-turn (server crash/restart, proxy cut) — never treat the
        # partial buffer as a complete answer.
        raise self._error("stream ended unexpectedly (no [DONE])")


def stream_chat(
    payload: dict,
    headers: dict,
    *,
    files: Any = None,
    disable_tools: bool = False,
    ephemeral: bool = False,
    reasoning_effort: Optional[str] = None,
    timeout: Optional[float] = None,
    kind: str = "interactive",
) -> StreamChat:
    """Start a streaming /v1/chat/completions call.

    ``await stream_chat(payload, headers)`` returns the accumulated answer
    text; ``async for event in stream_chat(...)`` yields SSE event dicts.
    *timeout* is the idle watchdog in seconds (default
    ``LLM_STREAM_IDLE_TIMEOUT_SECONDS``). See ``StreamChat``.
    """
    return StreamChat(
        payload, headers, files=files, disable_tools=disable_tools,
        ephemeral=ephemeral, reasoning_effort=reasoning_effort,
        timeout=timeout, kind=kind,
    )
