"""
LLM Backend for vLLM
Fully async, with native tool calling support via OpenAI-compatible API.

Robustness contract (shared with the agent loop):
- C1: transport errors / 5xx / in-band error chunks are retried internally with
  backoff ONLY while nothing has been yielded for the call. Once output has been
  yielded, a failure surfaces as VllmStreamError(partial=True). 4xx answers stay
  httpx.HTTPStatusError (context-overflow detection relies on that).
- C2: tool-call arguments that cannot be parsed into a dict become
  arguments={} plus ToolCallFunction.parse_error (never a `_raw` key).
- C3: every successful chat_stream ends with exactly ONE UsageEvent carrying the
  real finish_reason (tokens 0 when vLLM reported no usage). A stream that ends
  with neither a finish_reason nor [DONE] raises VllmStreamError.
- C4: chat() / chat_stream() accept a per-call chat_template_kwargs override
  (merged over config.CHAT_TEMPLATE_KWARGS) and max_tokens.
"""
import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple

import httpx

import config


# ============================================================================
# Response Types
# ============================================================================

@dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, Any]
    # C2: set when the model's arguments could not be parsed into a JSON object
    # (arguments is then {}). Dispatch returns a structured tool error instead
    # of executing the call, so the model gets told what went wrong.
    parse_error: Optional[str] = None

@dataclass
class ToolCall:
    id: str
    function: ToolCallFunction

@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"
    reasoning_content: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

@dataclass
class StreamEvent:
    pass

@dataclass
class TextEvent(StreamEvent):
    content: str = ""

@dataclass
class ReasoningEvent(StreamEvent):
    """Model-emitted reasoning/thinking content.

    MiniMax M2, Qwen3-Thinking, DeepSeek-R1, and other reasoning-trained
    models emit `<think>...</think>` chains either inline in `content` or
    via a separate `reasoning_content` delta field. These need to be
    preserved in the assistant turn that gets fed back to the model on the
    next iteration — stripping them severely degrades agentic performance —
    but they should NOT be streamed to the end user as visible text.
    """
    content: str = ""

@dataclass
class ToolCallDeltaEvent(StreamEvent):
    """Accumulated tool calls parsed from the stream.

    is_partial=True  → one tool call whose args just completed mid-stream;
                       more tool calls may follow in subsequent events.
    is_partial=False → final batch (stream ended); all remaining tool calls.
    """
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "tool"
    is_partial: bool = False

@dataclass
class ToolStatusEvent(StreamEvent):
    """Emitted before/after tool execution for streaming visibility."""
    tool_name: str = ""
    tool_call_id: str = ""
    status: str = ""        # "started" | "completed" | "failed"
    duration: float = 0.0
    activity: str = ""      # human-readable spinner text, e.g. "Reading file"
    user_name: str = ""     # display name for the tool, e.g. "File Reader"


@dataclass
class UsageEvent(StreamEvent):
    """Terminal end-of-stream event (C3): exactly one per successful chat_stream.

    Token counts are vLLM's real usage (0 when the server reported none —
    consumers must only calibrate on prompt_tokens > 0). finish_reason is the
    real finish reason ("stop" | "length" | "tool_calls"/"tool"); it is
    "length" whenever completion_tokens reached the max_tokens actually sent.
    The agent loop uses it to drive proactive compaction and to detect a
    truncated turn even when no tool call was emitted.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"


class VllmStreamError(Exception):
    """vLLM reported an error mid-stream (in-band error chunk, transport drop after output began, or stream ended without finish)."""
    def __init__(self, message: str, partial: bool = False, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.partial = partial
        self.status_code = status_code


# ============================================================================
# Small helpers
# ============================================================================

_JSON_DECODER = json.JSONDecoder()
_PARSE_ERROR_RAW_CHARS = 200
# Fixed prompt-token charge per image part (base64 bytes are NOT counted).
_IMAGE_TOKEN_COST = 1500
# Retry backoff schedule (seconds) for pre-first-yield failures.
_RETRY_BACKOFFS = (0.5, 1.5, 4.0)
# /v1/models window detection cache.
_WINDOW_TTL_OK = 600.0
_WINDOW_TTL_FAIL = 120.0
# Keys that select GLM-style thinking mode; a per-call override of any of them
# replaces the whole group from the global default (they are mutually exclusive).
_THINKING_MODE_KEYS = ("enable_thinking", "reasoning_effort", "thinking")
# Phrases that mark a genuine context-length error (never retried as transient).
_OVERFLOW_PHRASES = (
    "maximum context length", "context length", "too many tokens",
    "prompt is too long", "maximum model length",
)


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _pos_float(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def _fmt_exc(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else f"{type(exc).__name__}: {exc!r}"


def _utf8_len(s: str) -> int:
    """UTF-8 byte length; tolerant of lone surrogates (counted, never raised)."""
    try:
        return len(s.encode("utf-8"))
    except UnicodeEncodeError:
        return len(s.encode("utf-8", "surrogatepass"))


def _parse_error_msg(reason: str, raw: Any) -> str:
    """C2 parse_error text: short reason + first 200 chars of the raw input."""
    raw_s = raw if isinstance(raw, str) else repr(raw)
    return (f"{reason}. Raw arguments ({len(raw_s)} chars) began: "
            f"{raw_s[:_PARSE_ERROR_RAW_CHARS]}")


def _looks_like_overflow(message: str) -> bool:
    low = (message or "").lower()
    return any(p in low for p in _OVERFLOW_PHRASES)


def _stream_timeouts() -> Tuple[float, float]:
    """(first-byte timeout, inter-chunk idle timeout) in seconds."""
    first_byte = getattr(config, "STREAM_FIRST_BYTE_TIMEOUT_SECONDS", None)
    if first_byte is None:
        first_byte = getattr(config, "STREAM_READ_TIMEOUT", None)
    idle = getattr(config, "STREAM_IDLE_TIMEOUT_SECONDS", None)
    return _pos_float(first_byte, 1800.0), _pos_float(idle, 300.0)


def _merge_chat_template_kwargs(override: Optional[dict]) -> Optional[dict]:
    """C4: per-call chat_template_kwargs merged over config.CHAT_TEMPLATE_KWARGS.

    Thinking-mode keys (enable_thinking / reasoning_effort / thinking) are one
    group: if the override sets any of them, the default's thinking keys are
    dropped first, so {"enable_thinking": False} never ships next to a global
    {"reasoning_effort": "max"}.
    """
    base = getattr(config, "CHAT_TEMPLATE_KWARGS", None)
    merged: Dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    if isinstance(override, dict) and override:
        if any(k in override for k in _THINKING_MODE_KEYS):
            for k in _THINKING_MODE_KEYS:
                merged.pop(k, None)
        merged.update(override)
    return merged or None


def _normalize_tool_arguments(raw: Any, truncated: bool = False) -> Tuple[Dict[str, Any], Optional[str]]:
    """Turn model-produced tool arguments into (dict, parse_error).

    - dict → as-is; None / empty string → {}.
    - strings are stripped and the FIRST JSON value is raw_decoded (tolerates a
      duplicated/concatenated object such as `{...}{...}`).
    - a JSON string result (double-encoded arguments) is decoded once more.
    - anything that is not a JSON object → ({}, parse_error)  (C2).
    """
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return {}, _parse_error_msg(
            f"arguments must be a JSON object, got {type(raw).__name__}", raw)
    s = raw.strip()
    if not s:
        return {}, None
    try:
        obj, _end = _JSON_DECODER.raw_decode(s)
    except ValueError as e:
        if truncated:
            reason = ("arguments were cut off by the output token limit "
                      "(incomplete JSON)")
        else:
            reason = f"arguments are not valid JSON ({getattr(e, 'msg', e)})"
        return {}, _parse_error_msg(reason, raw)
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except ValueError:
            return {}, _parse_error_msg("arguments are a JSON string, not an object", raw)
    if obj is None:
        return {}, None
    if not isinstance(obj, dict):
        return {}, _parse_error_msg(
            f"arguments must be a JSON object, got {type(obj).__name__}", raw)
    return obj, None


def _strict_args(s: str) -> Optional[Dict[str, Any]]:
    """Whole-string strict parse used for mid-stream early dispatch."""
    try:
        obj = json.loads(s)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _call_signature(name: str, args: Dict[str, Any]) -> Tuple[str, str]:
    try:
        canon = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canon = repr(args)
    return name, canon


def _image_payload_stats(messages: List[Dict[str, Any]]) -> Tuple[int, int]:
    """(bytes of image URLs inside the serialized messages, number of image parts)."""
    img_bytes = 0
    n_images = 0
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("image_url", "image", "input_image"):
                continue
            n_images += 1
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if isinstance(url, str):
                img_bytes += len(url)  # base64 / URL text is ASCII: chars == bytes
    return img_bytes, n_images


# ============================================================================
# Inline reasoning extraction
# ============================================================================
#
# Reasoning models (GLM, Qwen3-Thinking, DeepSeek-R1, ...) emit their chain of
# thought as `<think>...</think>`. vLLM only lifts it into the separate
# `reasoning_content` delta field when launched with a matching
# `--reasoning-parser`; without that flag the think block arrives INLINE in the
# `content` stream. Left alone it streams straight to the user (full of draft
# code and deliberation) and pollutes the heartbeat planner/summariser parsing.
# _ThinkSplitter peels inline think blocks back out of `content` so they can be
# surfaced as ReasoningEvent (kept in history, never shown) exactly as if the
# parser had done it — making the backend robust regardless of vLLM's flags.

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_tag_suffix(buf: str, tag: str) -> int:
    """Length of the longest suffix of `buf` that is a proper prefix of `tag`.

    Lets us hold back the tail of a delta that might be the start of a tag split
    across SSE chunks (e.g. buf ends with "<thi", tag is "<think>")."""
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf[-k:] == tag[:k]:
            return k
    return 0


def _split_inline_reasoning(buf: str, in_think: bool) -> tuple[str, str, str, bool]:
    """Legacy stateless splitter, kept for backward compatibility.

    The streaming path uses _ThinkSplitter (start-of-content only, disabled by
    real reasoning deltas, never reclassifies emitted text)."""
    text_parts: list[str] = []
    reason_parts: list[str] = []
    while buf:
        if not in_think:
            i_open = buf.find(_THINK_OPEN)
            i_close = buf.find(_THINK_CLOSE)
            if i_close != -1 and (i_open == -1 or i_close < i_open):
                reason_parts.append(buf[:i_close])
                buf = buf[i_close + len(_THINK_CLOSE):]
                continue
            if i_open == -1:
                keep = max(_partial_tag_suffix(buf, _THINK_OPEN),
                           _partial_tag_suffix(buf, _THINK_CLOSE))
                text_parts.append(buf[:len(buf) - keep] if keep else buf)
                buf = buf[len(buf) - keep:] if keep else ""
                break
            text_parts.append(buf[:i_open])
            buf = buf[i_open + len(_THINK_OPEN):]
            in_think = True
        else:
            j = buf.find(_THINK_CLOSE)
            if j == -1:
                keep = _partial_tag_suffix(buf, _THINK_CLOSE)
                reason_parts.append(buf[:len(buf) - keep] if keep else buf)
                buf = buf[len(buf) - keep:] if keep else ""
                break
            reason_parts.append(buf[:j])
            buf = buf[j + len(_THINK_CLOSE):]
            in_think = False
    return "".join(text_parts), "".join(reason_parts), buf, in_think


class _ThinkSplitter:
    """Stateful per-stream splitter for inline `<think>` blocks.

    Rules:
    - `<think>` is honored only at the very start of the content (after
      whitespace); later occurrences are literal text.
    - Only the FIRST `</think>` of a stream that did not open with an explicit
      `<think>` is a tag (the chat template injected the open tag); any later
      one is literal text. It is an implicit close (leading content becomes
      reasoning) only while nothing has been emitted yet.
    - Seeing that first dangling close on a thinking-enabled request arms a
      process-wide latch ("this template injects <think>"): later
      thinking-enabled streams hold their opening content until a `</think>`
      arrives (or a large cap / end of stream) so their reasoning is not
      leaked. Requests that disable thinking (enable_thinking=False, e.g. the
      Step A recovery call) never hold, never latch and never split on a
      dangling `</think>`: their content is the answer.
    - Once a real reasoning delta arrives (`reasoning_content`/`reasoning`)
      the splitter is disabled for the stream and the latch is cleared — the
      server's reasoning parser is doing the job. From then on content is
      plain text in every mode.
    - A latched hold that ends normally without any `</think>` clears the
      latch (the evidence was wrong: this model does not think inline).
    - Text that was already emitted is never reclassified.
    """
    _latched = False            # process-wide (see class docstring)
    # Process-wide: a real reasoning delta has been seen at least once, i.e.
    # vLLM runs with a --reasoning-parser. From then on a dangling `</think>`
    # in content is literal (the model talking about the tag): it is never
    # split on and never arms the latch.
    _parser_seen = False
    _HOLD_CAP = 200_000         # latched hold gives up after this many chars

    def __init__(self, expect_thinking: bool = True):
        self.mode = "start"     # start | think | pass | off
        self.buf = ""
        # Per request: False when the request disabled thinking mode, so any
        # `</think>` in its content is literal and nothing is held.
        self.expect_thinking = expect_thinking
        self._closed = False    # this stream already consumed a </think> tag
        self._held = False      # this stream's opening was held by the latch

    def _inline_active(self) -> bool:
        """May a dangling `</think>` in this stream still be a real tag?"""
        return (self.expect_thinking and not self._closed
                and not _ThinkSplitter._parser_seen)

    def disable(self) -> Tuple[str, str]:
        """Real reasoning field seen: flush and switch to pass-through."""
        if self.mode == "off":
            return "", ""
        text, reasoning = "", ""
        if self.buf:
            if self.mode == "think":
                reasoning = self.buf
            else:
                text = self.buf
        self.buf = ""
        self.mode = "off"
        _ThinkSplitter._latched = False
        _ThinkSplitter._parser_seen = True
        return text, reasoning

    def feed(self, s: str) -> Tuple[str, str]:
        """Feed a content delta; returns (text, reasoning) ready to emit."""
        if self.mode == "off":
            return s, ""
        self.buf += s
        text_parts: List[str] = []
        reason_parts: List[str] = []
        while True:
            if self.mode == "start":
                if _ThinkSplitter._parser_seen:
                    # Server lifts reasoning itself: content is plain text.
                    self.mode = "pass"
                    continue
                stripped = self.buf.lstrip()
                if not stripped:
                    break  # only whitespace so far: undecided
                if stripped.startswith(_THINK_OPEN):
                    self.buf = stripped[len(_THINK_OPEN):]
                    self.mode = "think"
                    continue
                if _THINK_OPEN.startswith(stripped):
                    break  # possible split "<think>" tag
                if not self._inline_active():
                    self.mode = "pass"  # thinking disabled: content is the answer
                    continue
                j = self.buf.find(_THINK_CLOSE)
                if j != -1:
                    # Implicit close: the template injected <think> into the prompt.
                    reason_parts.append(self.buf[:j])
                    self.buf = self.buf[j + len(_THINK_CLOSE):]
                    self._closed = True
                    _ThinkSplitter._latched = True
                    self.mode = "pass"
                    continue
                if _ThinkSplitter._latched and len(self.buf) <= self._HOLD_CAP:
                    self._held = True
                    break  # hold until </think>, the cap, or end of stream
                self.mode = "pass"
                continue
            if self.mode == "think":
                j = self.buf.find(_THINK_CLOSE)
                if j == -1:
                    keep = _partial_tag_suffix(self.buf, _THINK_CLOSE)
                    reason_parts.append(self.buf[:len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                reason_parts.append(self.buf[:j])
                self.buf = self.buf[j + len(_THINK_CLOSE):]
                self._closed = True
                self.mode = "pass"
                continue
            # pass mode
            if not self._inline_active():
                # Server parser active, thinking disabled, or the stream's one
                # </think> tag was already consumed: content is plain text.
                text_parts.append(self.buf)
                self.buf = ""
                break
            j = self.buf.find(_THINK_CLOSE)
            if j != -1:
                # First dangling close of a thinking stream after text was
                # already emitted (the template injected <think>): cannot
                # reclassify, just drop the tag and hold later thinking
                # streams' openings.
                text_parts.append(self.buf[:j])
                self.buf = self.buf[j + len(_THINK_CLOSE):]
                self._closed = True
                _ThinkSplitter._latched = True
                continue
            keep = _partial_tag_suffix(self.buf, _THINK_CLOSE)
            text_parts.append(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            break
        return "".join(text_parts), "".join(reason_parts)

    def finish(self, finish_reason: str) -> Tuple[str, str]:
        """End of stream: flush held content as (text, reasoning)."""
        if self.mode == "off":
            return "", ""
        buf, self.buf = self.buf, ""
        if not buf:
            return "", ""
        if self.mode == "think":
            return "", buf  # unclosed explicit think → reasoning
        if self.mode == "start" and self._held and buf.strip():
            # Latched hold (thinking-enabled request only) never saw
            # </think>: a truncated turn was still thinking; a completed one
            # simply did not think — the latch was wrong, so clear it.
            if finish_reason == "length":
                return "", buf
            _ThinkSplitter._latched = False
            return buf, ""
        return buf, ""


# ============================================================================
# Inline tool-call extraction (defense-in-depth backstop)
# ============================================================================
#
# vLLM only turns model tool-call markup into structured `tool_calls` deltas
# when launched with `--enable-auto-tool-choice --tool-call-parser <parser>`
# matching the served family. When the parser misses a call, the raw
# `<tool_call>...</tool_call>` block falls through into the `content` stream.
# The extractor peels those blocks back out of `content`: cleanly-parsed
# blocks are re-emitted as real tool calls (so the tool actually runs), and
# unparseable ones become tool calls with parse_error set (so the model gets an
# error back instead of a silent drop). Tags inside ``` code fences are text.
# Both GLM-4.5/4.6/5.x and Hermes/Qwen3 use the same `<tool_call>` delimiters;
# only the payload differs (GLM: a name line + <arg_key>/<arg_value> pairs;
# Hermes: a JSON object). Prefer fixing the vLLM launch flag.

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_ARG_KEY_OPEN = "<arg_key>"
_ARG_PAIR_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_JSON_NAME_RE = re.compile(r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_.\-]*)"')


def _extract_leaked_tool_calls(buf: str, in_tool: bool) -> tuple[str, list[str], str, bool]:
    """Legacy stateless extractor, kept for backward compatibility.

    Returns (clean_text, raw_blocks, carry, in_tool). The streaming path uses
    _LeakedToolCallExtractor (fence-aware, bounded hold, never drops)."""
    text_parts: list[str] = []
    blocks: list[str] = []
    while buf:
        if not in_tool:
            i_open = buf.find(_TOOL_OPEN)
            if i_open == -1:
                keep = _partial_tag_suffix(buf, _TOOL_OPEN)
                text_parts.append(buf[:len(buf) - keep] if keep else buf)
                buf = buf[len(buf) - keep:] if keep else ""
                break
            text_parts.append(buf[:i_open])
            buf = buf[i_open + len(_TOOL_OPEN):]
            in_tool = True
        else:
            j = buf.find(_TOOL_CLOSE)
            if j == -1:
                break
            blocks.append(buf[:j])
            buf = buf[j + len(_TOOL_CLOSE):]
            in_tool = False
    return "".join(text_parts), blocks, buf, in_tool


def _looks_like_tool_call(head: str) -> bool:
    """Does the start of a <tool_call> body look like a real call payload?"""
    return _ARG_KEY_OPEN in head or head.lstrip().startswith("{")


class _LeakedToolCallExtractor:
    """Stateful, linear-time extractor of leaked `<tool_call>` blocks.

    feed(text) -> (clean_text, [(body, forced_error)]) for each closed block.
    finish() -> (clean_text, [(body, forced_error)], unclosed) where unclosed
    is (body, call_like) for a block still open at end of stream, else None.

    Hold bound: a body that does not look like a call (no <arg_key>, not JSON)
    is released as literal text after _RELEASE_CHARS; a call-looking body is
    held up to _MAX_HOLD chars, after which the payload is discarded and the
    block (when it closes) becomes a parse_error call.
    """
    _RELEASE_CHARS = 512
    _MAX_HOLD = 1_000_000
    _HEAD_CHARS = 1024

    def __init__(self, known_names: Optional[set] = None):
        self.buf = ""             # text-side carry (possible partial open tag / backticks)
        self.in_tool = False
        self.fence_open = False   # inside a ``` code fence in emitted text
        self._last_char = ""      # last emitted character (inline-code detection)
        # Tool names offered on this request; a closed block that neither
        # looks like a call payload nor names one of these is prose → text.
        self.known_names = known_names or set()
        self._reset_block()

    def _reset_block(self):
        self._parts: List[str] = []
        self._len = 0
        self._tail = ""
        self._head = ""
        self._call_like = False
        self._overflow = False

    def _emit(self, s: str, out: List[str]):
        if s:
            out.append(s)
            self._last_char = s[-1]
            if s.count("```") % 2 == 1:
                self.fence_open = not self.fence_open

    def _is_prose_block(self, body: str) -> bool:
        """A closed block that is clearly not a tool call (e.g. the model
        writing about the `<tool_call>` tag): no call payload shape and no
        known tool name on its first line."""
        if _looks_like_tool_call(body[:self._HEAD_CHARS]):
            return False
        name = _guess_tool_name(body)
        if name is None:
            return True
        return bool(self.known_names) and name not in self.known_names

    def _feed_block(self, s: str, out: List[str], blocks: list) -> str:
        """Feed body text while a block is open; returns unconsumed input."""
        probe = self._tail + s
        j = probe.find(_TOOL_CLOSE)
        if j != -1:
            cut = j - len(self._tail)
            if self._overflow:
                blocks.append((self._head, (
                    f"tool call markup exceeded {self._MAX_HOLD} characters and "
                    f"was discarded; write large content in smaller pieces")))
            else:
                joined = "".join(self._parts)
                body = (joined + s)[:len(joined) + cut]
                if self._is_prose_block(body):
                    self._emit(_TOOL_OPEN + body + _TOOL_CLOSE, out)
                else:
                    blocks.append((body, None))
            rest = s[cut + len(_TOOL_CLOSE):]
            self.in_tool = False
            self._reset_block()
            return rest
        keep_tail = len(_TOOL_CLOSE) - 1
        self._tail = probe[-keep_tail:]
        if len(self._head) < self._HEAD_CHARS:
            self._head += s[:self._HEAD_CHARS - len(self._head)]
            if not self._call_like and _looks_like_tool_call(self._head):
                self._call_like = True
        self._len += len(s)
        if not self._overflow:
            self._parts.append(s)
        if not self._call_like and self._len > self._RELEASE_CHARS:
            # Not a tool call (e.g. prose mentioning the tag): release as text
            # and re-scan the held body for a later real open tag.
            held = "".join(self._parts)
            self.in_tool = False
            self._reset_block()
            self._emit(_TOOL_OPEN, out)
            return held
        if not self._overflow and self._len > self._MAX_HOLD:
            self._overflow = True
            self._parts = []
        return ""

    def feed(self, s: str) -> Tuple[str, list]:
        out: List[str] = []
        blocks: list = []
        pending = s
        while pending:
            if self.in_tool:
                pending = self._feed_block(pending, out, blocks)
                continue
            buf = self.buf + pending
            self.buf = ""
            pending = ""
            i = buf.find(_TOOL_OPEN)
            if i == -1:
                keep = _partial_tag_suffix(buf, _TOOL_OPEN)
                ticks = len(buf) - len(buf.rstrip("`"))
                if 0 < ticks < 3:
                    keep = max(keep, ticks)  # a ``` fence split across chunks
                self._emit(buf[:len(buf) - keep], out)
                self.buf = buf[len(buf) - keep:]
                break
            self._emit(buf[:i], out)
            if self.fence_open or self._last_char == "`":
                # Inside a code fence / inline `code` span: literal text.
                self._emit(_TOOL_OPEN, out)
                pending = buf[i + len(_TOOL_OPEN):]
                continue
            self.in_tool = True
            self._reset_block()
            pending = buf[i + len(_TOOL_OPEN):]
        return "".join(out), blocks

    def finish(self) -> Tuple[str, list, Optional[Tuple[str, bool]]]:
        out: List[str] = []
        unclosed = None
        if self.in_tool:
            body = self._head if self._overflow else "".join(self._parts)
            unclosed = (body, self._call_like)
            self.in_tool = False
            self._reset_block()
        else:
            self._emit(self.buf, out)
        self.buf = ""
        return "".join(out), [], unclosed


def _tool_names(tools: Optional[List[Dict[str, Any]]]) -> set:
    """Names of the tools offered on a request (OpenAI tool schema list)."""
    names = set()
    for t in tools or ():
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _tool_param_types(name: str, tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Optional[str]]:
    """{param: json-schema type} for a tool, from the request's tools or TOOL_SCHEMAS."""
    props: Optional[Dict[str, Any]] = None
    for t in tools or ():
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if fn.get("name") == name:
            params = fn.get("parameters") or {}
            props = params.get("properties") if isinstance(params, dict) else None
            break
    if props is None:
        try:
            from tools.schemas import TOOL_SCHEMAS
            schema = TOOL_SCHEMAS.get(name) or {}
            params = schema.get("parameters") or {}
            props = params.get("properties") if isinstance(params, dict) else None
        except Exception:
            props = None
    types: Dict[str, Optional[str]] = {}
    for key, spec in (props or {}).items():
        types[key] = _schema_type(spec)
    return types


def _schema_type(spec: Any) -> Optional[str]:
    if not isinstance(spec, dict):
        return None
    t = spec.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if isinstance(t, str):
        return t
    for key in ("anyOf", "oneOf"):
        for sub in spec.get(key) or ():
            st = _schema_type(sub)
            if st and st != "null":
                return st
    enum = spec.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(x, str) for x in enum):
        return "string"
    return None


def _coerce_glm_value(raw: str, typ: Optional[str]) -> Any:
    """Coerce one GLM <arg_value> by its schema type (strings stay raw)."""
    if typ == "string":
        v = raw
        if v.startswith("\r\n"):
            v = v[2:]
        elif v.startswith("\n"):
            v = v[1:]
        if v.endswith("\r\n"):
            v = v[:-2]
        elif v.endswith("\n"):
            v = v[:-1]
        return v
    sv = raw.strip()
    if typ in ("integer", "number"):
        try:
            n = json.loads(sv)
        except ValueError:
            return sv
        if isinstance(n, bool) or not isinstance(n, (int, float)):
            return sv
        if typ == "integer" and isinstance(n, float) and n.is_integer():
            return int(n)
        return n
    if typ == "boolean":
        low = sv.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        return sv
    try:
        return json.loads(sv)  # object / array / unknown type (legacy behaviour)
    except ValueError:
        return sv


def _guess_tool_name(raw: str) -> Optional[str]:
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("{"):
        m = _JSON_NAME_RE.search(s[:2000])
        return m.group(1) if m else None
    first = s[:s.find(_ARG_KEY_OPEN)] if _ARG_KEY_OPEN in s[:2000] else s[:200]
    lines = first.strip().splitlines()
    name = lines[0].strip() if lines else ""
    return name if _TOOL_NAME_RE.match(name) else None


def _parse_leaked_block(raw: str, tools: Optional[List[Dict[str, Any]]] = None
                        ) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Parse one leaked `<tool_call>` body → (name, arguments, parse_error)."""
    s = (raw or "").strip()
    if not s:
        return None, {}, _parse_error_msg("empty <tool_call> block", raw)

    # Hermes / OpenAI JSON form.
    if s.startswith("{"):
        try:
            obj, _end = _JSON_DECODER.raw_decode(s)
        except ValueError as e:
            return (_guess_tool_name(s), {},
                    _parse_error_msg(f"tool call JSON is invalid ({getattr(e, 'msg', e)})", raw))
        if not isinstance(obj, dict):
            return None, {}, _parse_error_msg("tool call JSON is not an object", raw)
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = fn.get("name")
        if not isinstance(name, str) or not _TOOL_NAME_RE.match(name.strip()):
            return None, {}, _parse_error_msg("tool call JSON has no valid \"name\"", raw)
        args_raw = fn.get("arguments", fn.get("parameters", {}))
        args, err = _normalize_tool_arguments(args_raw)
        return name.strip(), args, err

    # GLM XML form: name line, then <arg_key>/<arg_value> pairs.
    first_pair = _ARG_PAIR_RE.search(s)
    head = (s[:first_pair.start()] if first_pair else s).strip()
    lines = head.splitlines()
    name = lines[0].strip() if lines else ""
    if not _TOOL_NAME_RE.match(name):
        return None, {}, _parse_error_msg("could not find a tool name in <tool_call>", raw)
    if first_pair is None:
        remainder = head[len(name):].strip()
        if remainder.startswith("{"):
            args, err = _normalize_tool_arguments(remainder)
            return name, args, err
        if remainder or _ARG_KEY_OPEN in s:
            return name, {}, _parse_error_msg(
                "malformed <arg_key>/<arg_value> pairs in <tool_call>", raw)
        return name, {}, None  # zero-argument call
    pairs = _ARG_PAIR_RE.findall(s)
    if s.count(_ARG_KEY_OPEN) != len(pairs):
        return name, {}, _parse_error_msg(
            "malformed <arg_key>/<arg_value> pairs in <tool_call>", raw)
    types = _tool_param_types(name, tools)
    args: Dict[str, Any] = {}
    for k, v in pairs:
        key = k.strip()
        args[key] = _coerce_glm_value(v, types.get(key))
    return name, args, None


def _parse_leaked_tool_call(raw: str, tools: Optional[List[Dict[str, Any]]] = None
                            ) -> Optional[tuple[str, dict]]:
    """Backward-compatible wrapper: (name, arguments) or None when unparseable."""
    name, args, err = _parse_leaked_block(raw, tools)
    if not name or err:
        return None
    return name, args


def extract_tool_calls_from_text(text: str, tools: Optional[List[Dict[str, Any]]] = None
                                 ) -> List[ToolCall]:
    """Parse every complete `<tool_call>...</tool_call>` block in `text`.

    For salvaging a call the model wrote inside its reasoning. Returns only
    cleanly parsed calls (no parse_error), each with a fresh `call_<uuid8>` id.
    """
    calls: List[ToolCall] = []
    if not text or _TOOL_OPEN not in text:
        return calls
    pos = 0
    while True:
        i = text.find(_TOOL_OPEN, pos)
        if i == -1:
            break
        j = text.find(_TOOL_CLOSE, i + len(_TOOL_OPEN))
        if j == -1:
            break
        body = text[i + len(_TOOL_OPEN):j]
        pos = j + len(_TOOL_CLOSE)
        name, args, err = _parse_leaked_block(body, tools)
        if name and not err:
            calls.append(ToolCall(id=_new_call_id(),
                                  function=ToolCallFunction(name=name, arguments=args)))
    return calls


# ============================================================================
# SSE stream parser (pure state machine: no I/O, unit-testable)
# ============================================================================

def _inband_error(chunk: Dict[str, Any]) -> Optional[Tuple[str, Optional[int]]]:
    """(message, status_code) when a chunk is a vLLM in-band error, else None."""
    err = chunk.get("error")
    if err is None and chunk.get("object") != "error":
        return None
    src = err if isinstance(err, dict) else chunk
    if isinstance(err, str):
        message = err
    else:
        message = str(src.get("message") or src.get("detail") or err or "unknown error")
    code = src.get("code") if isinstance(src, dict) else None
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None
    return message, code


class _StreamParser:
    """Turns vLLM SSE lines into StreamEvents for one request attempt."""

    def __init__(self, tools: Optional[List[Dict[str, Any]]], sent_max_tokens: Optional[int],
                 expect_thinking: bool = True):
        self.tools = tools
        self.sent_max_tokens = sent_max_tokens
        self.splitter = _ThinkSplitter(expect_thinking)
        # Set when a chunk carried finish_reason abort/error: the chunk's own
        # events are delivered first, then the stream loop raises this.
        self.pending_error: Optional[VllmStreamError] = None
        self.extractor = _LeakedToolCallExtractor(_tool_names(tools))
        self.pending: Dict[int, Dict[str, Any]] = {}  # index -> {id, vllm_id, name, args}
        self.dispatched: set = set()                  # indices already yielded mid-stream
        self.max_seen_idx = -1
        self.last_idx = -1
        self.structured_sigs: set = set()
        self.leaked_sigs: set = set()
        self.finish_reason: Optional[str] = None
        self.saw_finish = False
        self.saw_done = False
        self.usage: Optional[Dict[str, Any]] = None
        self.emitted = 0                              # events handed to the caller

    def _out(self, events: List[StreamEvent]) -> List[StreamEvent]:
        self.emitted += len(events)
        return events

    # ---------------- per-line ----------------

    def handle_line(self, raw_line: str) -> List[StreamEvent]:
        line = raw_line.strip()
        if not line.startswith("data:"):
            return []
        data = line[5:].strip()
        if not data:
            return []
        if data == "[DONE]":
            self.saw_done = True
            return []
        try:
            chunk = json.loads(data)
        except ValueError:
            return []
        if not isinstance(chunk, dict):
            return []
        return self.handle_chunk(chunk)

    def handle_chunk(self, chunk: Dict[str, Any]) -> List[StreamEvent]:
        err = _inband_error(chunk)
        if err is not None:
            message, code = err
            raise VllmStreamError(f"vLLM stream error: {message}",
                                  partial=self.emitted > 0, status_code=code)
        # Usage may arrive in its own trailing chunk (choices empty) or, with
        # continuous usage stats, on every chunk: last one wins.
        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
            self.usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return []
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}
        events: List[StreamEvent] = []

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning and isinstance(reasoning, str):
            text, held_reasoning = self.splitter.disable()
            if held_reasoning:
                events.append(ReasoningEvent(content=held_reasoning))
            if text:
                events.extend(self._text_events(text))
            events.append(ReasoningEvent(content=reasoning))

        content = delta.get("content")
        if content and isinstance(content, str):
            text, inline_reasoning = self.splitter.feed(content)
            if inline_reasoning:
                events.append(ReasoningEvent(content=inline_reasoning))
            if text:
                events.extend(self._text_events(text))

        tool_deltas = delta.get("tool_calls") or []
        if isinstance(tool_deltas, list):
            for tc_delta in tool_deltas:
                if isinstance(tc_delta, dict):
                    events.extend(self._tool_delta(tc_delta))

        finish = choice.get("finish_reason")
        if finish:
            finish = str(finish)
            if finish in ("abort", "error"):
                # The same chunk may carry the last tokens (MTP multi-token
                # chunks): deliver them first, then the caller raises.
                out = self._out(events)
                self.pending_error = VllmStreamError(
                    f"vLLM ended the request with finish_reason={finish!r}",
                    partial=self.emitted > 0)
                return out
            self.finish_reason = finish
            self.saw_finish = True
        return self._out(events)

    # ---------------- content / leaked calls ----------------

    def _text_events(self, text: str) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        clean, blocks = self.extractor.feed(text)
        if clean:
            events.append(TextEvent(content=clean))
        for body, forced_err in blocks:
            ev = self._leaked_call_event(body, forced_err)
            if ev is not None:
                events.append(ev)
        return events

    def _leaked_call_event(self, body: str, forced_err: Optional[str]) -> Optional[ToolCallDeltaEvent]:
        if forced_err:
            name = _guess_tool_name(body) or "invalid_tool_call"
            args: Dict[str, Any] = {}
            err: Optional[str] = _parse_error_msg(forced_err, body)
        else:
            name, args, err = _parse_leaked_block(body, self.tools)
            name = name or "invalid_tool_call"
        if err is None:
            sig = _call_signature(name, args)
            if sig in self.structured_sigs:
                return None  # the parser already delivered this call
            self.leaked_sigs.add(sig)
        print(f"[LLM] Recovered <tool_call> markup from content stream ({name}"
              f"{', parse_error' if err else ''}) — check vLLM --tool-call-parser")
        return ToolCallDeltaEvent(
            tool_calls=[ToolCall(
                id=_new_call_id(),
                function=ToolCallFunction(name=name, arguments=args, parse_error=err),
            )],
            finish_reason="tool",
            is_partial=True,
        )

    # ---------------- structured tool-call deltas ----------------

    def _tool_delta(self, tc_delta: Dict[str, Any]) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        idx = tc_delta.get("index")
        tc_id = tc_delta.get("id") or None
        if not isinstance(idx, int) or isinstance(idx, bool):
            if self.last_idx >= 0:
                last = self.pending[self.last_idx]
                if tc_id and last.get("vllm_id") and tc_id != last["vllm_id"]:
                    idx = self.max_seen_idx + 1
                else:
                    idx = self.last_idx
            else:
                idx = 0

        if idx not in self.pending:
            # A new index means the PREVIOUS max index is complete: dispatch it
            # now so the agent can start executing it.
            if self.max_seen_idx >= 0 and self.max_seen_idx not in self.dispatched:
                ev = self._try_early_dispatch(self.max_seen_idx)
                if ev is not None:
                    events.append(ev)
            self.pending[idx] = {
                "id": tc_id or _new_call_id(), "vllm_id": tc_id, "name": "", "args": "",
            }
        entry = self.pending[idx]
        if tc_id and not entry["vllm_id"] and idx not in self.dispatched:
            entry["id"] = tc_id
            entry["vllm_id"] = tc_id

        func_delta = tc_delta.get("function") or {}
        if isinstance(func_delta, dict):
            name_part = func_delta.get("name") or ""
            if isinstance(name_part, str) and name_part:
                cur = entry["name"]
                if not cur:
                    entry["name"] = name_part
                elif name_part == cur:
                    pass  # some servers repeat the full name on later deltas
                elif name_part.startswith(cur):
                    entry["name"] = name_part
                else:
                    entry["name"] = cur + name_part
            args_part = func_delta.get("arguments")
            if isinstance(args_part, dict):
                entry["args"] = json.dumps(args_part, ensure_ascii=False)
            elif isinstance(args_part, str):
                entry["args"] += args_part
            elif args_part is not None:
                entry["args"] += json.dumps(args_part, ensure_ascii=False, default=str)

        self.last_idx = idx
        if idx > self.max_seen_idx:
            self.max_seen_idx = idx

        # JSON-complete early dispatch: once the accumulated argument string is
        # a complete JSON object, yield immediately instead of waiting for the
        # next index or [DONE].
        if (idx not in self.dispatched and entry["name"]
                and entry["args"].rstrip().endswith("}")):
            ev = self._try_early_dispatch(idx)
            if ev is not None:
                events.append(ev)
        return events

    def _try_early_dispatch(self, idx: int) -> Optional[ToolCallDeltaEvent]:
        entry = self.pending[idx]
        name = entry["name"].strip()
        if not name:
            return None
        args = _strict_args(entry["args"])
        if args is None:
            return None  # incomplete — normalized at stream end
        self.dispatched.add(idx)
        sig = _call_signature(name, args)
        if sig in self.leaked_sigs:
            return None  # already recovered from leaked markup
        self.structured_sigs.add(sig)
        return ToolCallDeltaEvent(
            tool_calls=[ToolCall(id=entry["id"],
                                 function=ToolCallFunction(name=name, arguments=args))],
            finish_reason="tool",
            is_partial=True,
        )

    # ---------------- end of stream ----------------

    def is_complete(self) -> bool:
        return self.saw_finish or self.saw_done

    def effective_finish_reason(self) -> str:
        fr = self.finish_reason or "stop"
        if self.sent_max_tokens and self.usage:
            if _as_int(self.usage.get("completion_tokens")) >= int(self.sent_max_tokens):
                fr = "length"
        return fr

    def finalize(self) -> List[StreamEvent]:
        """Flush held state, remaining tool calls and the terminal UsageEvent."""
        eff = self.effective_finish_reason()
        events: List[StreamEvent] = []

        text, reasoning = self.splitter.finish(eff)
        if reasoning:
            events.append(ReasoningEvent(content=reasoning))
        if text:
            events.extend(self._text_events(text))

        tail_text, _blocks, unclosed = self.extractor.finish()
        if tail_text:
            events.append(TextEvent(content=tail_text))
        if unclosed is not None:
            body, call_like = unclosed
            ev = None
            if eff == "length" and self._unclosed_is_call(body, call_like):
                # Truncated mid-call: show nothing (half a call is useless
                # markup) — the terminal UsageEvent's finish_reason="length"
                # tells the agent loop the turn was cut off.
                print(f"[LLM] Dropped unclosed <tool_call> block ({len(body)} chars) "
                      f"truncated by the output token limit")
            elif self.extractor._is_prose_block(body):
                events.append(TextEvent(content=_TOOL_OPEN + body))
            else:
                name, _args, err = _parse_leaked_block(body, self.tools)
                if name and err is None:
                    ev = self._leaked_call_event(body, None)
                elif call_like:
                    ev = self._leaked_call_event(body, "unclosed <tool_call> block")
                else:
                    events.append(TextEvent(content=_TOOL_OPEN + body))
            if ev is not None:
                events.append(ev)

        remaining: List[ToolCall] = []
        for idx in sorted(self.pending):
            if idx in self.dispatched:
                continue
            entry = self.pending[idx]
            name = entry["name"].strip()
            raw = entry["args"]
            if not name and not raw.strip():
                continue  # empty phantom delta
            args, err = _normalize_tool_arguments(raw, truncated=(eff == "length"))
            if not name:
                name = "invalid_tool_call"
                err = err or _parse_error_msg("tool call had no function name", raw)
            if err is None:
                sig = _call_signature(name, args)
                if sig in self.leaked_sigs:
                    continue
                self.structured_sigs.add(sig)
            remaining.append(ToolCall(
                id=entry["id"],
                function=ToolCallFunction(name=name, arguments=args, parse_error=err),
            ))
        if remaining:
            events.append(ToolCallDeltaEvent(
                tool_calls=remaining, finish_reason=eff, is_partial=False))

        usage = self.usage or {}
        prompt_tokens = _as_int(usage.get("prompt_tokens"))
        completion_tokens = _as_int(usage.get("completion_tokens"))
        events.append(UsageEvent(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=_as_int(usage.get("total_tokens")) or (prompt_tokens + completion_tokens),
            finish_reason=eff,
        ))
        return self._out(events)

    def _unclosed_is_call(self, body: str, call_like: bool) -> bool:
        """Is a still-open <tool_call> body (likely) a truncated call rather
        than prose that merely mentions the tag? Empty bodies count as calls
        (a bare trailing open tag is half a call, not an answer)."""
        if call_like or not body.strip():
            return True
        return not self.extractor._is_prose_block(body)

    def drain_on_error(self) -> List[StreamEvent]:
        """Flush held visible text before an error surfaces after partial output.

        Ambiguous held content goes out as reasoning (never leak thinking); an
        open <tool_call> block that looks like a call is dropped (truncated
        markup is never shown), while held prose that merely mentions the tag
        is released as text."""
        events: List[StreamEvent] = []
        text, reasoning = self.splitter.finish("length")
        if reasoning:
            events.append(ReasoningEvent(content=reasoning))
        if text:
            clean, _blocks = self.extractor.feed(text)
            if clean:
                events.append(TextEvent(content=clean))
        tail_text, _blocks, unclosed = self.extractor.finish()
        if tail_text:
            events.append(TextEvent(content=tail_text))
        if unclosed is not None:
            body, call_like = unclosed
            if not self._unclosed_is_call(body, call_like):
                events.append(TextEvent(content=_TOOL_OPEN + body))
        return self._out(events)


@dataclass
class _PreparedRequest:
    body: bytes
    sent_max_tokens: Optional[int]
    raw_estimate: int
    expect_thinking: bool = True  # False when chat_template_kwargs disable thinking


def _thinking_requested(ctk: Any) -> bool:
    """Does this request leave GLM-style thinking mode on?

    False only when the effective chat_template_kwargs explicitly disable it
    (enable_thinking=False, thinking=False, or thinking={"type": "disabled"});
    absent kwargs mean the model's default, which for reasoning models thinks."""
    if not isinstance(ctk, dict):
        return True
    if ctk.get("enable_thinking") is False:
        return False
    thinking = ctk.get("thinking")
    if thinking is False:
        return False
    if isinstance(thinking, dict) and str(thinking.get("type", "")).lower() == "disabled":
        return False
    return True


# ============================================================================
# vLLM Backend
# ============================================================================

class VllmBackend:
    """vLLM backend: fully async, OpenAI-compatible, native tool calling."""

    def __init__(self, host: str = None):
        self.host = (host or config.VLLM_HOST).rstrip("/")
        self._ssl_verify = self._resolve_ssl()
        # Persistent connection pool sized for many concurrent agent streams.
        # read=None: streaming reads are bounded by the first-byte / idle
        # watchdogs in _chat_stream_once instead (spec-decode streams are
        # bursty). pool=None: waiting for a pooled connection is bounded by
        # the request semaphore, not a pool timeout.
        pool_size = max(1, int(getattr(config, "VLLM_POOL_MAX_CONNECTIONS",
                                       getattr(config, "VLLM_CONNECTION_POOL_SIZE", 128)) or 128))
        self._client = httpx.AsyncClient(
            verify=self._ssl_verify,
            timeout=httpx.Timeout(
                connect=_pos_float(getattr(config, "STREAM_CONNECT_TIMEOUT", 10.0), 10.0),
                read=None,
                write=60.0,
                pool=None,
            ),
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=min(64, pool_size),
            ),
        )
        # Request-concurrency semaphore (lazily bound to the running loop).
        self._sem: Optional[asyncio.Semaphore] = None
        self._sem_loop = None
        # Served-window detection (/v1/models max_model_len, TTL-cached).
        self._detected_window = 0
        self._window_checked_at = 0.0
        self._window_ok = False
        self._window_task: Optional[asyncio.Task] = None
        # Calibration of the byte-based prompt estimate (EMA of actual/estimate).
        self._prompt_scale = 1.0

    def _resolve_ssl(self):
        from pathlib import Path
        cert_path = Path("C:/DigitalCity.crt")
        if cert_path.exists():
            return str(cert_path)
        return True

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            n = max(1, int(getattr(config, "VLLM_MAX_CONCURRENT_REQUESTS", 64) or 64))
            self._sem = asyncio.Semaphore(n)
            self._sem_loop = loop
        return self._sem

    async def _select_available_host(self, *, prefer_active: bool = False) -> bool:
        """Kept for compatibility: True when /v1/models answers 200."""
        try:
            resp = await self._client.get(
                f"{self.host}/v1/models",
                timeout=httpx.Timeout(3.0),
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Shut down the persistent HTTP client."""
        await self._client.aclose()

    async def is_available(self) -> bool:
        return await self._select_available_host()

    async def list_models(self) -> List[str]:
        resp = await self._client.get(
            f"{self.host}/v1/models",
            timeout=httpx.Timeout(5.0),
        )
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    # ------------------------------------------------------------------
    # Context window detection
    # ------------------------------------------------------------------

    async def _fetch_max_model_len(self, model: Optional[str]) -> int:
        resp = await self._client.get(
            f"{self.host}/v1/models",
            timeout=httpx.Timeout(5.0),
        )
        resp.raise_for_status()
        entries = resp.json().get("data", []) or []
        for entry in entries:
            if model and entry.get("id") == model:
                return _as_int(entry.get("max_model_len"))
        if entries:
            return _as_int(entries[0].get("max_model_len"))
        return 0

    async def _refresh_window(self, model: Optional[str]) -> None:
        try:
            n = await self._fetch_max_model_len(model)
        except Exception:
            n = 0
        if n > 0:
            self._detected_window = n
            self._window_ok = True
        else:
            self._window_ok = False

    def _schedule_window_refresh(self, model: Optional[str]) -> None:
        """Lazy, TTL-cached /v1/models probe in the background — never per call."""
        now = time.monotonic()
        ttl = _WINDOW_TTL_OK if self._window_ok else _WINDOW_TTL_FAIL
        if self._window_checked_at and now - self._window_checked_at < ttl:
            return
        self._window_checked_at = now  # stampede guard: set before awaiting
        try:
            self._window_task = asyncio.get_running_loop().create_task(
                self._refresh_window(model))
        except RuntimeError:
            pass

    async def get_context_window(self, model: str) -> int:
        """Return the served model's max context length (vLLM's `max_model_len`
        field on /v1/models), or 0 if unavailable/unreported. Also refreshes
        the detected window used by effective_context_window()."""
        try:
            n = await self._fetch_max_model_len(model)
        except Exception:
            return 0
        if n > 0:
            self._detected_window = n
            self._window_ok = True
            self._window_checked_at = time.monotonic()
        return n

    def effective_context_window(self) -> int:
        """min(configured MODEL_CONTEXT_WINDOW, detected vLLM max_model_len); 0 if unknown."""
        configured = _as_int(getattr(config, "MODEL_CONTEXT_WINDOW", 1_000_000))
        detected = self._detected_window
        if configured > 0 and detected > 0:
            return min(configured, detected)
        return configured if configured > 0 else detected

    @property
    def _learned_window(self) -> int:
        """Effective window (read by the agent's compaction thresholds)."""
        return self.effective_context_window()

    @_learned_window.setter
    def _learned_window(self, value: int) -> None:
        self._detected_window = _as_int(value)

    # ------------------------------------------------------------------
    # Prompt-size estimate and max_tokens fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_from_serialized(messages_json: str, tools_json: str,
                                  messages: List[Dict[str, Any]]) -> int:
        nbytes = _utf8_len(messages_json)
        if tools_json:
            nbytes += _utf8_len(tools_json)
        img_bytes, n_images = _image_payload_stats(messages)
        return max(0, nbytes - img_bytes) // 3 + n_images * _IMAGE_TOKEN_COST

    @staticmethod
    def _estimate_prompt_tokens(
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> int:
        """Raw prompt estimate: JSON bytes // 3 (tool schemas included), base64
        image data excluded and a fixed charge per image part instead."""
        messages_json = json.dumps(messages, ensure_ascii=False, default=str)
        tools_json = json.dumps(tools, ensure_ascii=False, default=str) if tools else ""
        return VllmBackend._estimate_from_serialized(messages_json, tools_json, messages)

    def _update_prompt_scale(self, actual_tokens: int, raw_estimate: int) -> None:
        """EMA (alpha 0.3) of actual/estimated prompt tokens, clamped [0.4, 2.0]."""
        if actual_tokens <= 0 or raw_estimate <= 0:
            return
        ratio = min(2.0, max(0.4, actual_tokens / raw_estimate))
        self._prompt_scale = min(2.0, max(0.4, 0.7 * self._prompt_scale + 0.3 * ratio))

    def _fit_max_tokens(
        self,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        raw_estimate: Optional[int] = None,
    ) -> int:
        """Clamp max_tokens so prompt + output fits the served context window.

        vLLM rejects a request unless prompt_tokens + max_tokens <= max_model_len.
        The requested max_tokens is only a ceiling (a caller's small value is
        never raised); it shrinks to the room actually left (window - scaled
        prompt estimate - safety margin), floored at MIN_OUTPUT_TOKENS. No-op
        while the window is unknown (<= 0).
        """
        window = self.effective_context_window()
        if window <= 0:
            return max_tokens
        if raw_estimate is None:
            raw_estimate = self._estimate_prompt_tokens(messages, tools)
        prompt_est = int(raw_estimate * self._prompt_scale)
        margin = _as_int(getattr(config, "MAX_TOKENS_SAFETY_MARGIN", 2048))
        floor = _as_int(getattr(config, "MIN_OUTPUT_TOKENS", 8192))
        room = window - prompt_est - margin
        if room < floor:
            print(f"[LLM] Low output budget: window {window}, est. prompt {prompt_est}, "
                  f"room {room} < MIN_OUTPUT_TOKENS {floor} — sending {min(max_tokens, floor)}; "
                  f"context compaction is due")
        return min(max_tokens, max(floor, room))

    @staticmethod
    def _parse_context_overflow(body: str) -> Optional[tuple]:
        """Extract (max_model_len, prompt_tokens) from a vLLM context-overflow 400.

        Handles the wordings vLLM has used, including the max_tokens-too-large
        variant. Returns None when the body is not a recognizable overflow."""
        if not body:
            return None
        m = re.search(r"maximum context length is (\d+) tokens.*?request has (\d+) input tokens",
                      body, re.DOTALL)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"prompt \(length (\d+)\) is longer than the maximum model length of (\d+)", body)
        if m:
            return int(m.group(2)), int(m.group(1))
        m = re.search(r"\((\d+) > (\d+) - (\d+)\)", body)
        if m and "max_tokens" in body:
            return int(m.group(2)), int(m.group(3))
        m_limit = re.search(r"context length is (?:only )?(\d+)", body)
        m_prompt = (
            re.search(r"(\d+) tokens in the input messages", body)
            or re.search(r"\((\d+) in the messages", body)
            or re.search(r"passed (\d+) input tokens", body)
            or re.search(r"(\d+) input tokens", body)
        )
        if not (m_limit and m_prompt):
            return None
        return int(m_limit.group(1)), int(m_prompt.group(1))

    # ------------------------------------------------------------------
    # Request payload
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        guided_json: Optional[dict] = None,
        response_format: Optional[dict] = None,
        chat_template_kwargs: Optional[dict] = None,
        *,
        include_messages: bool = True,
        raw_estimate: Optional[int] = None,
        max_tokens_cap: Optional[int] = None,
    ) -> dict[str, Any]:
        """Assemble the request payload with all vLLM parameters.

        Always uses streaming — there is no non-streaming path in this backend.
        """
        payload: dict[str, Any] = {"model": model}
        if include_messages:
            payload["messages"] = messages
        payload.update({
            "temperature": temperature,
            "stream": True,
            # Final usage chunk → REAL token counts for compaction/calibration.
            "stream_options": {"include_usage": True},
        })
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = True
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if min_p is not None:
            payload["min_p"] = min_p
        if max_tokens is not None:
            fitted = self._fit_max_tokens(max_tokens, messages, tools, raw_estimate)
            if max_tokens_cap is not None:
                fitted = min(fitted, max_tokens_cap)
            payload["max_tokens"] = fitted
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        # Structured / guided decoding (vLLM). Opt-in per request.
        if guided_json is not None:
            payload["guided_json"] = guided_json
        if response_format is not None:
            payload["response_format"] = response_format
        # Chat-template kwargs (e.g. GLM-5.2 reasoning_effort / enable_thinking),
        # forwarded as a top-level body field; per-call override merged over
        # the global default (C4).
        effective_ctk = _merge_chat_template_kwargs(chat_template_kwargs)
        if effective_ctk:
            payload["chat_template_kwargs"] = effective_ctk
        return payload

    def _prepare_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float,
        tools: Optional[List[Dict[str, Any]]],
        sampling: Dict[str, Any],
        max_tokens_cap: Optional[int] = None,
    ) -> _PreparedRequest:
        """Serialize messages ONCE: the same JSON feeds the estimate and the body."""
        messages_json = json.dumps(messages, ensure_ascii=False, default=str)
        tools_json = json.dumps(tools, ensure_ascii=False, default=str) if tools else ""
        raw_est = self._estimate_from_serialized(messages_json, tools_json, messages)
        params = self._build_payload(
            messages, model, temperature, tools=tools,
            include_messages=False, raw_estimate=raw_est,
            max_tokens_cap=max_tokens_cap, **sampling,
        )
        rest_json = json.dumps(params, ensure_ascii=False, default=str)
        try:
            body = ('{"messages":' + messages_json + "," + rest_json[1:]).encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate (e.g. surrogateescape-decoded tool output) is not
            # encodable as UTF-8; \u-escape everything instead, exactly like
            # httpx's json= did, rather than failing every call of the session.
            body = ('{"messages":' + json.dumps(messages, default=str) + ","
                    + json.dumps(params, default=str)[1:]).encode("ascii")
        return _PreparedRequest(body=body, sent_max_tokens=params.get("max_tokens"),
                                raw_estimate=raw_est,
                                expect_thinking=_thinking_requested(
                                    params.get("chat_template_kwargs")))

    # ------------------------------------------------------------------
    # Non-streaming convenience (C4)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        guided_json: Optional[dict] = None,
        response_format: Optional[dict] = None,
        chat_template_kwargs: Optional[dict] = None,
        **_ignored: Any,
    ) -> LLMResponse:
        return await collect_llm_response(self.chat_stream(
            messages, model, temperature, tools=tools,
            top_p=top_p, top_k=top_k, min_p=min_p, max_tokens=max_tokens,
            repetition_penalty=repetition_penalty, guided_json=guided_json,
            response_format=response_format, chat_template_kwargs=chat_template_kwargs,
        ))

    # ------------------------------------------------------------------
    # Streaming chat: retries + overflow clamp around _chat_stream_once
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        guided_json: Optional[dict] = None,
        response_format: Optional[dict] = None,
        chat_template_kwargs: Optional[dict] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one completion with internal recovery.

        - Transport errors, first-byte/idle timeouts, 5xx and in-band errors
          are retried (VLLM_STREAM_MAX_RETRIES extra attempts, backoff 0.5s /
          1.5s / 4s) only while nothing has been yielded. After output was
          yielded → VllmStreamError(partial=True); retries exhausted →
          VllmStreamError(partial=False, status_code=...).
        - Context-overflow 400 (prompt + max_tokens > window): learn the real
          window and retry ONCE with max_tokens clamped to the room really left,
          only if that room is >= MIN_OUTPUT_TOKENS. Otherwise the 400 is
          re-raised as httpx.HTTPStatusError so the agent compacts.
        """
        sampling = {
            "top_p": top_p, "top_k": top_k, "min_p": min_p, "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty, "guided_json": guided_json,
            "response_format": response_format, "chat_template_kwargs": chat_template_kwargs,
        }
        self._schedule_window_refresh(model)
        prep = self._prepare_request(messages, model, temperature, tools, sampling)
        max_retries = max(0, _as_int(getattr(config, "VLLM_STREAM_MAX_RETRIES", 2)))
        attempt = 0
        overflow_retried = False

        while True:
            yielded = False
            last_err: Optional[BaseException] = None
            status_code: Optional[int] = None
            agen = self._chat_stream_once(prep, tools)
            try:
                async for event in agen:
                    yielded = True
                    if isinstance(event, UsageEvent):
                        self._update_prompt_scale(event.prompt_tokens, prep.raw_estimate)
                    yield event
                return
            except httpx.HTTPStatusError as e:
                status_code = _status_of(e)
                if yielded:
                    raise VllmStreamError(f"vLLM stream failed after partial output: {_fmt_exc(e)}",
                                          partial=True, status_code=status_code) from e
                if status_code is not None and status_code < 500:
                    if not overflow_retried and max_tokens:
                        clamp = self._overflow_clamp(str(e), max_tokens, prep.raw_estimate)
                        if clamp is not None:
                            overflow_retried = True
                            prep = self._prepare_request(messages, model, temperature, tools,
                                                         sampling, max_tokens_cap=clamp)
                            continue
                    raise
                last_err = e
            except VllmStreamError as e:
                if yielded or e.partial:
                    if e.partial:
                        raise
                    raise VllmStreamError(e.message, partial=True,
                                          status_code=e.status_code) from e
                status_code = e.status_code
                if _looks_like_overflow(e.message) or (
                        status_code is not None and 400 <= status_code < 500):
                    raise  # a genuine request error: retrying cannot help
                last_err = e
            except httpx.TransportError as e:
                if yielded:
                    raise VllmStreamError(
                        f"vLLM stream interrupted after partial output: {_fmt_exc(e)}",
                        partial=True) from e
                last_err = e
            finally:
                await agen.aclose()

            if attempt >= max_retries:
                detail = last_err.message if isinstance(last_err, VllmStreamError) else _fmt_exc(last_err)
                raise VllmStreamError(
                    f"vLLM request failed after {attempt + 1} attempt(s): {detail}",
                    partial=False, status_code=status_code) from last_err
            delay = _RETRY_BACKOFFS[min(attempt, len(_RETRY_BACKOFFS) - 1)]
            attempt += 1
            print(f"[LLM] vLLM request failed ({_fmt_exc(last_err)}); "
                  f"retry {attempt}/{max_retries} in {delay}s")
            await asyncio.sleep(delay)

    def _overflow_clamp(self, error_text: str, max_tokens: int, raw_estimate: int) -> Optional[int]:
        """max_tokens to retry with after a context-overflow 400, or None."""
        parsed = self._parse_context_overflow(error_text)
        if parsed is None:
            return None
        limit, prompt_tokens = parsed
        if limit > 0:
            self._detected_window = limit
        self._update_prompt_scale(prompt_tokens, raw_estimate)
        margin = _as_int(getattr(config, "MAX_TOKENS_SAFETY_MARGIN", 2048))
        floor = _as_int(getattr(config, "MIN_OUTPUT_TOKENS", 8192))
        room = limit - prompt_tokens - margin
        if room < floor:
            return None  # the prompt itself is too big: compaction, not a smaller output
        clamp = min(int(max_tokens), room)
        print(f"[LLM] Context overflow (window {limit}, prompt {prompt_tokens}): "
              f"retrying with max_tokens={clamp}")
        return clamp

    # ------------------------------------------------------------------
    # One HTTP attempt
    # ------------------------------------------------------------------

    async def _chat_stream_once(
        self,
        prep: _PreparedRequest,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamEvent]:
        first_byte_timeout, idle_timeout = _stream_timeouts()
        parser = _StreamParser(tools, prep.sent_max_tokens, prep.expect_thinking)
        url = f"{self.host}/v1/chat/completions"

        async with self._get_semaphore():
            request = self._client.build_request(
                "POST", url, content=prep.body,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            try:
                resp = await asyncio.wait_for(
                    self._client.send(request, stream=True), timeout=first_byte_timeout)
            except asyncio.TimeoutError:
                raise VllmStreamError(
                    f"vLLM did not respond within {first_byte_timeout:.0f}s (first-byte timeout)",
                    partial=False) from None
            try:
                if resp.status_code >= 400:
                    # Read the body so the actual vLLM error reaches the log
                    # instead of an opaque "400 Bad Request" with no detail.
                    try:
                        body_bytes = await asyncio.wait_for(resp.aread(), timeout=60.0)
                    except Exception:
                        body_bytes = b""
                    body = body_bytes.decode("utf-8", errors="replace")[:2000]
                    raise httpx.HTTPStatusError(
                        f"vLLM returned {resp.status_code}: {body}",
                        request=resp.request,
                        response=resp,
                    )

                lines = resp.aiter_lines()
                got_first = False
                try:
                    while True:
                        timeout = idle_timeout if got_first else first_byte_timeout
                        try:
                            raw_line = await asyncio.wait_for(lines.__anext__(), timeout=timeout)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            if parser.saw_finish:
                                # The answer is complete (finish_reason seen);
                                # only the trailing usage chunk / [DONE] is
                                # missing — do not turn it into a truncation.
                                print(f"[LLM] No [DONE] {timeout:.0f}s after finish_reason="
                                      f"{parser.finish_reason!r}; treating stream as complete")
                                break
                            kind = "idle" if got_first else "first-byte"
                            raise VllmStreamError(
                                f"vLLM stream stalled: no data for {timeout:.0f}s ({kind} timeout)",
                                partial=parser.emitted > 0) from None
                        if not raw_line:
                            continue
                        got_first = True
                        for event in parser.handle_line(raw_line):
                            yield event
                        if parser.pending_error is not None:
                            raise parser.pending_error  # abort/error finish
                        if parser.saw_done:
                            break
                except httpx.TransportError as e:
                    if not parser.saw_finish:
                        if parser.emitted:
                            for event in parser.drain_on_error():
                                yield event
                        raise
                    # Connection dropped after the finish chunk: the output is
                    # complete, only usage/[DONE] were lost. Finalize normally.
                    print(f"[LLM] Connection dropped after finish_reason="
                          f"{parser.finish_reason!r} ({_fmt_exc(e)}); treating stream as complete")
                except VllmStreamError:
                    if parser.emitted:
                        for event in parser.drain_on_error():
                            yield event
                    raise

                if not parser.is_complete():
                    if parser.emitted:
                        for event in parser.drain_on_error():
                            yield event
                    raise VllmStreamError(
                        "vLLM stream ended without a finish_reason or [DONE] "
                        "(connection closed early)",
                        partial=parser.emitted > 0)

                for event in parser.finalize():
                    yield event
            finally:
                await resp.aclose()


def _status_of(exc: httpx.HTTPStatusError) -> Optional[int]:
    try:
        return int(exc.response.status_code)
    except Exception:
        return None


async def collect_llm_response(stream: AsyncIterator[StreamEvent]) -> LLMResponse:
    """Accumulate a chat_stream into an LLMResponse with the real finish_reason."""
    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[ToolCall] = []
    tool_finish: Optional[str] = None
    usage: Optional[UsageEvent] = None
    async for event in stream:
        if isinstance(event, TextEvent):
            text_parts.append(event.content)
        elif isinstance(event, ReasoningEvent):
            reasoning_parts.append(event.content)
        elif isinstance(event, ToolCallDeltaEvent):
            tool_calls.extend(event.tool_calls)
            tool_finish = event.finish_reason
        elif isinstance(event, UsageEvent):
            usage = event
    finish_reason = usage.finish_reason if usage is not None else (tool_finish or "stop")
    return LLMResponse(
        content="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls or None,
        finish_reason=finish_reason,
        reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
        prompt_tokens=usage.prompt_tokens if usage is not None else 0,
        completion_tokens=usage.completion_tokens if usage is not None else 0,
    )


# ============================================================================
# Global instance (wrapped by interceptor)
# ============================================================================

from backend.core.llm_interceptor import LLMInterceptor

_backend = VllmBackend()
llm_backend = LLMInterceptor(_backend)
