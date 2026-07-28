"""
LLM Backend for vLLM
Fully async, with native tool calling support via OpenAI-compatible API.
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, AsyncIterator

import httpx

import config


# ============================================================================
# Response Types
# ============================================================================

@dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, Any]

@dataclass
class ToolCall:
    id: str
    function: ToolCallFunction

@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"

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
    """Real token usage reported by vLLM at end-of-stream.

    Emitted once per call when the request sets stream_options.include_usage.
    The agent loop uses prompt_tokens to drive *proactive* compaction (compact
    before the next call would overflow) instead of waiting for a 400 error.
    Internal-only: consumers that don't care simply ignore it.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


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
# The splitter below peels inline think blocks back out of `content` so they can
# be surfaced as ReasoningEvent (kept in history, never shown) exactly as if the
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
    """Split a running content buffer into (text, reasoning, carry, in_think).

    `carry` is the unresolved tail (a possible partial tag) to prepend to the
    next chunk; nothing is ever dropped. When no `<think>` tag is present this
    is a passthrough (text == buf, carry == "") apart from holding back a
    trailing partial-tag prefix by one chunk."""
    text_parts: list[str] = []
    reason_parts: list[str] = []
    while buf:
        if not in_think:
            i_open = buf.find(_THINK_OPEN)
            i_close = buf.find(_THINK_CLOSE)
            # A dangling close (no matching open) means the open tag was injected
            # into the prompt and the model streamed reasoning first, then closed
            # it — treat everything up to the close as reasoning and drop the tag.
            if i_close != -1 and (i_open == -1 or i_close < i_open):
                reason_parts.append(buf[:i_close])
                buf = buf[i_close + len(_THINK_CLOSE):]
                continue
            if i_open == -1:
                # Hold back a trailing partial that could start EITHER tag (both
                # begin with "<"), so a split tag isn't emitted as literal text.
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


# ============================================================================
# vLLM Backend
# ============================================================================

class VllmBackend:
    """vLLM backend: fully async, OpenAI-compatible, native tool calling."""

    def __init__(self, host: str = None):
        self.host = (host or config.VLLM_HOST).rstrip("/")
        self._ssl_verify = self._resolve_ssl()
        # Persistent connection pool — reuses TCP connections across requests
        pool_size = getattr(config, 'VLLM_CONNECTION_POOL_SIZE', 20)
        self._client = httpx.AsyncClient(
            verify=self._ssl_verify,
            # read=None → long agentic turns and slow prefill are never cut off
            # mid-stream; connect stays finite so a dead vLLM fails fast.
            timeout=httpx.Timeout(
                connect=getattr(config, "STREAM_CONNECT_TIMEOUT", 10.0),
                read=getattr(config, "STREAM_READ_TIMEOUT", None),
                write=60.0,
                pool=60.0,
            ),
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size // 2,
            ),
        )

    def _resolve_ssl(self):
        from pathlib import Path
        cert_path = Path("C:/DigitalCity.crt")
        if cert_path.exists():
            return str(cert_path)
        return True

    async def _select_available_host(self, *, prefer_active: bool = False) -> bool:
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
        await self._select_available_host(prefer_active=True)
        resp = await self._client.get(
            f"{self.host}/v1/models",
            timeout=httpx.Timeout(5.0),
        )
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    async def get_context_window(self, model: str) -> int:
        """Return the served model's max context length (vLLM's `max_model_len`
        field on /v1/models), or 0 if unavailable/unreported.

        Used at startup to auto-fill MODEL_CONTEXT_WINDOW when the operator
        hasn't set it explicitly, so proactive compaction works out of the box.
        """
        try:
            await self._select_available_host(prefer_active=True)
            resp = await self._client.get(
                f"{self.host}/v1/models",
                timeout=httpx.Timeout(5.0),
            )
            resp.raise_for_status()
            entries = resp.json().get("data", [])
            for entry in entries:
                if entry.get("id") == model:
                    return int(entry.get("max_model_len") or 0)
            if entries:
                return int(entries[0].get("max_model_len") or 0)
        except Exception:
            pass
        return 0

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
    ) -> dict[str, Any]:
        """Assemble the request payload with all vLLM parameters.

        Always uses streaming — there is no non-streaming path in this backend.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            # Ask vLLM to emit a final usage chunk so we get REAL token counts
            # (prompt/completion) instead of char-count estimates. Drives
            # proactive context compaction in the agent loop.
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = True
        # Sampling parameters. vLLM's OpenAI-compatible endpoint accepts top_k,
        # min_p and repetition_penalty as top-level fields (cache_prompt/id_slot
        # were llama.cpp-only and are gone — vLLM does prefix caching server-side).
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if min_p is not None:
            payload["min_p"] = min_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        # Structured / guided decoding (vLLM): constrain output to a JSON schema
        # (guided_json) or an OpenAI-style response_format. Opt-in per request.
        if guided_json is not None:
            payload["guided_json"] = guided_json
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    # ------------------------------------------------------------------
    # Streaming chat (with optional tool calling)
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
    ) -> AsyncIterator[StreamEvent]:
        await self._select_available_host(prefer_active=True)
        payload = self._build_payload(
            messages, model, temperature,
            tools=tools, top_p=top_p, top_k=top_k, min_p=min_p,
            max_tokens=max_tokens, repetition_penalty=repetition_penalty,
            guided_json=guided_json, response_format=response_format,
        )

        # State for accumulating tool call deltas across SSE chunks
        pending_tool_calls: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        yielded_indices: set[int] = set()          # indices already dispatched mid-stream
        max_seen_idx: int = -1
        finish_reason = "stop"
        usage: Optional[dict] = None                # final token usage, if reported
        # State for peeling inline <think>...</think> out of the content stream
        # (see _split_inline_reasoning). think_buf carries a possible partial tag
        # across chunks; in_think tracks whether we're inside a think block.
        think_buf = ""
        in_think = False

        async with self._client.stream(
            "POST",
            f"{self.host}/v1/chat/completions",
            json=payload,
        ) as resp:
            if resp.status_code >= 400:
                # Read the body so the actual vLLM error reaches the log
                # instead of an opaque "400 Bad Request" with no detail.
                body_bytes = await resp.aread()
                body = body_bytes.decode("utf-8", errors="replace")[:2000]
                raise httpx.HTTPStatusError(
                    f"vLLM returned {resp.status_code}: {body}",
                    request=resp.request,
                    response=resp,
                )
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage arrives in its own trailing chunk (choices is empty).
                # Capture it before the empty-choices skip below.
                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                chunk_finish = choices[0].get("finish_reason")
                if chunk_finish:
                    finish_reason = chunk_finish

                # Reasoning content (MiniMax M2, Qwen3-Thinking, DeepSeek-R1).
                # Preserved in history but not surfaced to the user.
                if "reasoning_content" in delta and delta["reasoning_content"]:
                    yield ReasoningEvent(content=delta["reasoning_content"])

                # Text content — peel any inline <think>...</think> back out so
                # reasoning is surfaced as ReasoningEvent (kept in history, not
                # shown) and only the real answer reaches the user as TextEvent.
                if "content" in delta and delta["content"]:
                    think_buf += delta["content"]
                    text_out, reason_out, think_buf, in_think = _split_inline_reasoning(
                        think_buf, in_think
                    )
                    if reason_out:
                        yield ReasoningEvent(content=reason_out)
                    if text_out:
                        yield TextEvent(content=text_out)

                # Tool call deltas
                if "tool_calls" in delta:
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)

                        # A new index appearing means the PREVIOUS max index is complete.
                        # Yield it immediately so the agent can start executing it now.
                        if idx not in pending_tool_calls:
                            if max_seen_idx >= 0 and max_seen_idx not in yielded_indices:
                                prev = pending_tool_calls[max_seen_idx]
                                if prev["name"]:
                                    try:
                                        args = json.loads(prev["arguments_str"])
                                        yield ToolCallDeltaEvent(
                                            tool_calls=[ToolCall(
                                                id=prev["id"],
                                                function=ToolCallFunction(
                                                    name=prev["name"], arguments=args
                                                ),
                                            )],
                                            finish_reason="tool",
                                            is_partial=True,
                                        )
                                        yielded_indices.add(max_seen_idx)
                                    except json.JSONDecodeError:
                                        pass  # incomplete JSON — will be caught at stream end

                            pending_tool_calls[idx] = {
                                "id": tc_delta.get("id", f"call_{idx}"),
                                "name": "",
                                "arguments_str": "",
                            }

                        entry = pending_tool_calls[idx]
                        func_delta = tc_delta.get("function", {})
                        if "name" in func_delta:
                            entry["name"] += func_delta["name"]
                        if "arguments" in func_delta:
                            entry["arguments_str"] += func_delta["arguments"]

                        if idx > max_seen_idx:
                            max_seen_idx = idx

                        # JSON-complete early dispatch: once the accumulated
                        # argument string forms a complete JSON object, yield
                        # immediately rather than waiting for the next index
                        # or [DONE]. This closes the gap for the last (or only)
                        # tool call in a turn — previously it always waited for
                        # stream end regardless of when arguments finished.
                        if (
                            entry["name"]
                            and idx not in yielded_indices
                            and entry["arguments_str"].rstrip().endswith("}")
                        ):
                            try:
                                args = json.loads(entry["arguments_str"])
                                yield ToolCallDeltaEvent(
                                    tool_calls=[ToolCall(
                                        id=entry["id"],
                                        function=ToolCallFunction(
                                            name=entry["name"], arguments=args
                                        ),
                                    )],
                                    finish_reason="tool",
                                    is_partial=True,
                                )
                                yielded_indices.add(idx)
                            except json.JSONDecodeError:
                                pass  # still accumulating

        # Flush any held-back content tail (a partial tag that never completed,
        # or reasoning left open because the stream ended mid-think).
        if think_buf:
            if in_think:
                yield ReasoningEvent(content=think_buf)
            else:
                yield TextEvent(content=think_buf)

        # After the stream finishes, yield any tool calls not yet dispatched
        remaining: list[ToolCall] = []
        for idx in sorted(pending_tool_calls.keys()):
            if idx in yielded_indices:
                continue
            entry = pending_tool_calls[idx]
            try:
                args = json.loads(entry["arguments_str"])
            except json.JSONDecodeError:
                args = {"_raw": entry["arguments_str"]}
            remaining.append(ToolCall(
                id=entry["id"],
                function=ToolCallFunction(name=entry["name"], arguments=args),
            ))
        if remaining:
            yield ToolCallDeltaEvent(
                tool_calls=remaining, finish_reason=finish_reason, is_partial=False
            )

        # Emit real token usage last so the agent loop can size the next call.
        if usage:
            yield UsageEvent(
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                total_tokens=int(usage.get("total_tokens", 0) or 0),
            )


# ============================================================================
# Global instance (wrapped by interceptor)
# ============================================================================

from backend.core.llm_interceptor import LLMInterceptor

_backend = VllmBackend()
llm_backend = LLMInterceptor(_backend)
