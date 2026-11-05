"""
LLM Interceptor for logging all LLM interactions.
Wraps VllmBackend and logs requests/responses to prompts.log.

With a 1M-token context the full prompt must never be dumped per call: each
request record carries metadata (role counts, tool count, cheap size estimate)
plus the LAST message only, truncated head+tail. File I/O runs in a worker
thread. The log is size-rotated (PROMPTS_LOG_MAX_BYTES, PROMPTS_LOG_BACKUPS
backups) in addition to the line cap in prompts_log_append.
"""
import asyncio
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncIterator
from pathlib import Path
import uuid

import config
from backend.utils.prompts_log_append import append_capped_prompts_log


# Max characters of the last message kept in a request record (head + tail).
_LAST_MESSAGE_LOG_CHARS = 8000
# Max characters of the response text kept in a response record.
_RESPONSE_LOG_CHARS = 2000


def _head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return (f"{text[:half]}\n...[{len(text) - 2 * half} chars omitted]...\n"
            f"{text[-half:]}")


def _content_len(content: Any) -> int:
    """Character size of a message's content without serializing it."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("image_url", "image", "input_image"):
                    total += 4500  # ~1500 tokens per image, base64 not counted
                    continue
                for key in ("text", "content"):
                    val = part.get(key)
                    if isinstance(val, str):
                        total += len(val)
            elif isinstance(part, str):
                total += len(part)
        return total
    return len(str(content))


def _estimate_input_tokens(messages: List[Dict[str, Any]]) -> int:
    """Cheap O(#messages) estimate (~3 chars/token) — no tokenizer on the event loop."""
    chars = 0
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        chars += _content_len(m.get("content"))
        for tc in m.get("tool_calls") or ():
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                args = fn.get("arguments") if isinstance(fn, dict) else None
                chars += len(args) if isinstance(args, str) else 0
    return chars // 3


def _render_message(m: Dict[str, Any]) -> str:
    """Render one message for the log (content head+tail truncated)."""
    lines: List[str] = []
    content = m.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                lines.append(str(part))
                continue
            ptype = part.get("type", "")
            if ptype == "text":
                lines.append(str(part.get("text", "")))
            elif ptype == "tool_result":
                lines.append(f"<tool_result id={part.get('tool_use_id', '')}>\n"
                             f"{part.get('content', '')}\n</tool_result>")
            elif ptype in ("image_url", "image", "input_image"):
                lines.append(f"[{ptype} part omitted]")
            else:
                lines.append(json.dumps(part, ensure_ascii=False, default=str)[:500])
    elif content is not None:
        lines.append(str(content))
    for tc in m.get("tool_calls") or ():
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            lines.append(f"  → tool_call: {fn.get('name', '?')}({str(fn.get('arguments', ''))[:500]})")
    return _head_tail("\n".join(lines), _LAST_MESSAGE_LOG_CHARS)


def _snapshot_request(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Small, immutable view of the request (the live list keeps mutating)."""
    role_counts: Dict[str, int] = {}
    for m in messages or ():
        r = m.get("role", "unknown") if isinstance(m, dict) else "unknown"
        role_counts[r] = role_counts.get(r, 0) + 1
    last = messages[-1] if messages else None
    snap: Dict[str, Any] = {"count": len(messages or ()), "role_counts": role_counts}
    if isinstance(last, dict):
        snap["last_index"] = len(messages) - 1
        snap["last_role"] = str(last.get("role", "unknown"))
        snap["last_tool_call_id"] = last.get("tool_call_id")
        snap["last_text"] = _render_message(last)
    return snap


class LLMInterceptor:
    """Wraps the LLM backend, adding logging for every call."""

    def __init__(self, backend, log_path: Path = None):
        self.backend = backend
        self.log_path = Path(log_path or config.PROMPTS_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _format_exception(self, exc: Exception) -> str:
        """Return a useful error string even for timeout exceptions with empty str()."""
        message = str(exc).strip()
        if message:
            return f"{type(exc).__name__}: {message}"
        return f"{type(exc).__name__}: {repr(exc)}"

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _format_human_readable(self, log_data: Dict) -> str:
        lines = []

        response = log_data.get("response", "")
        is_request = response == "[STREAMING...]"

        lines.append("")
        lines.append("=" * 80)
        lines.append(">>> REQUEST TO LLM" if is_request else "<<< RESPONSE FROM LLM")
        lines.append("=" * 80)

        if is_request:
            snap = log_data.get("request") or {}
            role_counts = snap.get("role_counts") or {}
            role_summary = ", ".join(f"{v} {k}" for k, v in role_counts.items())
            lines.append(f"  Messages:    {snap.get('count', 0)} ({role_summary})")
            if log_data.get("tools_provided"):
                lines.append(f"  Tools:       {log_data['tools_provided']} schema(s) provided")
            if log_data.get("max_tokens") is not None:
                lines.append(f"  Max tokens:  {log_data['max_tokens']}")
            if log_data.get("chat_template_kwargs"):
                lines.append(f"  Template:    {log_data['chat_template_kwargs']}")
            lines.append("")
            if "last_text" in snap:
                tcid = snap.get("last_tool_call_id")
                lines.append(
                    f"[{snap.get('last_index', 0)}] {snap.get('last_role', 'unknown').upper()} (last message)"
                    + (f" (tool_call_id={tcid})" if tcid else "")
                )
                lines.append(snap.get("last_text", ""))
                lines.append("")
        else:
            response_text = log_data.get("response", log_data.get("partial_response", ""))
            lines.append("")
            lines.append(_head_tail(str(response_text), _RESPONSE_LOG_CHARS))
            if log_data.get("response_tool_calls"):
                lines.append(f"\n  tool_calls: {log_data['response_tool_calls']}")
            lines.append("")

        lines.append("-" * 80)
        lines.append("STATS:")
        lines.append(f"  Timestamp:   {log_data.get('timestamp', 'N/A')}")
        lines.append(f"  Model:       {log_data.get('model', 'N/A')}")
        lines.append(f"  Backend:     {log_data.get('backend', 'N/A')}")
        lines.append(f"  Temperature: {log_data.get('temperature', 'N/A')}")

        sid = log_data.get('session_id')
        if sid and sid != 'N/A':
            lines.append(f"  Session:     {sid}")
        at = log_data.get('agent_type')
        if at and at != 'N/A':
            lines.append(f"  Agent:       {at}")

        lines.append(f"  Streaming:   {'Yes' if log_data.get('streaming', False) else 'No'}")

        if is_request:
            et = log_data.get("estimated_tokens", {})
            lines.append(f"  Est. input:  ~{et.get('input', 0)} tokens")
        else:
            duration = log_data.get("duration_seconds", 0)
            lines.append(f"  Duration:    {duration:.2f}s")
            et = log_data.get("estimated_tokens", {})
            t_in = et.get("input", 0)
            t_out = et.get("output", 0)
            lines.append(f"  Tokens:      {t_in} in + {t_out} out = {t_in + t_out} total")
            if duration > 0 and t_out > 0:
                lines.append(f"  Speed:       {t_out / duration:.1f} tokens/sec")
            if log_data.get("finish_reason"):
                lines.append(f"  Finish:      {log_data['finish_reason']}")
            if log_data.get("reasoning_chars"):
                lines.append(f"  Reasoning:   {log_data['reasoning_chars']} chars (not shown)")

            if log_data.get("success", False):
                if log_data.get("empty_text"):
                    lines.append(f"  Status:      SUCCESS (EMPTY_TEXT: no visible text and no tool calls)")
                else:
                    lines.append(f"  Status:      SUCCESS")
            else:
                lines.append(f"  Status:      FAILED")
                if log_data.get("error"):
                    lines.append(f"  Error:       {log_data['error']}")

        lines.append("=" * 80)
        lines.append("")
        return '\n'.join(lines)

    def _rotate_by_size(self) -> None:
        """Size-based rotation: prompts.log → .1 → .2 … (keep PROMPTS_LOG_BACKUPS).

        Runs under the same lock file as append_capped_prompts_log, but as a
        separate acquisition (never nested) so the two cannot deadlock."""
        try:
            max_bytes = int(getattr(config, "PROMPTS_LOG_MAX_BYTES", 50 * 1024 * 1024) or 0)
            backups = int(getattr(config, "PROMPTS_LOG_BACKUPS", 3) or 0)
        except (TypeError, ValueError):
            return
        if max_bytes <= 0:
            return
        path = self.log_path
        try:
            if path.stat().st_size < max_bytes:
                return
        except OSError:
            return
        try:
            from filelock import FileLock
            lock = FileLock(str(path.with_name(path.name + ".lock")), timeout=120)
            with lock:
                try:
                    if path.stat().st_size < max_bytes:
                        return  # another process rotated it meanwhile
                except OSError:
                    return
                if backups <= 0:
                    os.replace(str(path), str(path) + ".old")
                    try:
                        os.remove(str(path) + ".old")
                    except OSError:
                        pass
                    return
                oldest = Path(f"{path}.{backups}")
                if oldest.exists():
                    try:
                        oldest.unlink()
                    except OSError:
                        pass
                for i in range(backups - 1, 0, -1):
                    src = Path(f"{path}.{i}")
                    if src.exists():
                        try:
                            os.replace(str(src), f"{path}.{i + 1}")
                        except OSError:
                            pass
                os.replace(str(path), f"{path}.1")
        except Exception as e:
            print(f"Warning: prompts.log size rotation failed: {e}")

    def _log_interaction_sync(self, log_data: Dict):
        """Synchronous log write — called from a thread pool to avoid blocking."""
        try:
            if "id" not in log_data:
                log_data["id"] = str(uuid.uuid4())[:8]
            formatted = self._format_human_readable(log_data)
            self._rotate_by_size()
            append_capped_prompts_log(
                formatted if formatted.endswith("\n") else formatted + "\n",
                path=self.log_path,
            )
        except Exception as e:
            print(f"Warning: Failed to log LLM interaction: {e}")

    def _log_interaction(self, log_data: Dict):
        """Non-blocking log: offloads formatting + file I/O to a thread pool."""
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._log_interaction_sync, log_data)
        except RuntimeError:
            # No running loop (e.g. during shutdown) — fall back to sync
            self._log_interaction_sync(log_data)

    # ------------------------------------------------------------------
    # Non-streaming chat — thin accumulator over chat_stream().
    # Everything is streaming under the hood; this method just collects
    # the full response for callers that want a single LLMResponse.
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        session_id: str = None,
        agent_type: str = None,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        guided_json: dict = None,
        response_format: dict = None,
        chat_template_kwargs: dict = None,
    ):
        """Collect a full response. finish_reason is the real one from the
        terminal UsageEvent ("stop" | "length" | "tool_calls"/"tool")."""
        from backend.core.llm_backend import collect_llm_response

        return await collect_llm_response(self.chat_stream(
            messages, model, temperature,
            tools=tools, session_id=session_id, agent_type=agent_type,
            top_p=top_p, top_k=top_k, min_p=min_p,
            max_tokens=max_tokens, repetition_penalty=repetition_penalty,
            guided_json=guided_json, response_format=response_format,
            chat_template_kwargs=chat_template_kwargs,
        ))

    # ------------------------------------------------------------------
    # Streaming chat
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        tools: Optional[List[Dict[str, Any]]] = None,
        session_id: str = None,
        agent_type: str = None,
        top_p: float = None,
        top_k: int = None,
        min_p: float = None,
        max_tokens: int = None,
        repetition_penalty: float = None,
        guided_json: dict = None,
        response_format: dict = None,
        chat_template_kwargs: dict = None,
    ) -> AsyncIterator:
        """Yields StreamEvent objects from backend.chat_stream()."""
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_id = str(uuid.uuid4())[:8]
        backend_name = type(self.backend).__name__
        input_tokens = _estimate_input_tokens(messages)

        self._log_interaction({
            "id": log_id, "timestamp": timestamp, "streaming": True,
            "model": model, "temperature": temperature,
            "request": _snapshot_request(messages),
            "backend": backend_name, "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
            "tools_provided": len(tools) if tools else 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": chat_template_kwargs,
            "response": "[STREAMING...]",
            "estimated_tokens": {"input": input_tokens, "output": 0, "total": input_tokens},
        })

        response_log: Dict[str, Any] = {
            "id": log_id, "timestamp": timestamp, "streaming": True,
            "model": model, "temperature": temperature,
            "backend": backend_name, "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
        }

        text_parts: List[str] = []
        reasoning_chars = 0
        collected_tool_calls: list = []
        real_usage = None

        # Import event types once per call, not inside the loop
        from backend.core.llm_backend import (
            TextEvent, ReasoningEvent, ToolCallDeltaEvent, UsageEvent,
        )

        def _out_estimate() -> int:
            return sum(len(p) for p in text_parts) // 3

        inner = self.backend.chat_stream(
            messages, model, temperature, tools=tools,
            top_p=top_p, top_k=top_k, min_p=min_p,
            max_tokens=max_tokens, repetition_penalty=repetition_penalty,
            guided_json=guided_json, response_format=response_format,
            chat_template_kwargs=chat_template_kwargs,
        )
        try:
            async for event in inner:
                if isinstance(event, TextEvent):
                    text_parts.append(event.content)
                elif isinstance(event, ReasoningEvent):
                    reasoning_chars += len(event.content or "")
                elif isinstance(event, ToolCallDeltaEvent):
                    collected_tool_calls.extend(event.tool_calls)
                elif isinstance(event, UsageEvent):
                    real_usage = event
                yield event

            duration = time.time() - start_time
            collected_text = "".join(text_parts)
            # Prefer vLLM's real token counts; fall back to estimates only when
            # the backend didn't report usage.
            if real_usage is not None and (real_usage.prompt_tokens or real_usage.completion_tokens):
                token_stats = {
                    "input": real_usage.prompt_tokens,
                    "output": real_usage.completion_tokens,
                    "total": real_usage.total_tokens or (real_usage.prompt_tokens + real_usage.completion_tokens),
                }
            else:
                output_tokens = _out_estimate()
                token_stats = {
                    "input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens,
                }
            response_log["response"] = collected_text
            response_log["duration_seconds"] = duration
            response_log["success"] = True
            response_log["estimated_tokens"] = token_stats
            response_log["finish_reason"] = real_usage.finish_reason if real_usage is not None else "unknown"
            response_log["reasoning_chars"] = reasoning_chars
            response_log["empty_text"] = not collected_text.strip() and not collected_tool_calls

            if collected_tool_calls:
                tc_summary = []
                for tc in collected_tool_calls:
                    item: Dict[str, Any] = {"name": tc.function.name, "args": tc.function.arguments}
                    if getattr(tc.function, "parse_error", None):
                        item["parse_error"] = tc.function.parse_error
                    tc_summary.append(item)
                response_log["response_tool_calls"] = json.dumps(
                    tc_summary, ensure_ascii=False, default=str
                )[:3000]

        except BaseException as e:
            collected_text = "".join(text_parts)
            response_log["success"] = False
            response_log["error"] = self._format_exception(e)
            response_log["duration_seconds"] = time.time() - start_time
            response_log["partial_response"] = collected_text
            response_log["reasoning_chars"] = reasoning_chars
            out = _out_estimate()
            response_log["estimated_tokens"] = {
                "input": input_tokens, "output": out, "total": input_tokens + out,
            }
            raise
        finally:
            self._log_interaction(response_log)
            # A consumer that stops early (stop request, cancellation, early
            # return) must release the vLLM connection + request-semaphore slot
            # now, not whenever the abandoned generator is garbage-collected.
            aclose = getattr(inner, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Pass-through
    # ------------------------------------------------------------------

    async def list_models(self) -> List[str]:
        return await self.backend.list_models()

    async def get_context_window(self, model: str) -> int:
        return await self.backend.get_context_window(model)

    async def is_available(self) -> bool:
        return await self.backend.is_available()

    def effective_context_window(self) -> int:
        fn = getattr(self.backend, "effective_context_window", None)
        return int(fn()) if callable(fn) else int(getattr(self.backend, "_learned_window", 0) or 0)

    @property
    def _learned_window(self) -> int:
        return int(getattr(self.backend, "_learned_window", 0) or 0)

    @property
    def _prompt_scale(self) -> float:
        return float(getattr(self.backend, "_prompt_scale", 1.0) or 1.0)
