"""
LLM Interceptor for logging all LLM interactions.
Wraps VllmBackend and logs requests/responses to prompts.log.
"""
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncIterator
from pathlib import Path
import uuid

import config
from backend.utils.prompts_log_append import append_capped_prompts_log


class LLMInterceptor:
    """Wraps the LLM backend, adding logging for every call."""

    def __init__(self, backend, log_path: Path = None):
        self.backend = backend
        self.log_path = log_path or config.PROMPTS_LOG_PATH
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
            messages = log_data.get("messages", [])
            role_counts: Dict[str, int] = {}
            for m in messages:
                r = m.get("role", "unknown")
                role_counts[r] = role_counts.get(r, 0) + 1
            role_summary = ", ".join(f"{v} {k}" for k, v in role_counts.items())
            lines.append(f"  Messages:    {len(messages)} ({role_summary})")
            if log_data.get("tools_provided"):
                lines.append(f"  Tools:       {log_data['tools_provided']} schema(s) provided")
            lines.append("")
            for i, m in enumerate(messages):
                role = m.get("role", "unknown").upper()
                content = m.get("content")
                tool_calls = m.get("tool_calls")
                tool_call_id = m.get("tool_call_id")
                lines.append(f"[{i}] {role}" + (f" (tool_call_id={tool_call_id})" if tool_call_id else ""))
                if isinstance(content, list):
                    for part in content:
                        ptype = part.get("type", "")
                        if ptype == "text":
                            lines.append(part.get("text", ""))
                        elif ptype == "tool_result":
                            lines.append(f"<tool_result id={part.get('tool_use_id', '')}>\n{part.get('content', '')}\n</tool_result>")
                        else:
                            lines.append(json.dumps(part, ensure_ascii=False))
                elif content is not None:
                    lines.append(str(content))
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        lines.append(f"  → tool_call: {fn.get('name', '?')}({fn.get('arguments', '')})")
                lines.append("")
        else:
            response_text = log_data.get("response", log_data.get("partial_response", ""))
            lines.append("")
            lines.append(str(response_text)[:2000])
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

        if not is_request:
            duration = log_data.get("duration_seconds", 0)
            lines.append(f"  Duration:    {duration:.2f}s")
            et = log_data.get("estimated_tokens", {})
            t_in = et.get("input", 0)
            t_out = et.get("output", 0)
            lines.append(f"  Tokens:      {t_in} in + {t_out} out = {t_in + t_out} total")
            if duration > 0 and t_out > 0:
                lines.append(f"  Speed:       {t_out / duration:.1f} tokens/sec")

            if log_data.get("success", False):
                lines.append(f"  Status:      SUCCESS")
            else:
                lines.append(f"  Status:      FAILED")
                if log_data.get("error"):
                    lines.append(f"  Error:       {log_data['error']}")

        lines.append("=" * 80)
        lines.append("")
        return '\n'.join(lines)

    def _log_interaction_sync(self, log_data: Dict):
        """Synchronous log write — called from a thread pool to avoid blocking."""
        try:
            if "id" not in log_data:
                log_data["id"] = str(uuid.uuid4())[:8]
            formatted = self._format_human_readable(log_data)
            append_capped_prompts_log(
                formatted if formatted.endswith("\n") else formatted + "\n",
                path=self.log_path,
            )
        except Exception as e:
            print(f"Warning: Failed to log LLM interaction: {e}")

    def _log_interaction(self, log_data: Dict):
        """Non-blocking log: offloads file I/O to a thread pool."""
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
    ):
        from backend.core.llm_backend import (
            LLMResponse, TextEvent, ToolCallDeltaEvent, ToolCall,
        )

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "stop"

        async for event in self.chat_stream(
            messages, model, temperature,
            tools=tools, session_id=session_id, agent_type=agent_type,
            top_p=top_p, top_k=top_k, min_p=min_p,
            max_tokens=max_tokens, repetition_penalty=repetition_penalty,
            guided_json=guided_json, response_format=response_format,
        ):
            if isinstance(event, TextEvent):
                content_parts.append(event.content)
            elif isinstance(event, ToolCallDeltaEvent):
                tool_calls.extend(event.tool_calls)
                finish_reason = event.finish_reason

        content = "".join(content_parts) if content_parts else None
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
        )

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
        backend_name = "VllmBackend"
        from backend.utils.tokens import total_message_tokens
        input_tokens = total_message_tokens(messages)

        self._log_interaction({
            "id": log_id, "timestamp": timestamp, "streaming": True,
            "model": model, "temperature": temperature,
            "messages": messages,  # kept for role-count summary only
            "backend": backend_name, "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
            "tools_provided": len(tools) if tools else 0,
            "response": "[STREAMING...]",
            "estimated_tokens": {"input": input_tokens, "output": 0, "total": input_tokens},
        })

        response_log: Dict[str, Any] = {
            "id": log_id, "timestamp": timestamp, "streaming": True,
            "model": model, "temperature": temperature,
            "messages": messages,  # kept for role-count summary only
            "backend": backend_name, "session_id": session_id or "N/A",
            "agent_type": agent_type or "N/A",
        }

        collected_text = ""
        collected_tool_calls = None
        real_usage = None

        # Import event types once per call, not inside the loop
        from backend.core.llm_backend import TextEvent, ToolCallDeltaEvent, UsageEvent

        try:
            async for event in self.backend.chat_stream(
                messages, model, temperature, tools=tools,
                top_p=top_p, top_k=top_k, min_p=min_p,
                max_tokens=max_tokens, repetition_penalty=repetition_penalty,
                guided_json=guided_json, response_format=response_format,
                chat_template_kwargs=chat_template_kwargs,
            ):
                if isinstance(event, TextEvent):
                    collected_text += event.content
                elif isinstance(event, ToolCallDeltaEvent):
                    collected_tool_calls = event.tool_calls
                elif isinstance(event, UsageEvent):
                    real_usage = event
                yield event

            duration = time.time() - start_time
            # Prefer vLLM's real token counts; fall back to estimates only when
            # the backend didn't report usage.
            if real_usage is not None:
                token_stats = {
                    "input": real_usage.prompt_tokens,
                    "output": real_usage.completion_tokens,
                    "total": real_usage.total_tokens or (real_usage.prompt_tokens + real_usage.completion_tokens),
                }
            else:
                output_tokens = int(len(collected_text.split()) * 1.3)
                token_stats = {
                    "input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens,
                }
            response_log["response"] = collected_text
            response_log["duration_seconds"] = duration
            response_log["success"] = True
            response_log["estimated_tokens"] = token_stats

            if collected_tool_calls:
                tc_summary = [
                    {"name": tc.function.name, "args": tc.function.arguments}
                    for tc in collected_tool_calls
                ]
                response_log["response_tool_calls"] = json.dumps(
                    tc_summary, ensure_ascii=False
                )[:3000]

        except Exception as e:
            response_log["success"] = False
            response_log["error"] = self._format_exception(e)
            response_log["duration_seconds"] = time.time() - start_time
            response_log["partial_response"] = collected_text
            response_log["estimated_tokens"] = {
                "input": input_tokens, "output": int(len(collected_text.split()) * 1.3),
                "total": input_tokens + int(len(collected_text.split()) * 1.3),
            }
            raise
        finally:
            self._log_interaction(response_log)

    # ------------------------------------------------------------------
    # Pass-through
    # ------------------------------------------------------------------

    async def list_models(self) -> List[str]:
        return await self.backend.list_models()

    async def get_context_window(self, model: str) -> int:
        return await self.backend.get_context_window(model)

    async def is_available(self) -> bool:
        return await self.backend.is_available()
