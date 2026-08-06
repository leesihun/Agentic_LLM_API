"""CompactionMixin: microcompaction, history limits, and auto-compact on context overflow."""
import asyncio
import json
import re
from typing import List, Dict, Any, AsyncIterator, Optional

import config
from backend.core.llm_backend import StreamEvent, TextEvent, ToolStatusEvent, UsageEvent


# Substrings (case-insensitive) that mark a tool result worth keeping in full
# through microcompaction — failure signals are the most useful context for
# the agent to recover from a spiral.
_FAILURE_SUBSTRINGS = (
    "error", "traceback", "exception", "failed", "permission denied",
    "no such file", "not found", "fatal:", "syntax error", "exit code",
    "exitcode", "killed", "timeout", "timed out", "denied", "refused",
)
_ACTIVE_GOAL_LINE_RE = re.compile(r"(?im)^\s*active\s*goal\s*:\s*(.+?)\s*$")


def _looks_like_failure(content: str) -> bool:
    """Cheap heuristic — preserve verbatim through microcompaction."""
    if not content:
        return False
    lower = content[:400].lower()
    return any(s in lower for s in _FAILURE_SUBSTRINGS)


class CompactionMixin:
    """Compresses old tool results and auto-compacts on vLLM context overflow."""

    # Real prompt-token count from the previous LLM call (vLLM usage). 0 = unknown.
    # Set by _stream_with_autocompact; AgentLoop.__init__ also initializes it.
    _last_prompt_tokens: int = 0

    def _should_proactively_compact(self) -> bool:
        """True when the last measured prompt-token count crossed the budget.

        Budget = MODEL_CONTEXT_WINDOW * AGENT_COMPACT_AT_TOKEN_FRACTION. Disabled
        (returns False) when the context window is unknown (<= 0) or no real
        token count has been observed yet.
        """
        window = int(getattr(config, "MODEL_CONTEXT_WINDOW", 0) or 0)
        last = int(getattr(self, "_last_prompt_tokens", 0) or 0)
        if window <= 0 or last <= 0:
            return False
        frac = float(getattr(config, "AGENT_COMPACT_AT_TOKEN_FRACTION", 0.75))
        return last >= window * frac

    def _compress_old_iterations(self, msgs: List[Dict[str, Any]], current_iteration: int) -> List[Dict[str, Any]]:
        """
        Return a copy of msgs with old-iteration tool results and assistant
        tool-call messages compressed to short summaries. The original msgs list
        is NEVER mutated so the vLLM KV-cache prefix stays byte-stable
        across iterations.

        Only messages before the warm-window boundary are compressed; the hot
        tail (recent `warm` iterations) is passed through unchanged.
        """
        warm = config.AGENT_COMPACTION_WARM_WINDOW
        # Even when no microcompaction is due, we still want to run the
        # overlay hook so reflection / turn-boundary / tail-goal reminders
        # land on the first iteration too. A shallow list copy is cheap and
        # keeps the original msgs list (and its KV-cache prefix) untouched.
        if current_iteration == 0 or len(self._iteration_boundaries) <= warm:
            result = list(msgs)
            overlay = getattr(self, "_apply_loop_overlays", None)
            if callable(overlay):
                try:
                    overlay(result, msgs, current_iteration)
                except Exception as e:
                    self._log(f"  [MICROCOMPACT] overlay hook failed: {e}")
            return result

        old_boundary = self._iteration_boundaries[-(warm + 1)]
        summary_cap = config.AGENT_OLD_TOOL_RESULT_SUMMARY_MAX_CHARS
        tool_result_min = 120
        assistant_min = 80

        compressed_count = 0
        result: List[Dict[str, Any]] = []

        for i, msg in enumerate(msgs):
            if i >= old_boundary:
                result.append(msg)
                continue

            role = msg.get("role")

            if role == "tool":
                content = msg.get("content", "")
                if len(content) > tool_result_min:
                    if _looks_like_failure(content):
                        # Keep failure signals verbatim — the agent needs the
                        # full error text to recover, not a 120-char prefix.
                        pass
                    else:
                        tool_name = msg.get("name", "tool")
                        summary = content[:summary_cap].replace('\n', ' ')
                        msg = {**msg, "content": f"[{tool_name}: {summary}...]"}
                        compressed_count += 1

            elif role == "assistant" and msg.get("tool_calls"):
                tc_list = msg["tool_calls"]
                raw_size = sum(
                    len(tc.get("function", {}).get("arguments", ""))
                    for tc in tc_list
                )
                if raw_size > assistant_min:
                    names = ", ".join(
                        tc.get("function", {}).get("name", "?") for tc in tc_list
                    )
                    msg = {
                        **msg,
                        "content": f"[called: {names}]",
                        "tool_calls": [
                            {
                                "id": tc.get("id", f"call_{j}"),
                                "type": "function",
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": "{}",
                                },
                            }
                            for j, tc in enumerate(tc_list)
                        ],
                    }
                    compressed_count += 1

            result.append(msg)

        if compressed_count:
            self._log(f"  [MICROCOMPACT] Compressed {compressed_count} old message(s) "
                      f"from iterations before {current_iteration + 1}")
            # Inject a system-reminder into the VIEW so the model knows context was compressed.
            # Insert at the boundary index so it appears right before the uncompressed tail.
            # CRITICAL: mutate `result` (the copy), never `msgs` — mutating msgs would
            # invalidate the vLLM KV-cache prefix for all future iterations.
            sid = getattr(self, "session_id", None) or "session"
            reminder = {
                "role": "system",
                "content": (
                    f"<system-reminder>{compressed_count} earlier tool result(s) were compressed "
                    f"to save context. If you need the full content of a previous result, find the "
                    f"disk path in the truncation marker (data/tool_results/{sid}/<call_id>.json) "
                    f"and retrieve it with file_reader.</system-reminder>"
                ),
            }
            # Insert just before the first uncompressed message (old_boundary index in result)
            insert_at = min(old_boundary, len(result))
            result.insert(insert_at, reminder)

        # Hook: let the concrete agent layer apply additional view-only
        # overlays (turn-boundary marker, tail-goal reminder, stuck reminders).
        # Default implementation is a no-op; AgentLoop overrides it.
        overlay = getattr(self, "_apply_loop_overlays", None)
        if callable(overlay):
            try:
                overlay(result, msgs, current_iteration)
            except Exception as e:
                self._log(f"  [MICROCOMPACT] overlay hook failed: {e}")

        return result

    def _enforce_history_limit(self, msgs: List[Dict[str, Any]]):
        """Enforce the recent-history limit by dropping old messages.

        The limit is the more restrictive of MAX_CONVERSATION_HISTORY (message
        count) and MAX_CONVERSATION_TOKENS (estimated token budget, charged
        against the retained system prompt). When the conversation exceeds it:
        1. Keep system messages at the front (indices 0, 1, ...)
        2. Drop oldest non-system messages, replacing with a compaction notice
        3. Compress tool results in remaining old messages

        Operates in-place on *msgs*.
        """
        from backend.utils.tokens import total_message_tokens, trim_to_token_budget

        limit = config.MAX_CONVERSATION_HISTORY

        # Find where system messages end
        system_end = 0
        for i, msg in enumerate(msgs):
            if msg.get("role") == "system":
                system_end = i + 1
            else:
                break

        non_system = msgs[system_end:]

        # Token budget for non-system messages = total budget minus the system
        # prompt we always keep. Kept newest-first within both count and tokens.
        max_tokens = int(getattr(config, "MAX_CONVERSATION_TOKENS", 200_000))
        system_tokens = total_message_tokens(msgs[:system_end])
        token_budget = max(1000, max_tokens - system_tokens)
        kept = trim_to_token_budget(non_system, token_budget, max_messages=limit)

        excess = len(non_system) - len(kept)

        if excess > 0:
            dropped = non_system[:excess]
            # Build compaction notice
            dropped_roles = {}
            for m in dropped:
                r = m.get("role", "unknown")
                dropped_roles[r] = dropped_roles.get(r, 0) + 1
            summary_parts = [f"{v} {k}" for k, v in dropped_roles.items()]
            notice = f"[Compacted {len(dropped)} earlier messages: {', '.join(summary_parts)}]"
            # Rebuild msgs in-place
            msgs[system_end:] = [{"role": "system", "content": notice}] + kept
            print(f"[AGENT] History compacted: dropped {len(dropped)} old messages "
                  f"({len(msgs)} remaining, limit {limit})")

        # Additionally compress old messages in the kept portion
        # (compress everything except the last limit//2 messages)
        compress_boundary = len(msgs) - limit // 2
        masked = 0
        for i in range(system_end, compress_boundary):
            msg = msgs[i]
            role = msg.get("role")

            if role == "tool":
                content = msg.get("content", "")
                if len(content) > 80:
                    tool_name = msg.get("name", "tool")
                    summary = content[:40].replace("\n", " ")
                    msg["content"] = f"[{tool_name}: {summary}...]"
                    masked += 1

            elif role == "assistant" and msg.get("tool_calls"):
                tc_list = msg["tool_calls"]
                raw_size = sum(
                    len(tc.get("function", {}).get("arguments", ""))
                    for tc in tc_list
                )
                if raw_size > 60:
                    names = ", ".join(
                        tc.get("function", {}).get("name", "?") for tc in tc_list
                    )
                    msg["content"] = f"[called: {names}]"
                    msg["tool_calls"] = [
                        {
                            "id": tc.get("id", f"call_{j}"),
                            "type": "function",
                            "function": {"name": tc.get("function", {}).get("name", ""), "arguments": "{}"},
                        }
                        for j, tc in enumerate(tc_list)
                    ]
                    masked += 1

        if masked:
            print(f"[AGENT] Compressed {masked} old message(s)")

    @staticmethod
    def _is_context_overflow_error(exc: BaseException) -> bool:
        """True if exc is a vLLM HTTP error caused by context-window overflow."""
        import httpx
        if not isinstance(exc, httpx.HTTPStatusError):
            return False
        msg = str(exc).lower()
        # vLLM phrases overflow as: "This model's maximum context length is N
        # tokens. However, you requested M ... Please reduce the length of the
        # messages." Keep generic needles too so detection stays backend-agnostic.
        needles = (
            "context", "exceed", "too large", "too long",
            "maximum context length", "maximum model length",
            "please reduce the length", "longer than the maximum",
            "reduce the length of the messages",
        )
        return any(n in msg for n in needles)

    async def _summarize_and_compact_msgs(self, msgs: List[Dict[str, Any]]) -> bool:
        """Summarize the older half of msgs into one system message. Mutates msgs in place.
        Returns True if compaction happened, False if there isn't enough to compact."""
        system_end = 0
        for i, m in enumerate(msgs):
            if m.get("role") == "system":
                system_end = i + 1
            else:
                break

        keep_recent = max(0, getattr(config, "AGENT_AUTOCOMPACT_KEEP_RECENT", 4))
        middle_start = system_end
        middle_end = max(system_end, len(msgs) - keep_recent)
        middle_count = middle_end - middle_start
        if middle_count < 4:
            return False

        half = middle_count // 2
        to_compact = msgs[middle_start:middle_start + half]

        per_msg_cap = max(200, getattr(config, "AGENT_AUTOCOMPACT_PER_MSG_CHARS", 1500))
        summary_max_tokens = max(100, getattr(config, "AGENT_AUTOCOMPACT_SUMMARY_MAX_TOKENS", 1500))

        lines = []
        for i, m in enumerate(to_compact):
            role = m.get("role", "unknown")
            content = m.get("content")
            if content is None:
                content = ""
            elif not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False, default=str)
                except Exception:
                    content = str(content)
            if len(content) > per_msg_cap:
                content = content[:per_msg_cap] + f"...[+{len(content) - per_msg_cap} chars]"
            tc_summary = ""
            if m.get("tool_calls"):
                names = ", ".join(
                    tc.get("function", {}).get("name", "?") for tc in m["tool_calls"]
                )
                tc_summary = f" [tool_calls: {names}]"
            name_label = f" ({m['name']})" if m.get("name") else ""
            lines.append(f"[{i}] {role.upper()}{name_label}{tc_summary}\n{content}")
        convo_text = "\n\n".join(lines)

        summary_prompt = config.read_prompt("agent/autocompact_summary.txt").format(
            convo_text=convo_text,
        )

        total_chars = sum(len(str(m.get("content") or "")) for m in to_compact)
        self._log(f"  [AUTOCOMPACT] Summarizing {len(to_compact)} messages ({total_chars} chars)")

        summary_text = ""
        try:
            resp = await self.llm.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                model=self.model,
                temperature=0.0,
                session_id=self.session_id,
                agent_type="agent:autocompact",
                max_tokens=summary_max_tokens,
            )
            summary_text = (resp.content or "").strip()
        except Exception as e:
            self._log(f"  [AUTOCOMPACT] Summarizer call failed ({e}); dropping without summary")

        # Extract the trailing `Active goal: ...` line so the agent loop can
        # re-inject it as a tail reminder on subsequent iterations. Empty if
        # the summarizer didn't include one (older model output may not).
        active_goal: Optional[str] = None
        if summary_text:
            m = _ACTIVE_GOAL_LINE_RE.search(summary_text)
            if m:
                active_goal = m.group(1).strip()
        if active_goal and hasattr(self, "_carried_active_goal"):
            self._carried_active_goal = active_goal

        sid = getattr(self, "session_id", None) or "session"
        if summary_text:
            notice = (
                f"<system-reminder>Conversation auto-compacted. {len(to_compact)} older messages "
                f"summarized below. Full tool results are at data/tool_results/{sid}/. "
                f"Use file_reader on the disk path in a truncation marker "
                f"to retrieve any result you need. "
                f"Re-read AGENTS.md or critical files if the summary is missing details.</system-reminder>\n"
                f"[Earlier conversation summary]\n{summary_text}"
            )
        else:
            notice = (
                f"<system-reminder>Conversation auto-compacted: dropped {len(to_compact)} "
                f"older messages without summary. Full tool results at data/tool_results/{sid}/."
                f"</system-reminder>"
            )

        msgs[middle_start:middle_start + half] = [{"role": "system", "content": notice}]
        self._log(f"  [AUTOCOMPACT] Replaced {len(to_compact)} msgs with 1 system summary "
                  f"({len(notice)} chars). Total msgs now: {len(msgs)}")
        return True

    async def _stream_with_autocompact(
        self,
        msgs: List[Dict[str, Any]],
        model: str,
        temperature: float,
        *,
        iteration: int,
        use_compressed_view: bool = True,
        **chat_kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Wraps self.llm.chat_stream with reactive auto-compaction.

        On a vLLM context-overflow error, summarizes the older half of msgs
        (mutating in place) and retries from scratch. Safe because vLLM
        rejects context overflow BEFORE streaming any tokens — we re-raise
        immediately if any events were yielded prior to the error.
        """
        enabled = getattr(config, "AGENT_AUTOCOMPACT_ENABLED", True)
        max_retries = max(0, getattr(config, "AGENT_AUTOCOMPACT_MAX_RETRIES", 2))
        # Reset per-stream; the terminal UsageEvent (swallowed below) refreshes it
        # so the loop can read this call's finish_reason after the stream drains.
        self._last_finish_reason = "stop"

        # PROACTIVE compaction: if the previous call's REAL prompt-token count
        # (from vLLM usage) crossed the budget, summarize now — before sending —
        # so we avoid the wasted 400-then-resend round-trip on the hot path.
        if enabled and use_compressed_view and self._should_proactively_compact():
            self._log(
                f"  [AUTOCOMPACT] Proactive: last prompt was {self._last_prompt_tokens} tokens "
                f"(>= budget); compacting before send"
            )
            compacted = await self._summarize_and_compact_msgs(msgs)
            if compacted:
                # Force a fresh measurement next call; don't re-trigger on stale count.
                self._last_prompt_tokens = 0
                yield ToolStatusEvent(
                    tool_name="autocompact",
                    tool_call_id="autocompact",
                    status="started",
                    activity="Compacting context (proactive)",
                    user_name="System",
                )

        attempt = 0
        while True:
            events_yielded = False
            view = self._compress_old_iterations(msgs, iteration) if use_compressed_view else msgs
            try:
                async for event in self.llm.chat_stream(view, model, temperature, **chat_kwargs):
                    # Capture real token usage to steer the NEXT call's budget,
                    # then swallow it — downstream consumers don't handle it.
                    if isinstance(event, UsageEvent):
                        self._last_prompt_tokens = event.prompt_tokens
                        self._last_finish_reason = getattr(event, "finish_reason", "stop")
                        continue
                    events_yielded = True
                    yield event
                return
            except Exception as e:
                if not enabled or events_yielded or attempt >= max_retries:
                    raise
                if not self._is_context_overflow_error(e):
                    raise
                attempt += 1
                self._log(f"  [AUTOCOMPACT] Context-overflow error caught "
                          f"(attempt {attempt}/{max_retries}): {str(e)[:200]}")
                compacted = await self._summarize_and_compact_msgs(msgs)
                if not compacted:
                    self._log("  [AUTOCOMPACT] Nothing left to compact; propagating error")
                    raise
                if attempt == 1:
                    yield ToolStatusEvent(
                        tool_name="autocompact",
                        tool_call_id="autocompact",
                        status="started",
                        activity="Auto-compacting context",
                        user_name="System",
                    )
