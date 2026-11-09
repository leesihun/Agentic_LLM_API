"""CompactionMixin: microcompaction, history limits, and auto-compact on context overflow."""
import asyncio
import json
import re
from typing import List, Dict, Any, AsyncIterator, Optional

import config
from backend.core.llm_backend import (
    StreamEvent, TextEvent, ToolStatusEvent, UsageEvent, ToolCallDeltaEvent,
)
from backend.agent.result_formatting import _head_tail, _pageable_spill_text

try:  # C1 (partition A). Fallback keeps this module importable on its own.
    from backend.core.llm_backend import VllmStreamError
except ImportError:  # pragma: no cover
    class VllmStreamError(Exception):  # type: ignore[no-redef]
        def __init__(self, message: str, partial: bool = False, status_code: Optional[int] = None):
            super().__init__(message)
            self.message = message
            self.partial = partial
            self.status_code = status_code


# Substrings (case-insensitive) that mark a tool result worth keeping in full
# through microcompaction — failure signals are the most useful context for
# the agent to recover from a spiral.
_FAILURE_SUBSTRINGS = (
    "error", "traceback", "exception", "failed", "permission denied",
    "no such file", "not found", "fatal:", "syntax error", "exit code",
    "exitcode", "killed", "timeout", "timed out", "denied", "refused",
)
_ACTIVE_GOAL_LINE_RE = re.compile(r"(?im)^\s*active\s*goal\s*:\s*(.+?)\s*$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>(?:(?!</think>).)*\Z", re.DOTALL)

# Explicit context-overflow phrasings (vLLM / OpenAI-compatible servers). Only
# these count as overflow — a generic "error"/"exceed"/"context" substring used
# to turn transport errors and unrelated 4xx into destructive compactions.
_OVERFLOW_PHRASES = (
    "maximum context length",
    "context length",
    "too many tokens",
    "prompt is too long",
    "maximum model length",
    "reduce the length of the messages",
    "longer than the maximum model length",
)

# Bound on the transcript text sent to the summarizer. With a 1M window the
# "older half" can be enormous; summarizing 1.5M chars in one call is slow and
# can itself overflow, so per-message caps shrink to fit this total.
_SUMMARIZER_MAX_INPUT_CHARS = 600_000

# Proactive-compaction hysteresis: after a compaction, don't compact again until
# the prompt grows by this fraction of the effective window.
_COMPACT_HYSTERESIS_FRACTION = 0.10


def _looks_like_failure(content: str) -> bool:
    """Cheap heuristic — preserve verbatim through microcompaction."""
    if not content or not isinstance(content, str):
        return False
    lower = content[:400].lower()
    return any(s in lower for s in _FAILURE_SUBSTRINGS)


def _strip_think(text: str) -> str:
    """Remove closed <think> blocks and a trailing unclosed one."""
    if not text or "<think>" not in text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def _content_text(m: Dict[str, Any]) -> str:
    content = m.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return " ".join(parts)
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


class CompactionMixin:
    """Compresses old tool results and auto-compacts on vLLM context overflow."""

    # Real prompt-token count from the previous LLM call (vLLM usage). 0 = unknown.
    # Set by _stream_with_autocompact; AgentLoop.__init__ also initializes it.
    _last_prompt_tokens: int = 0
    # Prompt-token count measured on the first call after the last compaction
    # (hysteresis baseline). None = no compaction yet / not measured yet.
    _post_compact_tokens: Optional[int] = None
    _compact_baseline_pending: bool = False
    # Number of system messages the loop itself put at the front of msgs
    # (static prompt, goal preamble, dynamic context). Messages after that are
    # caller-supplied and may be truncated as a last resort on overflow.
    _own_system_count: int = 1

    # ------------------------------------------------------------------
    # Window / thresholds
    # ------------------------------------------------------------------

    def _effective_window(self) -> int:
        """MODEL_CONTEXT_WINDOW, tightened to the backend's learned vLLM window."""
        window = int(getattr(config, "MODEL_CONTEXT_WINDOW", 1_000_000) or 0)
        llm = getattr(self, "llm", None)
        backend = getattr(llm, "backend", llm)
        learned = 0
        for obj in (backend, llm):
            try:
                learned = int(getattr(obj, "_learned_window", 0) or 0)
            except (TypeError, ValueError):
                learned = 0
            if learned > 0:
                break
        if learned > 0:
            window = min(window, learned) if window > 0 else learned
        return window

    def _should_proactively_compact(self) -> bool:
        """True when the last measured prompt-token count crossed the budget.

        Budget = effective window * AGENT_COMPACT_AT_TOKEN_FRACTION. Disabled
        (returns False) when the context window is unknown (<= 0) or no real
        token count has been observed yet. Hysteresis: after a compaction the
        prompt must grow by >= 10% of the window past the first post-compaction
        measurement before compacting again (prevents compaction thrash when a
        compaction can't get below the threshold).
        """
        window = self._effective_window()
        last = int(getattr(self, "_last_prompt_tokens", 0) or 0)
        if window <= 0 or last <= 0:
            return False
        frac = float(getattr(config, "AGENT_COMPACT_AT_TOKEN_FRACTION", 0.75))
        if last < window * frac:
            return False
        baseline = getattr(self, "_post_compact_tokens", None)
        if baseline is not None and last < baseline + window * _COMPACT_HYSTERESIS_FRACTION:
            return False
        return True

    def _note_compaction(self) -> None:
        """Record that a compaction happened (for hysteresis + diagnostics)."""
        self._compact_baseline_pending = True
        self._post_compact_tokens = None
        self._last_prompt_tokens = 0

    def _record_usage(self, event: Any) -> None:
        """Consume a terminal UsageEvent (C3)."""
        prompt_tokens = int(getattr(event, "prompt_tokens", 0) or 0)
        if prompt_tokens > 0:
            self._last_prompt_tokens = prompt_tokens
            if getattr(self, "_compact_baseline_pending", False):
                self._post_compact_tokens = prompt_tokens
                self._compact_baseline_pending = False
        self._last_completion_tokens = int(getattr(event, "completion_tokens", 0) or 0)
        fr = getattr(event, "finish_reason", None)
        self._last_finish_reason = str(fr) if fr else None

    # ------------------------------------------------------------------
    # Microcompaction (view only)
    # ------------------------------------------------------------------

    def _compress_old_iterations(self, msgs: List[Dict[str, Any]], current_iteration: int) -> List[Dict[str, Any]]:
        """
        Return a copy of msgs with old-iteration tool results and assistant
        tool-call messages compressed to short summaries. The original msgs list
        is NEVER mutated so the vLLM KV-cache prefix stays byte-stable
        across iterations.

        Only messages before the warm-window boundary are compressed; the hot
        tail (recent `warm` iterations) is passed through unchanged.
        """
        warm = int(getattr(
            config, "AGENT_MICROCOMPACT_WARM_WINDOW",
            getattr(config, "AGENT_COMPACTION_WARM_WINDOW", 20),
        ))
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
        summary_cap = int(getattr(config, "AGENT_OLD_TOOL_RESULT_SUMMARY_MAX_CHARS", 4000))
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
                if isinstance(content, str) and len(content) > tool_result_min:
                    if _looks_like_failure(content):
                        # Keep failure signals verbatim — the agent needs the
                        # full error text to recover, not a 120-char prefix.
                        pass
                    elif len(content) > summary_cap:
                        tool_name = msg.get("name", "tool")
                        # Head+tail: a log's end (errors, final status) matters most.
                        summary = _head_tail(content, summary_cap).replace('\n', ' ')
                        spill = self._spill_for_microcompact(msg, content)
                        spill_note = f" (full result: {spill})" if spill else ""
                        msg = {**msg, "content": f"[{tool_name}: {summary}...]{spill_note}"}
                        compressed_count += 1

            elif role == "assistant":
                # Old reasoning is the biggest context hog at reasoning_effort=max
                # and is not needed once the step it planned is done.
                content = msg.get("content")
                if isinstance(content, str) and "<think>" in content:
                    stripped = _strip_think(content)
                    if stripped != content:
                        msg = {**msg, "content": stripped}
                        if not msg.get("tool_calls") and not stripped:
                            msg["content"] = "[reasoning omitted]"
                if msg.get("tool_calls"):
                    tc_list = msg["tool_calls"]
                    raw_size = sum(
                        len(str(tc.get("function", {}).get("arguments", "")))
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
            reminder = {
                "role": "system",
                "content": (
                    f"<system-reminder>{compressed_count} earlier tool result(s) were compressed "
                    f"to save context. A compressed result that shows a "
                    f"`(full result: <path>)` / full_result_path was saved in full under "
                    f"{self._tool_results_dir_abs()} — open that absolute path with "
                    f"file_reader. Results without such a path were not saved."
                    f"</system-reminder>"
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

    def _spill_for_microcompact(self, msg: Dict[str, Any], content: str) -> Optional[str]:
        """Absolute spill path for an old tool result about to be cut.

        Results that fit their per-tool budget were never spilled; save them
        now (once — the path is memoized in _spill_paths, keeping the view
        byte-stable across iterations) so the cut part stays recoverable."""
        call_id = msg.get("tool_call_id")
        spills = getattr(self, "_spill_paths", None) or {}
        if call_id and call_id in spills:
            return spills[call_id]
        save = getattr(self, "_save_tool_result_to_disk", None)
        if not call_id or not getattr(self, "session_id", None) or not callable(save):
            return None
        try:
            text = _pageable_spill_text(json.loads(content))
        except Exception:
            text = content
        try:
            return save(call_id, text, suffix=".txt")
        except Exception as e:
            self._log(f"  [MICROCOMPACT] spill of old tool result failed: {e}")
            return None

    def _tool_results_dir_abs(self) -> str:
        sid = getattr(self, "session_id", None) or "default"
        try:
            return str((config.TOOL_RESULTS_DIR / sid).resolve())
        except Exception:
            return f"data/tool_results/{sid}"

    # ------------------------------------------------------------------
    # History limit (in place, at run start)
    # ------------------------------------------------------------------

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

        limit = int(getattr(config, "MAX_CONVERSATION_HISTORY", 400))

        # Find where system messages end
        system_end = 0
        for i, msg in enumerate(msgs):
            if msg.get("role") == "system":
                system_end = i + 1
            else:
                break

        non_system = msgs[system_end:]

        # Token budget for non-system messages = total budget minus the system
        # prompt we always keep, but never less than half the budget: a huge
        # caller system prompt (e.g. a big memory file) must not starve history
        # down to nothing.
        max_tokens = int(getattr(config, "MAX_CONVERSATION_TOKENS", 300_000))
        system_tokens = total_message_tokens(msgs[:system_end])
        token_budget = max(1000, max_tokens // 2, max_tokens - system_tokens)
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
                if isinstance(content, str) and len(content) > 80:
                    tool_name = msg.get("name", "tool")
                    summary = content[:40].replace("\n", " ")
                    msg["content"] = f"[{tool_name}: {summary}...]"
                    masked += 1

            elif role == "assistant" and msg.get("tool_calls"):
                tc_list = msg["tool_calls"]
                raw_size = sum(
                    len(str(tc.get("function", {}).get("arguments", "")))
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

    # ------------------------------------------------------------------
    # Overflow detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_context_overflow_error(exc: BaseException) -> bool:
        """True only for a genuine context-window overflow.

        Requires an explicit context-length phrase in the error text, and — when
        a status code is known — a 400/413. A VllmStreamError counts only when
        its message is a context-length message and nothing was streamed yet.
        Transport errors, timeouts, 5xx and unrelated 4xx are NOT overflow
        (compacting on those destroyed history for no benefit).
        """
        import httpx
        status: Optional[int] = None
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                status = exc.response.status_code
            except Exception:
                status = None
        elif isinstance(exc, VllmStreamError):
            if getattr(exc, "partial", False):
                return False
            status = getattr(exc, "status_code", None)
        else:
            return False
        msg = (str(getattr(exc, "message", "") or "") + " " + str(exc)).lower()
        if not any(p in msg for p in _OVERFLOW_PHRASES):
            return False
        if status is not None and status not in (400, 413):
            return False
        return True

    @staticmethod
    def _is_transient_stream_error(exc: BaseException) -> bool:
        """Errors worth one loop-level retry when nothing was streamed yet."""
        import httpx
        if isinstance(exc, VllmStreamError):
            if getattr(exc, "partial", False):
                return False
            # The backend re-raises a 4xx / in-band request error immediately
            # because retrying cannot help; don't undo that decision here.
            status = getattr(exc, "status_code", None)
            return not (isinstance(status, int) and 400 <= status < 500)
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                return exc.response.status_code >= 500
            except Exception:
                return False
        return isinstance(exc, (httpx.TransportError, asyncio.TimeoutError, TimeoutError))

    # ------------------------------------------------------------------
    # Summarize-and-compact
    # ------------------------------------------------------------------

    def _extractive_summary(self, to_compact: List[Dict[str, Any]]) -> str:
        """Deterministic fallback summary (no LLM): the user requests verbatim
        (truncated), tools called with counts, and the last assistant text."""
        user_lines: List[str] = []
        tool_counts: Dict[str, int] = {}
        failures: List[str] = []
        last_assistant = ""
        for m in to_compact:
            role = m.get("role")
            if role == "user":
                text = _content_text(m).strip()
                # The loop's own supervisor nudge is not a user request.
                if text and not text.startswith("Supervisor check after"):
                    user_lines.append(f"- {text[:600]}{'…' if len(text) > 600 else ''}")
            elif role == "assistant":
                for tc in m.get("tool_calls") or []:
                    name = (tc.get("function") or {}).get("name", "?")
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                text = _strip_think(_content_text(m)).strip()
                if text and not text.startswith("[called:"):
                    last_assistant = text
            elif role == "tool":
                content = _content_text(m)
                if '"success": false' in content[:200].lower():
                    failures.append(f"- {m.get('name', 'tool')}: {content[:200]}")
        parts = ["(Extractive summary — the summarizer was unavailable.)"]
        if user_lines:
            parts.append("User requests in this segment:\n" + "\n".join(user_lines[-20:]))
        if tool_counts:
            parts.append("Tools called: " + ", ".join(
                f"{k}×{v}" for k, v in sorted(tool_counts.items(), key=lambda kv: -kv[1])
            ))
        if failures:
            parts.append("Failed tool calls (latest):\n" + "\n".join(failures[-5:]))
        if last_assistant:
            parts.append("Last assistant message in segment:\n" + last_assistant[:1500])
        return "\n\n".join(parts)

    async def _call_summarizer(self, prompt: str, max_tokens: int) -> str:
        """Run the summarizer with thinking disabled (C4). Returns '' on failure."""
        kwargs = dict(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.0,
            session_id=self.session_id,
            agent_type="agent:autocompact",
            max_tokens=max_tokens,
        )
        try:
            try:
                resp = await self.llm.chat(
                    chat_template_kwargs={"enable_thinking": False}, **kwargs,
                )
            except TypeError as te:
                # Older interceptor without the C4 kwarg — retry without it.
                if "chat_template_kwargs" not in str(te):
                    raise
                resp = await self.llm.chat(**kwargs)
            text = _strip_think((getattr(resp, "content", None) or "").strip())
            return text
        except asyncio.CancelledError:
            # A stopped/abandoned compaction (POST /sessions/{id}/compact
            # cancels on stop or client disconnect) must abort, not fall back
            # to the extractive summary and rewrite history anyway.
            raise
        except Exception as e:
            self._log(f"  [AUTOCOMPACT] Summarizer call failed ({type(e).__name__}: {e})")
            return ""

    async def _summarize_and_compact_msgs(self, msgs: List[Dict[str, Any]]) -> bool:
        """Summarize the older half of msgs into one system message. Mutates msgs in place.
        Returns True if compaction happened, False if there isn't enough to compact.
        Never drops history without a summary: when the summarizer fails a
        deterministic extractive summary is used instead.

        Cancellation-safe: msgs is only mutated in one synchronous step after
        the summarizer returns, so cancelling the awaiting task (stop / client
        disconnect) raises CancelledError and leaves msgs untouched."""
        system_end = 0
        for i, m in enumerate(msgs):
            if m.get("role") == "system":
                system_end = i + 1
            else:
                break

        keep_recent = max(0, int(getattr(config, "AGENT_AUTOCOMPACT_KEEP_RECENT", 16)))
        middle_start = system_end
        middle_end = max(system_end, len(msgs) - keep_recent)
        middle_count = middle_end - middle_start
        if middle_count < 4:
            # Long tool-heavy turns can have all their history inside keep_recent;
            # fall back to a smaller keep window before giving up.
            middle_end = max(system_end, len(msgs) - min(keep_recent, 4))
            middle_count = middle_end - middle_start
            if middle_count < 4:
                return False

        half = middle_count // 2
        # Don't split an assistant tool-call message from its tool results:
        # extend the cut over any trailing tool messages.
        cut = middle_start + half
        while cut < middle_end and msgs[cut].get("role") == "tool":
            cut += 1
        to_compact = msgs[middle_start:cut]

        per_msg_cap = max(200, int(getattr(config, "AGENT_AUTOCOMPACT_PER_MSG_CHARS", 4000)))
        if per_msg_cap * len(to_compact) > _SUMMARIZER_MAX_INPUT_CHARS:
            per_msg_cap = max(200, _SUMMARIZER_MAX_INPUT_CHARS // max(1, len(to_compact)))
        summary_max_tokens = max(256, int(getattr(config, "AGENT_AUTOCOMPACT_SUMMARY_MAX_TOKENS", 8000)))

        lines = []
        for i, m in enumerate(to_compact):
            role = m.get("role", "unknown")
            content = _content_text(m)
            if role == "assistant":
                content = _strip_think(content)
            if len(content) > per_msg_cap:
                content = content[:per_msg_cap] + f"...[+{len(content) - per_msg_cap} chars]"
            tc_summary = ""
            if m.get("tool_calls"):
                names = ", ".join(
                    (tc.get("function") or {}).get("name", "?") for tc in m["tool_calls"]
                )
                tc_summary = f" [tool_calls: {names}]"
            name_label = f" ({m['name']})" if m.get("name") else ""
            lines.append(f"[{i}] {role.upper()}{name_label}{tc_summary}\n{content}")
        convo_text = "\n\n".join(lines)

        try:
            template = config.read_prompt("agent/autocompact_summary.txt")
            summary_prompt = template.replace("{convo_text}", convo_text)
            if "{convo_text}" not in template:
                summary_prompt = template + "\n\n" + convo_text
        except Exception as e:
            self._log(f"  [AUTOCOMPACT] Could not load summary prompt ({e}); using inline prompt")
            summary_prompt = (
                "Summarize this conversation segment so an agent can continue the "
                "same task. Preserve the user goal, decisions, file paths, commands "
                "and their outcomes, and open questions. End with a line "
                "'Active goal: <goal>'.\n\n" + convo_text
            )

        total_chars = sum(len(_content_text(m)) for m in to_compact)
        self._log(f"  [AUTOCOMPACT] Summarizing {len(to_compact)} messages ({total_chars} chars)")

        summary_text = await self._call_summarizer(summary_prompt, summary_max_tokens)
        used_fallback = False
        if not summary_text:
            summary_text = self._extractive_summary(to_compact)
            used_fallback = True
            self._log("  [AUTOCOMPACT] Using deterministic extractive summary")

        # Extract the trailing `Active goal: ...` line so the agent loop can
        # re-inject it as a tail reminder on subsequent iterations. Empty if
        # the summarizer didn't include one (older model output may not).
        active_goal: Optional[str] = None
        if summary_text and not used_fallback:
            m = _ACTIVE_GOAL_LINE_RE.search(summary_text)
            if m:
                active_goal = m.group(1).strip()
        if active_goal and hasattr(self, "_carried_active_goal"):
            self._carried_active_goal = active_goal

        notice = (
            f"<system-reminder>Conversation auto-compacted. {len(to_compact)} older messages "
            f"summarized below. Oversized tool results were saved under "
            f"{self._tool_results_dir_abs()} — open the absolute path from a "
            f"full_result_path / truncation marker with file_reader if you need one. "
            f"Re-read AGENTS.md or critical files if the summary is missing details.</system-reminder>\n"
            f"[Earlier conversation summary]\n{summary_text}"
        )

        msgs[middle_start:cut] = [{"role": "system", "content": notice}]
        self._note_compaction()
        self._log(f"  [AUTOCOMPACT] Replaced {len(to_compact)} msgs with 1 system summary "
                  f"({len(notice)} chars, fallback={used_fallback}). Total msgs now: {len(msgs)}")
        return True

    def _truncate_oversize_system_message(self, msgs: List[Dict[str, Any]]) -> bool:
        """Last resort on overflow when nothing is left to summarize: shrink the
        largest caller-supplied system message (head + tail kept, with a note).
        The loop's own static system prompt (index 0) is never touched, so the
        prefix cache stays valid. Returns True if something was truncated."""
        own = max(1, int(getattr(self, "_own_system_count", 1) or 1))
        best_idx, best_len = -1, 0
        for i, m in enumerate(msgs):
            if i < own or m.get("role") != "system":
                continue
            content = m.get("content")
            if isinstance(content, str) and len(content) > best_len:
                best_idx, best_len = i, len(content)
        if best_idx < 0:
            # No caller system message — allow the loop's dynamic context (never index 0).
            for i, m in enumerate(msgs[1:own], start=1):
                content = m.get("content")
                if m.get("role") == "system" and isinstance(content, str) and len(content) > best_len:
                    best_idx, best_len = i, len(content)
        if best_idx < 0 or best_len < 8000:
            return False
        content = msgs[best_idx]["content"]
        target = best_len // 2
        head = int(target * 0.4)
        tail = target - head
        omitted = best_len - head - tail
        msgs[best_idx] = {
            **msgs[best_idx],
            "content": (
                content[:head]
                + f"\n\n[… {omitted} chars of this system message omitted to fit the "
                  f"context window …]\n\n"
                + content[-tail:]
            ),
        }
        self._log(f"  [AUTOCOMPACT] Last resort: truncated system message #{best_idx} "
                  f"{best_len} -> {len(msgs[best_idx]['content'])} chars")
        self._note_compaction()
        return True

    async def _compact_with_status(self, msgs: List[Dict[str, Any]], activity: str):
        """Run a compaction bracketed by ToolStatusEvents. Yields events; the
        outcome is stored on self._last_compaction_ok."""
        self._last_compaction_ok = False
        yield ToolStatusEvent(
            tool_name="autocompact",
            tool_call_id="autocompact",
            status="started",
            activity=activity,
            user_name="System",
        )
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        ok = False
        try:
            ok = await self._summarize_and_compact_msgs(msgs)
            if not ok:
                ok = self._truncate_oversize_system_message(msgs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._log(f"  [AUTOCOMPACT] Compaction failed: {type(e).__name__}: {e}")
            ok = False
        self._last_compaction_ok = ok
        if ok:
            self._compactions = int(getattr(self, "_compactions", 0) or 0) + 1
        yield ToolStatusEvent(
            tool_name="autocompact",
            tool_call_id="autocompact",
            status="completed" if ok else "failed",
            duration=round(loop.time() - t0, 2),
            activity=activity,
            user_name="System",
        )

    # ------------------------------------------------------------------
    # Streaming wrapper
    # ------------------------------------------------------------------

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

        On a genuine context-overflow error, summarizes the older half of msgs
        (mutating in place) and retries from scratch — only when nothing was
        streamed yet. A transient error before the first event (transport drop,
        5xx, VllmStreamError(partial=False) after the backend's own retries)
        gets ONE more attempt after 2s. Anything after streamed output is
        re-raised for the loop to handle (VllmStreamError(partial=True) is kept
        as a partial answer there).

        The terminal UsageEvent (C3) is consumed here: it sets
        self._last_finish_reason and — only when prompt_tokens > 0 — the
        prompt-token measurement that drives proactive compaction.
        """
        enabled = getattr(config, "AGENT_AUTOCOMPACT_ENABLED", True)
        max_retries = max(0, int(getattr(config, "AGENT_AUTOCOMPACT_MAX_RETRIES", 2)))
        # Reset per-stream; None means "no terminal UsageEvent seen".
        self._last_finish_reason = None

        # PROACTIVE compaction: if the previous call's REAL prompt-token count
        # (from vLLM usage) crossed the budget, summarize now — before sending —
        # so we avoid the wasted 400-then-resend round-trip on the hot path.
        if enabled and use_compressed_view and self._should_proactively_compact():
            self._log(
                f"  [AUTOCOMPACT] Proactive: last prompt was {self._last_prompt_tokens} tokens "
                f"(>= {getattr(config, 'AGENT_COMPACT_AT_TOKEN_FRACTION', 0.75)} of "
                f"{self._effective_window()}); compacting before send"
            )
            async for ev in self._compact_with_status(msgs, "Compacting context (proactive)"):
                yield ev
            if not getattr(self, "_last_compaction_ok", False):
                # Don't re-trigger on every call when nothing can be compacted.
                self._note_compaction()

        attempt = 0
        transient_retried = False
        while True:
            events_yielded = False
            self._last_finish_reason = None
            view = self._compress_old_iterations(msgs, iteration) if use_compressed_view else msgs
            try:
                async for event in self.llm.chat_stream(view, model, temperature, **chat_kwargs):
                    # Capture real token usage to steer the NEXT call's budget,
                    # then swallow it — downstream consumers don't handle it.
                    if isinstance(event, UsageEvent):
                        self._record_usage(event)
                        continue
                    if (
                        isinstance(event, ToolCallDeltaEvent)
                        and not event.is_partial
                        and self._last_finish_reason is None
                    ):
                        # Provisional; the terminal UsageEvent overrides it.
                        self._last_finish_reason = event.finish_reason or None
                    events_yielded = True
                    yield event
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if events_yielded:
                    raise
                if enabled and attempt < max_retries and self._is_context_overflow_error(e):
                    attempt += 1
                    self._log(f"  [AUTOCOMPACT] Context-overflow error caught "
                              f"(attempt {attempt}/{max_retries}): {str(e)[:200]}")
                    async for ev in self._compact_with_status(msgs, "Auto-compacting context"):
                        yield ev
                    if not getattr(self, "_last_compaction_ok", False):
                        self._log("  [AUTOCOMPACT] Nothing left to compact; propagating error")
                        raise
                    continue
                if (
                    not transient_retried
                    and self._is_transient_stream_error(e)
                    # An overflow that compaction could not (or may no longer)
                    # fix is deterministic; resending the same prompt can't help.
                    and not self._is_context_overflow_error(e)
                ):
                    transient_retried = True
                    self._log(f"  [AGENT] Transient LLM error before any output "
                              f"({type(e).__name__}: {str(e)[:200]}); retrying once in 2s")
                    await asyncio.sleep(2.0)
                    continue
                raise
