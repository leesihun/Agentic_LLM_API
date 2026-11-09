"""AgentLoop: the main single-while-loop agent that composes all the mixins."""
import asyncio
import hashlib
import json as _json
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator, Deque, Tuple

import config
from backend.core.llm_backend import (
    StreamEvent, TextEvent, ReasoningEvent,
    ToolCallDeltaEvent, ToolCall, ToolCallFunction, ToolStatusEvent, llm_backend,
)
from backend.utils.stop_signal import check_stop, StopInferenceError
from tools.schemas import TOOL_METADATA

from backend.agent._cache import _CACHED_GOAL_MODE_PROMPT
from backend.agent import watchdog as _wd
from backend.agent.logging_helpers import LoggingMixin
from backend.agent.prompt_assembly import PromptMixin
from backend.agent.tool_dispatch import DispatchMixin, _format_exception
from backend.agent.result_formatting import FormattingMixin
from backend.agent.compaction import CompactionMixin, VllmStreamError


# ---- Loop nudges / notices -------------------------------------------------
# Nudges go into the in-run message list only (never persisted: chat.py stores
# only the streamed TextEvents). Notices are user-visible TextEvents.
_TRUNCATION_NUDGE = (
    "Your previous output was cut off by the token limit. Continue exactly "
    "where it stopped. Do not repeat anything."
)
_STREAM_ERROR_NUDGE = (
    "Your previous output was interrupted by a connection error. Continue "
    "exactly where it stopped. Do not repeat anything."
)
_TRUNCATED_TOOL_NUDGE = (
    "Your tool call was truncated. Write large content with file_writer in "
    "smaller chunks (append mode) or apply_patch."
)
_REISSUE_TOOL_NUDGE = (
    "Your previous tool call did not arrive (it was empty or could not be "
    "read). Re-issue the tool call with complete, valid JSON arguments."
)
_FINAL_ANSWER_NUDGE = (
    "Write the final answer to the user now, in the user's language, based on "
    "the work above."
)
_TOOLS_UNAVAILABLE_NUDGE = (
    "<system-reminder>Tools are unavailable now; answer in text. Do not call "
    "any tool — summarize what was done and what remains.</system-reminder>"
)
_LOOP_NUDGES = frozenset({
    _TRUNCATION_NUDGE,
    _STREAM_ERROR_NUDGE,
    _TRUNCATED_TOOL_NUDGE,
    _REISSUE_TOOL_NUDGE,
    _FINAL_ANSWER_NUDGE,
})
# Injected by the turn review (supervisor) at an iteration boundary. Matched by
# prefix in _is_loop_nudge so it is never mistaken for the user's request.
_SUPERVISOR_NUDGE_PREFIX = "Supervisor check after"
_LENGTH_NOTICE = "\n\n[출력이 길이 제한으로 잘렸습니다]"
_STREAM_ERROR_NOTICE = "\n\n[응답 생성 중 연결 오류로 출력이 중단되었습니다]"
_TOOL_DROPPED_NOTICE = "\n\n[반복 한도에 도달하여 추가 도구 호출은 실행하지 않았습니다]"

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_MAX_REASONING_TOOL_CALLS = 8
_FINAL_NOTES_MAX_CHARS = 6000
_MAX_REISSUE_NUDGES = 2
# Continuations of a recovered final answer (Step A/B) that was itself cut by
# its token cap or a mid-answer stream drop, before labelling it as cut.
_FINAL_ANSWER_MAX_CONTINUATIONS = 2


def _new_recovery_out() -> Dict[str, Any]:
    """Result holder for one _stream_recovery_once call."""
    return {"text": [], "reasoning": [], "tool_calls": [], "failed": False, "finish": None}


def _recovery_cut(out: Dict[str, Any]) -> Optional[str]:
    """"length" / "error" when a recovery stream that produced output was cut
    (token cap, or an error after partial output); None when it completed."""
    if out.get("failed") or out.get("finish") == "error":
        return "error"
    if out.get("finish") == "length":
        return "length"
    return None


def _strip_unclosed_think(text: str) -> str:
    """Drop a trailing `<think>` block that never closed (cut off mid-thought)."""
    if not text or "<think>" not in text:
        return text
    last_open = text.rfind("<think>")
    if text.find("</think>", last_open) == -1:
        return text[:last_open]
    return text


def _text_after_think(reasoning: str) -> str:
    """Answer text the model wrote after a `</think>` inside the reasoning
    channel (template/parser mismatch). Tool-call markup is removed."""
    if not reasoning or "</think>" not in reasoning:
        return ""
    tail = reasoning.rsplit("</think>", 1)[1]
    tail = _TOOL_CALL_BLOCK_RE.sub("", tail)
    if "<tool_call>" in tail:
        tail = tail.split("<tool_call>", 1)[0]
    return tail.strip()


def _compose_assistant_content(text_body: str, reasoning_body: str) -> str:
    """Assistant history content: reasoning kept inline as <think>…</think>."""
    if reasoning_body:
        if text_body:
            return f"<think>\n{reasoning_body}\n</think>\n\n{text_body}"
        return f"<think>\n{reasoning_body}\n</think>"
    return text_body


def _result_headline(result: Dict[str, Any], cap: int = 160) -> str:
    """One-line summary of a tool result for the deterministic fallback."""
    if not isinstance(result, dict):
        return ""
    if not result.get("success", True):
        raw = result.get("error") or result.get("stderr") or ""
    else:
        raw = ""
        for key in ("message", "summary", "result", "stdout", "output", "content", "status"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                raw = val
                break
    for line in str(raw).splitlines():
        line = line.strip()
        if line:
            return line[:cap] + ("…" if len(line) > cap else "")
    return ""


class AgentLoop(LoggingMixin, CompactionMixin, DispatchMixin, FormattingMixin, PromptMixin):
    """
    Single agent loop that uses llama.cpp native tool calling.

    The LLM receives tool schemas and returns structured tool_calls.
    The loop executes tools in-process (parallel when possible) and
    feeds results back until the LLM responds with plain text.

    Features:
    - Parallel read-only tool execution
    - Microcompaction (save large results to disk, compress old iterations)
    - Prompt caching (byte-stable prefix for llama.cpp KV reuse)
    - Tool status events (streaming visibility)
    """

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        session_id: str = None,
        username: str = None,
        tools: Optional[List[str]] = None,
        workspace_dir: Optional[str] = None,
        response_format: Optional[dict] = None,
        guided_json: Optional[dict] = None,
        mode: str = "normal",
        chat_template_kwargs: Optional[dict] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or config.VLLM_MODEL
        # Per-request output ceiling (OpenAI `max_tokens`). None = DEFAULT_MAX_TOKENS.
        # Only a ceiling: llm_backend still fits it to the context headroom.
        try:
            self.max_tokens: Optional[int] = int(max_tokens) if max_tokens is not None else None
        except (TypeError, ValueError):
            self.max_tokens = None
        if self.max_tokens is not None and self.max_tokens <= 0:
            self.max_tokens = None
        self.temperature = self._resolve_temperature(temperature)
        self.session_id = session_id
        self.username = username
        self.llm = llm_backend
        # Goal mode: an extensive, goal-focused run. Raises the iteration budget
        # and enforces a plan->execute->verify done-gate (see _run_stream_body).
        self.goal_mode = str(mode or "normal").lower() == "goal"
        self.max_iterations = (
            config.AGENT_GOAL_MAX_ITERATIONS if self.goal_mode
            else config.AGENT_MAX_ITERATIONS
        )
        # Done-gate: consecutive no-progress refusals to let the agent stop with
        # an incomplete todo list (capped by AGENT_GOAL_MAX_COMPLETION_BLOCKS).
        # Reset whenever more todos complete, so a steadily-progressing long task
        # is gated as long as it needs; only repeated stalled quits trip the cap.
        self._completion_blocks = 0
        self._last_gate_completed = 0
        # C5: an explicit empty list means "no tools at all" (disable_tools);
        # only None falls back to the configured default set.
        self.enabled_tools = list(tools) if tools is not None else list(config.AVAILABLE_TOOLS)
        # Per-request chat-template override (C5), e.g. {"enable_thinking": False}
        # or {"reasoning_effort": "high"}. Wins over AGENT_REASONING_EFFORT.
        self.chat_template_kwargs: Optional[dict] = (
            dict(chat_template_kwargs) if isinstance(chat_template_kwargs, dict) and chat_template_kwargs
            else None
        )
        # Structured/guided decoding (opt-in, forwarded to vLLM on each call).
        self.response_format = response_format
        self.guided_json = guided_json
        self.tool_calls_log: List[Dict[str, Any]] = []
        self._iteration_boundaries: List[int] = []
        self._available_rag_collections: Optional[List[str]] = None
        self._tool_cache: Dict[str, Any] = {}
        self._filtered_tool_schemas: Optional[List[Dict[str, Any]]] = None
        self._compressed_up_to: int = 0  # tracks how far old-iteration compression has processed
        # Real prompt-token count from the last LLM call (vLLM usage), used to
        # drive proactive compaction. 0 until the first call reports usage.
        self._last_prompt_tokens: int = 0
        # finish_reason of the last streamed LLM call (set by _stream_with_autocompact
        # from the terminal UsageEvent, C3). None = unknown / no terminal event.
        # "length" means the turn was truncated at the token ceiling.
        self._last_finish_reason: Optional[str] = None
        # CONSECUTIVE truncation auto-continues (bounded by
        # AGENT_MAX_TRUNCATION_CONTINUATIONS); reset after any non-truncated
        # iteration. The run total goes to diagnostics.
        self._truncation_continuations: int = 0
        self._total_truncation_continuations: int = 0
        # Run diagnostics (C7). Populated by run_stream's finally block.
        self._empty_retries: int = 0
        self._compactions: int = 0
        self._fallback_used: bool = False
        self._iterations_run: int = 0
        self._tool_calls_dispatched: int = 0
        self._reissue_nudges: int = 0
        # Compact per-run ledger of executed tools, used by the deterministic
        # final-answer fallback (never contains model reasoning).
        self._worklog: List[Dict[str, Any]] = []
        self._enabled_tool_names: set = set()
        self._recovered_tool_calls: List[ToolCall] = []
        self._final_answer_text: str = ""
        # "length" / "error" when the last _ensure_final_answer recovery was
        # itself cut off (after its own continuations); None when complete.
        self._final_answer_cut: Optional[str] = None
        self.final_finish_reason: Optional[str] = None
        self.diagnostics: Dict[str, Any] = self._build_diagnostics(None)
        # Per-run memoization of read-only/concurrency-safe tool results, keyed by
        # (tool_name, arg_hash). Serves duplicate reads instantly; cleared by any
        # mutating tool. See DispatchMixin.execute_tool.
        self._read_result_cache: Dict[str, Any] = {}
        # Cache log verbosity once (avoids getattr + .lower() on every log call)
        self._log_level: str = str(getattr(config, "AGENT_LOG_VERBOSITY", "summary")).lower()
        # Session-scoped todo list (injected into dynamic context each iteration)
        self._session_todos: List[Dict[str, Any]] = []

        # Per-session workspace dir. None preserves legacy behaviour (relative
        # paths resolve against scratch/uploads/CWD). When set, file_reader,
        # file_navigator, file_editor, file_writer, apply_patch,
        # and shell_exec all treat it as the project root.
        self.workspace_dir: Optional[Path] = self._validate_workspace(workspace_dir)

        # ----- Reflection / anti-spiral state -----
        # (tool_name, sha1(args_canonical_json)[:16]) tuples in arrival order.
        self._tool_call_history: Deque[Tuple[str, str]] = deque(maxlen=64)
        # Iteration index where the last repeat-call reminder was injected.
        self._last_repeat_reminder_iter: int = -10**6
        # Per-iteration tool-success ledger (1 = all-failed, 0 = at least one success).
        self._iteration_failures: Deque[int] = deque(maxlen=8)
        # Carry-forward goal extracted from autocompact summary (active goal line).
        self._carried_active_goal: Optional[str] = None
        # Iterations at which milestone goal reminders have been emitted (so we
        # don't double-inject when the loop reattempts after autocompact).
        self._emitted_goal_reminder_iters: set = set()
        # True after we inject the first-iteration plan nudge so it doesn't fire
        # repeatedly across autocompact retries.
        self._plan_nudge_emitted: bool = False

        # New files created by file_writer/apply_patch during this run that
        # weren't flagged persist=True. Swept at end of run_stream so the agent
        # doesn't litter the workspace with temp scripts/scratch files.
        self._tracked_new_files: set[str] = set()

        # AGENTS.md / CLAUDE.md files already injected this session. The
        # tool_dispatch walk-up uses this set to inject each subtree
        # instruction file at most once. Resolved paths as strings.
        self._agents_md_seen: set[str] = set()

        # Post-edit check failure carried from the previous iteration. Set by
        # _run_stream_body when any tool result has post_edit_check.status=="failed";
        # consumed (and cleared) by _apply_loop_overlays on the next iteration.
        self._pending_postcheck_reminder: Optional[str] = None

        # Tool tasks launched with create_task (mid-stream early starts and
        # concurrency-safe groups). Reaped — cancelled AND awaited — by
        # run_stream's finally, so a closed/cancelled run (subagent deadline,
        # client disconnect) never leaves tools or their processes running.
        self._inflight_tasks: set = set()
        # Subagent progress (child tool starts) queued by the agent dispatcher
        # and forwarded to this stream while a tool group is awaited.
        self._child_progress: Deque[ToolStatusEvent] = deque(maxlen=64)
        self._child_progress_signal: Optional[asyncio.Event] = None
        # Live-visibility / supervisor state (reset per run in _reset_run_state).
        self._tool_launch_times: Dict[str, float] = {}
        self._completion_emitted: set = set()  # call ids whose terminal event is out
        self._agent_child_sessions: Dict[str, str] = {}
        self._msgs_ref: Optional[List[Dict[str, Any]]] = None
        self._turn_started: float = time.monotonic()
        self._last_touch: float = 0.0
        self._last_review_at: Optional[float] = None
        self._last_review_iter: int = 0
        self._supervisor_reviews: int = 0
        self._watchdog_stops: int = 0

    # ------------------------------------------------------------------
    # Sampling parameters forwarded to vLLM
    # ------------------------------------------------------------------

    def _resolve_temperature(self, requested: Optional[float]) -> float:
        """Pick the temperature for this session.

        Caller-supplied value wins. Otherwise look up MODEL_TEMPERATURE_OVERRIDES
        for a substring match on the model name (e.g. 'minimax' -> 1.0 per
        MiniMax M2 official recommendation). Falls back to DEFAULT_TEMPERATURE.
        """
        if requested is not None:
            return requested
        overrides = getattr(config, "MODEL_TEMPERATURE_OVERRIDES", None) or {}
        if overrides:
            model_lower = (self.model or "").lower()
            for needle, temp in overrides.items():
                if needle and needle.lower() in model_lower:
                    return float(temp)
        return config.DEFAULT_TEMPERATURE

    # ------------------------------------------------------------------
    # Workspace validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_workspace(raw: Optional[str]) -> Optional[Path]:
        """Resolve and validate a workspace path supplied by the caller.

        Returns None for unset / invalid values rather than raising — an
        unknown workspace should degrade to legacy behaviour, not break the
        chat request. Validation errors are printed for the operator.
        """
        if not raw:
            default = getattr(config, "AGENT_DEFAULT_WORKSPACE", None)
            raw = default if default else None
        if not raw:
            return None
        try:
            p = Path(raw).expanduser().resolve()
        except Exception as e:
            print(f"[AGENT] Ignoring invalid workspace path '{raw}': {e}")
            return None
        if not p.exists() or not p.is_dir():
            print(f"[AGENT] Ignoring workspace '{p}' - directory does not exist")
            return None
        # Refuse to point the agent at its own install root by default — the
        # agent editing itself is almost always unintentional.
        try:
            api_root = config.APP_DIR.resolve()
            if p == api_root:
                print(f"[AGENT] Refusing workspace == llm-api install dir ({p})")
                return None
        except Exception:
            pass
        return p

    # ------------------------------------------------------------------
    # Reflection helpers (anti-spiral)
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_signature(name: str, arguments: Any) -> Tuple[str, str]:
        """Return (tool_name, short_arg_hash) — used for repeat-call detection."""
        try:
            if isinstance(arguments, str):
                canon = arguments
            else:
                canon = _json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)
        except Exception:
            canon = repr(arguments)
        h = hashlib.sha1(canon.encode("utf-8", errors="replace")).hexdigest()[:16]
        return name, h

    def _record_iteration_outcome(self, results: List[Dict[str, Any]]) -> None:
        """Append 1 to the failure ledger if every result in this iteration failed."""
        if not results:
            return
        any_success = any(bool(r.get("success", True)) for r in results)
        self._iteration_failures.append(0 if any_success else 1)

    def _detect_repeat_call(self, current_iteration: int) -> Optional[str]:
        """Return a reminder string when the most recent tool signature has
        appeared >= threshold times within the recent window. Otherwise None.

        Honors a cooldown so the same reminder isn't injected every iteration.
        """
        threshold = max(2, getattr(config, "AGENT_STUCK_REPEAT_THRESHOLD", 3))
        window = max(threshold, getattr(config, "AGENT_STUCK_REPEAT_WINDOW", 6))
        cooldown = max(0, getattr(config, "AGENT_STUCK_COOLDOWN_ITERATIONS", 4))
        if current_iteration - self._last_repeat_reminder_iter < cooldown:
            return None
        if len(self._tool_call_history) < threshold:
            return None
        recent = list(self._tool_call_history)[-window:]
        if not recent:
            return None
        latest_sig = recent[-1]
        count = sum(1 for sig in recent if sig == latest_sig)
        if count < threshold:
            return None
        self._last_repeat_reminder_iter = current_iteration
        tool_name = latest_sig[0]
        return (
            f"<system-reminder>The `{tool_name}` call has run {count} times with "
            f"similar arguments in the last {window} tool rounds and isn't "
            f"producing new information. Stop repeating it. Choose one: "
            f"(a) try a fundamentally different approach, (b) ask the user "
            f"for clarification, or (c) report what you've found and what's "
            f"blocking you.</system-reminder>"
        )

    def _detect_consecutive_failures(self) -> Optional[str]:
        """Return a reflection reminder when the last N iterations all failed."""
        threshold = max(2, getattr(config, "AGENT_CONSECUTIVE_FAILURE_THRESHOLD", 2))
        recent = list(self._iteration_failures)[-threshold:]
        if len(recent) < threshold or not all(recent):
            return None
        return (
            f"<system-reminder>The last {threshold} tool rounds all failed. "
            f"Step back: re-read the user's latest request, identify the "
            f"assumption that's wrong, and decide whether to (a) investigate "
            f"the failure (read the error), (b) try a different tool or path, "
            f"or (c) ask the user. Do not retry the same call.</system-reminder>"
        )

    def _milestone_goal_reminder(self, current_iteration: int,
                                 latest_user_text: Optional[str]) -> Optional[str]:
        """At configured milestones, re-inject the user's latest request to
        combat lost-in-the-middle on long iteration tails."""
        milestones = tuple(getattr(config, "AGENT_GOAL_REMINDER_ITERATIONS", ()))
        if not milestones or not latest_user_text:
            return None
        if current_iteration not in milestones:
            return None
        if current_iteration in self._emitted_goal_reminder_iters:
            return None
        self._emitted_goal_reminder_iters.add(current_iteration)
        cap = max(200, getattr(config, "AGENT_TAIL_GOAL_MAX_CHARS", 1500))
        text = latest_user_text.strip()
        if len(text) > cap:
            text = text[:cap] + "...[truncated]"
        return (
            f"<system-reminder>User's request (reminder at iteration "
            f"{current_iteration + 1}/{self.max_iterations}): {text}"
            f"</system-reminder>"
        )

    def _plan_nudge(self, current_iteration: int,
                    latest_user_text: Optional[str]) -> Optional[str]:
        """On the first iteration of a session, prompt a brief plan before any
        tool calls. Fires unconditionally — even trivial-looking requests
        benefit from a one-line plan, and the cost (one short reminder) is
        negligible compared to the tool-spam it prevents."""
        if self._plan_nudge_emitted or current_iteration != 0 or not latest_user_text:
            return None
        self._plan_nudge_emitted = True
        return (
            "<system-reminder>Before any tool calls, state a brief plan "
            "(1–5 bullets, scaled to the task). Read provided docs/files "
            "if you haven't. Avoid exploratory grep/search loops — converge "
            "fast on the specific files the task touches.</system-reminder>"
        )

    # ------------------------------------------------------------------
    # Goal mode (extensive, goal-focused run)
    # ------------------------------------------------------------------

    @staticmethod
    def _goal_mode_preamble() -> str:
        """The goal-mode instruction block, injected as its own system message."""
        return _CACHED_GOAL_MODE_PROMPT

    def _goal_gate_would_block(self) -> bool:
        """Side-effect-free prediction of `_goal_completion_block() is not None`
        for the current todo state (todos only change via tool calls, so the
        prediction holds for a no-tool-call iteration)."""
        todos = self._session_todos or []
        incomplete = [t for t in todos if t.get("status") != "completed"]
        completed = len(todos) - len(incomplete)
        blocks = 0 if completed > self._last_gate_completed else self._completion_blocks
        cap = max(0, getattr(config, "AGENT_GOAL_MAX_COMPLETION_BLOCKS", 3))
        if blocks >= cap:
            return False
        return not todos or bool(incomplete)

    def _goal_completion_block(self) -> Optional[str]:
        """Done-gate for goal mode. Called when the model tries to finish (no
        tool calls). Returns a `<system-reminder>` string to force the loop to
        continue, or None to allow completion.

        The agent's own todo list is the definition of done: the run may only
        finish once every todo is `completed`. A safety cap
        (AGENT_GOAL_MAX_COMPLETION_BLOCKS) guarantees we eventually let it stop
        even if it never converges, and max_iterations is the absolute backstop.
        """
        todos = self._session_todos or []
        incomplete = [t for t in todos if t.get("status") != "completed"]

        # Reset the stall counter whenever the model has completed more todos
        # since the last block — the cap only guards against repeated quits that
        # make no progress, not against a long task that keeps advancing.
        completed = len(todos) - len(incomplete)
        if completed > self._last_gate_completed:
            self._completion_blocks = 0
        self._last_gate_completed = completed

        cap = max(0, getattr(config, "AGENT_GOAL_MAX_COMPLETION_BLOCKS", 3))
        if self._completion_blocks >= cap:
            return None

        if not todos:
            return (
                "<system-reminder>Goal mode: you are trying to finish but have "
                "not produced a plan yet. Call `todo_write` with a concrete, "
                "verifiable checklist that fully covers the goal, then execute "
                "it. Do not stop until every item is completed.</system-reminder>"
            )
        if incomplete:
            open_items = "; ".join(
                f"{t.get('id', '?')}: {t.get('content', '')}" for t in incomplete[:10]
            )
            return (
                "<system-reminder>Goal mode: do NOT stop yet — "
                f"{len(incomplete)} task(s) are still open: {open_items}. "
                "Continue working through them (or update the list if the plan "
                "changed), verify each one, and mark it completed. Only finish "
                "once every task is completed.</system-reminder>"
            )
        return None

    # ------------------------------------------------------------------
    # Static helpers for view overlays
    # ------------------------------------------------------------------

    @staticmethod
    def _find_last_user_message(msgs: List[Dict[str, Any]]) -> Optional[Tuple[int, str]]:
        """Return (index, text) of the last role=user message, or None.

        Strips list-content (multimodal) blocks to plain text.
        """
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                text = " ".join(parts)
            else:
                text = str(content) if content is not None else ""
            if AgentLoop._is_loop_nudge(text):
                # Loop-injected nudges are not the user's request.
                continue
            return i, text
        return None

    @staticmethod
    def _is_loop_nudge(text: str) -> bool:
        """True for user-role messages the loop itself injected."""
        stripped = (text or "").strip()
        if not stripped:
            return False
        if stripped in _LOOP_NUDGES:
            return True
        return (
            stripped.endswith(_FINAL_ANSWER_NUDGE)
            or stripped.startswith("<internal_notes>")
            or stripped.startswith(_SUPERVISOR_NUDGE_PREFIX)
        )

    @staticmethod
    def _has_prior_assistant_turn(msgs: List[Dict[str, Any]], user_idx: int) -> bool:
        """True if there's any role=assistant message at index < user_idx."""
        for i in range(user_idx):
            if msgs[i].get("role") == "assistant":
                return True
        return False

    # ------------------------------------------------------------------
    # View overlay hook called by CompactionMixin._compress_old_iterations
    # ------------------------------------------------------------------

    def _apply_loop_overlays(
        self,
        view: List[Dict[str, Any]],
        msgs: List[Dict[str, Any]],
        current_iteration: int,
    ) -> None:
        """Insert view-only `<system-reminder>` blocks for:

        - D1: turn-boundary marker before a new user message that follows
          prior assistant turns,
        - B5: plan-first nudge on iteration 0 for long/multi-step requests,
        - B4: milestone goal re-injection at configured iterations,
        - B2: repeat-call detection reminder,
        - B3: consecutive-failure reflection reminder,
        - D3: carried "Active goal" from autocompact,
        - C1: tail-pinned user-goal echo.

        Mutates *view* in place. Never touches *msgs*.
        """
        last_user = self._find_last_user_message(view)
        latest_user_text = last_user[1] if last_user else None
        # Capture pre-overlay view length so the tail-goal min-turns gate uses
        # the real conversation size, not the size after we inserted markers.
        view_len_pre_overlay = len(view)

        # ---- D1: turn-boundary marker ----
        if getattr(config, "AGENT_TURN_BOUNDARY_MARKER_ENABLED", True) and last_user is not None:
            user_idx, _ = last_user
            if self._has_prior_assistant_turn(view, user_idx):
                view.insert(user_idx, {
                    "role": "system",
                    "content": (
                        "<system-reminder>New user request boundary. Earlier "
                        "turns in this session are completed work — treat them "
                        "as background context, not as ongoing constraints, "
                        "unless the user explicitly references them. The "
                        "user's *next* message (immediately after this "
                        "reminder) is the active request.</system-reminder>"
                    ),
                })

        # ---- Tail-injected reminders (appended to view) ----
        tail_blocks: List[str] = []

        # P1: post-edit syntax/type check failure — inject first, highest priority
        pending_pec = getattr(self, "_pending_postcheck_reminder", None)
        if pending_pec:
            tail_blocks.append(pending_pec)
            self._pending_postcheck_reminder = None

        # B5: plan nudge (iteration 0 only)
        plan = self._plan_nudge(current_iteration, latest_user_text)
        if plan:
            tail_blocks.append(plan)

        # B4: milestone goal re-injection
        milestone = self._milestone_goal_reminder(current_iteration, latest_user_text)
        if milestone:
            tail_blocks.append(milestone)

        # B2: repeat-call detection
        repeat = self._detect_repeat_call(current_iteration)
        if repeat:
            tail_blocks.append(repeat)

        # B3: consecutive-failure reflection
        failure = self._detect_consecutive_failures()
        if failure:
            tail_blocks.append(failure)

        # D3: carried active goal from autocompact summary
        if self._carried_active_goal:
            cap = max(200, getattr(config, "AGENT_TAIL_GOAL_MAX_CHARS", 1500))
            goal_text = self._carried_active_goal.strip()
            if len(goal_text) > cap:
                goal_text = goal_text[:cap] + "...[truncated]"
            tail_blocks.append(
                f"<system-reminder>Active goal carried forward from earlier "
                f"summary: {goal_text}</system-reminder>"
            )

        # C1: tail-pinned latest user goal (always last, wins recency)
        if (
            getattr(config, "AGENT_TAIL_GOAL_REMINDER_ENABLED", True)
            and latest_user_text
            and view_len_pre_overlay >= getattr(config, "AGENT_TAIL_GOAL_MIN_TURNS", 4)
        ):
            cap = max(200, getattr(config, "AGENT_TAIL_GOAL_MAX_CHARS", 1500))
            text = latest_user_text.strip()
            if len(text) > cap:
                text = text[:cap] + "...[truncated]"
            tail_blocks.append(
                f"<system-reminder>The user's latest request, repeated for "
                f"fidelity (highest priority after system instructions): "
                f"{text}</system-reminder>"
            )

        for block in tail_blocks:
            view.append({"role": "system", "content": block})

    def _sampling_kwargs(self, final_response: bool = False) -> Dict[str, Any]:
        """Return sampling params to forward to the LLM backend."""
        kwargs: Dict[str, Any] = {
            "top_p": config.DEFAULT_TOP_P,
            "top_k": config.DEFAULT_TOP_K,
            "min_p": config.DEFAULT_MIN_P,
            "repetition_penalty": config.DEFAULT_REPETITION_PENALTY,
        }
        # One output ceiling for tool-loop and final turns: the request's
        # max_tokens when given, else DEFAULT_MAX_TOKENS. llm_backend fits it
        # to the real context headroom on every call.
        kwargs["max_tokens"] = (
            getattr(self, "max_tokens", None)
            or int(getattr(config, "DEFAULT_MAX_TOKENS", 100000))
        )
        # Agent-loop reasoning depth. "high" is a big latency win over the global
        # "max" default across a multi-iteration tool loop; the per-call override
        # (see llm_backend._build_payload) wins over CHAT_TEMPLATE_KWARGS. None
        # falls back to the global default. A per-request override (C5) wins.
        effort = getattr(config, "AGENT_REASONING_EFFORT", None)
        if effort:
            kwargs["chat_template_kwargs"] = {"reasoning_effort": effort}
        if getattr(self, "chat_template_kwargs", None):
            kwargs["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        # Structured/guided decoding is opt-in per request. When the caller sets
        # it they want a constrained answer, so forward it on the agent's calls.
        if getattr(self, "response_format", None) is not None:
            kwargs["response_format"] = self.response_format
        if getattr(self, "guided_json", None) is not None:
            kwargs["guided_json"] = self.guided_json
        return kwargs

    def _can_start_tool_immediately(self, tool_name: str, deferred_seen: bool) -> bool:
        """Only start read-only concurrency-safe tools during model streaming."""
        if deferred_seen:
            return False
        meta = TOOL_METADATA.get(tool_name, {})
        return bool(meta.get("is_read_only") and meta.get("is_concurrency_safe"))

    # ------------------------------------------------------------------
    # Non-streaming run — thin accumulator over run_stream().
    # The entire loop is streaming under the hood; this method just
    # collects text events into a final string for callers that don't
    # need incremental output.
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: List[Dict[str, Any]],
        attached_files: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        parts: list[str] = []
        async for event in self.run_stream(messages, attached_files):
            if isinstance(event, TextEvent):
                parts.append(event.content)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Streaming run
    # ------------------------------------------------------------------

    def _cleanup_temp_files(self) -> None:
        """Delete files created during this run that weren't flagged persist=True.

        Called from run_stream's finally block — runs even on stop/exception.
        Empty parent dirs are NOT cleaned (would surprise the user if they
        wrote into an existing project dir).
        """
        if not self._tracked_new_files:
            return
        for path in list(self._tracked_new_files):
            try:
                p = Path(path)
                if p.is_file():
                    p.unlink()
                    print(f"[AGENT] Removed temp file: {p}")
            except Exception as exc:
                print(f"[AGENT] Failed to remove temp file {path}: {exc}")
        self._tracked_new_files.clear()

    # ------------------------------------------------------------------
    # Run diagnostics (C7)
    # ------------------------------------------------------------------

    def _build_diagnostics(self, finish_reason: Optional[str]) -> Dict[str, Any]:
        return {
            "finish_reason": finish_reason,
            "iterations": getattr(self, "_iterations_run", 0),
            "truncation_continuations": getattr(self, "_total_truncation_continuations", 0),
            "empty_retries": getattr(self, "_empty_retries", 0),
            "tool_calls": getattr(self, "_tool_calls_dispatched", 0),
            "compactions": getattr(self, "_compactions", 0),
            "fallback_used": bool(getattr(self, "_fallback_used", False)),
            "supervisor_reviews": getattr(self, "_supervisor_reviews", 0),
            "watchdog_stops": getattr(self, "_watchdog_stops", 0),
        }

    def _reset_run_state(self) -> None:
        self._last_finish_reason = None
        self._truncation_continuations = 0
        self._total_truncation_continuations = 0
        self._empty_retries = 0
        self._compactions = 0
        self._fallback_used = False
        self._iterations_run = 0
        self._tool_calls_dispatched = 0
        self._reissue_nudges = 0
        self._worklog = []
        self._recovered_tool_calls = []
        self._final_answer_text = ""
        self._final_answer_cut = None
        self.final_finish_reason = None
        self._tool_launch_times = {}
        self._completion_emitted = set()
        self._agent_child_sessions = {}
        self._msgs_ref = None
        self._turn_started = time.monotonic()
        self._last_touch = 0.0
        self._last_review_at = None
        self._last_review_iter = 0
        self._supervisor_reviews = 0
        self._watchdog_stops = 0
        self.diagnostics = self._build_diagnostics(None)

    def _touch_turn(self, **fields: Any) -> None:
        """Refresh this session's turn record in the watchdog registry. Bare
        activity pings (no fields) are throttled to ~1/s so streaming deltas stay
        cheap; state/iteration changes always go through."""
        try:
            now = time.monotonic()
            if not fields and now - self._last_touch < 1.0:
                return
            self._last_touch = now
            _wd.note_turn(self.session_id, **fields)
        except Exception:
            pass

    async def run_stream(
        self,
        messages: List[Dict[str, Any]],
        attached_files: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run the agent loop and yield events. Wraps _run_stream_body so we
        can sweep non-persisted temp files and record diagnostics in a single
        finally block, even on early return / exception / generator close.

        After the run, `self.final_finish_reason` is one of "stop" | "length" |
        "error" | "interrupted" and `self.diagnostics` holds the C7 dict."""
        self._reset_run_state()
        self._touch_turn(iteration=0, state="llm")
        body = self._run_stream_body(messages, attached_files)
        try:
            async for event in body:
                yield event
            if not self.final_finish_reason:
                self.final_finish_reason = "stop"
        except StopInferenceError:
            self.final_finish_reason = "interrupted"
            raise
        except (asyncio.CancelledError, GeneratorExit):
            self.final_finish_reason = "interrupted"
            raise
        except BaseException:
            if self.final_finish_reason in (None, "stop", "length"):
                self.final_finish_reason = "error"
            raise
        finally:
            try:
                # Close the body explicitly (not via GC) so its own cleanup runs,
                # then cancel AND await every tool task it launched — before the
                # temp-file sweep, so no tool is still writing/running.
                try:
                    await body.aclose()
                except Exception:
                    pass
                await self._reap_inflight_tasks()
            finally:
                _wd.clear_turn(self.session_id)
                self._msgs_ref = None
                if not self.final_finish_reason:
                    self.final_finish_reason = "interrupted"
                self.diagnostics = self._build_diagnostics(self.final_finish_reason)
                try:
                    self._log(f"[AGENT] Run diagnostics: {_json.dumps(self.diagnostics)}")
                except Exception:
                    pass
                self._cleanup_temp_files()

    # ------------------------------------------------------------------
    # Loop helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cancel_tasks(tasks) -> None:
        for t in tasks:
            if t is not None and not t.done():
                t.cancel()

    def _launch_tool(self, tc: ToolCall) -> asyncio.Task:
        """create_task(_safe_execute) tracked in _inflight_tasks until done."""
        self._tool_launch_times[tc.id] = time.monotonic()
        task = asyncio.create_task(self._safe_execute(tc))
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)
        return task

    async def _reap_inflight_tasks(self, timeout: float = 15.0) -> None:
        """Cancel every still-running tool task and wait (bounded) for its
        cancellation cleanup — e.g. process-tree kills — to finish."""
        tasks = [t for t in self._inflight_tasks if not t.done()]
        if not tasks:
            return
        for t in tasks:
            t.cancel()
        try:
            _, still = await asyncio.wait(tasks, timeout=timeout)
        except Exception:
            return
        if still:
            self._log(f"[AGENT] {len(still)} tool task(s) still running {timeout:.0f}s "
                      f"after cancellation; abandoning them")

    def _emit_child_progress(self, event: ToolStatusEvent) -> None:
        """Queue a subagent progress event for the stream (never raises)."""
        try:
            self._child_progress.append(event)
            if self._child_progress_signal is not None:
                self._child_progress_signal.set()
        except Exception:
            pass

    def _started_event(self, tc: ToolCall) -> ToolStatusEvent:
        meta = TOOL_METADATA.get(tc.function.name, {})
        try:
            detail = _wd.describe_call(tc.function.name, tc.function.arguments)
        except Exception:
            detail = ""
        return ToolStatusEvent(
            tool_name=tc.function.name,
            tool_call_id=tc.id,
            status="started",
            activity=meta.get("activity", ""),
            user_name=meta.get("user_name", ""),
            detail=detail,
        )

    async def _await_with_progress(
        self, calls: List[Tuple[ToolCall, asyncio.Task]],
    ) -> AsyncIterator[StreamEvent]:
        """Wait for the tool tasks of *calls* to finish while yielding (a) queued
        subagent progress events and (b) a status="progress" event per still-running
        tool after TOOL_PROGRESS_AFTER_SECONDS, then every
        TOOL_PROGRESS_INTERVAL_SECONDS. Results are read from the tasks afterwards.

        Idle cost is one asyncio.wait with a timeout (no polling); it never delays
        a result because it wakes on task completion."""
        if self._child_progress_signal is None:
            self._child_progress_signal = asyncio.Event()
        signal = self._child_progress_signal
        after = max(0.5, float(getattr(config, "TOOL_PROGRESS_AFTER_SECONDS", 6)))
        interval = max(1.0, float(getattr(config, "TOOL_PROGRESS_INTERVAL_SECONDS", 10)))
        now = time.monotonic()
        next_tick: Dict[str, float] = {}
        for tc, _t in calls:
            next_tick[tc.id] = self._tool_launch_times.get(tc.id, now) + after
        emitted = self._completion_emitted

        def _finished_events() -> List[ToolStatusEvent]:
            # "completed"/"failed" as each task finishes (not after the whole
            # iteration), so a turn cut off mid-iteration (stop / preempt) still
            # has accurate per-call status for the tools that already ended.
            out: List[ToolStatusEvent] = []
            for f_tc, f_t in calls:
                if not f_t.done() or f_tc.id in emitted:
                    continue
                emitted.add(f_tc.id)
                try:
                    out.append(self._completion_event(
                        f_tc, self._coerce_tool_result(self._task_outcome(f_t)),
                    ))
                except Exception:
                    pass
            return out

        pending = {t for _tc, t in calls if not t.done()}
        for ev in _finished_events():
            yield ev
        while pending:
            running_ids = [tc.id for tc, t in calls if not t.done()]
            now = time.monotonic()
            due = min((next_tick[i] for i in running_ids), default=now + interval)
            waiter = asyncio.ensure_future(signal.wait())
            try:
                await asyncio.wait(
                    pending | {waiter},
                    timeout=max(0.05, due - now),
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                if not waiter.done():
                    waiter.cancel()
            pending = {t for t in pending if not t.done()}
            signal.clear()
            while self._child_progress:
                yield self._child_progress.popleft()
            for ev in _finished_events():
                yield ev
            now = time.monotonic()
            for tc, t in calls:
                if not t.done() and now >= next_tick[tc.id]:
                    next_tick[tc.id] = now + interval
                    try:
                        yield self._make_progress_event(tc)
                    except Exception:
                        pass
        self._child_progress.clear()

    def _completion_event(self, tc: ToolCall, result: Dict[str, Any]) -> ToolStatusEvent:
        """The terminal "completed"/"failed" ToolStatusEvent of one tool call."""
        duration = 0.0
        for e in reversed(self.tool_calls_log):
            if e.get("tool_call_id") == tc.id:
                duration = e.get("duration", 0) or 0
                break
        status = "completed" if result.get("success", True) else "failed"
        meta = TOOL_METADATA.get(tc.function.name, {})
        launch_ts = self._tool_launch_times.get(tc.id)
        stopped = result.get("stopped_reason")
        if stopped:
            status = "failed"
            detail = f"stopped: {stopped}"
        else:
            try:
                detail = _wd.describe_call(tc.function.name, tc.function.arguments)
            except Exception:
                detail = ""
        elapsed = float(duration) if duration else (
            max(0.0, time.monotonic() - launch_ts) if launch_ts is not None else 0.0
        )
        return ToolStatusEvent(
            tool_name=tc.function.name,
            tool_call_id=tc.id,
            status=status,
            duration=round(float(duration), 2),
            activity=meta.get("activity", ""),
            user_name=meta.get("user_name", ""),
            detail=detail,
            elapsed=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Turn review (iteration-boundary supervisor for very long jobs)
    # ------------------------------------------------------------------

    def _turn_review_due(self, iteration: int) -> bool:
        if not getattr(config, "TURN_REVIEW_ENABLED", True) or iteration <= 0:
            return False
        if "__sub_" in (self.session_id or ""):
            return False  # subagents are bounded by their own deadline
        now = time.monotonic()
        after = max(1.0, float(getattr(config, "TURN_REVIEW_AFTER_SECONDS", 900)))
        if now - self._turn_started < after:
            return False
        if self._last_review_at is None:
            return True
        interval = max(1.0, float(getattr(config, "TURN_REVIEW_INTERVAL_SECONDS", 900)))
        every = int(getattr(config, "TURN_REVIEW_EVERY_ITERATIONS", 40) or 0)
        return (
            now - self._last_review_at >= interval
            or (every > 0 and iteration - self._last_review_iter >= every)
        )

    def _build_turn_review_prompt(self, elapsed: float, iteration: int) -> str:
        lines = [
            f"User request (headline): {_wd.mask_text(self._request_headline(400)) or '(unknown)'}",
            f"Running for: {_wd.fmt_duration(elapsed)}; LLM iterations so far: {iteration}",
        ]
        log = list(self._worklog or [])
        if log:
            counts: Dict[str, int] = {}
            failed = 0
            for e in log:
                counts[e["name"]] = counts.get(e["name"], 0) + 1
                if not e.get("success", True):
                    failed += 1
            ledger = ", ".join(
                f"{n} x{c}" for n, c in sorted(counts.items(), key=lambda kv: -kv[1])[:10]
            )
            lines.append(f"Tool calls so far: {len(log)} (failed {failed}): {ledger}")
            lines.append("Last tool results:")
            for e in log[-5:]:
                head = _wd.mask_text((e.get("headline") or "")[:120])
                lines.append(f"- {e['name']} ({'ok' if e.get('success', True) else 'failed'}): {head}")
        else:
            lines.append("Tool calls so far: none")
        todos = self._session_todos or []
        if todos:
            done = sum(1 for t in todos if t.get("status") == "completed")
            lines.append(f"Todo list: {done}/{len(todos)} completed")
            for t in [t for t in todos if t.get("status") != "completed"][:5]:
                lines.append(f"- [{t.get('status', 'pending')}] {str(t.get('content', ''))[:100]}")
        else:
            lines.append("Todo list: none")
        lines.append("Judge whether the agent is on track. Answer with the JSON object only.")
        return "\n".join(lines)

    async def _review_turn(
        self, msgs: List[Dict[str, Any]], iteration: int,
    ) -> AsyncIterator[StreamEvent]:
        """Supervisor check of a long-running turn at an iteration boundary.
        continue -> nothing; replan / wrap_up -> a non-persisted user-role nudge
        is appended to *msgs* (before the next LLM call) and a visible
        ToolStatusEvent is yielded. Fails open on every error."""
        if not self._turn_review_due(iteration):
            return
        now = time.monotonic()
        self._last_review_at = now
        self._last_review_iter = iteration
        elapsed = now - self._turn_started
        self._touch_turn(state="review")
        reply = await self._supervisor_llm_text(
            _wd.TURN_REVIEW_SYSTEM,
            self._build_turn_review_prompt(elapsed, iteration),
            "agent:turn_review",
        )
        self._touch_turn(state="llm")
        self._supervisor_reviews += 1
        verdict, note = _wd.parse_turn_verdict(reply)
        if verdict == "wrap_up" and self.goal_mode and self._goal_gate_would_block():
            verdict = "continue"  # the done-gate owns completion in goal mode
        self._log(f"[TURN_REVIEW] after {_wd.fmt_duration(elapsed)} / iteration {iteration}: "
                  f"{verdict} {note[:200]}")
        if verdict == "continue":
            return
        minutes = max(1, int(round(elapsed / 60.0)))
        if verdict == "wrap_up":
            guidance = ("Deliver what you have now and state clearly what remains "
                        "to be done.")
        elif "agent" in (getattr(self, "_enabled_tool_names", None) or ()):
            guidance = ("Narrow the plan, or delegate independent parts to parallel "
                        "subagents with the agent tool.")
        else:
            guidance = "Narrow the plan and focus on the most important remaining work."
        if note and note[-1] not in ".!?。…":
            note_txt = note + ". "
        else:
            note_txt = note + " " if note else ""
        text = f"{_SUPERVISOR_NUDGE_PREFIX} {minutes} min: {note_txt}{guidance}"
        msgs.append({"role": "user", "content": text})
        yield ToolStatusEvent(
            tool_name="supervisor",
            tool_call_id=f"supervisor_{iteration}",
            status="completed",
            activity=f"진행 점검: {verdict}",
            detail=note,
        )

    @staticmethod
    def _task_outcome(task: asyncio.Task) -> Any:
        """A finished task's result in asyncio.gather(return_exceptions=True) form."""
        if task.cancelled():
            return asyncio.CancelledError()
        exc = task.exception()
        return exc if exc is not None else task.result()

    async def _safe_execute(self, tc: ToolCall) -> Dict[str, Any]:
        """execute_tool that never raises (except cancellation)."""
        try:
            return await self.execute_tool(
                tc.function.name, tc.function.arguments, tc.id,
                parse_error=getattr(tc.function, "parse_error", None),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {"success": False, "error": _format_exception(e)}

    @staticmethod
    def _coerce_tool_result(r: Any) -> Dict[str, Any]:
        if isinstance(r, BaseException):
            if isinstance(r, asyncio.CancelledError):
                return {"success": False, "error": "Tool call was cancelled"}
            return {"success": False, "error": _format_exception(r)}
        if r is None:
            return {"success": False, "error": "Tool produced no result"}
        if not isinstance(r, dict):
            return {"success": r is not None, "output": r}
        return r

    @staticmethod
    def _is_partial_stream_failure(exc: BaseException, saw_output: bool) -> bool:
        """True when a stream died AFTER producing output (keep the partial)."""
        import httpx
        if isinstance(exc, VllmStreamError):
            return bool(getattr(exc, "partial", False)) or saw_output
        if isinstance(exc, (httpx.TransportError, httpx.HTTPStatusError,
                            asyncio.TimeoutError, TimeoutError)):
            return saw_output
        return False

    def _tool_calls_from_reasoning(self, reasoning: str) -> List[ToolCall]:
        """Parse `<tool_call>` blocks the model wrote inside its reasoning
        channel (backend helper) into dispatchable ToolCalls."""
        names = self._enabled_tool_names
        if not reasoning or "<tool_call>" not in reasoning or not names:
            return []
        try:
            from backend.core.llm_backend import _parse_leaked_tool_call
        except ImportError:
            return []
        calls: List[ToolCall] = []
        seen = set()
        for m in _TOOL_CALL_BLOCK_RE.finditer(reasoning):
            try:
                parsed = _parse_leaked_tool_call(m.group(1))
            except Exception:
                parsed = None
            if not parsed or not isinstance(parsed, (tuple, list)) or len(parsed) < 2:
                continue
            name, args = parsed[0], parsed[1]
            if name not in names:
                continue
            if not isinstance(args, dict):
                args = {}
            sig = self._tool_signature(name, args)
            if sig in seen:
                continue
            seen.add(sig)
            calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                function=ToolCallFunction(name=name, arguments=args),
            ))
            if len(calls) >= _MAX_REASONING_TOOL_CALLS:
                break
        return calls

    def _record_worklog(self, tc: ToolCall, result: Dict[str, Any]) -> None:
        name = tc.function.name
        args = tc.function.arguments if isinstance(tc.function.arguments, dict) else {}
        success = bool(result.get("success", True))
        paths: List[str] = []
        if success and name in ("file_writer", "file_edit", "apply_patch"):
            for key in ("path", "file_path"):
                v = args.get(key) or result.get(key)
                if isinstance(v, str) and v:
                    paths.append(v)
                    break
            files = result.get("files") or result.get("files_changed")
            if isinstance(files, list):
                paths.extend(str(f) for f in files[:10] if isinstance(f, (str, Path)))
        self._worklog.append({
            "name": name,
            "success": success,
            "headline": _result_headline(result),
            "paths": paths,
        })

    def _deterministic_fallback_answer(self) -> str:
        """Korean summary of the work done, built from the tool ledger only.
        Never a bare "완료." and never model reasoning."""
        log = self._worklog
        if not log:
            return (
                "죄송합니다. 이번 요청에 대한 답변을 생성하지 못했습니다 "
                "(모델이 빈 응답을 반환했습니다). 같은 요청을 다시 보내 주시거나 "
                "질문을 조금 더 구체적으로 적어 주세요."
            )
        counts: Dict[str, int] = {}
        failures = 0
        temp_files = {str(p) for p in (getattr(self, "_tracked_new_files", None) or set())}
        files: List[str] = []
        for e in log:
            counts[e["name"]] = counts.get(e["name"], 0) + 1
            if not e["success"]:
                failures += 1
            for p in e.get("paths") or []:
                if p not in files and p not in temp_files:
                    files.append(p)
        tools_str = ", ".join(
            f"{name} {n}회" for name, n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        total = f"총 {len(log)}회" + (f", 실패 {failures}회" if failures else "")
        lines = [
            "모델이 최종 답변을 작성하지 못해, 지금까지 수행한 작업을 자동으로 정리했습니다.",
            "",
            f"- 실행한 도구: {tools_str} ({total})",
        ]
        if files:
            more = f" 외 {len(files) - 10}개" if len(files) > 10 else ""
            lines.append(f"- 작성/수정한 파일: {', '.join(files[:10])}{more}")
        last = log[-1]
        status = "성공" if last["success"] else "실패"
        headline = f": {last['headline']}" if last.get("headline") else ""
        lines.append(f"- 마지막 도구 결과: {last['name']} ({status}){headline}")
        lines.append("")
        lines.append("결과를 확인하신 뒤 필요한 부분을 다시 요청해 주세요.")
        return "\n".join(lines)

    async def _stream_recovery_once(
        self,
        msgs: List[Dict[str, Any]],
        iteration: int,
        agent_type: str,
        kwargs: Dict[str, Any],
        out: Dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """One tool-less recovery LLM call. Yields TextEvents / autocompact
        ToolStatusEvents and fills *out* (text, reasoning, tool_calls, failed,
        finish). A non-stop error is swallowed (out["failed"]) — after partial
        output that marks the answer as cut, before it the caller moves on."""
        try:
            async for ev in self._stream_with_autocompact(
                msgs, self.model, self.temperature,
                iteration=iteration,
                session_id=self.session_id,
                agent_type=agent_type,
                **kwargs,
            ):
                if isinstance(ev, TextEvent):
                    out["text"].append(ev.content)
                    yield ev
                elif isinstance(ev, ReasoningEvent):
                    out["reasoning"].append(ev.content)
                elif isinstance(ev, ToolStatusEvent):
                    yield ev
                elif isinstance(ev, ToolCallDeltaEvent):
                    out["tool_calls"].extend(ev.tool_calls)
        except (StopInferenceError, asyncio.CancelledError):
            raise
        except Exception as e:
            out["failed"] = True
            self._log(f"[EMPTY_ANSWER] {agent_type} failed: {_format_exception(e)[:300]}")
        out["finish"] = self._last_finish_reason

    async def _ensure_final_answer(
        self,
        msgs: List[Dict[str, Any]],
        reasoning_body: str,
        iteration: int,
        allow_tools: bool,
    ) -> AsyncIterator[StreamEvent]:
        """Recover a final answer after an iteration ended with no visible text
        and no tool calls (the "완료." / empty-answer bug).

        Order: text after `</think>` inside reasoning → Step A (thinking off,
        short budget, no tools) → Step B (reasoning_effort=high) → Step C
        deterministic Korean summary of the tool ledger. Yields TextEvents (and
        autocompact ToolStatusEvents). Sets self._final_answer_text, or — when
        a retry produced tool calls and tools are allowed —
        self._recovered_tool_calls for the caller to dispatch. A Step A/B
        answer cut by its token cap or a mid-answer stream drop is continued
        up to _FINAL_ANSWER_MAX_CONTINUATIONS times; if still cut, a notice is
        yielded and self._final_answer_cut is set ("length" / "error") for the
        caller's finish_reason.
        """
        self._touch_turn(state="final")
        self._recovered_tool_calls = []
        self._final_answer_text = ""
        self._final_answer_cut = None
        self._empty_retries += 1
        self._log(
            f"[EMPTY_ANSWER] iteration={iteration + 1} reasoning_chars={len(reasoning_body or '')} "
            f"finish={self._last_finish_reason} — recovering final answer"
        )

        promoted = _text_after_think(reasoning_body)
        if promoted:
            self._log(f"[EMPTY_ANSWER] promoted {len(promoted)} chars written after </think> in reasoning")
            yield TextEvent(content=promoted)
            msgs.append({"role": "assistant", "content": promoted})
            self._final_answer_text = promoted
            return

        notes = _TOOL_CALL_BLOCK_RE.sub("", reasoning_body or "").strip()
        if len(notes) > _FINAL_NOTES_MAX_CHARS:
            notes = "…" + notes[-_FINAL_NOTES_MAX_CHARS:]
        if notes:
            nudge = (
                "<internal_notes>\n(Your own reasoning from the previous step. Use it; "
                "do not quote it verbatim.)\n"
                f"{notes}\n</internal_notes>\n\n{_FINAL_ANSWER_NUDGE}"
            )
        else:
            nudge = _FINAL_ANSWER_NUDGE
        msgs.append({"role": "user", "content": nudge})

        short_max = max(256, int(getattr(config, "AGENT_FINAL_ANSWER_RETRY_MAX_TOKENS", 4096)))
        min_out = int(getattr(config, "MIN_OUTPUT_TOKENS", 8192))
        steps = (
            ("A", {"enable_thinking": False}, short_max),
            # Thinking stays on in B, so give it room to finish reasoning AND answer.
            ("B", {"reasoning_effort": "high"}, max(short_max * 4, min_out)),
        )
        tool_dropped = False
        for label, ctk, max_tok in steps:
            check_stop(self.session_id)
            kwargs = self._sampling_kwargs(final_response=True)
            kwargs["chat_template_kwargs"] = ctk
            kwargs["max_tokens"] = min(max_tok, self.max_tokens) if self.max_tokens else max_tok
            out = _new_recovery_out()
            async for ev in self._stream_recovery_once(
                msgs, iteration, f"agent:stream:final-answer-{label}", kwargs, out,
            ):
                yield ev
            text = "".join(out["text"])
            reason_parts: List[str] = out["reasoning"]
            tool_calls: List[ToolCall] = out["tool_calls"]
            if not text.strip():
                promoted = _text_after_think("".join(reason_parts))
                if promoted:
                    yield TextEvent(content=promoted)
                    text = promoted
            if text.strip():
                # A recovery answer that hit its own token cap (finish=length)
                # or whose stream died mid-answer is NOT complete: continue it
                # a bounded number of times, then label it as cut.
                cut = _recovery_cut(out)
                chunk = text
                conts = 0
                while cut and conts < _FINAL_ANSWER_MAX_CONTINUATIONS:
                    check_stop(self.session_id)
                    conts += 1
                    self._total_truncation_continuations += 1
                    nudge_c = _TRUNCATION_NUDGE if cut == "length" else _STREAM_ERROR_NUDGE
                    msgs.append({"role": "assistant", "content": chunk.strip()})
                    msgs.append({"role": "user", "content": nudge_c})
                    self._log(f"[EMPTY_ANSWER] step {label} answer cut (finish={cut}) — "
                              f"continuing ({conts}/{_FINAL_ANSWER_MAX_CONTINUATIONS})")
                    out = _new_recovery_out()
                    async for ev in self._stream_recovery_once(
                        msgs, iteration, f"agent:stream:final-answer-{label}-cont", kwargs, out,
                    ):
                        yield ev
                    chunk = "".join(out["text"])
                    if not chunk.strip():
                        chunk = _text_after_think("".join(out["reasoning"]))
                        if chunk:
                            yield TextEvent(content=chunk)
                    if not chunk.strip():
                        # Continuation produced nothing: drop its nudge so the
                        # transcript ends on the recorded partial answer.
                        if msgs and msgs[-1].get("content") == nudge_c:
                            msgs.pop()
                        chunk = ""
                        break
                    text += chunk
                    cut = _recovery_cut(out)
                if chunk.strip():
                    msgs.append({"role": "assistant", "content": chunk.strip()})
                if cut:
                    self._log(f"[EMPTY_ANSWER] step {label} answer still cut "
                              f"(finish={cut}) — labelling the reply as {cut}")
                    yield TextEvent(content=_LENGTH_NOTICE if cut == "length" else _STREAM_ERROR_NOTICE)
                    self._final_answer_cut = cut
                self._log(f"[EMPTY_ANSWER] recovered via step {label} ({len(text)} chars)")
                self._final_answer_text = text
                return

            if not tool_calls and allow_tools:
                tool_calls = self._tool_calls_from_reasoning("".join(reason_parts))
            if tool_calls:
                good = [tc for tc in tool_calls if not getattr(tc.function, "parse_error", None)]
                if allow_tools and good:
                    self._log(f"[EMPTY_ANSWER] step {label} produced {len(good)} "
                              f"tool call(s) instead of text — dispatching them")
                    self._recovered_tool_calls = good
                    return
                if not allow_tools:
                    tool_dropped = True
            self._log(f"[EMPTY_ANSWER] step {label} produced no text")

        fallback = self._deterministic_fallback_answer()
        if tool_dropped:
            fallback += _TOOL_DROPPED_NOTICE
        self._fallback_used = True
        self._log("[EMPTY_ANSWER] steps A/B failed — using deterministic fallback summary")
        yield TextEvent(content=fallback)
        msgs.append({"role": "assistant", "content": fallback})
        self._final_answer_text = fallback

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run_stream_body(
        self,
        messages: List[Dict[str, Any]],
        attached_files: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[StreamEvent]:
        # RAG listing, git and the workspace walk are blocking IO — keep them
        # off the event loop so concurrent sessions aren't stalled.
        await asyncio.to_thread(self._refresh_available_rag_collections)
        system_prompt = self._build_system_prompt()
        msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        # Goal-mode preamble as its own system message after the (byte-stable,
        # KV-cached) static prompt, so it can't be truncated by the dynamic
        # context budget and doesn't disturb prefix-cache reuse.
        if self.goal_mode:
            preamble = self._goal_mode_preamble()
            if preamble:
                msgs.append({"role": "system", "content": preamble})
        dynamic_ctx = await asyncio.to_thread(self._build_dynamic_context, attached_files)
        if dynamic_ctx:
            msgs.append({"role": "system", "content": dynamic_ctx})
        # Everything after this index is caller-supplied (compaction's last
        # resort may shrink an oversize caller system message, never ours).
        self._own_system_count = len(msgs)
        msgs.extend(messages)
        self._enforce_history_limit(msgs)
        self._msgs_ref = msgs
        tool_schemas = self._get_tool_schemas()
        self._enabled_tool_names = {
            s["function"]["name"] for s in (tool_schemas or [])
            if isinstance(s, dict) and isinstance(s.get("function"), dict)
        }
        self._truncation_continuations = 0
        max_cont = max(0, int(getattr(config, "AGENT_MAX_TRUNCATION_CONTINUATIONS", 6)))
        # Visible text streamed since the last tool round (spans truncation
        # continuations), used to decide whether an answer exists.
        answer_parts: List[str] = []

        for iteration in range(self.max_iterations):
            check_stop(self.session_id)
            self._iterations_run = iteration + 1
            self._touch_turn(iteration=iteration + 1, state="llm")
            print(f"\n[AGENT-STREAM] Iteration {iteration + 1}/{self.max_iterations}")
            self._log_iteration_start(iteration, streaming=True)

            self._iteration_boundaries.append(len(msgs))
            async for _review_ev in self._review_turn(msgs, iteration):
                yield _review_ev

            # Collect all tool calls. Concurrency-safe tools (independent reads
            # and subagent fan-out) run in parallel during resolution; mutating
            # tools are serial barriers. Read-only safe tools additionally start
            # mid-stream as soon as their args parse (see resolution below).
            all_tool_calls: List[ToolCall] = []
            # (tc, task_or_none) pairs; None means deferred serial execution.
            pending_tasks: List[Tuple[ToolCall, Optional[asyncio.Task]]] = []
            deferred_seen = False
            # Accumulate any text the model streams alongside tool calls so the
            # assistant turn we record back into msgs preserves it. Without this,
            # the model sees content=None next iteration and may emit empty replies.
            streamed_text_parts: List[str] = []
            # Reasoning/thinking deltas (separate field on reasoning models).
            # Preserved in history but NOT yielded to the caller, so the user
            # never sees raw chain-of-thought.
            streamed_reasoning_parts: List[str] = []
            stream_error: Optional[BaseException] = None
            # Goal mode with the done-gate armed: a no-tool-call answer this
            # iteration would be blocked, so hold its text back until we know
            # (released as soon as a tool call arrives or the gate passes;
            # dropped if the gate blocks — otherwise the client would receive
            # every blocked "final answer" concatenated).
            hold_text = self.goal_mode and self._goal_gate_would_block()
            held_text: List[TextEvent] = []

            # Compressed view is rebuilt per attempt inside the wrapper. msgs
            # is the source-of-truth list; auto-compact mutates it on overflow.
            try:
                async for event in self._stream_with_autocompact(
                    msgs, self.model, self.temperature,
                    iteration=iteration,
                    tools=tool_schemas,
                    session_id=self.session_id,
                    agent_type="agent:stream",
                    **self._sampling_kwargs(final_response=True),
                ):
                    self._touch_turn()
                    if isinstance(event, TextEvent):
                        streamed_text_parts.append(event.content)
                        if hold_text and not all_tool_calls:
                            held_text.append(event)
                        else:
                            yield event
                    elif isinstance(event, ReasoningEvent):
                        streamed_reasoning_parts.append(event.content)
                        # Internal-only: NOT yielded to the caller (chat.py / hoonbot).
                    elif isinstance(event, ToolStatusEvent):
                        # Autocompact progress — forward so clients see activity.
                        yield event
                    elif isinstance(event, ToolCallDeltaEvent):
                        if held_text and event.tool_calls:
                            # It's tool narration after all — release it in order.
                            for h in held_text:
                                yield h
                            held_text.clear()
                        for tc in event.tool_calls:
                            all_tool_calls.append(tc)
                            task = None
                            has_parse_error = bool(getattr(tc.function, "parse_error", None))
                            if not has_parse_error and self._can_start_tool_immediately(
                                tc.function.name, deferred_seen
                            ):
                                task = self._launch_tool(tc)
                            elif not has_parse_error:
                                deferred_seen = True
                            pending_tasks.append((tc, task))
                            if task is not None:
                                yield self._started_event(tc)
            except BaseException as e:
                saw_output = bool(streamed_text_parts or streamed_reasoning_parts or all_tool_calls)
                if (
                    isinstance(e, Exception)
                    and not isinstance(e, StopInferenceError)
                    and self._is_partial_stream_failure(e, saw_output)
                ):
                    stream_error = e
                    self._log(f"[AGENT] LLM stream failed after partial output "
                              f"({_format_exception(e)[:300]}) — keeping the partial")
                else:
                    self._cancel_tasks([t for _, t in pending_tasks])
                    if isinstance(e, Exception):
                        if not isinstance(e, StopInferenceError):
                            self.final_finish_reason = "error"
                        # Never swallow text the user would have seen unheld.
                        for h in held_text:
                            yield h
                        held_text.clear()
                    raise

            finish = self._last_finish_reason
            saw_output = bool(streamed_text_parts or streamed_reasoning_parts or all_tool_calls)
            if stream_error is not None or (finish == "error" and saw_output):
                finish = "error_partial"
            truncated = finish in ("length", "error_partial")
            if not truncated:
                self._truncation_continuations = 0

            text_body = "".join(streamed_text_parts)
            reasoning_body = "".join(streamed_reasoning_parts).strip()

            # Under truncation, a tool call whose arguments failed to parse was
            # cut off mid-JSON: drop it (never execute half a file write) and
            # tell the model how to avoid it. Fully parsed calls are kept.
            truncated_tool_dropped = False
            if truncated and all_tool_calls:
                kept = [(tc, t) for tc, t in pending_tasks
                        if not getattr(tc.function, "parse_error", None)]
                if len(kept) != len(pending_tasks):
                    truncated_tool_dropped = True
                    self._log(f"[AGENT] Dropped {len(pending_tasks) - len(kept)} truncated "
                              f"tool call(s) (finish_reason={finish})")
                    pending_tasks = kept
                    all_tool_calls = [tc for tc, _ in kept]

            if not all_tool_calls:
                answer_parts.append(text_body)
                visible_so_far = "".join(answer_parts)
                fallback_before = self._fallback_used

                # ---- Truncation / partial-stream continuation ----
                if truncated:
                    # Mid-answer cut: the gate decision waits for the end of
                    # the answer, so release the held part now.
                    for h in held_text:
                        yield h
                    held_text.clear()
                    if self._truncation_continuations < max_cont:
                        self._truncation_continuations += 1
                        self._total_truncation_continuations += 1
                        partial = _compose_assistant_content(
                            _strip_unclosed_think(text_body).strip(), reasoning_body,
                        )
                        if partial:
                            msgs.append({"role": "assistant", "content": partial})
                        if truncated_tool_dropped:
                            nudge = _TRUNCATED_TOOL_NUDGE
                        elif finish == "error_partial":
                            nudge = _STREAM_ERROR_NUDGE
                        else:
                            nudge = _TRUNCATION_NUDGE
                        msgs.append({"role": "user", "content": nudge})
                        self._log(
                            f"[AGENT] Output truncated (finish_reason={finish}) — continuing "
                            f"({self._truncation_continuations}/{max_cont})"
                        )
                        continue
                    # Cap reached: stop with a visible notice rather than loop forever.
                    self._log(f"[AGENT] Truncation continuation cap ({max_cont}) reached")
                    if not visible_so_far.strip():
                        # Only reasoning was cut; the recovery writes a fresh
                        # answer, so label the run by THAT answer (complete →
                        # "stop"; the recovery adds its own notice if cut).
                        async for ev in self._ensure_final_answer(
                            msgs, reasoning_body, iteration, allow_tools=False,
                        ):
                            yield ev
                        self.final_finish_reason = self._final_answer_cut or "stop"
                    else:
                        # The visible answer really is cut: one inline notice
                        # (persisted with the turn) + the matching finish_reason.
                        yield TextEvent(
                            content=_LENGTH_NOTICE if finish == "length" else _STREAM_ERROR_NOTICE
                        )
                        self.final_finish_reason = "length" if finish == "length" else "error"
                    self._log_agent_complete(
                        f"Truncation cap reached (finish_reason={finish})", iteration + 1,
                    )
                    return

                # ---- finish_reason says "tool call" but none arrived ----
                if (
                    finish in ("tool_calls", "tool")
                    and self._enabled_tool_names
                    and self._reissue_nudges < _MAX_REISSUE_NUDGES
                ):
                    self._reissue_nudges += 1
                    partial = _compose_assistant_content(text_body.strip(), reasoning_body)
                    if partial:
                        msgs.append({"role": "assistant", "content": partial})
                    msgs.append({"role": "user", "content": _REISSUE_TOOL_NUDGE})
                    self._log(f"[AGENT] finish_reason={finish} but no tool calls — asking "
                              f"the model to re-issue ({self._reissue_nudges}/{_MAX_REISSUE_NUDGES})")
                    # This text was pre-tool narration ("X를 확인해 보겠습니다"),
                    # not an answer: don't let it satisfy the empty-answer check
                    # if the re-issued turn comes back empty.
                    if answer_parts:
                        answer_parts.pop()
                    for h in held_text:
                        yield h
                    held_text.clear()
                    continue

                # ---- Empty answer: no visible text, no tool calls ----
                answer_recorded = False
                if not visible_so_far.strip():
                    recovered = self._tool_calls_from_reasoning(reasoning_body)
                    if recovered:
                        self._log(f"[EMPTY_ANSWER] dispatching {len(recovered)} tool call(s) "
                                  f"found inside reasoning")
                        reasoning_body = _TOOL_CALL_BLOCK_RE.sub("", reasoning_body).strip()
                        all_tool_calls = recovered
                        pending_tasks = [(tc, None) for tc in recovered]
                    else:
                        async for ev in self._ensure_final_answer(
                            msgs, reasoning_body, iteration,
                            allow_tools=bool(self._enabled_tool_names),
                        ):
                            if hold_text and isinstance(ev, TextEvent):
                                held_text.append(ev)
                            else:
                                yield ev
                        if self._recovered_tool_calls:
                            all_tool_calls = list(self._recovered_tool_calls)
                            pending_tasks = [(tc, None) for tc in all_tool_calls]
                            # The nudge is already in msgs; the recovered call's
                            # assistant turn carries no text of its own.
                            text_body, reasoning_body = "", ""
                        else:
                            answer_recorded = True

                if not all_tool_calls:
                    # Goal-mode done-gate (only once a non-empty answer exists):
                    # refuse to finish while todos are still open. Record the
                    # answer so the loop stays coherent, inject the reminder and
                    # continue. Capped so it always eventually terminates.
                    if self.goal_mode:
                        block = self._goal_completion_block()
                        if block is not None:
                            final_text = text_body.strip()
                            if final_text and not answer_recorded:
                                msgs.append({"role": "assistant", "content": final_text})
                            msgs.append({"role": "system", "content": block})
                            self._completion_blocks += 1
                            answer_parts.clear()
                            if held_text:
                                # The blocked answer never reached the client.
                                self._log(f"[AGENT] Goal-mode done-gate: withheld "
                                          f"{sum(len(h.content) for h in held_text)} chars "
                                          f"of the blocked answer")
                                held_text.clear()
                                # A withheld fallback was never delivered.
                                self._fallback_used = fallback_before
                            self._log(
                                f"[AGENT] Goal-mode done-gate: blocking completion "
                                f"({self._completion_blocks}/"
                                f"{getattr(config, 'AGENT_GOAL_MAX_COMPLETION_BLOCKS', 3)})"
                            )
                            continue
                    for h in held_text:
                        yield h
                    held_text.clear()
                    # A recovered answer that was itself cut keeps its label;
                    # anything else that reaches here completed normally.
                    self.final_finish_reason = (
                        (self._final_answer_cut if answer_recorded else None)
                        or self.final_finish_reason or "stop"
                    )
                    self._log_agent_complete("LLM returned final text response (stream)", iteration + 1)
                    return

            # ---------------- Tool execution ----------------
            for h in held_text:
                yield h
            held_text.clear()
            answer_parts.clear()

            # Log what the LLM requested
            self._log_tool_calls_requested(all_tool_calls, iteration)

            # Append assistant message (must contain ALL tool calls before results).
            # Reasoning content (if emitted by the model) is preserved inline as
            # `<think>...</think>` so the model can build on its own chain of
            # thought on the next iteration. MiniMax M2 in particular degrades
            # severely when prior reasoning is stripped from history.
            combined_content = _compose_assistant_content(
                _strip_unclosed_think(text_body).strip(), reasoning_body,
            )
            msgs.append(self._build_assistant_tool_msg(
                all_tool_calls,
                content=combined_content or None,
            ))

            # Record tool-call signatures for repeat detection BEFORE awaiting
            # results — we want every issued call counted even if a later one
            # in the same iteration fails.
            for tc in all_tool_calls:
                self._tool_call_history.append(
                    self._tool_signature(tc.function.name, tc.function.arguments)
                )
            self._tool_calls_dispatched += len(all_tool_calls)

            # Resolve results. Like Claude Code / Cursor / Codex, independent
            # work runs concurrently while mutations stay ordered: a maximal run
            # of consecutive concurrency-safe calls (reads, subagent fan-out) is
            # gathered in parallel, and every non-concurrency-safe call (writes,
            # shell_exec, code_exec, memo, todo, process_monitor) is a serial
            # barrier that runs alone. Because each group is awaited before the
            # loop advances, all earlier tools finish before a barrier runs and
            # the barrier finishes before any later tool starts — preserving
            # read-after-write and write-after-write ordering within the turn.
            # Tools already launched mid-stream (task is not None) are awaited in
            # place; the rest are launched here. Results stay in model order.
            # gather(return_exceptions=True): one tool blowing up becomes an error
            # result for that call only. On cancellation / stop / generator close
            # every launched task is cancelled before re-raising.
            self._touch_turn(state="tools")
            n_pending = len(pending_tasks)
            results: List[Optional[Dict[str, Any]]] = [None] * n_pending
            launched: List[asyncio.Task] = [t for _, t in pending_tasks if t is not None]
            try:
                idx = 0
                while idx < n_pending:
                    meta = TOOL_METADATA.get(pending_tasks[idx][0].function.name, {})
                    if meta.get("is_concurrency_safe"):
                        group_idx: List[int] = []
                        group_tasks: List[asyncio.Task] = []
                        group_calls: List[Tuple[ToolCall, asyncio.Task]] = []
                        while idx < n_pending:
                            g_tc, g_task = pending_tasks[idx]
                            g_meta = TOOL_METADATA.get(g_tc.function.name, {})
                            if not g_meta.get("is_concurrency_safe"):
                                break
                            if g_task is None:
                                # Not started mid-stream — launch now and announce it.
                                g_task = self._launch_tool(g_tc)
                                launched.append(g_task)
                                yield self._started_event(g_tc)
                            group_tasks.append(g_task)
                            group_calls.append((g_tc, g_task))
                            group_idx.append(idx)
                            idx += 1
                        # Every group is awaited with a ticker: subagent progress
                        # (child tool starts) and per-tool "progress" events are
                        # forwarded while waiting; results are read afterwards.
                        progress = self._await_with_progress(group_calls)
                        try:
                            async for ev in progress:
                                yield ev
                        finally:
                            await progress.aclose()
                        group_results = [self._task_outcome(t) for t in group_tasks]
                        for r_idx, r in zip(group_idx, group_results):
                            results[r_idx] = self._coerce_tool_result(r)
                    else:
                        # Serial barrier: mutating / stateful tool runs alone.
                        tc, _ = pending_tasks[idx]
                        yield self._started_event(tc)
                        s_task = self._launch_tool(tc)
                        launched.append(s_task)
                        progress = self._await_with_progress([(tc, s_task)])
                        try:
                            async for ev in progress:
                                yield ev
                        finally:
                            await progress.aclose()
                        results[idx] = self._coerce_tool_result(self._task_outcome(s_task))
                        idx += 1
            except BaseException:
                self._cancel_tasks(launched)
                raise
            results = [self._coerce_tool_result(r) for r in results]

            # Record iteration outcome (all-failed vs at-least-one-success)
            # for the consecutive-failure detector.
            self._record_iteration_outcome(results)
            self._touch_turn(state="llm")

            # Carry post-edit check failures into next iteration as a high-priority
            # system-reminder so the model must address them before continuing.
            self._pending_postcheck_reminder = None
            for r in results:
                pec = r.get("post_edit_check")
                if isinstance(pec, dict) and pec.get("status") == "failed":
                    lines = "; ".join(
                        f"{f.get('path', '?')}: {f.get('error', '?')}"
                        for f in pec.get("failures", []) if isinstance(f, dict)
                    )
                    self._pending_postcheck_reminder = (
                        f"<system-reminder>Post-edit check FAILED: {lines} — "
                        f"fix the error before continuing. "
                        f"Re-read the file with file_reader if needed.</system-reminder>"
                    )
                    break

            # Durations by call id (tool_calls_log is capped, so look ids up
            # rather than slicing by a pre-iteration offset).
            call_ids = {tc.id for tc in all_tool_calls}
            duration_by_call_id = {
                e.get("tool_call_id"): e.get("duration", 0)
                for e in self.tool_calls_log
                if e.get("tool_call_id") in call_ids
            }
            durations = [duration_by_call_id.get(tc.id, 0) for tc in all_tool_calls]
            self._log_execution_summary(all_tool_calls, results, durations, iteration)

            # Emit "completed"/"failed" events and append results
            for tc, result in zip(all_tool_calls, results):
                if tc.id in self._completion_emitted:
                    # Already announced the moment the task finished
                    # (_await_with_progress); never emit a duplicate.
                    self._completion_emitted.discard(tc.id)
                else:
                    yield self._completion_event(tc, result)
                self._tool_launch_times.pop(tc.id, None)
                msgs.append(self._build_tool_result_msg(tc, result))
                try:
                    self._record_worklog(tc, result)
                except Exception:
                    pass

            if truncated_tool_dropped:
                msgs.append({"role": "user", "content": _TRUNCATED_TOOL_NUDGE})

            # If any tool requested clarification, relay the question to the user
            # and pause the loop. The user's next message resumes with full context.
            clarifications = [
                (tc, r) for tc, r in zip(all_tool_calls, results)
                if r.get("needs_clarification")
            ]
            if clarifications:
                if len(clarifications) == 1:
                    tc, r = clarifications[0]
                    question = r.get("question") or r.get("error") or "Need more information to proceed."
                    relay = question
                else:
                    lines = ["I need some clarification before I can continue:\n"]
                    for i, (tc, r) in enumerate(clarifications, 1):
                        q = r.get("question") or r.get("error") or "?"
                        lines.append(f"{i}. **{tc.function.name}**: {q}")
                    relay = "\n".join(lines)
                self._log(f"  [CLARIFICATION] Pausing — relaying question to user: {relay[:300]}")
                self._log_agent_complete("Tool requested clarification from user", iteration + 1)
                self.final_finish_reason = "stop"
                yield TextEvent(content=relay)
                return

        # Max iterations — final answer without tools
        print(f"[AGENT-STREAM] Max iterations ({self.max_iterations}) reached, requesting final answer")
        self._log_agent_complete(f"Max iterations ({self.max_iterations}) reached (stream)", self.max_iterations)
        check_stop(self.session_id)
        self._touch_turn(state="final")
        msgs.append({"role": "system", "content": _TOOLS_UNAVAILABLE_NUDGE})
        final_text_parts: List[str] = []
        final_reason_parts: List[str] = []
        tool_dropped = False
        stream_error = None
        try:
            async for event in self._stream_with_autocompact(
                msgs, self.model, self.temperature,
                iteration=self.max_iterations,
                use_compressed_view=False,
                session_id=self.session_id,
                agent_type="agent:stream:final",
                **self._sampling_kwargs(final_response=True),
            ):
                if isinstance(event, TextEvent):
                    final_text_parts.append(event.content)
                    yield event
                elif isinstance(event, ReasoningEvent):
                    final_reason_parts.append(event.content)
                elif isinstance(event, ToolStatusEvent):
                    yield event
                elif isinstance(event, ToolCallDeltaEvent):
                    tool_dropped = True
        except Exception as e:
            saw_output = bool(final_text_parts or final_reason_parts)
            if not isinstance(e, StopInferenceError) and self._is_partial_stream_failure(e, saw_output):
                stream_error = e
                self._log(f"[AGENT] Final-answer stream failed after partial output "
                          f"({_format_exception(e)[:300]})")
            else:
                self.final_finish_reason = "error" if not isinstance(e, StopInferenceError) else None
                raise

        final_text = "".join(final_text_parts)
        if not final_text.strip():
            async for ev in self._ensure_final_answer(
                msgs, "".join(final_reason_parts).strip(), self.max_iterations, allow_tools=False,
            ):
                yield ev
            if tool_dropped and not self._fallback_used:
                yield TextEvent(content=_TOOL_DROPPED_NOTICE)
            if self._final_answer_cut:
                self.final_finish_reason = self._final_answer_cut
        else:
            if tool_dropped:
                yield TextEvent(content=_TOOL_DROPPED_NOTICE)
            if stream_error is not None:
                yield TextEvent(content=_STREAM_ERROR_NOTICE)
                self.final_finish_reason = "error"
            elif self._last_finish_reason == "length":
                yield TextEvent(content=_LENGTH_NOTICE)
                self.final_finish_reason = "length"
        self.final_finish_reason = self.final_finish_reason or "stop"
