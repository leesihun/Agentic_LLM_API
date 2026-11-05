"""
SubAgent Tool
Spawns a fresh-context AgentLoop (sub-agent) to handle research or multi-step
subtasks without polluting the main conversation context.

  - "explore"  subagent: read-only tools (file_reader, grep, file_navigator,
                rag, shell_lint; websearch only when not AIRGAPPED)
  - "general"  subagent: full toolset minus "agent" (no infinite recursion)

Throughput / robustness:
  - Several `agent` calls in one assistant message run concurrently (the tool is
    concurrency-safe in TOOL_METADATA). Admission is bounded by a process-wide
    semaphore (SUBAGENT_MAX_CONCURRENT) plus a per-parent-turn semaphore
    (SUBAGENT_MAX_PER_TURN). Excess calls queue; they are never rejected.
  - Children inherit the parent's workspace_dir (and temperature when the parent
    passes it), always run in normal mode (never goal mode) and use
    SUBAGENT_REASONING_EFFORT via chat_template_kwargs.
  - Only the child's FINAL iteration text is returned (head+tail capped at the
    agent tool-result budget); the full transcript is spilled to
    data/tool_results/<parent session>/subagent_<child id>.json and its absolute
    path is included so the parent can file_reader it.

  - Child tool starts are reported through an optional progress_callback (the
    parent dispatcher re-emits them on its stream as "sub:<tool>" activity).

File writes (file_edit / file_writer / apply_patch) of the parent and parallel
general subagents are serialized per resolved path by a process-wide lock
registry in backend/agent/tool_dispatch.py. shell_exec / code_exec are NOT
serialized, so the child preamble still tells subagents to touch only the files
their task owns and to avoid workspace-wide state changes.
"""
import asyncio
import inspect
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import config


_SUBAGENT_PREAMBLE = (
    "You are a subagent: a parent agent delegated the task above to you. "
    "You cannot ask the user or the parent anything — make reasonable "
    "assumptions and state them. Other subagents may run in parallel in the "
    "same workspace, so touch only the files your task needs and do not run "
    "workspace-wide state changes (git checkout/reset/stash/clean, mass "
    "deletes, killing processes you did not start) unless the task explicitly "
    "requires it. Batch independent tool calls in one message.\n"
    "Finish with a complete, self-contained final report as your last message: "
    "the answer/results with concrete evidence (file paths with line numbers, "
    "commands run and their outcomes, exact values), what you changed if "
    "anything, and what is still unresolved. Your final text message is the "
    "ONLY thing the parent sees — earlier messages and tool outputs are not "
    "forwarded — so never end on a tool call or a bare acknowledgement like "
    "\"done\" / \"완료\"."
)

_EXPLORE_NOTE = (
    "You have read-only tools only. Do not try to modify files or run commands."
)

_EXPLORE_ALIASES = {"explore", "explorer", "read", "readonly", "read-only", "read_only", "research"}
_GENERAL_ALIASES = {"general", "general-purpose", "general_purpose", "full", "worker"}

# Process-wide admission control, created lazily per event loop.
_GLOBAL_SEM: Optional[asyncio.Semaphore] = None
_GLOBAL_SEM_LOOP: Optional[asyncio.AbstractEventLoop] = None

# Cached AgentLoop constructor parameter names (None = accepts **kwargs).
_AGENTLOOP_PARAMS: Optional[set] = None
_AGENTLOOP_PARAMS_LOADED = False


def _airgapped() -> bool:
    return bool(getattr(config, "AIRGAPPED", True))


def _int_cfg(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(getattr(config, name, default)))
    except (TypeError, ValueError):
        return default


def _global_semaphore() -> asyncio.Semaphore:
    global _GLOBAL_SEM, _GLOBAL_SEM_LOOP
    loop = asyncio.get_running_loop()
    if _GLOBAL_SEM is None or _GLOBAL_SEM_LOOP is not loop:
        _GLOBAL_SEM = asyncio.Semaphore(_int_cfg("SUBAGENT_MAX_CONCURRENT", 32))
        _GLOBAL_SEM_LOOP = loop
    return _GLOBAL_SEM


def _agent_budget() -> int:
    budgets = getattr(config, "TOOL_RESULT_BUDGETS", None)
    if not isinstance(budgets, dict):
        budgets = getattr(config, "TOOL_RESULT_BUDGET", None)
    value = budgets.get("agent") if isinstance(budgets, dict) else None
    try:
        return max(2000, int(value)) if value is not None else 16000
    except (TypeError, ValueError):
        return 16000


def _subagent_template_kwargs() -> Optional[Dict[str, Any]]:
    effort = getattr(config, "SUBAGENT_REASONING_EFFORT", "high")
    if not effort:
        return None
    effort = str(effort).strip().lower()
    if effort == "none":
        return {"enable_thinking": False}
    return {"reasoning_effort": effort}


def _general_tools() -> List[str]:
    """Full toolset minus 'agent' (no recursion); websearch dropped when airgapped."""
    airgapped = _airgapped()
    return [
        t for t in config.AVAILABLE_TOOLS
        if t != "agent" and not (airgapped and t == "websearch")
    ]


def _explore_tools() -> List[str]:
    """Read-only subset of AVAILABLE_TOOLS (per TOOL_METADATA)."""
    from tools.schemas import TOOL_METADATA
    airgapped = _airgapped()
    return [
        t for t in config.AVAILABLE_TOOLS
        if t != "agent"
        and TOOL_METADATA.get(t, {}).get("is_read_only")
        and not (airgapped and t == "websearch")
    ]


def _normalize_type(value: Any) -> str:
    v = str(value or "").strip().lower()
    if v in _GENERAL_ALIASES:
        return "general"
    # Unknown values fall back to the safe read-only type.
    return "explore"


def _format_exception(exc: BaseException) -> str:
    """Type name + message; never an empty string (httpx/asyncio errors often
    stringify to '')."""
    msg = getattr(exc, "message", None) or str(exc).strip() or repr(exc)
    return f"{type(exc).__name__}: {msg}"[:1000]


def _head_tail(text: str, limit: int, head_frac: float = 0.35) -> tuple:
    """Return (text, truncated). Keeps the head and the (larger) tail."""
    if len(text) <= limit:
        return text, False
    head = max(0, int(limit * head_frac))
    tail = max(0, limit - head)
    omitted = len(text) - head - tail
    marker = f"\n\n[... {omitted} chars omitted - full text in transcript_path ...]\n\n"
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value or "")[:120] or "nosession"


def _agentloop_kwargs(loop_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop constructor kwargs the current AgentLoop does not accept, so a
    signature mismatch degrades (e.g. no per-call reasoning override) instead of
    failing every subagent with TypeError."""
    global _AGENTLOOP_PARAMS, _AGENTLOOP_PARAMS_LOADED
    if not _AGENTLOOP_PARAMS_LOADED:
        try:
            params = inspect.signature(loop_cls.__init__).parameters
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                _AGENTLOOP_PARAMS = None
            else:
                _AGENTLOOP_PARAMS = set(params)
        except (TypeError, ValueError):
            _AGENTLOOP_PARAMS = None
        _AGENTLOOP_PARAMS_LOADED = True
    if _AGENTLOOP_PARAMS is None:
        return kwargs
    dropped = [k for k in kwargs if k not in _AGENTLOOP_PARAMS]
    if dropped:
        print(f"[AGENT-TOOL] AgentLoop does not accept {dropped}; ignoring")
    return {k: v for k, v in kwargs.items() if k in _AGENTLOOP_PARAMS}


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=1, default=str)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows: target briefly held open by a reader/AV scanner.
            if attempt == 4:
                raise
            time.sleep(0.05)


class _ChildRun:
    """Mutable record of a child run, filled while streaming so a timeout or
    error still has the partial output."""

    def __init__(self) -> None:
        self.segments: List[str] = []      # text per tool-delimited segment
        self.current: List[str] = []       # text since the last tool event
        self.tool_events: List[Dict[str, Any]] = []
        self.tool_counts: Dict[str, int] = {}
        self.tool_failures: int = 0

    def add_text(self, text: str) -> None:
        if text:
            self.current.append(text)

    def boundary(self) -> None:
        seg = "".join(self.current)
        if seg.strip():
            self.segments.append(seg)
        self.current = []

    def final_text(self) -> str:
        return "".join(self.current)

    def report_text(self) -> str:
        """Final answer for the parent. A child often writes its report and
        then makes one last bookkeeping call (todo_write, memo) followed by a
        one-liner like "Done." — when the final segment is that short and the
        previous segment is longer, keep both so the report isn't lost."""
        final = self.final_text().strip()
        if not final:
            return ""
        prev = self.segments[-1].strip() if self.segments else ""
        if prev and len(final) < 500 and len(prev) > len(final):
            return f"{prev}\n\n{final}"
        return final

    def last_nonempty(self) -> str:
        final = self.final_text()
        if final.strip():
            return final
        return self.segments[-1] if self.segments else ""


class SubAgentTool:
    """Spawn a sub-AgentLoop and return its final text result."""

    def __init__(
        self,
        session_id: str = None,
        username: str = None,
        workspace_dir: Optional[str] = None,
        temperature: Optional[float] = None,
        **_ignored: Any,
    ):
        self.session_id = session_id
        self.username = username
        self.workspace_dir = str(workspace_dir) if workspace_dir else None
        self.temperature = temperature
        # Per-parent-turn cap (one SubAgentTool instance per parent AgentLoop).
        self._turn_sem: Optional[asyncio.Semaphore] = None
        self._turn_sem_loop: Optional[asyncio.AbstractEventLoop] = None

    def _turn_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._turn_sem is None or self._turn_sem_loop is not loop:
            self._turn_sem = asyncio.Semaphore(_int_cfg("SUBAGENT_MAX_PER_TURN", 16))
            self._turn_sem_loop = loop
        return self._turn_sem

    def _new_child_id(self) -> str:
        base = self.session_id or "anon"
        return f"{base}__sub_{uuid.uuid4().hex[:8]}"

    def _transcript_path(self, child_id: str) -> Path:
        root = Path(getattr(config, "TOOL_RESULTS_DIR", Path(config.DATA_DIR) / "tool_results"))
        return (root / _safe_name(self.session_id or "nosession") / f"subagent_{_safe_name(child_id)}.json").resolve()

    async def _spill(self, child_id: str, payload: Dict[str, Any]) -> Optional[str]:
        path = self._transcript_path(child_id)
        try:
            await asyncio.to_thread(_write_json_atomic, path, payload)
            return str(path)
        except Exception as exc:
            print(f"[AGENT-TOOL] Failed to write transcript {path}: {_format_exception(exc)}")
            return None

    @staticmethod
    def _stop_requested(child_id: str) -> bool:
        try:
            from backend.utils import stop_signal
            fn = getattr(stop_signal, "is_stop_requested", None)
            if fn is not None:
                return bool(fn(child_id, consume=False))
        except Exception:
            pass
        return False

    async def execute(
        self,
        prompt: str = None,
        subagent_type: str = "explore",
        description: Optional[str] = None,
        child_session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        **_ignored: Any,
    ) -> Dict[str, Any]:
        """
        Spawn a sub-agent and return its final text response.

        Args:
            prompt:           Task or question for the subagent.
            subagent_type:    "explore" (read-only) or "general" (full toolset).
            description:      Short label for logging (3-5 words).
            child_session_id: Session id for the child (parent passes
                              "{parent}__sub_{uuid8}"); generated when omitted.
            progress_callback: Optional sync callable, called with the tool
                              name each time the child starts a tool.

        Returns:
            {"success": bool, "subagent_type": str, "result": str, ...}
            On failure: "error" (+ "partial_output" when the child produced text).
        """
        stype = _normalize_type(subagent_type)
        if not isinstance(prompt, str) or not prompt.strip():
            return {
                "success": False,
                "subagent_type": stype,
                "error": "Missing required argument 'prompt' (the subagent's task).",
                "hint": "Re-issue the agent call with a self-contained 'prompt' describing the task and the expected output.",
            }

        label = description or f"{stype} subagent"
        child_id = child_session_id or self._new_child_id()
        tools = _explore_tools() if stype == "explore" else _general_tools()

        queued_at = time.monotonic()
        async with self._turn_semaphore():
            async with _global_semaphore():
                queue_wait = time.monotonic() - queued_at
                if queue_wait > 1.0:
                    print(f"[AGENT-TOOL] {label} waited {queue_wait:.1f}s for a subagent slot")
                if self._stop_requested(child_id):
                    return {
                        "success": False,
                        "subagent_type": stype,
                        "error": "Stopped: a stop was requested for this session before the subagent started.",
                    }
                return await self._run_child(
                    prompt=prompt,
                    stype=stype,
                    label=label,
                    description=description,
                    child_id=child_id,
                    tools=tools,
                    queue_wait=queue_wait,
                    progress_callback=progress_callback,
                )

    async def _run_child(
        self,
        prompt: str,
        stype: str,
        label: str,
        description: Optional[str],
        child_id: str,
        tools: List[str],
        queue_wait: float,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        print(f"\n[AGENT-TOOL] Spawning {label} ({stype}) as {child_id}")

        # Import here to avoid circular import at module load
        from backend.agent import AgentLoop
        from backend.core.llm_backend import TextEvent, ToolStatusEvent
        from tools.schemas import TOOL_METADATA

        loop_kwargs: Dict[str, Any] = {
            "session_id": child_id,
            "username": self.username,
            "tools": tools,
            "workspace_dir": self.workspace_dir,
            "mode": "normal",
        }
        if self.temperature is not None:
            loop_kwargs["temperature"] = self.temperature
        template_kwargs = _subagent_template_kwargs()
        if template_kwargs:
            loop_kwargs["chat_template_kwargs"] = template_kwargs

        preamble = _SUBAGENT_PREAMBLE
        if stype == "explore":
            preamble = f"{preamble}\n{_EXPLORE_NOTE}"
        # Task first (it is what tail goal reminders echo), instructions after.
        # Kept in the user message so the parent's byte-stable system prefix is
        # shared with the child for vLLM prefix caching.
        messages = [{
            "role": "user",
            "content": f"{prompt}\n\n<subagent_instructions>\n{preamble}\n</subagent_instructions>",
        }]

        state = _ChildRun()
        started = time.monotonic()
        timeout = _int_cfg("SUBAGENT_TIMEOUT_SECONDS", 1800)
        error: Optional[str] = None
        timed_out = False
        stopped = False
        sub_loop = None

        async def _consume() -> None:
            agen = sub_loop.run_stream(messages)
            try:
                async for event in agen:
                    if isinstance(event, TextEvent):
                        state.add_text(event.content or "")
                    elif isinstance(event, ToolStatusEvent):
                        # Any tool activity closes the current text segment, so
                        # the text after the last tool event is the final answer.
                        state.boundary()
                        name = event.tool_name or ""
                        status = event.status or ""
                        state.tool_events.append({
                            "tool": name,
                            "status": status,
                            "duration": round(float(event.duration or 0.0), 3),
                        })
                        if name in TOOL_METADATA and status in ("completed", "failed"):
                            state.tool_counts[name] = state.tool_counts.get(name, 0) + 1
                            if status == "failed":
                                state.tool_failures += 1
                        if progress_callback is not None and status == "started":
                            try:
                                progress_callback(name)
                            except Exception:
                                pass
            finally:
                await agen.aclose()

        try:
            sub_loop = AgentLoop(**_agentloop_kwargs(AgentLoop, loop_kwargs))
            await asyncio.wait_for(_consume(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            # asyncio.TimeoutError is the builtin TimeoutError on 3.11+, so a
            # timeout raised INSIDE the child (an idle-stream or tool timeout
            # that escaped) lands here too — only call it the wall-clock
            # deadline when the deadline was actually reached.
            if time.monotonic() - started >= timeout - 1:
                timed_out = True
                error = f"Subagent exceeded its {timeout}s wall-clock deadline and was cancelled"
            else:
                error = f"Subagent failed: {_format_exception(exc)}"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if type(exc).__name__ == "StopInferenceError":
                stopped = True
                error = f"Subagent stopped: {_format_exception(exc)}"
            else:
                error = f"Subagent failed: {_format_exception(exc)}"

        duration = time.monotonic() - started
        diagnostics = getattr(sub_loop, "diagnostics", None) if sub_loop is not None else None
        if not isinstance(diagnostics, dict):
            diagnostics = {}
        finish_reason = (
            getattr(sub_loop, "final_finish_reason", None) if sub_loop is not None else None
        ) or diagnostics.get("finish_reason")
        iterations = diagnostics.get("iterations")
        if iterations is None and sub_loop is not None:
            iterations = len(getattr(sub_loop, "_iteration_boundaries", []) or []) or None

        final_text = state.final_text()
        has_final = bool(final_text.strip()) and error is None
        if error is None and not has_final:
            error = (
                f"Subagent produced no final answer "
                f"(finish_reason={finish_reason or 'unknown'}, iterations={iterations or '?'}, "
                f"tool_calls={sum(state.tool_counts.values())})"
            )

        transcript_path = await self._spill(child_id, {
            "child_session_id": child_id,
            "parent_session_id": self.session_id,
            "subagent_type": stype,
            "description": description,
            "prompt": prompt,
            "success": has_final,
            "error": error,
            "timed_out": timed_out,
            "finish_reason": finish_reason,
            "iterations": iterations,
            "duration_s": round(duration, 2),
            "queue_wait_s": round(queue_wait, 2),
            "diagnostics": diagnostics,
            "tool_counts": state.tool_counts,
            "tool_events": state.tool_events,
            "text_segments": state.segments + ([final_text] if final_text.strip() else []),
            "final_text": final_text,
        })

        budget = _agent_budget()
        result: Dict[str, Any] = {"success": has_final, "subagent_type": stype}
        if description:
            result["description"] = description
        if has_final:
            text, truncated = _head_tail(state.report_text(), budget)
            result["result"] = text
            if truncated:
                result["truncated"] = True
            if finish_reason == "length":
                result["warning"] = "Subagent output hit the token limit; its report may be incomplete."
            elif diagnostics.get("fallback_used"):
                result["warning"] = (
                    "Subagent did not write its own final answer; the text is an "
                    "automatic summary of what it did. Verify before relying on it."
                )
            print(f"[AGENT-TOOL] {label} completed ({len(final_text)} chars, {duration:.1f}s)")
        else:
            result["error"] = error
            partial = state.last_nonempty().strip()
            if partial:
                result["partial_output"], _ = _head_tail(partial, max(2000, budget // 2))
            result["hint"] = (
                "A stop was requested; do not retry. Report what was done so far."
                if stopped else
                "Read transcript_path for the subagent's full output, then retry with "
                "a narrower, more specific prompt or do the step yourself."
            )
            print(f"[AGENT-TOOL] {label} failed after {duration:.1f}s: {error}")
        result["finish_reason"] = finish_reason
        result["iterations"] = iterations
        result["tool_calls"] = dict(state.tool_counts)
        if state.tool_failures:
            result["tool_failures"] = state.tool_failures
        result["duration_s"] = round(duration, 1)
        if queue_wait >= 0.5:
            result["queue_wait_s"] = round(queue_wait, 1)
        if transcript_path:
            result["transcript_path"] = transcript_path
        return result
