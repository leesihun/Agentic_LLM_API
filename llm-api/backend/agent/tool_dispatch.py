"""DispatchMixin: in-process tool execution with parallel support."""
import asyncio
import contextlib
import contextvars
import os
import re
import threading
import time
import uuid
import weakref
from typing import Callable, Iterable, List, Dict, Any, Optional

import config
from backend.core.llm_backend import ToolCall, ToolStatusEvent
from backend.agent import watchdog as _wd
try:  # K2 contract (owned by the process-lifecycle partition): used defensively.
    from backend.utils import proc_lifecycle as _proc_lifecycle
except Exception:  # pragma: no cover - module always exists in this repo
    _proc_lifecycle = None
from pathlib import Path

from tools.schemas import TOOL_METADATA
try:
    from tools.schemas import TOOL_SCHEMAS as _TOOL_SCHEMAS
except Exception:  # pragma: no cover - schemas module always defines it today
    _TOOL_SCHEMAS = {}
from tools.file_ops._postcheck import (
    check_python, is_python_path,
    check_typescript, is_typescript_path,
)
from tools.file_ops._agents_md import walk_up_for_agents_md, attach_to_result


def _format_exception(exc: BaseException) -> str:
    """`TypeName: message` — str() alone is empty for e.g. TimeoutError."""
    msg = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


# ---- Daemon-thread runner for in-process (blocking) tools -------------------
# asyncio.to_thread uses the loop's small default executor: a few hung tool
# threads (a thread cannot be killed) would saturate it and wedge EVERY later
# to_thread call in the server. Tool work therefore runs on its own daemon
# thread per call. The caller gets an asyncio.Future: cancelling / timing out
# the await abandons the thread (its late result is dropped); being a daemon it
# never blocks interpreter shutdown. The caller's contextvars (proc_lifecycle's
# call context) are propagated like asyncio.to_thread does.
def _to_tool_thread(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> "asyncio.Future":
    loop = asyncio.get_running_loop()
    fut: "asyncio.Future" = loop.create_future()
    ctx = contextvars.copy_context()

    def _deliver(result: Any, exc: Optional[BaseException]) -> None:
        if fut.done():  # cancelled / timed out: late result is dropped
            return
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    def _run() -> None:
        result: Any = None
        exc: Optional[BaseException] = None
        try:
            result = ctx.run(fn, *args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - relayed to the awaiting task
            exc = e if isinstance(e, Exception) else RuntimeError(f"{type(e).__name__}: {e}")
        try:
            loop.call_soon_threadsafe(_deliver, result, exc)
        except RuntimeError:  # loop already closed (shutdown)
            pass

    name = getattr(fn, "__name__", "tool")
    threading.Thread(target=_run, name=f"tool-{name}", daemon=True).start()
    return fut


# ---- proc_lifecycle call context (K2) — every use is defensive --------------
def _enter_call_context(session_id: Optional[str], tool_call_id: str, tool_name: str) -> Any:
    fn = getattr(_proc_lifecycle, "set_call_context", None) if _proc_lifecycle else None
    if fn is None:
        return None
    try:
        return fn(session_id, tool_call_id, tool_name)
    except Exception:
        return None


def _exit_call_context(token: Any) -> None:
    if token is None:
        return
    fn = getattr(_proc_lifecycle, "reset_call_context", None) if _proc_lifecycle else None
    if fn is None:
        return
    try:
        fn(token)
    except Exception:  # token from another context / already reset
        pass


async def _kill_call_processes(tool_call_id: Optional[str], reason: str,
                               session_id: Optional[str] = None,
                               timeout: float = 10.0) -> int:
    """Kill the process tree(s) registered for one tool call (scoped to
    *session_id* so a reused call id in another session is never touched).
    Never called without a tool_call_id (an empty filter would match every
    process). Bounded: kill_processes runs its kills on the loop's shared
    default executor, which hung threads can saturate; after *timeout* seconds
    the kill is left to finish in the background and the caller moves on."""
    fn = getattr(_proc_lifecycle, "kill_processes", None) if _proc_lifecycle else None
    if fn is None or not tool_call_id:
        return 0
    kill_task: Optional["asyncio.Future"] = None
    try:
        kill_task = asyncio.ensure_future(fn(
            session_id=session_id, tool_call_id=tool_call_id, reason=reason,
            include_children_sessions=False,
        ))
        kill_task.add_done_callback(_swallow_task_exception)
        done, _ = await asyncio.wait({kill_task}, timeout=timeout)
        if kill_task in done and not kill_task.cancelled() and kill_task.exception() is None:
            return int(kill_task.result() or 0)
        return 0
    except asyncio.CancelledError:
        raise  # the kill task keeps running on its own
    except Exception:
        return 0


def _swallow_task_exception(task: "asyncio.Future") -> None:
    """done-callback for abandoned tasks (no 'exception never retrieved')."""
    try:
        if not task.cancelled():
            task.exception()
    except Exception:
        pass


class _ToolControl:
    """Stop request channel between a tool's supervisor and its runner."""
    __slots__ = ("event", "reason", "killed")

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.reason: Optional[str] = None
        self.killed: int = 0


# Never judged by the LLM watchdog: subagents supervise themselves, and every
# process_monitor operation is a quick call (its background processes live on).
_UNSUPERVISED_TOOLS = frozenset({"agent", "process_monitor"})
# Own timeouts/caps instead of the in-process hard timeout.
_NO_HARD_TIMEOUT_TOOLS = frozenset({"shell_exec", "code_exec", "agent"})
_STOP_HINT = (
    "The call was stopped before it finished. Retry with a narrower path/pattern/scope or "
    "a shorter timeout, use shell_exec (find/ls/rg) with a timeout, or run long jobs "
    "through process_monitor."
)


def _process_owner_session(session_id: Optional[str]) -> Optional[str]:
    """Root session id owning process_monitor handles: a subagent session
    `{root}__sub_{a}__sub_{b}` maps to `{root}`; a top-level id is unchanged.
    The `ephemeral` root (parent ran without a session id) maps back to None."""
    if not session_id:
        return session_id
    from backend.utils.stop_signal import _ancestor_session_ids
    ancestors = _ancestor_session_ids(session_id)
    if not ancestors:
        return session_id
    root = ancestors[-1]
    return None if root == "ephemeral" else root


# ---- Per-file write locks ---------------------------------------------------
# Process-wide, keyed by resolved absolute path. A parent and its parallel
# subagents are separate AgentLoops sharing one workspace, so their
# file_edit / file_writer / apply_patch calls could interleave on the same
# file. Weak values: an entry lives only while a coroutine holds or awaits it.
# shell_exec / code_exec are deliberately NOT serialized (throughput).
_PATH_LOCKS: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()
_PATCH_PATH_RE = re.compile(
    r"^\s*\*\*\* (?:(?:Add|Delete|Update) File|Move to):\s*(.+?)\s*$", re.MULTILINE,
)
_FILE_MUTATING_TOOLS = frozenset({"file_edit", "file_writer", "apply_patch"})
# Thread tools whose abandoned worker thread keeps writing after a stop/timeout.
_BACKGROUND_CONTINUES_TOOLS = _FILE_MUTATING_TOOLS | frozenset({"rag"})


def _path_lock(key: str) -> asyncio.Lock:
    lock = _PATH_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _PATH_LOCKS[key] = lock
    return lock


@contextlib.asynccontextmanager
async def _hold_path_locks(keys: Iterable[str]):
    """Acquire the locks for *keys* in sorted order (no lock-order deadlock)."""
    locks = [_path_lock(k) for k in sorted(set(keys))]  # strong refs while held
    async with contextlib.AsyncExitStack() as stack:
        for lock in locks:
            await stack.enter_async_context(lock)
        yield


def _required_params(name: str) -> List[str]:
    """Required parameter names for *name* from the tool schema (empty if unknown)."""
    schema = _TOOL_SCHEMAS.get(name) or {}
    params = schema.get("parameters") or {}
    req = params.get("required") or []
    return [r for r in req if isinstance(r, str)]


class DispatchMixin:
    """Executes tool calls in-process; parallel via asyncio.gather."""

    def _tool_parameters(self, name: str) -> Dict[str, Any]:
        return config.TOOL_PARAMETERS.get(name, {})

    def _tool_timeout(
        self,
        name: str,
        arguments: Dict[str, Any],
        default: Optional[float] = None,
        hard_cap: Optional[float] = None,
    ) -> Optional[float]:
        """Resolve a tool timeout in seconds.

        The model-supplied value is coerced to a number; missing, unparseable,
        or <= 0 values (e.g. a model that learned "timeout=0 means unlimited")
        fall back to the tool default instead of killing the process
        instantly. The result is clamped to *hard_cap*. Integral values are
        returned as int so tool messages read "300s", not "300.0s".
        """
        tool_default = self._tool_parameters(name).get("timeout", default)
        try:
            tool_default = float(tool_default) if tool_default is not None else None
        except (TypeError, ValueError):
            tool_default = float(default) if default is not None else None
        if tool_default is not None and tool_default <= 0:
            tool_default = float(default) if default is not None and default > 0 else None

        raw = arguments.get("timeout") if isinstance(arguments, dict) else None
        timeout: Optional[float]
        try:
            timeout = float(raw) if raw is not None and raw != "" else None
        except (TypeError, ValueError):
            timeout = None
        if timeout is None or timeout != timeout or timeout <= 0:  # NaN / <=0 / invalid
            timeout = tool_default
        if hard_cap is not None and timeout is not None:
            try:
                cap = float(hard_cap)
                if cap > 0 and timeout > cap:
                    timeout = cap
            except (TypeError, ValueError):
                pass
        if timeout is None:
            return None
        if timeout == float("inf"):
            return float(hard_cap) if hard_cap else None
        return int(timeout) if float(timeout).is_integer() else timeout

    def _validate_tool_arguments(
        self, name: str, arguments: Any, parse_error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a structured error result when the call can't be executed as
        given (unparseable / truncated arguments, non-dict arguments, missing
        required params), or None when the call is valid. The error goes back
        to the model as the tool result so it can re-issue a correct call."""
        if parse_error:
            return {
                "success": False,
                "error": f"Tool call arguments could not be parsed: {parse_error}",
                "hint": (
                    f"Re-issue the `{name}` call with a single valid JSON object as "
                    "arguments. If the content is large, write it in smaller chunks "
                    "(file_writer mode='append') or use apply_patch."
                ),
            }
        if not isinstance(arguments, dict):
            return {
                "success": False,
                "error": (
                    f"Tool call arguments must be a JSON object, got "
                    f"{type(arguments).__name__}."
                ),
                "hint": f"Re-issue the `{name}` call with arguments as a JSON object of named parameters.",
            }
        enabled = getattr(self, "enabled_tools", None)
        if enabled is not None and name not in enabled:
            # Leaked / salvaged tool-call markup can name a tool this session
            # never offered (tools=[] or a restricted sub-agent list).
            return {
                "success": False,
                "error": f"Tool `{name}` is not available in this session.",
                "hint": (
                    "Use only the tools listed in your instructions"
                    + (f" ({', '.join(enabled)})" if enabled else "")
                    + ", or answer in text."
                ),
            }
        missing = [
            p for p in _required_params(name)
            if p not in arguments or arguments.get(p) is None
        ]
        if missing:
            return {
                "success": False,
                "error": f"Missing required parameter(s) for `{name}`: {', '.join(missing)}.",
                "hint": (
                    f"Re-issue the `{name}` call including: {', '.join(missing)}. "
                    "If the call was cut off because the content was too large, "
                    "split it into smaller chunks."
                ),
            }
        return None

    def _read_cache_key(self, name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """Per-run cache key for a read-only, concurrency-safe tool call.

        Returns None for tools that must always re-run (anything that can mutate
        state). The signature reuses AgentLoop._tool_signature so it matches the
        anti-spiral repeat detector.
        """
        meta = TOOL_METADATA.get(name, {})
        if not (meta.get("is_read_only") and meta.get("is_concurrency_safe")):
            return None
        sig = getattr(self, "_tool_signature", None)
        if sig is None:
            return None
        tool_name, arg_hash = sig(name, arguments)
        return f"{tool_name}:{arg_hash}"

    async def execute_tool(self, name: str, arguments: Dict[str, Any],
                           tool_call_id: str = None,
                           parse_error: Optional[str] = None) -> Dict[str, Any]:
        start = time.time()
        print(f"\n{'='*70}")
        print(f"[TOOL] Executing: {name}")
        print(f"{'='*70}")
        if isinstance(arguments, dict):
            for k, v in arguments.items():
                sv = str(v)
                print(f"  {k}: {sv[:150]}{'...' if len(sv) > 150 else ''}")
        else:
            print(f"  (non-dict arguments: {type(arguments).__name__})")

        # ---- Argument validation (C2): a call whose arguments failed to parse
        # (truncated / malformed JSON), non-dict arguments, or missing required
        # params is never executed — a structured error goes back to the model.
        invalid = None
        try:
            invalid = self._validate_tool_arguments(name, arguments, parse_error)
        except Exception as e:  # validation must never crash the turn
            invalid = {"success": False, "error": f"Argument validation failed: {_format_exception(e)}"}
        if invalid is not None:
            duration = time.time() - start
            print(f"[TOOL] {name} rejected — {invalid.get('error')}")
            self.tool_calls_log.append({
                "name": name, "input": arguments if isinstance(arguments, dict) else {},
                "tool_call_id": tool_call_id,
                "success": False, "error": invalid.get("error"), "duration": duration,
            })
            if len(self.tool_calls_log) > 200:
                self.tool_calls_log = self.tool_calls_log[-200:]
            self._log_tool_result(name, tool_call_id, invalid, duration)
            return invalid

        # ---- Duplicate-call hard-stop: serve an identical earlier read from the
        # per-run cache instead of re-running it. Turns the anti-spiral *reminder*
        # into enforcement and reclaims wasted iterations. The cache is cleared
        # whenever a mutating tool runs (see below), so reads can't go stale.
        cache: Optional[Dict[str, Any]] = getattr(self, "_read_result_cache", None)
        try:
            cache_key = self._read_cache_key(name, arguments) if cache is not None else None
        except Exception:
            cache_key = None
        if cache_key is not None and cache_key in cache:
            duration = time.time() - start
            cached = dict(cache[cache_key])
            cached["cached"] = True
            cached["cache_note"] = (
                "Identical call already ran this turn — reusing the prior result "
                "instead of repeating it. If you need different information, change "
                "the arguments or use a different tool/approach."
            )
            print(f"[TOOL] {name} served from per-run cache (duplicate call)")
            self.tool_calls_log.append({
                "name": name, "input": arguments, "tool_call_id": tool_call_id,
                "success": cached.get("success", True), "duration": duration, "cached": True,
            })
            if len(self.tool_calls_log) > 200:
                self.tool_calls_log = self.tool_calls_log[-200:]
            self._log_tool_result(name, tool_call_id, cached, duration)
            return cached

        try:
            result = await self._run_guarded_tool(name, arguments, tool_call_id=tool_call_id)
            if not isinstance(result, dict):
                # Tools must return dicts; wrap anything else so downstream
                # formatting (result.get(...)) can't crash the turn.
                result = {"success": result is not None, "output": result}
            duration = time.time() - start
            print(f"[TOOL] {name} completed in {duration:.2f}s — success={result.get('success', '?')}")
            self.tool_calls_log.append({
                "name": name, "input": arguments,
                "tool_call_id": tool_call_id,
                "success": result.get("success", True), "duration": duration,
            })
            if len(self.tool_calls_log) > 200:
                self.tool_calls_log = self.tool_calls_log[-200:]
            self._log_tool_result(name, tool_call_id, result, duration)
            # Post-processing is best-effort: a failure here must not turn a
            # tool that already ran (possibly mutating files) into an error.
            try:
                self._track_temp_files(name, result)
                await self._run_postchecks(name, result)
                self._inject_subtree_agents_md(name, arguments, result)
            except asyncio.CancelledError:
                raise
            except Exception as post_exc:
                print(f"[TOOL] {name} post-processing failed: {_format_exception(post_exc)}")
            # Invalidate cached reads after anything that can mutate state, then
            # memoize the fully-formed result of a successful read.
            if cache is not None:
                if not TOOL_METADATA.get(name, {}).get("is_read_only"):
                    cache.clear()
                elif cache_key is not None and result.get("success", True):
                    cache[cache_key] = result
            return result
        except Exception as e:
            duration = time.time() - start
            err_text = _format_exception(e)
            print(f"[TOOL] {name} FAILED in {duration:.2f}s — {err_text}")
            err_result = {"success": False, "error": err_text}
            if isinstance(e, KeyError):
                err_result["hint"] = (
                    f"A required parameter ({e}) was missing. Re-issue the `{name}` "
                    "call with all required parameters."
                )
            self.tool_calls_log.append({
                "name": name, "input": arguments,
                "tool_call_id": tool_call_id,
                "success": False, "error": err_text, "duration": duration,
            })
            if len(self.tool_calls_log) > 200:
                self.tool_calls_log = self.tool_calls_log[-200:]
            self._log_tool_result(name, tool_call_id, err_result, duration)
            return err_result

    # ------------------------------------------------------------------
    # Supervised execution: call context, registry, hard timeout, watchdog
    # ------------------------------------------------------------------

    def _child_session_map(self) -> Dict[str, str]:
        m = getattr(self, "_agent_child_sessions", None)
        if m is None:
            m = {}
            self._agent_child_sessions = m
        return m

    def _hard_timeout_seconds(self, name: str) -> Optional[float]:
        """Wall-clock cap for in-process (thread) tools; None = the tool has
        its own timeout (shell_exec / code_exec / agent)."""
        if name in _NO_HARD_TIMEOUT_TOOLS:
            return None
        per_tool = getattr(config, "TOOL_HARD_TIMEOUTS", None) or {}
        raw = per_tool.get(name, getattr(config, "TOOL_DEFAULT_HARD_TIMEOUT_SECONDS", 180))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _mark_stopped(result: Any, reason: str) -> Dict[str, Any]:
        """Turn a (partial) tool result into the 'stopped by supervisor' result
        fed back to the model: success False, the reason, and partial output."""
        res: Dict[str, Any] = dict(result) if isinstance(result, dict) else {"output": result}
        parts: List[str] = []
        for key, cap in (("stdout", 600), ("output", 600), ("stderr", 300)):
            v = res.get(key)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip()[-cap:])
                if key in ("stdout", "output"):
                    break  # `output` duplicates stdout for some tools
        msg = f"stopped by supervisor: {reason}"
        if parts:
            msg += "; partial output: " + " | ".join(parts)
        res["success"] = False
        res["error"] = msg
        res["stopped_reason"] = reason
        res.setdefault("hint", _STOP_HINT)
        return res

    async def _abandon_task(self, task: "asyncio.Future", wait: float = 5.0) -> None:
        """Cancel *task* and give its cancellation cleanup a bounded time. A
        thread-backed await is abandoned (the thread keeps running but its
        result is dropped)."""
        if task is None or task.done():
            return
        task.cancel()
        task.add_done_callback(_swallow_task_exception)
        with contextlib.suppress(Exception):
            await asyncio.wait({task}, timeout=wait)

    async def _run_guarded_tool(self, name: str, arguments: Dict[str, Any],
                                tool_call_id: Optional[str] = None) -> Dict[str, Any]:
        """_dispatch_tool under the live-visibility + backup system:

        - tags child processes with (session, call id, tool) via proc_lifecycle's
          call context and registers the call in the watchdog registry;
        - enforces the hard wall-clock cap of in-process tools (structured error;
          the worker thread is abandoned, never awaited forever);
        - runs the supervisor (tool watchdog) as a background task that may stop
          the call; a stopped call returns 'stopped by supervisor: <reason>;
          partial output: ...' so the agent can adapt;
        - is cancellation-safe: on cancel the tool task is cancelled AND awaited
          (bounded) so process-tree kills finish before the turn lease is freed.
        """
        cid = tool_call_id or f"call_{uuid.uuid4().hex[:8]}"
        detail = _wd.describe_call(name, arguments)
        hard = self._hard_timeout_seconds(name)
        ctl = _ToolControl()
        started = time.monotonic()
        token = _enter_call_context(self.session_id, cid, name)
        _wd.register_tool(self.session_id, cid, name, detail)
        inner: Optional[asyncio.Future] = None
        sup: Optional[asyncio.Future] = None
        stop_waiter: Optional[asyncio.Future] = None
        try:
            inner = asyncio.ensure_future(self._dispatch_tool(name, arguments, tool_call_id=tool_call_id))
            stop_waiter = asyncio.ensure_future(ctl.event.wait())
            if (
                name not in _UNSUPERVISED_TOOLS
                and getattr(config, "WATCHDOG_ENABLED", True)
            ):
                sup = asyncio.ensure_future(
                    self._supervise_tool(ctl, name, cid, detail, started, arguments)
                )
            done, _ = await asyncio.wait(
                {inner, stop_waiter}, timeout=hard, return_when=asyncio.FIRST_COMPLETED,
            )
            if inner in done:
                result = inner.result()
                if ctl.reason and isinstance(result, dict) and not result.get("success", True):
                    result = self._mark_stopped(result, ctl.reason)
                return result
            if stop_waiter in done:
                reason = ctl.reason or "stopped"
                if ctl.killed:
                    # The kill makes run_process return promptly with partial output.
                    with contextlib.suppress(Exception):
                        await asyncio.wait({inner}, timeout=10)
                    if inner.done() and not inner.cancelled() and inner.exception() is None:
                        return self._mark_stopped(inner.result(), reason)
                await self._abandon_task(inner)
                await _kill_call_processes(cid, f"stopped by supervisor: {reason}", self.session_id)
                stopped = self._mark_stopped({}, reason)
                if name in _BACKGROUND_CONTINUES_TOOLS:
                    # The worker thread cannot be killed: tell the model so it does
                    # not blindly re-issue the same write / index job.
                    stopped["error"] += (
                        " The call may still complete in the background; verify the "
                        "current state before repeating it."
                    )
                return stopped
            # Hard timeout (hard is not None here): stop waiting, never hang the agent.
            await self._abandon_task(inner)
            await _kill_call_processes(cid, "tool timed out", self.session_id)
            limit = int(hard) if float(hard).is_integer() else hard
            note = (
                " The call may still complete in the background."
                if name in _BACKGROUND_CONTINUES_TOOLS else ""
            )
            reason = (
                f"{name} timed out after {limit}s (still running in a background thread "
                f"and was abandoned)"
            )
            return {
                "success": False,
                "error": reason + "." + note,
                "stopped_reason": f"timed out after {limit}s",
                "hint": "narrow the path/pattern, or use shell_exec (find/ls/rg) with a timeout",
            }
        except BaseException:
            # Cancelled (stop / disconnect / deadline) or crashed: cancel the tool
            # and wait (bounded) for its cleanup, e.g. process-tree kills.
            if inner is not None and not inner.done():
                inner.cancel()
                inner.add_done_callback(_swallow_task_exception)
                with contextlib.suppress(BaseException):
                    await asyncio.wait({inner}, timeout=12)
            raise
        finally:
            pending = [t for t in (sup, stop_waiter) if t is not None]
            for t in pending:
                if not t.done():
                    t.cancel()
                t.add_done_callback(_swallow_task_exception)
            _wd.unregister_tool(cid, self.session_id)
            _exit_call_context(token)
            if tool_call_id:
                self._child_session_map().pop(tool_call_id, None)

    # ---- Tool watchdog (supervisor) -------------------------------------

    def _request_headline(self, cap: int = 300) -> str:
        """First chars of the user's latest request, for supervisor prompts."""
        msgs = getattr(self, "_msgs_ref", None)
        finder = getattr(self, "_find_last_user_message", None)
        if not msgs or finder is None:
            return ""
        try:
            found = finder(msgs)
        except Exception:
            return ""
        if not found:
            return ""
        raw = found[1] or ""

        def _clean(s: str) -> str:
            s = re.sub(r"<[^>]{1,200}>", " ", s)
            return re.sub(r"\s+", " ", s).strip()

        def _clip(s: str, n: int) -> str:
            return s[:n] + ("..." if len(s) > n else "")

        # A merged / preempted turn arrives as labelled <message> blocks, oldest
        # first, with the interrupted earlier request(s) flagged. The goal is the
        # NEWEST request, so it leads the headline; the interrupted one follows.
        blocks = re.findall(r"<message\b([^>]*)>(.*?)</message>", raw, flags=re.DOTALL)
        if blocks:
            latest = [_clean(b) for a, b in blocks if "interrupted-earlier-request" not in a]
            earlier = [_clean(b) for a, b in blocks if "interrupted-earlier-request" in a]
            latest_txt = _clean(" / ".join(t for t in latest if t))
            earlier_txt = _clean(" / ".join(t for t in earlier if t))
            if latest_txt and earlier_txt:
                head = _clip(latest_txt, max(80, cap * 2 // 3))
                tail_cap = max(40, cap - len(head))
                return f"{head} (earlier, interrupted request: {_clip(earlier_txt, tail_cap)})"
            return _clip(latest_txt or earlier_txt, cap)
        text = re.sub(r"^\s*\[Room:[^\]]*\]", " ", raw)
        return _clip(_clean(text), cap)

    async def _supervisor_llm_text(self, system: str, user: str, agent_type: str) -> Optional[str]:
        """One separate, tool-less, non-thinking supervisor LLM call. Returns the
        reply text, or None on ANY failure / timeout (callers fail open).
        Re-raises CancelledError so a finished tool / stopped turn aborts it."""
        timeout = float(getattr(config, "WATCHDOG_LLM_TIMEOUT_SECONDS", 90))
        kwargs = dict(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=self.model,
            temperature=0.0,
            session_id=self.session_id,
            agent_type=agent_type,
            max_tokens=int(getattr(config, "WATCHDOG_MAX_TOKENS", 300)),
        )
        try:
            try:
                coro = self.llm.chat(chat_template_kwargs={"enable_thinking": False}, **kwargs)
            except TypeError as te:
                if "chat_template_kwargs" not in str(te):
                    raise
                coro = self.llm.chat(**kwargs)
            resp = await asyncio.wait_for(coro, timeout=timeout)
            return (getattr(resp, "content", None) or "").strip() or None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._log(f"  [SUPERVISOR] {agent_type} call failed ({type(e).__name__}: {str(e)[:160]})")
            return None

    async def _watchdog_stop(self, ctl: "_ToolControl", cid: str, name: str, reason: str) -> None:
        """Record the stop reason FIRST (so the runner labels the result), kill
        the call's processes, then wake the runner."""
        ctl.reason = reason
        try:
            self._watchdog_stops = getattr(self, "_watchdog_stops", 0) + 1
        except Exception:
            pass
        self._log(f"  [WATCHDOG] stopping {name} ({cid}): {reason}")
        # Wake the runner no matter how the kill goes: _kill_call_processes is
        # bounded, and the event is set in `finally` so a slow/failed kill can
        # never leave the call stuck as "running" (the runner abandons it).
        try:
            ctl.killed = await _kill_call_processes(
                cid, f"stopped by supervisor: {reason}", self.session_id,
            )
        finally:
            ctl.event.set()

    @staticmethod
    def _explicit_timeout(name: str, arguments: Any) -> Optional[float]:
        """The `timeout` the model itself passed to shell_exec / code_exec
        (None = it left the default). An explicit timeout is the model's own
        bound and is respected by the supervisor."""
        if name not in ("shell_exec", "code_exec") or not isinstance(arguments, dict):
            return None
        raw = arguments.get("timeout")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _declared_timeout(name: str, explicit: Optional[float]) -> Optional[float]:
        """Seconds after which the tool kills itself (explicit or default)."""
        if explicit is not None:
            return explicit
        key = {"shell_exec": "SHELL_DEFAULT_TIMEOUT_SECONDS",
               "code_exec": "CODE_EXEC_DEFAULT_TIMEOUT_SECONDS"}.get(name)
        if key is None:
            return None
        try:
            return float(getattr(config, key, 300))
        except (TypeError, ValueError):
            return None

    async def _supervise_tool(self, ctl: "_ToolControl", name: str, cid: str,
                              detail: str, started: float,
                              arguments: Any = None) -> None:
        """Background supervisor of ONE tool call (cancelled when the call ends).

        Only calls with a live OS process are judged (an in-process thread tool
        has no output signal and its own hard timeout). Deterministic rule (no
        LLM): a foreground process silent for WATCHDOG_STALL_SECONDS is stopped,
        unless the model passed an explicit `timeout` (its own bound stands: a
        silent `python train.py > log` with timeout=3600 is legitimate). After
        WATCHDOG_AFTER_SECONDS (then every WATCHDOG_INTERVAL_SECONDS) the
        supervisor LLM is asked continue/stop, but only while the process has also
        been silent for WATCHDOG_INTERVAL_SECONDS (a job that is printing is never
        judged). Every failure means continue."""
        try:
            after = max(1.0, float(getattr(config, "WATCHDOG_AFTER_SECONDS", 180)))
            interval = max(5.0, float(getattr(config, "WATCHDOG_INTERVAL_SECONDS", 120)))
            stall = max(1.0, float(getattr(config, "WATCHDOG_STALL_SECONDS", 600)))
            step = max(5.0, min(interval, 60.0))
            explicit = self._explicit_timeout(name, arguments)
            declared = self._declared_timeout(name, explicit)
            checks = 0
            last_llm: Optional[float] = None
            await asyncio.sleep(min(after, stall))
            while not ctl.event.is_set():
                now = time.monotonic()
                elapsed = now - started
                procs = _wd.live_processes(
                    tool_call_id=cid, session_id=self.session_id, include_children=False,
                )
                fg = [
                    p for p in procs
                    if isinstance(p, dict) and p.get("alive") is not False
                    and p.get("tool") != "process_monitor"
                ]
                dead = _wd.stalled_process(fg, stall) if explicit is None else None
                if dead is not None:
                    idle = int(float(dead.get("idle_seconds") or dead.get("idle") or 0))
                    await self._watchdog_stop(
                        ctl, cid, name,
                        f"no output for {idle}s (process pid {dead.get('pid')} looks stalled)",
                    )
                    return
                min_idle = min(
                    (float(p.get("idle_seconds") or 0.0) for p in fg), default=0.0,
                )
                if (
                    fg
                    and min_idle >= interval
                    and elapsed >= after
                    and (last_llm is None or now - last_llm >= interval)
                ):
                    checks += 1
                    last_llm = now
                    reply = await self._supervisor_llm_text(
                        _wd.TOOL_REVIEW_SYSTEM,
                        _wd.build_tool_review_prompt(
                            name, detail, elapsed, checks, self._request_headline(),
                            procs, declared_timeout=declared,
                            explicit_timeout=explicit is not None,
                        ),
                        "agent:watchdog",
                    )
                    action, reason = _wd.parse_tool_verdict(reply)
                    if action == "stop":
                        # The verdict can be ~90s old: if the process printed
                        # something since the snapshot, it is alive - keep waiting.
                        fresh = [
                            p for p in _wd.live_processes(
                                tool_call_id=cid, session_id=self.session_id,
                                include_children=False,
                            )
                            if isinstance(p, dict) and p.get("alive") is not False
                            and p.get("tool") != "process_monitor"
                        ]
                        if fresh and min(
                            float(p.get("idle_seconds") or 0.0) for p in fresh
                        ) < min(30.0, interval):
                            action = "continue"
                    if action == "stop":
                        await self._watchdog_stop(ctl, cid, name, reason or "judged stuck by the supervisor")
                        return
                await asyncio.sleep(step)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # supervisor bugs must never touch the tool
            self._log(f"  [WATCHDOG] supervisor for {name} crashed ({type(e).__name__}: {e}) - ignoring")

    def _make_progress_event(self, tc: ToolCall) -> ToolStatusEvent:
        """A status="progress" event for a still-running tool call (K1)."""
        name = tc.function.name
        cid = tc.id
        now = time.monotonic()
        launch = (getattr(self, "_tool_launch_times", None) or {}).get(cid, now)
        meta = TOOL_METADATA.get(name, {})
        procs: List[Dict[str, Any]] = []
        try:
            if name == "agent":
                child = self._child_session_map().get(cid)
                if child:
                    procs = _wd.live_processes(session_id=child, include_children=True)
            elif cid:
                procs = _wd.live_processes(
                    tool_call_id=cid, session_id=self.session_id, include_children=False,
                )
        except Exception:
            procs = []
        return ToolStatusEvent(
            tool_name=name,
            tool_call_id=cid,
            status="progress",
            activity=meta.get("activity", ""),
            user_name=meta.get("user_name", ""),
            detail=_wd.describe_call(name, tc.function.arguments),
            elapsed=round(max(0.0, now - launch), 1),
            processes=_wd.trim_processes(procs),
        )

    def _track_temp_files(self, name: str, result: Dict[str, Any]) -> None:
        """Register newly-created files (not flagged persist=True) for cleanup
        at end of run_stream. file_writer and apply_patch are the only tools
        that can create new files; file_edit only modifies existing ones."""
        if not result.get("success"):
            return
        tracker = getattr(self, "_tracked_new_files", None)
        if tracker is None:
            return
        if name == "file_writer":
            if result.get("new_file") and not result.get("persist"):
                path = result.get("path")
                if path:
                    tracker.add(path)
        elif name == "apply_patch":
            if result.get("persist"):
                return
            for change in result.get("files_changed", []) or []:
                if change.get("op") == "added":
                    path = change.get("path")
                    if path:
                        tracker.add(path)

    async def _run_postchecks(self, name: str, result: Dict[str, Any]) -> None:
        """Run lightweight post-write verification (Python syntax + TS typecheck).

        Attaches a `post_edit_check` field to *result* in-place so the model
        sees the verification outcome in the next turn. Never raises - check
        failures become structured data the model can react to.
        """
        if not result.get("success"):
            return
        py_paths: List[str] = []
        ts_paths: List[str] = []
        if name in ("file_writer", "file_edit"):
            path = result.get("path")
            if is_python_path(path):
                py_paths.append(path)
            elif is_typescript_path(path):
                ts_paths.append(path)
        elif name == "apply_patch":
            for change in result.get("files_changed", []) or []:
                if change.get("op") == "deleted":
                    continue
                path = change.get("path")
                if is_python_path(path):
                    py_paths.append(path)
                elif is_typescript_path(path):
                    ts_paths.append(path)
        if not py_paths and not ts_paths:
            return

        failures: List[Dict[str, Any]] = []
        all_paths: List[str] = py_paths + ts_paths

        if py_paths:
            outcomes = await asyncio.gather(*(check_python(p) for p in py_paths))
            failures.extend(
                {"path": p, "error": o.get("error", "")}
                for p, o in zip(py_paths, outcomes)
                if o.get("status") == "failed"
            )
        if ts_paths:
            # Typecheck is project-wide — one call covers all changed TS files.
            ts_outcome = await check_typescript(ts_paths[0])
            if ts_outcome.get("status") == "failed":
                failures.append({
                    "path": ", ".join(ts_paths),
                    "error": ts_outcome.get("error", ""),
                })

        if failures:
            result["post_edit_check"] = {
                "status": "failed",
                "failures": failures,
                "hint": "Fix the error before continuing. Re-read the file with file_reader if needed.",
            }
        else:
            result["post_edit_check"] = {"status": "passed", "files": all_paths}

    def _inject_subtree_agents_md(
        self, name: str, arguments: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """When the agent reads into a subtree, surface that subtree's
        AGENTS.md / CLAUDE.md (if any, and not already seen this session).

        Only fires for the discovery tools - reader/navigator/grep. Walks
        parents of the accessed path up to the workspace root, capped by
        the per-session seen set on the agent loop instance.
        """
        if name not in ("file_reader", "file_navigator", "grep"):
            return
        if not result.get("success"):
            return

        accessed_raw = (
            result.get("path")
            or result.get("root")
            or arguments.get("path")
        )
        if not accessed_raw:
            return

        try:
            accessed = Path(accessed_raw)
        except Exception:
            return

        workspace_root = self.workspace_dir or getattr(
            config, "AGENT_DEFAULT_WORKSPACE", None
        )
        if not workspace_root:
            return

        seen = getattr(self, "_agents_md_seen", None)
        if seen is None:
            return

        try:
            discovered = walk_up_for_agents_md(accessed, Path(workspace_root), seen)
        except Exception as exc:
            self._log(f"  [AGENTS.md] walk-up failed for {accessed}: {exc}")
            return

        attach_to_result(result, discovered)

    def _workspace_str(self) -> Optional[str]:
        """self.workspace_dir is a Path (or None); tool constructors take str."""
        ws = getattr(self, "workspace_dir", None)
        return str(ws) if ws else None

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        """Lenient int coercion for model-supplied numeric args ("200", 200.0)."""
        if value is None or value == "":
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _write_lock_keys(self, name: str, tool: Any, arguments: Dict[str, Any]) -> List[str]:
        """Resolved absolute target paths of a file-mutating call, using the
        tool's own resolver so parent and subagents (same workspace/username)
        map the same argument to the same key. Unresolvable paths (the tool
        will reject them anyway) are skipped."""
        if name == "apply_patch":
            raws = [m.group(1) for m in _PATCH_PATH_RE.finditer(str(arguments.get("patch") or ""))]
        else:
            raws = [arguments.get("path")]
        keys: List[str] = []
        for raw in raws:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                if name == "file_writer":
                    resolved = tool._resolve_target(raw)
                elif name == "file_edit":
                    resolved = tool._resolve_path(raw)
                else:
                    resolved = tool._resolve_path(raw, must_exist=False)
            except Exception:
                continue
            keys.append(os.path.normcase(str(resolved)))
        return keys

    async def _locked_file_write(self, name: str, tool: Any, arguments: Dict[str, Any],
                                 fn: Callable[..., Any], **kwargs: Any) -> Any:
        """Run a file-mutating tool call in a thread while holding the
        process-wide locks for every file it targets."""
        try:
            keys = await _to_tool_thread(self._write_lock_keys, name, tool, arguments)
        except Exception:
            keys = []
        async with _hold_path_locks(keys):
            fut = _to_tool_thread(fn, **kwargs)
            try:
                return await asyncio.shield(fut)
            except asyncio.CancelledError:
                # The worker thread can't be interrupted: keep the lock until
                # its write lands so a sibling never sees a half-written file.
                with contextlib.suppress(BaseException):
                    await asyncio.wait({fut}, timeout=30)
                raise

    def _subagent_progress_callback(self, tool_call_id: Optional[str]) -> Optional[Callable[[str], None]]:
        """Callback for SubAgentTool: each child tool start is re-emitted on
        this loop's stream as a "started" ToolStatusEvent for the parent's
        `agent` call (same tool_call_id, activity "sub:<tool>"). Started-only,
        so clients never count child tools as completed parent tools."""
        emit = getattr(self, "_emit_child_progress", None)
        if emit is None or not tool_call_id:
            return None
        meta = TOOL_METADATA.get("agent", {})

        def _progress(child_tool: str) -> None:
            emit(ToolStatusEvent(
                tool_name="agent",
                tool_call_id=tool_call_id,
                status="started",
                activity=f"sub:{child_tool}",
                user_name=meta.get("user_name", ""),
            ))

        return _progress

    async def _dispatch_tool(self, name: str, arguments: Dict[str, Any], tool_call_id: str = None) -> Dict[str, Any]:
        cache = self._tool_cache

        if name == "websearch":
            if "websearch" not in cache:
                from tools.web_search import WebSearchTool
                cache["websearch"] = WebSearchTool()
            return await _to_tool_thread(
                cache["websearch"].search,
                query=arguments["query"],
                max_results=arguments.get("max_results"),
            )

        elif name == "code_exec":
            if "code_exec" not in cache:
                from tools.code_exec import CodeExecTool
                cache["code_exec"] = CodeExecTool(
                    session_id=self.session_id,
                    workspace_dir=self._workspace_str(),
                )
            return await cache["code_exec"].execute(
                code=arguments["code"],
                timeout=self._tool_timeout(
                    "code_exec",
                    arguments,
                    getattr(config, "CODE_EXEC_DEFAULT_TIMEOUT_SECONDS", 300),
                    hard_cap=getattr(config, "CODE_EXEC_HARD_CAP_SECONDS", 1800),
                ),
            )

        elif name == "rag":
            if "rag" not in cache:
                from tools.rag import RAGTool
                cache["rag"] = RAGTool(username=self.username)
            available_collections = self._get_available_rag_collections()
            requested_collection = arguments.get("collection_name")

            if not available_collections:
                return {
                    "success": False,
                    "error": "No RAG collections are available for this user. Create a collection first.",
                    "available_collections": [],
                }

            if requested_collection not in available_collections:
                return {
                    "success": False,
                    "error": (
                        f"Invalid collection_name '{requested_collection}'. "
                        "Use one of the available collections."
                    ),
                    "available_collections": available_collections,
                }

            return await _to_tool_thread(
                cache["rag"].retrieve,
                collection_name=requested_collection,
                query=arguments["query"],
                max_results=arguments.get("max_results"),
            )

        elif name == "file_reader":
            if "file_reader" not in cache:
                from tools.file_ops import FileReaderTool
                cache["file_reader"] = FileReaderTool(
                    username=self.username, session_id=self.session_id,
                    workspace_dir=self.workspace_dir,
                )
            return await _to_tool_thread(
                cache["file_reader"].read,
                path=arguments["path"],
                offset=arguments.get("offset"),
                limit=arguments.get("limit"),
            )

        elif name == "file_writer":
            if "file_writer" not in cache:
                from tools.file_ops import FileWriterTool
                cache["file_writer"] = FileWriterTool(
                    session_id=self.session_id,
                    workspace_dir=self.workspace_dir,
                )
            return await self._locked_file_write(
                name, cache["file_writer"], arguments,
                cache["file_writer"].write,
                path=arguments["path"],
                content=arguments["content"],
                mode=arguments.get("mode", "write"),
                persist=bool(arguments.get("persist", True)),
            )

        elif name == "apply_patch":
            if "apply_patch" not in cache:
                from tools.file_ops import ApplyPatchTool
                cache["apply_patch"] = ApplyPatchTool(
                    session_id=self.session_id, username=self.username,
                    workspace_dir=self.workspace_dir,
                )
            return await self._locked_file_write(
                name, cache["apply_patch"], arguments,
                cache["apply_patch"].apply,
                patch=arguments["patch"],
                persist=bool(arguments.get("persist", True)),
            )

        elif name == "shell_lint":
            if "shell_lint" not in cache:
                from tools.shell.lint import ShellLintTool
                cache["shell_lint"] = ShellLintTool(workspace_dir=self._workspace_str())
            return await _to_tool_thread(
                cache["shell_lint"].lint,
                path=arguments["path"],
            )

        elif name == "file_navigator":
            if "file_navigator" not in cache:
                from tools.file_ops import FileNavigatorTool
                cache["file_navigator"] = FileNavigatorTool(
                    username=self.username, session_id=self.session_id,
                    workspace_dir=self.workspace_dir,
                )
            return await _to_tool_thread(
                cache["file_navigator"].navigate,
                operation=arguments["operation"],
                path=arguments.get("path"),
                pattern=arguments.get("pattern"),
            )

        elif name == "shell_exec":
            if "shell_exec" not in cache:
                from tools.shell import ShellExecTool
                cache["shell_exec"] = ShellExecTool(
                    session_id=self.session_id,
                    workspace_dir=self.workspace_dir,
                )
            desc = arguments.get("description")
            if desc:
                print(f"  [shell_exec] {desc}")
            return await cache["shell_exec"].execute(
                command=arguments["command"],
                timeout=self._tool_timeout(
                    "shell_exec",
                    arguments,
                    getattr(config, "SHELL_DEFAULT_TIMEOUT_SECONDS", 300),
                    hard_cap=getattr(config, "SHELL_EXEC_HARD_CAP_SECONDS", 3600),
                ),
                working_directory=arguments.get("working_directory"),
            )

        elif name == "memo":
            if "memo" not in cache:
                from tools.memo import MemoTool
                cache["memo"] = MemoTool(username=self.username)
            return await _to_tool_thread(
                cache["memo"].execute,
                operation=arguments["operation"],
                key=arguments.get("key"),
                value=arguments.get("value"),
            )

        elif name == "process_monitor":
            if "process_monitor" not in cache:
                from tools.process_monitor import ProcessMonitorTool
                # Background processes belong to the ROOT session: a subagent
                # runs under a throwaway `{parent}__sub_{uuid8}` id, so a
                # process it started would otherwise be invisible to (and
                # unkillable by) the parent once the subagent returns.
                cache["process_monitor"] = ProcessMonitorTool(
                    session_id=_process_owner_session(self.session_id),
                    workspace_dir=self._workspace_str(),
                )
            # The model routinely omits max_lines/stream; passing None through
            # as an explicit kwarg defeats the tool's own defaults (stream=None
            # returned no output at all), so default them here.
            return await _to_tool_thread(
                cache["process_monitor"].execute,
                operation=arguments["operation"],
                command=arguments.get("command"),
                handle=arguments.get("handle"),
                working_directory=arguments.get("working_directory"),
                # offset=None is meaningful (tail of the buffer) — keep it.
                offset=(
                    None if arguments.get("offset") in (None, "")
                    else self._coerce_int(arguments.get("offset"), 0)
                ),
                max_lines=self._coerce_int(arguments.get("max_lines"), 200) or 200,
                stream=arguments.get("stream") or "both",
            )

        elif name == "file_edit":
            if "file_edit" not in cache:
                from tools.file_ops import FileEditorTool
                cache["file_edit"] = FileEditorTool(
                    session_id=self.session_id, username=self.username,
                    workspace_dir=self.workspace_dir,
                )
            return await self._locked_file_write(
                name, cache["file_edit"], arguments,
                cache["file_edit"].edit,
                path=arguments["path"],
                old_string=arguments["old_string"],
                new_string=arguments["new_string"],
                replace_all=arguments.get("replace_all", False),
            )

        elif name == "grep":
            if "grep" not in cache:
                from tools.grep import GrepTool
                cache["grep"] = GrepTool(
                    username=self.username, session_id=self.session_id,
                    workspace_dir=self.workspace_dir,
                )
            return await _to_tool_thread(
                cache["grep"].search,
                pattern=arguments["pattern"],
                path=arguments.get("path"),
                glob=arguments.get("glob"),
                output_mode=arguments.get("output_mode", "files_with_matches"),
                context=arguments.get("context", arguments.get("-C", 0)),
                before=arguments.get("-B", 0),
                after=arguments.get("-A", 0),
                case_insensitive=arguments.get("-i", False),
                file_type=arguments.get("type"),
                head_limit=arguments.get("head_limit", 250),
                offset=arguments.get("offset", 0),
                multiline=arguments.get("multiline", False),
            )

        elif name == "todo_write":
            if "todo_write" not in cache:
                from tools.todo import TodoTool
                cache["todo_write"] = TodoTool()
            result = await _to_tool_thread(
                cache["todo_write"].write,
                todos=arguments.get("todos", []),
            )
            if result.get("success"):
                self._session_todos = result["todos"]
            return result

        elif name == "agent":
            if "agent" not in cache:
                from tools.agent import SubAgentTool
                cache["agent"] = SubAgentTool(
                    session_id=self.session_id,
                    username=self.username,
                    workspace_dir=self._workspace_str(),
                    temperature=getattr(self, "temperature", None),
                )
            # Derive a unique session_id per sub-agent invocation so parallel
            # children never share state/log files. Format `{parent}__sub_{uuid8}`
            # (no "::" — unsafe in Windows filenames); the stop-signal helper
            # maps it back to the parent prefix before `__sub_`.
            child_session_id = f"{self.session_id or 'ephemeral'}__sub_{uuid.uuid4().hex[:8]}"
            if tool_call_id:  # lets progress events show the child's live processes
                self._child_session_map()[tool_call_id] = child_session_id
            # SubAgentTool.execute is async — await directly (not to_thread)
            return await cache["agent"].execute(
                prompt=arguments["prompt"],
                subagent_type=arguments.get("subagent_type", "explore"),
                description=arguments.get("description"),
                child_session_id=child_session_id,
                progress_callback=self._subagent_progress_callback(tool_call_id),
            )

        else:
            return {"success": False, "error": f"Unknown tool: {name}"}
