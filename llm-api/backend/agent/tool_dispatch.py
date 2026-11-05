"""DispatchMixin: in-process tool execution with parallel support."""
import asyncio
import contextlib
import os
import re
import time
import uuid
import weakref
from typing import Callable, Iterable, List, Dict, Any, Optional

import config
from backend.core.llm_backend import ToolCall, ToolStatusEvent
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
            result = await self._dispatch_tool(name, arguments, tool_call_id=tool_call_id)
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
            keys = await asyncio.to_thread(self._write_lock_keys, name, tool, arguments)
        except Exception:
            keys = []
        async with _hold_path_locks(keys):
            fut = asyncio.ensure_future(asyncio.to_thread(fn, **kwargs))
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
            return await asyncio.to_thread(
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

            return await asyncio.to_thread(
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
            return await asyncio.to_thread(
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
            return await asyncio.to_thread(
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
            return await asyncio.to_thread(
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
            return await asyncio.to_thread(
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
            return await asyncio.to_thread(
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
                cache["grep"] = GrepTool()
            return await asyncio.to_thread(
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
            result = await asyncio.to_thread(
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
