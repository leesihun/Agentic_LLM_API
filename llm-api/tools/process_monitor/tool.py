"""
Process Monitor Tool
Launch background processes, check status, read output incrementally, kill them.
Processes are scoped per session and tracked in an in-memory registry
(per server process — run llm-api with SERVER_WORKERS = 1).

Processes are spawned through the same shell as shell_exec (bash on POSIX,
PowerShell on Windows) in their own process group/session with
stdin=DEVNULL and the non-interactive env from backend.utils.proc_lifecycle;
`kill` terminates the whole process tree, not just the shell wrapper.
"""
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List

import config
from backend.utils import proc_lifecycle

MAX_BUFFER_LINES = getattr(config, "PROCESS_MONITOR_MAX_BUFFER_LINES", 5000)
MAX_PER_SESSION = getattr(config, "PROCESS_MONITOR_MAX_PER_SESSION", 20)
MAX_LINE_LENGTH = 4096
DEFAULT_READ_LINES = 200
DEAD_CLEANUP_SECONDS = 300  # Remove dead processes after 5 minutes


# ======================================================================
# ManagedProcess — state for one tracked background process
# ======================================================================

@dataclass
class ManagedProcess:
    handle: str
    command: str
    pid: int
    proc: subprocess.Popen
    started_at: float
    working_directory: str
    stdout_buf: deque = field(default_factory=lambda: deque(maxlen=MAX_BUFFER_LINES))
    stderr_buf: deque = field(default_factory=lambda: deque(maxlen=MAX_BUFFER_LINES))
    total_stdout_lines: int = 0
    total_stderr_lines: int = 0
    exited_at: Optional[float] = None   # first time poll() saw the exit
    # Per-stream read cursor: next_offset last returned for each stream, so
    # read_output(stream="both", offset=<one stream's next_offset>) continues
    # each stream from its own position instead of applying one offset to both.
    read_cursor: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"stdout": None, "stderr": None})
    _group_gone: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Observability entry in proc_lifecycle's registry (idle/tail/elapsed).
    tracker: Any = field(default=None, repr=False)

    def live_info(self, running: bool) -> Dict[str, Any]:
        """elapsed / idle / tail for status+list (keys are additive)."""
        info: Dict[str, Any] = {"elapsed_seconds": round(time.time() - self.started_at, 1)}
        tracker = self.tracker
        if tracker is not None:
            try:
                if running:
                    info["idle_seconds"] = round(tracker.idle_seconds(), 1)
                info["output_bytes"] = tracker.output_bytes
                tail = tracker.tail_text(200)
                if tail:
                    info["tail"] = tail
            except Exception:
                pass
        return info

    def group_alive(self) -> bool:
        """POSIX: True while any member of the process group (the wrapper's
        backgrounded children, e.g. `server &`) is still running.

        Latched: once the group is seen empty its pgid may be reused by an
        unrelated process, so it is never probed (or signalled) again."""
        if self._group_gone:
            return False
        if self.proc.poll() is None:
            return True
        if proc_lifecycle.group_alive(self.pid):
            return True
        self._group_gone = True
        return False

    def poll(self) -> Optional[int]:
        """proc.poll() that also records when the exit was first observed.

        The process group stays registered for the atexit tree-kill while any
        member is alive — the shell wrapper exiting does not mean its
        backgrounded children did."""
        code = self.proc.poll()
        if code is not None:
            if self.exited_at is None:
                self.exited_at = time.time()
            if not self.group_alive():
                proc_lifecycle.untrack(self.proc)
        return code

    def active(self) -> bool:
        """Wrapper still running, or its process group still has live members."""
        return self.poll() is None or self.group_alive()


def _start_drain_threads(managed: ManagedProcess):
    """Spawn daemon threads that continuously drain stdout/stderr into ring buffers."""

    def drain(pipe, buf_attr: str, counter_attr: str):
        buf: deque = getattr(managed, buf_attr)
        skipping = False  # inside the remainder of an over-long line
        while True:
            try:
                # Bounded read: a child that never writes a newline (binary
                # dump, endless progress output) must not grow server memory.
                line = pipe.readline(MAX_LINE_LENGTH + 1)
            except (ValueError, OSError):
                break
            if not line:
                break
            tracker = managed.tracker
            if tracker is not None:
                tracker.note_output(line)   # idle/tail tracking (counts skipped long lines too)
            complete = line.endswith("\n")
            if skipping:
                if complete:
                    skipping = False
                continue
            if len(line) > MAX_LINE_LENGTH:
                line = line[:MAX_LINE_LENGTH] + "...[truncated]\n"
                skipping = not complete
            with managed._lock:
                buf.append(line)
                setattr(managed, counter_attr, getattr(managed, counter_attr) + 1)

    t_out = threading.Thread(
        target=drain,
        args=(managed.proc.stdout, "stdout_buf", "total_stdout_lines"),
        daemon=True,
    )
    t_err = threading.Thread(
        target=drain,
        args=(managed.proc.stderr, "stderr_buf", "total_stderr_lines"),
        daemon=True,
    )
    t_out.start()
    t_err.start()


# ======================================================================
# ProcessRegistry — module-level singleton
# ======================================================================

class ProcessRegistry:
    """Thread-safe registry mapping (session_id, handle) -> ManagedProcess."""

    def __init__(self):
        self._lock = threading.Lock()
        self._processes: Dict[str, Dict[str, ManagedProcess]] = {}
        self._counters: Dict[str, int] = {}

    def register(self, session_id: str, proc: subprocess.Popen,
                 command: str, cwd: str) -> str:
        with self._lock:
            if session_id not in self._processes:
                self._processes[session_id] = {}
                self._counters[session_id] = 0
            self._counters[session_id] += 1
            handle = f"proc_{self._counters[session_id]}"
            managed = ManagedProcess(
                handle=handle,
                command=command,
                pid=proc.pid,
                proc=proc,
                started_at=time.time(),
                working_directory=cwd,
            )
            self._processes[session_id][handle] = managed
            return handle

    def get(self, session_id: str, handle: str) -> Optional[ManagedProcess]:
        with self._lock:
            return self._processes.get(session_id, {}).get(handle)

    def remove(self, session_id: str, handle: str) -> bool:
        with self._lock:
            session = self._processes.get(session_id, {})
            if handle in session:
                del session[handle]
                return True
            return False

    def list_for_session(self, session_id: str) -> List[ManagedProcess]:
        with self._lock:
            return list(self._processes.get(session_id, {}).values())

    def cleanup_dead(self, session_id: str):
        """Remove entries for processes that exited more than DEAD_CLEANUP_SECONDS ago.

        A handle whose wrapper exited but whose process group still has live
        members (backgrounded `server &`) is kept, so the model can still see
        and kill it; it stays registered for the atexit tree-kill as well."""
        now = time.time()
        with self._lock:
            session = self._processes.get(session_id, {})
            to_remove = []
            for handle, managed in session.items():
                if not managed.active():
                    exited_duration = now - (managed.exited_at or now)
                    if exited_duration > DEAD_CLEANUP_SECONDS:
                        to_remove.append(handle)
            for handle in to_remove:
                del session[handle]


# Module-level singleton
_registry = ProcessRegistry()


# ======================================================================
# ProcessMonitorTool
# ======================================================================

class ProcessMonitorTool:
    """Manage background processes for a session."""

    def __init__(self, session_id: str, workspace_dir: Optional[str] = None):
        self.session_id = session_id or "default"
        self.workspace_dir: Optional[Path] = None
        if workspace_dir:
            try:
                self.workspace_dir = Path(workspace_dir).expanduser().resolve()
            except Exception:
                self.workspace_dir = Path(workspace_dir)
        # Default cwd + resolution base for relative working_directory —
        # same rule as shell_exec: session workspace, else AGENT_DEFAULT_WORKSPACE.
        default_base = getattr(config, "AGENT_DEFAULT_WORKSPACE", None) or Path.cwd()
        self.workspace = self.workspace_dir or Path(default_base)

    @staticmethod
    def _as_int(value: Any, default: Optional[int]) -> Optional[int]:
        if value is None or value == "" or isinstance(value, bool):
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        operation = (operation or "").strip().lower() if isinstance(operation, str) else operation
        handle = kwargs.get("handle") or ""
        if operation == "start":
            return self._start(kwargs.get("command"), kwargs.get("working_directory"))
        elif operation == "status":
            return self._status(handle)
        elif operation == "read_output":
            max_lines = self._as_int(kwargs.get("max_lines"), DEFAULT_READ_LINES)
            if max_lines is None or max_lines <= 0:
                max_lines = DEFAULT_READ_LINES
            offset = self._as_int(kwargs.get("offset"), None)
            if offset is not None and offset < 0:
                offset = None
            stream = str(kwargs.get("stream") or "both").strip().lower()
            if stream not in ("stdout", "stderr", "both"):
                stream = "both"
            return self._read_output(handle, offset=offset, max_lines=max_lines, stream=stream)
        elif operation == "kill":
            return self._kill(handle)
        elif operation == "list":
            return self._list()
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}",
                "hint": "Use one of: start, status, read_output, kill, list.",
            }

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _start(self, command: Optional[str], working_directory: Optional[str]) -> Dict[str, Any]:
        if not command:
            return {"success": False, "error": "command is required for 'start' operation."}

        # Lazy cleanup of dead processes
        _registry.cleanup_dead(self.session_id)

        if not isinstance(command, str):
            command = str(command)

        # Enforce per-session limit
        existing = _registry.list_for_session(self.session_id)
        alive = [p for p in existing if p.active()]
        if len(alive) >= MAX_PER_SESSION:
            return {
                "success": False,
                "error": f"Limit of {MAX_PER_SESSION} concurrent background processes reached. "
                         f"Kill a process first.",
            }

        cwd = self._resolve_working_directory(working_directory)
        if not cwd.exists():
            if working_directory:
                return {
                    "success": False,
                    "error": f"Working directory does not exist: {cwd}",
                    "hint": "Relative working_directory paths resolve from the session workspace "
                            f"({self.workspace}); omit it to run there.",
                }
            try:
                cwd.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return {"success": False, "error": f"Cannot create working directory {cwd}: {e}"}
        if not cwd.is_dir():
            return {"success": False, "error": f"Working directory is not a directory: {cwd}"}

        try:
            proc = subprocess.Popen(
                proc_lifecycle.shell_argv(command),
                cwd=str(cwd),
                env=proc_lifecycle.child_env(str(self.workspace_dir) if self.workspace_dir else None),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **proc_lifecycle.spawn_kwargs(),
            )
        except Exception as e:
            return {"success": False, "error": f"Failed to start process: {type(e).__name__}: {e}"}
        proc_lifecycle.track(proc)

        handle = _registry.register(self.session_id, proc, command, str(cwd))
        managed = _registry.get(self.session_id, handle)
        # Observable registry entry (live UI / supervisor): tagged with the
        # calling session + tool call; a background job stays registered until
        # it (and its process group) exits and is NOT killed by a turn stop.
        ctx_sid = proc_lifecycle.get_call_context()[0]
        try:
            managed.tracker = proc_lifecycle.register_process(
                proc,
                command=command,
                tool="process_monitor",
                cwd=str(cwd),
                session_id=ctx_sid or self.session_id,
                kind="popen",
                background=True,
            )
        except Exception:
            managed.tracker = None
        _start_drain_threads(managed)

        return {
            "success": True,
            "handle": handle,
            "pid": proc.pid,
            "command": command,
            "working_directory": str(cwd),
            "shell": proc_lifecycle.shell_name(),
            "hint": f"Poll with process_monitor read_output handle='{handle}'; stop with kill.",
        }

    def _status(self, handle: str) -> Dict[str, Any]:
        managed = _registry.get(self.session_id, handle)
        if not managed:
            return self._no_handle(handle)

        exit_code = managed.poll()
        running = exit_code is None
        uptime = time.time() - managed.started_at

        return {
            "success": True,
            "handle": handle,
            "pid": managed.pid,
            "command": managed.command,
            "running": running,
            "exit_code": exit_code,
            "uptime_seconds": round(uptime, 1),
            "total_stdout_lines": managed.total_stdout_lines,
            "total_stderr_lines": managed.total_stderr_lines,
            **managed.live_info(running),
            **self._orphan_fields(managed, exit_code),
        }

    @staticmethod
    def _orphan_fields(managed: ManagedProcess, exit_code: Optional[int]) -> Dict[str, Any]:
        """Extra keys when the shell wrapper exited but processes it
        backgrounded are still running in its process group."""
        if exit_code is None or not managed.group_alive():
            return {}
        return {
            "background_children_running": True,
            "note": "The shell wrapper exited but processes it started in the background "
                    "are still running. Use kill on this handle to stop them.",
        }

    def _read_output(self, handle: str, offset: Optional[int],
                     max_lines: int, stream: str) -> Dict[str, Any]:
        managed = _registry.get(self.session_id, handle)
        if not managed:
            return self._no_handle(handle)

        exit_code = managed.poll()
        result: Dict[str, Any] = {
            "success": True,
            "handle": handle,
            "running": exit_code is None,
            "exit_code": exit_code,
        }

        streams = {
            "stdout": (managed.stdout_buf, "total_stdout_lines"),
            "stderr": (managed.stderr_buf, "total_stderr_lines"),
        }
        names = ["stdout", "stderr"] if stream == "both" else [stream]
        with managed._lock:
            cursors = managed.read_cursor
            totals = {name: getattr(managed, streams[name][1]) for name in names}
            # stdout and stderr count lines independently, but the tool takes a
            # single offset. In "both" mode a non-zero offset equal to either
            # stream's last next_offset means "continue": each stream resumes
            # from its own cursor. Any other value is an explicit position;
            # one past a stream's end falls back to that stream's cursor (or
            # its tail) rather than silently hiding its unread lines.
            continuation = (
                stream == "both" and offset is not None and offset != 0
                and offset in (cursors["stdout"], cursors["stderr"])
            )
            for name in names:
                buf = streams[name][0]
                total = totals[name]
                stream_offset = offset
                if stream == "both" and offset is not None:
                    if continuation:
                        stream_offset = cursors[name] if cursors[name] is not None else 0
                    elif offset > total:
                        stream_offset = cursors[name]
                res = self._extract_lines(buf, total, stream_offset, max_lines)
                cursors[name] = res["next_offset"]
                result[name] = res

        if stream == "both":
            result["offset_note"] = (
                "stdout and stderr have separate offsets; passing either next_offset "
                "continues each stream from where it was last read."
            )
        return result

    def _kill(self, handle: str) -> Dict[str, Any]:
        managed = _registry.get(self.session_id, handle)
        if not managed:
            return self._no_handle(handle)

        if managed.poll() is not None:
            # The shell wrapper exited, but backgrounded children in its
            # process group may still be alive (e.g. `server &`): reap them too.
            # group_alive() is latched, so a group already seen empty (whose
            # pgid may since have been reused) is never signalled.
            killed_children = False
            if managed.group_alive():
                try:
                    proc_lifecycle.kill_tree_sync(managed.proc, term_grace=1.0, wait_grace=1.0)
                    killed_children = True
                except Exception:
                    pass
                managed.poll()  # re-probe the group; untracks it once empty
            return {
                "success": True,
                "handle": handle,
                "already_exited": True,
                "exit_code": managed.proc.returncode,
                **({"killed_background_children": True} if killed_children else {}),
            }

        # Whole tree: SIGTERM the process group, escalate to SIGKILL after a
        # grace period (Windows: taskkill /T /F).
        proc_lifecycle.kill_tree_sync(managed.proc)
        exit_code = managed.poll()

        return {
            "success": exit_code is not None,
            "handle": handle,
            "killed": exit_code is not None,
            "exit_code": exit_code,
            **({} if exit_code is not None else {"error": f"PID {managed.pid} did not exit after kill."}),
        }

    def _no_handle(self, handle: str) -> Dict[str, Any]:
        known = [m.handle for m in _registry.list_for_session(self.session_id)]
        return {
            "success": False,
            "error": f"No process with handle '{handle}' in this session.",
            "known_handles": known,
            "hint": "Use process_monitor list to see this session's handles.",
        }

    def _list(self) -> Dict[str, Any]:
        _registry.cleanup_dead(self.session_id)
        procs = _registry.list_for_session(self.session_id)
        entries = []
        for m in procs:
            exit_code = m.poll()
            entry = {
                "handle": m.handle,
                "pid": m.pid,
                "command": m.command[:100],
                "running": exit_code is None,
                "exit_code": exit_code,
                "uptime_seconds": round(time.time() - m.started_at, 1),
                **m.live_info(exit_code is None),
            }
            if exit_code is not None and m.group_alive():
                entry["background_children_running"] = True
            entries.append(entry)
        return {"success": True, "processes": entries}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_lines(buf: deque, total: int, offset: Optional[int],
                       max_lines: int) -> Dict[str, Any]:
        """Extract lines from a ring buffer with offset tracking."""
        buf_list = list(buf)
        oldest_available = total - len(buf_list)

        if offset is None:
            # Tail mode: return the last max_lines
            lines = buf_list[-max_lines:] if len(buf_list) > max_lines else buf_list
            start_offset = total - len(lines)
            return {
                "lines": "".join(lines),
                "start_offset": start_offset,
                "next_offset": total,
                "total_lines": total,
            }

        if offset < oldest_available:
            # Requested lines have been evicted from the ring buffer
            gap = oldest_available - offset
            lines = buf_list[:max_lines]
            return {
                "lines": "".join(lines),
                "start_offset": oldest_available,
                "next_offset": oldest_available + len(lines),
                "total_lines": total,
                "gap_lines": gap,
                "note": f"{gap} lines were evicted from the buffer before they could be read.",
            }

        # Never hand back a next_offset past the stream's end: a later read
        # from it would skip lines written in between.
        offset = min(offset, total)
        buf_offset = offset - oldest_available
        lines = buf_list[buf_offset:buf_offset + max_lines]
        return {
            "lines": "".join(lines),
            "start_offset": offset,
            "next_offset": offset + len(lines),
            "total_lines": total,
        }

    def _resolve_working_directory(self, working_directory: Optional[str]) -> Path:
        """Same rule as shell_exec: absolute as-is; relative from the session
        workspace (else AGENT_DEFAULT_WORKSPACE); unset -> that base."""
        if not working_directory:
            return Path(self.workspace).resolve()
        cwd = Path(str(working_directory)).expanduser()
        if cwd.is_absolute():
            return cwd.resolve()
        return (Path(self.workspace) / cwd).resolve()
