"""
Shared child-process lifecycle helpers for shell_exec, code_exec and
process_monitor.

Why this exists: the agent's tools used to spawn children in the server's own
process group with inherited stdin, kill only the direct child on timeout, and
then wait UNBOUNDED for stdout/stderr EOF. A backgrounded grandchild (`cmd &`,
`nohup server &`, a Popen() the script left behind) keeps the pipe write-ends
open, so EOF never arrives and the tool call hangs forever. This module gives
every spawner the same hardened lifecycle:

  - spawn in a NEW process group/session (POSIX: start_new_session=True,
    Windows: CREATE_NEW_PROCESS_GROUP) with stdin=DEVNULL and a
    non-interactive environment (no git/pip/apt prompts, unbuffered UTF-8
    Python, headless matplotlib, pip offline when AIRGAPPED);
  - kill the whole TREE: POSIX killpg SIGTERM -> grace -> SIGKILL;
    Windows `taskkill /T /F /PID` run in a worker thread;
  - kill in `finally`, including on asyncio.CancelledError (turn cancelled,
    @stop, subagent timeout, server shutdown);
  - bounded proc.wait() and bounded output drains after exit/kill, with a
    note when a background process still holds the pipes;
  - head+tail output capture (first HEAD + last TAIL bytes) so tracebacks and
    compiler errors at the END of output survive truncation;
  - an atexit hook that kills still-running tracked process groups, since the
    new session detaches them from the server's Ctrl+C / SIGTERM.
"""
from __future__ import annotations

import asyncio
import atexit
import base64
import html
import locale
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

IS_WINDOWS = os.name == "nt"

# Output capture defaults (bytes). Head keeps the start (command echo, early
# errors); tail keeps the end (tracebacks, final status). Downstream result
# formatting applies its own per-tool char budget on top of this.
DEFAULT_HEAD_BYTES = 20_000
DEFAULT_TAIL_BYTES = 60_000

TERM_GRACE_SECONDS = 2.0     # SIGTERM -> SIGKILL escalation delay
WAIT_GRACE_SECONDS = 5.0     # bounded proc.wait() after a kill
DRAIN_GRACE_SECONDS = 3.0    # bounded stdout/stderr drain after exit/kill

_READ_CHUNK = 65536

BACKGROUND_PIPE_NOTE = (
    "a background process still holds stdout/stderr; stopped reading. "
    "Redirect background jobs (`cmd > log.txt 2>&1 &`) or use process_monitor "
    "for long-running/background processes."
)


# ======================================================================
# Environment / spawn flags
# ======================================================================

def _airgapped() -> bool:
    try:
        import config  # local import: keep this module importable standalone
        return bool(getattr(config, "AIRGAPPED", True))
    except Exception:
        return True


def child_env(
    workspace_dir: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Environment for agent-spawned children: inherited + non-interactive."""
    env = dict(os.environ)
    env.update({
        "GIT_TERMINAL_PROMPT": "0",          # never prompt for credentials
        "GCM_INTERACTIVE": "never",          # git-credential-manager (Windows)
        "DEBIAN_FRONTEND": "noninteractive",
        "PIP_NO_INPUT": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONUNBUFFERED": "1",             # output survives a kill
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "MPLBACKEND": "Agg",                 # plt.show() must never block
    })
    # Fail fast instead of retrying an unreachable index — unless the host
    # explicitly configures an internal mirror via env (don't break it).
    if _airgapped() and not (env.get("PIP_INDEX_URL") or env.get("PIP_EXTRA_INDEX_URL")):
        env["PIP_NO_INDEX"] = "1"
    if workspace_dir:
        env["WORKSPACE_DIR"] = str(workspace_dir)
    if extra:
        env.update({k: str(v) for k, v in extra.items() if v is not None})
    return env


def spawn_kwargs() -> Dict[str, Any]:
    """Platform kwargs that put the child in its own process group/session."""
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {"start_new_session": True}


# ======================================================================
# Shell selection
# ======================================================================

_POSIX_SHELL: Optional[str] = None

# Prepended to every PowerShell command: UTF-8 output (the parent decodes
# UTF-8), no progress-bar noise in stderr.
_PS_PREFIX = (
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
    "$OutputEncoding=[Text.Encoding]::UTF8;"
    "$ProgressPreference='SilentlyContinue'\n"
)
# Appended so a failing native command's real exit code reaches the caller
# (powershell.exe otherwise collapses every failure to 1).
_PS_SUFFIX = "\nif (-not $?) { if ($LASTEXITCODE) { exit $LASTEXITCODE } else { exit 1 } }"


def posix_shell() -> str:
    """bash if available (the prompt promises bash), else /bin/sh."""
    global _POSIX_SHELL
    if _POSIX_SHELL is None:
        found = shutil.which("bash")
        if not found and os.path.exists("/bin/bash"):
            found = "/bin/bash"
        _POSIX_SHELL = found or "/bin/sh"
    return _POSIX_SHELL


def shell_name() -> str:
    if IS_WINDOWS:
        return "powershell"
    return os.path.basename(posix_shell())


def shell_argv(command: str) -> List[str]:
    """argv that runs *command* in the platform shell.

    POSIX: [bash, -c, command]. Windows: PowerShell 5.1 with -EncodedCommand
    (base64 UTF-16LE) — passing the script via -Command through Windows argv
    quoting strips embedded double quotes.
    """
    if not IS_WINDOWS:
        return [posix_shell(), "-c", command]
    script = _PS_PREFIX + command + _PS_SUFFIX
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return [
        "powershell.exe", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass", "-OutputFormat", "Text",
        "-EncodedCommand", encoded,
    ]


# Longest command passed inline. Linux caps a single argv string at
# MAX_ARG_STRLEN (128 KiB) -> E2BIG "Argument list too long"; Windows caps the
# whole command line at 32,767 chars and -EncodedCommand inflates the script
# ~2.7x. Longer commands (big heredocs, inline file contents) run from a temp
# script file instead.
_POSIX_INLINE_MAX_BYTES = 100_000
_WIN_INLINE_MAX_CHARS = 30_000


def shell_argv_or_script(command: str) -> Tuple[List[str], Optional[str]]:
    """Like shell_argv(), but spills an over-long command to a temp script.

    Returns (argv, script_path_or_None); the caller deletes the script
    (best-effort) once the process is done.
    """
    argv = shell_argv(command)
    if IS_WINDOWS:
        too_long = sum(len(a) + 3 for a in argv) > _WIN_INLINE_MAX_CHARS
    else:
        too_long = len(command.encode("utf-8", errors="replace")) > _POSIX_INLINE_MAX_BYTES
    if not too_long:
        return argv, None
    if IS_WINDOWS:
        fd, path = tempfile.mkstemp(prefix="shell_exec_", suffix=".ps1")
        # UTF-8 *with BOM*: Windows PowerShell 5.1 reads BOM-less scripts as ANSI.
        with os.fdopen(fd, "w", encoding="utf-8-sig", errors="replace", newline="") as f:
            f.write(_PS_PREFIX + command + _PS_SUFFIX)
        return [
            "powershell.exe", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass", "-OutputFormat", "Text",
            "-File", path,
        ], path
    fd, path = tempfile.mkstemp(prefix="shell_exec_", suffix=".sh")
    with os.fdopen(fd, "w", encoding="utf-8", errors="replace", newline="") as f:
        f.write(command)
    return [posix_shell(), path], path


def remove_script(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


# ======================================================================
# Decoding
# ======================================================================

_CLIXML_MARK = "#< CLIXML"
_CLIXML_OBJS_RE = re.compile(r"<Objs\b.*?(?:</Objs>|$)", re.S)
_CLIXML_S_RE = re.compile(r'<S S="Error">(.*?)</S>', re.S)
_CLIXML_ESC_RE = re.compile(r"_x([0-9A-Fa-f]{4})_")


def _flatten_clixml_block(match: "re.Match[str]") -> str:
    parts = _CLIXML_S_RE.findall(match.group(0))
    joined = "".join(parts)
    joined = _CLIXML_ESC_RE.sub(lambda m: chr(int(m.group(1), 16)), joined)
    return html.unescape(joined)


def _strip_clixml(text: str) -> str:
    """powershell.exe -EncodedCommand serialises its error stream as CLIXML
    (native commands may still write raw bytes around it); flatten the XML
    blocks to their error text and keep everything else."""
    if _CLIXML_MARK not in text:
        return text
    text = text.replace(_CLIXML_MARK + "\r\n", "").replace(_CLIXML_MARK + "\n", "").replace(_CLIXML_MARK, "")
    text = _CLIXML_OBJS_RE.sub(_flatten_clixml_block, text)
    return text.replace("\r\n", "\n").strip("\n")


def _decode_bytes(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if not IS_WINDOWS:
        return data.decode("utf-8", errors="replace")
    # Windows dev box: native tools / localized error messages may emit the
    # ANSI/OEM codepage (cp949 on Korean Windows) mixed with UTF-8. Decode per
    # line, preferring UTF-8.
    fallback = locale.getpreferredencoding(False) or "cp949"
    out = []
    for line in data.split(b"\n"):
        try:
            out.append(line.decode("utf-8"))
            continue
        except UnicodeDecodeError:
            pass
        try:
            out.append(line.decode(fallback))
        except Exception:
            out.append(line.decode("utf-8", errors="replace"))
    return "\n".join(out)


def _trim_partial_utf8_head(data: bytes) -> bytes:
    """Drop a trailing incomplete UTF-8 sequence (cut at the head boundary)."""
    for back in range(1, min(4, len(data)) + 1):
        b = data[-back]
        if b < 0x80:
            return data
        if b >= 0xC0:  # lead byte: complete only if enough continuation bytes
            need = 2 if b < 0xE0 else 3 if b < 0xF0 else 4
            return data if back >= need else data[:-back]
    return data


def _trim_partial_utf8_tail(data: bytes) -> bytes:
    """Drop leading UTF-8 continuation bytes (cut at the tail boundary)."""
    i = 0
    while i < min(3, len(data)) and 0x80 <= data[i] < 0xC0:
        i += 1
    return data[i:]


_PS_ECHO_MARK = _PS_PREFIX.split(";", 1)[0]
# The echo-matching regex has one optional-newline group per script char and
# is compiled on the event loop; keep it small.
_PS_ECHO_MAX_SCRIPT = 4_000


def strip_ps_script_echo(text: str, argv: List[str]) -> str:
    """Write-Error at script scope prefixes its message with the *whole*
    invoking script (our UTF-8 prefix + command + exit-code suffix), wrapped at
    the host width. Remove that echo so stderr reads "<message>"."""
    if not text or _PS_ECHO_MARK not in text or "-EncodedCommand" not in argv:
        return text
    try:
        idx = argv.index("-EncodedCommand")
        script = base64.b64decode(argv[idx + 1]).decode("utf-16-le")
    except Exception:
        return text
    if len(script) > _PS_ECHO_MAX_SCRIPT:
        return text
    # The host may hard-wrap the echoed script anywhere and normalise CRLF.
    flat = script.replace("\r\n", "\n")
    pattern = r"(?:\r?\n)?".join(re.escape(ch) for ch in flat) + r"[ \t]*:[ \t]?"
    try:
        return re.sub(pattern, "", text)
    except re.error:
        return text


def decode_output(data: bytes, clixml: bool = False) -> str:
    text = _decode_bytes(bytes(data))
    if clixml and IS_WINDOWS:
        text = _strip_clixml(text)
    return text


# ======================================================================
# Head + tail capture buffer
# ======================================================================

class HeadTailBuffer:
    """Keep the first `head_max` and last `tail_max` bytes of a stream."""

    def __init__(self, head_max: int = DEFAULT_HEAD_BYTES, tail_max: int = DEFAULT_TAIL_BYTES):
        self.head_max = max(0, int(head_max))
        self.tail_max = max(0, int(tail_max))
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total += len(chunk)
        room = self.head_max - len(self.head)
        if room > 0:
            self.head.extend(chunk[:room])
            chunk = chunk[room:]
        if chunk and self.tail_max > 0:
            self.tail.extend(chunk)
            # Amortised trim: only compact once we're well past the cap.
            if len(self.tail) > 2 * self.tail_max + _READ_CHUNK:
                del self.tail[: len(self.tail) - self.tail_max]

    @property
    def omitted(self) -> int:
        return max(0, self.total - len(self.head) - min(len(self.tail), self.tail_max))

    def text(self, clixml: bool = False) -> str:
        tail = bytes(self.tail[-self.tail_max:]) if self.tail_max else b""
        omitted = self.omitted
        if not tail:
            return decode_output(bytes(self.head), clixml)
        if omitted == 0:
            return decode_output(bytes(self.head) + tail, clixml)
        head_txt = decode_output(_trim_partial_utf8_head(bytes(self.head)), clixml)
        tail_txt = decode_output(_trim_partial_utf8_tail(tail), clixml)
        return f"{head_txt}\n[… {omitted:,} bytes omitted …]\n{tail_txt}"


# ======================================================================
# Live-process registry (atexit cleanup)
# ======================================================================

_live_lock = threading.Lock()
_live: Dict[int, Any] = {}        # pid -> asyncio Process | subprocess.Popen
_bg_tasks: "set[asyncio.Task]" = set()


def track(proc: Any) -> None:
    with _live_lock:
        _live[proc.pid] = proc


def untrack(proc: Any) -> None:
    with _live_lock:
        _live.pop(getattr(proc, "pid", None), None)


def _returncode(proc: Any) -> Optional[int]:
    if isinstance(proc, subprocess.Popen):
        try:
            return proc.poll()
        except Exception:
            return proc.returncode
    return getattr(proc, "returncode", None)


def _kill_all_live() -> None:
    """atexit: kill tracked process groups that still have live members —
    a running leader, or (POSIX) children the exited shell wrapper
    backgrounded (`server &`), which stay in its process group."""
    with _live_lock:
        procs = list(_live.values())
        _live.clear()
    for proc in procs:
        try:
            if _returncode(proc) is None or _group_alive(proc.pid):
                _hard_kill_tree(proc.pid)
        except Exception:
            pass


atexit.register(_kill_all_live)


# ======================================================================
# Tree kill (sync primitives)
# ======================================================================

def _signal_group(pid: int, sig: int) -> bool:
    """POSIX: signal the process group led by *pid*. False if it is gone."""
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    except OSError:
        return False


def _group_alive(pid: int) -> bool:
    if IS_WINDOWS:
        return False
    return _signal_group(pid, 0)


def group_alive(pid: int) -> bool:
    """POSIX: True while the process group led by *pid* has any live member
    (children spawned with spawn_kwargs() lead their own group). Always False
    on Windows, where there is no group to probe."""
    return _group_alive(pid)


def _taskkill(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except Exception:
        pass


def _hard_kill_tree(pid: int) -> None:
    """Immediate, synchronous SIGKILL of the tree (atexit / last resort)."""
    if IS_WINDOWS:
        _taskkill(pid)
    else:
        _signal_group(pid, signal.SIGKILL)


def kill_tree_sync(
    proc: subprocess.Popen,
    term_grace: float = TERM_GRACE_SECONDS,
    wait_grace: float = WAIT_GRACE_SECONDS,
) -> None:
    """Blocking tree kill for subprocess.Popen children (process_monitor)."""
    pid = proc.pid
    try:
        if IS_WINDOWS:
            if proc.poll() is not None:
                return  # taskkill /T needs a live root; nothing left to walk
            _taskkill(pid)
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        else:
            if not _signal_group(pid, signal.SIGTERM) and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            deadline = time.monotonic() + term_grace
            while time.monotonic() < deadline:
                if proc.poll() is not None and not _group_alive(pid):
                    break
                time.sleep(0.1)
            if _group_alive(pid):
                _signal_group(pid, signal.SIGKILL)
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            proc.wait(timeout=wait_grace)
        except Exception:
            pass
    finally:
        if proc.poll() is not None:
            untrack(proc)


# ======================================================================
# Tree kill (asyncio)
# ======================================================================

async def _wait_proc(proc: asyncio.subprocess.Process, timeout: float) -> bool:
    """Bounded proc.wait(); True if the process has exited."""
    if proc.returncode is not None:
        return True
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False
    except Exception:
        return proc.returncode is not None


async def kill_tree(
    proc: asyncio.subprocess.Process,
    term_grace: float = TERM_GRACE_SECONDS,
    wait_grace: float = WAIT_GRACE_SECONDS,
) -> None:
    """Kill *proc* and all its descendants; never blocks unboundedly."""
    pid = proc.pid
    try:
        if IS_WINDOWS:
            if proc.returncode is not None:
                return  # taskkill /T needs a live root; nothing left to walk
            await asyncio.to_thread(_taskkill, pid)
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        else:
            if not _signal_group(pid, signal.SIGTERM) and proc.returncode is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
            loop = asyncio.get_running_loop()
            deadline = loop.time() + term_grace
            while loop.time() < deadline:
                if proc.returncode is not None and not _group_alive(pid):
                    break
                await asyncio.sleep(0.1)
            if _group_alive(pid):
                _signal_group(pid, signal.SIGKILL)
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        await _wait_proc(proc, wait_grace)
    finally:
        if proc.returncode is not None:
            untrack(proc)


def kill_tree_nowait(proc: asyncio.subprocess.Process) -> None:
    """Start killing the tree from a context that must not await (finally on
    cancellation). Sends the first signal synchronously, then escalates in a
    background task that holds its own strong reference."""
    pid = proc.pid
    # Windows: no synchronous first step — killing the root first would
    # orphan its children from `taskkill /T`, which walks from a live root.
    if not IS_WINDOWS:
        try:
            _signal_group(pid, signal.SIGTERM)
        except Exception:
            pass
    try:
        task = asyncio.get_running_loop().create_task(kill_tree(proc))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)
    except RuntimeError:
        # No running loop (interpreter shutdown): hard kill synchronously.
        _hard_kill_tree(pid)


# ======================================================================
# Async runner
# ======================================================================

@dataclass
class ProcResult:
    stdout: str
    stderr: str
    returncode: int
    pid: Optional[int] = None
    timed_out: bool = False
    idle_killed: bool = False
    still_running: bool = False          # timeout with kill_on_timeout=False
    stdout_omitted: int = 0              # bytes dropped from the middle
    stderr_omitted: int = 0
    drain_incomplete: bool = False       # stopped reading: a background child held the pipe
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)


async def _drain(stream: Optional[asyncio.StreamReader], buf: HeadTailBuffer,
                 last_seen: Optional[list] = None) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(_READ_CHUNK)
        if not chunk:
            return
        buf.feed(chunk)
        if last_seen is not None:
            last_seen[0] = asyncio.get_running_loop().time()


async def _finish_drains(proc, tasks: Sequence[asyncio.Task], grace: float) -> bool:
    """Wait up to *grace* seconds for the drains to hit EOF. On expiry, cancel
    them and close the transport so our pipe FDs are released. Returns True if
    the drains completed on their own."""
    pending = [t for t in tasks if not t.done()]
    if pending:
        _, pending_set = await asyncio.wait(pending, timeout=grace)
        pending = list(pending_set)
    if not pending:
        return True
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass
    return False


def _spawn_bg(coro) -> None:
    """Run *coro* as a background task holding its own strong reference."""
    task = asyncio.get_running_loop().create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _reap_left_running(proc, drains: Sequence[asyncio.Task],
                             wait_task: asyncio.Task) -> None:
    """kill_on_timeout=False: the drains keep consuming output after
    run_process returns; once the child exits, finish the drains (bounded —
    closes the pipes if a background grandchild still holds them) and untrack."""
    try:
        await wait_task
        await _finish_drains(proc, drains, DRAIN_GRACE_SECONDS)
    except asyncio.CancelledError:
        for t in (*drains, wait_task):
            if not t.done():
                t.cancel()
        raise
    except Exception:
        pass
    finally:
        if proc.returncode is not None:
            untrack(proc)


async def run_process(
    argv: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    idle_timeout: Optional[float] = None,
    kill_on_timeout: bool = True,
    head_bytes: int = DEFAULT_HEAD_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    drain_grace: float = DRAIN_GRACE_SECONDS,
    clixml_stderr: bool = False,
) -> ProcResult:
    """Run *argv* with the hardened lifecycle described in the module docstring.

    timeout: wall-clock limit in seconds (None/<=0 disables).
    idle_timeout: kill when no stdout/stderr bytes arrive for this long.
    kill_on_timeout=False: on timeout leave the process running (legacy
    SHELL_EXEC_KILL_ON_TIMEOUT=False) and return what was captured so far.
    Raises whatever create_subprocess_exec raises (e.g. FileNotFoundError).
    """
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env if env is not None else child_env(),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **spawn_kwargs(),
    )
    track(proc)

    loop = asyncio.get_running_loop()
    out_buf = HeadTailBuffer(head_bytes, tail_bytes)
    err_buf = HeadTailBuffer(head_bytes, tail_bytes)
    last_seen = [loop.time()]
    out_task = asyncio.create_task(_drain(proc.stdout, out_buf, last_seen))
    err_task = asyncio.create_task(_drain(proc.stderr, err_buf, last_seen))
    wait_task = asyncio.create_task(proc.wait())

    timed_out = False
    idle_killed = False
    leave_running = False
    finished = False
    notes: List[str] = []
    try:
        deadline = None
        if timeout is not None and timeout > 0:
            deadline = loop.time() + float(timeout)
        while True:
            step = 1.0
            if deadline is not None:
                step = max(0.05, min(step, deadline - loop.time()))
            done, _ = await asyncio.wait({wait_task}, timeout=step)
            # returncode check too: on newer CPython a wait() already pending
            # at exit resolves only once the pipes close, which a background
            # grandchild (`cmd &`) can hold open until the deadline.
            if wait_task in done or proc.returncode is not None:
                break
            now = loop.time()
            if deadline is not None and now >= deadline:
                timed_out = True
                break
            if idle_timeout is not None and idle_timeout > 0 and now - last_seen[0] >= idle_timeout:
                timed_out = True
                idle_killed = True
                break

        if timed_out and not kill_on_timeout:
            # Leave the child running, but keep its pipes drained (into the
            # bounded buffers; the result below is a snapshot) so it never
            # blocks on a full pipe. A background reaper closes the pipes and
            # untracks it once it exits.
            leave_running = True
            _spawn_bg(_reap_left_running(proc, (out_task, err_task), wait_task))
        else:
            if timed_out:
                await kill_tree(proc)
            elif not await _wait_proc(proc, WAIT_GRACE_SECONDS):
                await kill_tree(proc)
            if not await _finish_drains(proc, (out_task, err_task), drain_grace):
                notes.append(BACKGROUND_PIPE_NOTE)
        finished = True
    finally:
        if not finished and not leave_running:
            # CancelledError or unexpected failure: never orphan the tree.
            if proc.returncode is None or _group_alive(proc.pid):
                kill_tree_nowait(proc)
        if not leave_running:  # left running: the reaper owns these tasks
            for t in (out_task, err_task, wait_task):
                if not t.done():
                    t.cancel()
        if proc.returncode is not None and not leave_running:
            untrack(proc)

    stdout = out_buf.text()
    stderr = err_buf.text(clixml=clixml_stderr)
    if clixml_stderr and IS_WINDOWS:
        stderr = strip_ps_script_echo(stderr, list(argv))
    if out_buf.omitted:
        notes.append(f"stdout: {out_buf.omitted:,} bytes omitted from the middle (head+tail kept)")
    if err_buf.omitted:
        notes.append(f"stderr: {err_buf.omitted:,} bytes omitted from the middle (head+tail kept)")
    rc = proc.returncode
    return ProcResult(
        stdout=stdout,
        stderr=stderr,
        returncode=rc if rc is not None else -1,
        pid=proc.pid,
        timed_out=timed_out,
        idle_killed=idle_killed,
        still_running=leave_running,
        stdout_omitted=out_buf.omitted,
        stderr_omitted=err_buf.omitted,
        drain_incomplete=BACKGROUND_PIPE_NOTE in notes,
        duration=time.monotonic() - start,
        notes=notes,
    )


# ======================================================================
# Misc helpers
# ======================================================================

def coerce_timeout(value: Any, default: float, cap: Optional[float] = None) -> float:
    """Model-supplied timeout -> positive seconds. None/invalid/<=0 -> default;
    accepts "600", "600s", 600.0. Clamped to *cap* when given."""
    result: Optional[float] = None
    if value is not None and not isinstance(value, bool):
        try:
            if isinstance(value, str):
                value = value.strip().lower().rstrip("s").strip()
            result = float(value)
        except (TypeError, ValueError):
            result = None
    if result is None or result != result or result <= 0:  # NaN check
        result = float(default)
    if cap is not None and cap > 0 and result > cap:
        result = float(cap)
    return result


def fmt_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "never"
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:.1f}s"


def head_tail_text(text: str, head: int = 2000, tail: int = 2000) -> str:
    """Head+tail clip for log records."""
    if not text or len(text) <= head + tail:
        return text or ""
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[… {omitted:,} chars omitted …]\n{text[-tail:]}"
