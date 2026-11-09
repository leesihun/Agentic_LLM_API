"""
Shell Execution Tool
Run shell commands via async subprocess with incremental output streaming.

Shell: bash on POSIX (falls back to /bin/sh when bash is missing) and
PowerShell 5.1 on Windows (`-NoProfile -NonInteractive -EncodedCommand`, UTF-8
output) — the same shells the system prompt promises the model.

Process lifecycle comes from backend.utils.proc_lifecycle: the command runs in
its own process group/session with stdin=DEVNULL and a non-interactive env;
on timeout or cancellation the WHOLE process tree is killed; waits and output
drains after exit/kill are bounded, so a background job that still holds the
pipes can no longer hang the call; output is kept head+tail so the end of a
long log (errors, final status) survives.
"""
import asyncio
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import config
from backend.utils import proc_lifecycle

# Head+tail capture per stream (bytes). Result formatting applies the
# per-tool char budget (TOOL_RESULT_BUDGET["shell_exec"]) on top of this.
HEAD_BYTES = 20_000
TAIL_BYTES = 60_000

# A command word is at "command position" when it starts a line or follows a
# separator: ; & | (covers && and ||), newline, ( / $( , or a backtick —
# optionally after VAR=value assignments. Words inside arguments
# (`grep sudo file`) do not match. These patterns are applied to
# _mask_inert_spans(command), so heredoc bodies, quoted-string contents and
# comments (`echo "a; sudo b"`, `cat > s.sh <<EOF\nsudo ...\nEOF`) never match.
# Also allows wrapper/keyword prefixes that still run the next word as a
# command (`then sudo ...`, `time curl ...`, `nohup sudo ...`, `! curl ...`).
_CMD_POS = (
    r"(?:^|(?<=[;&|\n(`]))[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*=\S*[ \t]+)*"
    r"(?:(?:then|do|else|elif|exec|time|nohup|env|command|xargs|!)[ \t]+)*"
)
_SUDO_RE = re.compile(_CMD_POS + r"sudo(?=[ \t]|$)", re.M)
_CURL_CMD_RE = re.compile(_CMD_POS + r"(?P<curl>curl(?:\.exe)?)(?=[ \t]|$)", re.M | re.I)
_HTTP_URL_RE = re.compile(r"(?i)https?://")
_CURL_FAIL_LONG_RE = re.compile(r"(?<!\S)--fail(?:-with-body)?(?=\s|=|$)")
_CURL_FAIL_SHORT_RE = re.compile(r"(?<!\S)-[A-Za-z]*f[A-Za-z]*(?=\s|$)")
# -w/--write-out: the caller inspects the status code itself (health probes
# like `curl -s -o /dev/null -w "%{http_code}" URL`); --fail would change the
# exit status those checks rely on.
_CURL_WRITE_OUT_RE = re.compile(r"(?<!\S)(?:-[A-Za-z]*w\S*|--write-out\b)")
# End of one curl invocation: the next command separator. Searched in the
# masked command, so separators inside quotes do not end the segment.
_SEGMENT_END_RE = re.compile(r"[;\n]|&&|\|\||\|")
_HEREDOC_OP_RE = re.compile(r"<<(-?)[ \t]*")
_HEREDOC_WORD_END = set(" \t\r\n;|&<>()")
_PS_HERESTRING_RE = re.compile(r"@(['\"])[ \t]*\r?\n")
_PS_IWR_PARAM_RE = re.compile(
    r"(?i)(?<!\S)-(?:Uri|Method|Body|Headers|UseBasicParsing|OutFile|ContentType|InFile)\b"
)

_CURL_FAIL_FLAG: Optional[str] = None
_CURL_FAIL_FLAG_LOCK = threading.Lock()


def _match_paren(s: str, open_idx: int) -> Optional[int]:
    """Index of the `)` closing the `(` at *open_idx* (naive depth count)."""
    depth = 0
    for k in range(open_idx, len(s)):
        if s[k] == "(":
            depth += 1
        elif s[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return None


def _mask_inert_spans(command: str) -> str:
    """Same-length copy of *command* with text the shell does not run as a
    command blanked to spaces: heredoc bodies (bash) / here-strings
    (PowerShell), quoted-string contents and `#` comments.

    Command substitutions inside double quotes (`"$(...)"`, bash backticks)
    stay visible because they do execute. Indices map 1:1 onto the original,
    so callers search the mask and slice the original. Unterminated quotes or
    heredocs are left visible (conservative: the guards still see them).
    """
    win = proc_lifecycle.IS_WINDOWS
    esc = "`" if win else "\\"
    n = len(command)
    out = list(command)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            out[k] = " "

    pending: List[Tuple[str, bool]] = []  # (delimiter, strip_tabs) opened on this line

    def consume_heredocs(body_start: int) -> int:
        """Blank the queued heredoc bodies starting at *body_start*; return the
        index to resume scanning (the newline after the last delimiter)."""
        resume = body_start
        for delim, strip_tabs in pending:
            pos, end = body_start, None
            while True:
                le = command.find("\n", pos)
                le = n if le == -1 else le
                line = command[pos:le].rstrip("\r")
                if (line.lstrip("\t") if strip_tabs else line) == delim:
                    end = le
                    break
                if le >= n:
                    break
                pos = le + 1
            if end is None:
                break  # unterminated heredoc: leave the rest visible
            blank(body_start, end)
            body_start = end + 1
            resume = end
        pending.clear()
        return resume

    i = 0
    while i < n:
        c = command[i]
        if c == "\n":
            i = consume_heredocs(i + 1) if pending else i + 1
            continue
        if c == esc:
            i += 2
            continue
        if c == "#" and (i == 0 or command[i - 1] in " \t\r\n;&|("):
            le = command.find("\n", i)
            le = n if le == -1 else le
            blank(i, le)
            i = le
            continue
        if win and c == "@":
            m = _PS_HERESTRING_RE.match(command, i)
            if m:
                close = command.find("\n" + m.group(1) + "@", m.end() - 1)
                if close != -1:
                    blank(m.end(), close + 1)
                    i = close + 3
                    continue
            i += 1
            continue
        if not win and c == "<" and command.startswith("<<", i):
            if command.startswith("<<<", i):  # here-string: an argument, not a body
                i += 3
                continue
            m = _HEREDOC_OP_RE.match(command, i)
            j = m.end()
            word: List[str] = []
            while j < n and command[j] not in _HEREDOC_WORD_END:
                ch = command[j]
                if ch in "'\"":
                    k = command.find(ch, j + 1)
                    if k == -1:
                        break
                    word.append(command[j + 1:k])
                    j = k + 1
                elif ch == "\\":
                    word.append(command[j + 1:j + 2])
                    j += 2
                else:
                    word.append(ch)
                    j += 1
            delim = "".join(word)
            if delim:
                pending.append((delim, m.group(1) == "-"))
            i = max(j, i + 2)
            continue
        if c == "'":
            ansi_c = not win and i > 0 and command[i - 1] == "$"  # $'...' honours \'
            j = i + 1
            while j < n and command[j] != "'":
                j += 2 if (ansi_c and command[j] == "\\") else 1
            if j >= n:
                i += 1
                continue
            blank(i + 1, j)
            i = j + 1
            continue
        if c == '"':
            j = i + 1
            keep: List[Tuple[int, int]] = []  # command substitutions stay visible
            closed = False
            while j < n:
                ch = command[j]
                if ch == esc:
                    j += 2
                elif ch == '"':
                    closed = True
                    break
                elif ch == "$" and command.startswith("$(", j):
                    k = _match_paren(command, j + 1)
                    if k is None:
                        break
                    keep.append((j, k + 1))
                    j = k + 1
                elif not win and ch == "`":
                    k = command.find("`", j + 1)
                    if k == -1:
                        break
                    keep.append((j, k + 1))
                    j = k + 1
                else:
                    j += 1
            if not closed:
                i += 1
                continue
            pos = i + 1
            for a, b in keep:
                blank(pos, a)
                pos = b
            blank(pos, j)
            i = j + 1
            continue
        i += 1
    return "".join(out)


def _uses_sudo(command: str) -> bool:
    """Return True if the command runs sudo as a command (not as an argument,
    quoted text, comment or heredoc/file content)."""
    return bool(_SUDO_RE.search(_mask_inert_spans(command)))


def _curl_fail_flag() -> str:
    """`--fail-with-body` when the local curl supports it (>= 7.76), else `-f`.

    Probed once per process (`curl --version`, 5s timeout); any failure falls
    back to `-f`, which every curl understands.
    """
    global _CURL_FAIL_FLAG
    if _CURL_FAIL_FLAG is not None:
        return _CURL_FAIL_FLAG
    with _CURL_FAIL_FLAG_LOCK:
        if _CURL_FAIL_FLAG is not None:
            return _CURL_FAIL_FLAG
        flag = "-f"
        try:
            exe = "curl.exe" if proc_lifecycle.IS_WINDOWS else "curl"
            out = subprocess.run(
                [exe, "--version"],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=5,
            ).stdout.decode("utf-8", errors="replace")
            m = re.search(r"curl\s+(\d+)\.(\d+)", out)
            if m and (int(m.group(1)), int(m.group(2))) >= (7, 76):
                flag = "--fail-with-body"
        except Exception:
            pass
        _CURL_FAIL_FLAG = flag
        return flag


def _curl_segment(command: str, masked: str, m: "re.Match[str]") -> str:
    """Arguments of the curl invocation *m* (matched in *masked*), sliced from
    the original so quoted URLs/flags stay readable."""
    seg_start = m.end("curl")
    end_m = _SEGMENT_END_RE.search(masked, seg_start)
    return command[seg_start: end_m.start() if end_m else len(command)]


def _is_powershell_iwr(m: "re.Match[str]", segment: str) -> bool:
    """Windows: bare `curl` used deliberately as PowerShell's Invoke-WebRequest
    alias (PowerShell-style parameters). It already throws on HTTP errors."""
    return (
        proc_lifecycle.IS_WINDOWS
        and m.group("curl").lower() == "curl"
        and bool(_PS_IWR_PARAM_RE.search(segment))
    )


def _curl_calls_missing_fail_flag(
    command: str, masked: Optional[str] = None,
) -> List[Tuple[int, int, str]]:
    """(start, end, token) of curl invocations that hit an http(s) URL without
    --fail / --fail-with-body / -f — those would report HTTP 4xx/5xx as exit 0.

    Only real invocations count: curl text inside heredoc bodies, quoted
    strings or comments is file/argument content, not a command. Calls using
    -w/--write-out are status-code probes and are left alone."""
    if masked is None:
        masked = _mask_inert_spans(command)
    found = []
    for m in _CURL_CMD_RE.finditer(masked):
        segment = _curl_segment(command, masked, m)
        if _is_powershell_iwr(m, segment):
            continue
        if not _HTTP_URL_RE.search(segment):
            continue
        if _CURL_FAIL_LONG_RE.search(segment) or _CURL_FAIL_SHORT_RE.search(segment):
            continue
        if _CURL_WRITE_OUT_RE.search(segment):
            continue
        found.append((m.start("curl"), m.end("curl"), m.group("curl")))
    return found


def _needs_curl_fail_flag(command: str) -> bool:
    """Return True for curl HTTP calls that would hide HTTP 4xx/5xx as exit 0."""
    return bool(_curl_calls_missing_fail_flag(command))


def _apply_curl_guard(command: str) -> Tuple[str, Optional[str]]:
    """Make curl HTTP calls fail on HTTP errors.

    Instead of rejecting the command (which costs the model a whole
    iteration), insert the strict-failure flag right after each offending
    curl token. On Windows (PowerShell) a bare `curl` is an alias for
    Invoke-WebRequest, which rejects curl flags — rewrite it to `curl.exe`.
    Returns (possibly rewritten command, note or None).
    """
    masked = _mask_inert_spans(command)
    calls = _curl_calls_missing_fail_flag(command, masked)
    rewrites_alias = []
    if proc_lifecycle.IS_WINDOWS:
        rewrites_alias = [
            (m.start("curl"), m.end("curl"))
            for m in _CURL_CMD_RE.finditer(masked)
            if m.group("curl").lower() == "curl"
            and not _is_powershell_iwr(m, _curl_segment(command, masked, m))
        ]
    if not calls and not rewrites_alias:
        return command, None

    flag = _curl_fail_flag() if calls else ""
    edits: Dict[int, Tuple[int, str]] = {}
    for start, end in rewrites_alias:
        edits[start] = (end, "curl.exe")
    for start, end, token in calls:
        new_token = "curl.exe" if (proc_lifecycle.IS_WINDOWS and token.lower() == "curl") else token
        edits[start] = (end, f"{new_token} {flag}")
    out = command
    for start in sorted(edits, reverse=True):
        end, replacement = edits[start]
        out = out[:start] + replacement + out[end:]

    curl_exe = "curl.exe" if proc_lifecycle.IS_WINDOWS else "curl"
    parts = []
    if calls:
        parts.append(
            f"Added `{flag}` to curl so HTTP 4xx/5xx responses fail the command "
            f"(convention: `{curl_exe} -sS {flag} ...`)."
        )
    if rewrites_alias:
        parts.append("Rewrote `curl` to `curl.exe` (in PowerShell `curl` is Invoke-WebRequest).")
    return out, " ".join(parts)


class ShellExecTool:
    """Execute shell commands via async subprocess with incremental streaming."""

    def __init__(self, session_id: str, workspace_dir: Optional[Path] = None):
        self.session_id = session_id or "default"
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        # Resolution base for relative `working_directory` args, and the default
        # when none is supplied. workspace_dir wins; otherwise we use the
        # configured AGENT_DEFAULT_WORKSPACE (typically $HOME / /home/<user>).
        default_base = getattr(config, "AGENT_DEFAULT_WORKSPACE", None) or Path.cwd()
        self.repo_root = self.workspace_dir or Path(default_base)

    def _resolve_working_directory(self, working_directory: Optional[str]) -> Path:
        """
        Resolve working directory for command execution.

        Absolute paths are used directly.
        Relative paths are resolved from the project root (workspace if set,
        otherwise AGENT_DEFAULT_WORKSPACE).
        If unset, default to the project root.
        """
        if not working_directory:
            return self.repo_root

        cwd = Path(str(working_directory)).expanduser()
        if cwd.is_absolute():
            return cwd.resolve()
        return (self.repo_root / cwd).resolve()

    @staticmethod
    def _default_timeout() -> float:
        return float(getattr(config, "SHELL_DEFAULT_TIMEOUT_SECONDS", 300) or 300)

    @staticmethod
    def _hard_cap() -> float:
        return float(getattr(config, "SHELL_EXEC_HARD_CAP_SECONDS", 3600) or 3600)

    async def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a shell command asynchronously with streaming output drain.

        On timeout:
        - SHELL_EXEC_KILL_ON_TIMEOUT=True (default): kill the process tree,
          return whatever was streamed so far.
        - False: leave the process running, return partial output + live PID.
        """
        start = time.time()
        if not isinstance(command, str) or not command.strip():
            return {
                "success": False,
                "error": "command is required (a non-empty string).",
                "command": command,
            }
        timeout_s = proc_lifecycle.coerce_timeout(timeout, self._default_timeout(), self._hard_cap())

        cwd = self._resolve_working_directory(working_directory)
        if not cwd.exists():
            return {
                "success": False,
                "error": f"Working directory does not exist: {cwd}",
                "hint": "Relative working_directory paths resolve from the session workspace "
                        f"({self.repo_root}); omit it to run there.",
                "command": command,
            }
        if not cwd.is_dir():
            return {
                "success": False,
                "error": f"Working directory is not a directory: {cwd}",
                "command": command,
            }

        if _uses_sudo(command):
            return {
                "success": False,
                "error": (
                    "sudo is not permitted in shell_exec. Run the command without sudo "
                    "(read-only `sysctl`, `cat /proc/...`, `ss`, etc. work as a normal user)."
                ),
                "hint": "Drop the `sudo` prefix. If you genuinely need root, ask the human operator.",
                "duration": round(time.time() - start, 2),
                "command": command,
            }

        notes: List[str] = []
        try:
            run_command, curl_note = await asyncio.to_thread(_apply_curl_guard, command)
        except Exception:
            run_command, curl_note = command, None
        if curl_note:
            notes.append(curl_note)

        kill_on_timeout = getattr(config, "SHELL_EXEC_KILL_ON_TIMEOUT", True)
        script_path: Optional[str] = None
        try:
            # Over-long commands (big heredocs) exceed the OS argv limit when
            # passed inline (E2BIG on Linux) — run them from a temp script.
            argv, script_path = await asyncio.to_thread(
                proc_lifecycle.shell_argv_or_script, run_command,
            )
            res = await proc_lifecycle.run_process(
                argv,
                cwd=str(cwd),
                env=proc_lifecycle.child_env(str(self.workspace_dir) if self.workspace_dir else None),
                timeout=timeout_s,
                kill_on_timeout=kill_on_timeout,
                head_bytes=HEAD_BYTES,
                tail_bytes=TAIL_BYTES,
                clixml_stderr=proc_lifecycle.IS_WINDOWS,
                display_command=command,
                tool="shell_exec",
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to start {proc_lifecycle.shell_name()}: {type(e).__name__}: {e}",
                "duration": round(time.time() - start, 2),
                "command": command,
            }
        finally:
            proc_lifecycle.remove_script(script_path)

        notes.extend(res.notes)
        duration = round(time.time() - start, 2)
        shell = proc_lifecycle.shell_name()

        if res.stopped:
            # kill_processes(): user @stop / supervisor verdict. Partial output.
            result = {
                "success": False,
                "exit_code": res.returncode,
                "killed": True,
                "stopped": True,
                "error": f"stopped: {res.stop_reason} (process tree killed)",
                "stdout": res.stdout,
                "stderr": res.stderr,
                "duration": duration,
                "command": command,
                "pid": res.pid,
                "shell": shell,
            }
            if run_command != command:
                result["executed_command"] = run_command
            if notes:
                result["notes"] = notes
            return result

        if res.timed_out and res.still_running:
            # Legacy mode: process still running — partial buffers only.
            result = {
                "success": False,
                "still_running": True,
                "timed_out": True,
                "error": f"Process still running after {proc_lifecycle.fmt_seconds(timeout_s)}",
                "stdout": res.stdout,
                "stderr": res.stderr,
                "duration": duration,
                "command": command,
                "pid": res.pid,
                "shell": shell,
                "note": f"PID {res.pid} is alive. Call shell_exec with 'kill {res.pid}' to terminate if needed.",
            }
            if run_command != command:
                result["executed_command"] = run_command
            if notes:
                result["notes"] = notes
            return result

        if res.timed_out:
            msg = f"timed out after {proc_lifecycle.fmt_seconds(timeout_s)} (process tree killed)"
            notes.append(
                "For long-running commands raise `timeout` (max "
                f"{proc_lifecycle.fmt_seconds(self._hard_cap())}) or start them with process_monitor "
                "and poll read_output."
            )
            result = {
                "success": False,
                "exit_code": res.returncode,
                "killed": True,
                "timed_out": True,
                "error": msg,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "duration": duration,
                "command": command,
                "pid": res.pid,
                "shell": shell,
            }
            if run_command != command:
                result["executed_command"] = run_command
            result["notes"] = notes
            return result

        success = res.returncode == 0
        # Status keys first so they survive any downstream truncation.
        result = {
            "success": success,
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "duration": duration,
            "command": command,
            "pid": res.pid,
            "shell": shell,
        }
        if run_command != command:
            result["executed_command"] = run_command
        if not success:
            result["error"] = f"exit code {res.returncode}"
        if success and res.stderr.strip() and not res.stdout.strip():
            result["warning"] = (
                "Command exited 0 but wrote to stderr and produced no stdout. "
                "For API calls, verify the response body or use strict HTTP failure flags."
            )
        if notes:
            result["notes"] = notes
        return result
