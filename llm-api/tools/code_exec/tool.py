"""
Direct Python code executor — no second LLM call.

The agent LLM writes the code directly in the tool-call arguments.
This tool writes it to a fresh temp dir and runs it there with
`python -u <abs script path>`; the temp dir (the script file only) is
deleted on return.

Working directory: the session workspace when one is set (relative paths,
`import mymodule` and files the script writes resolve/persist there; the
workspace is also prepended to PYTHONPATH and exported as WORKSPACE_DIR).
Without a workspace the script runs inside the ephemeral temp dir, so
anything it writes there is discarded.

Process lifecycle (new process group, stdin=DEVNULL, non-interactive UTF-8
env, tree kill on timeout/cancel, bounded drains, head+tail output) comes
from backend.utils.proc_lifecycle via run_streaming.
"""
import asyncio
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional

import config
from backend.utils import proc_lifecycle
from backend.utils.prompts_log_append import log_to_prompts_file
from backend.utils.subprocess_stream import run_streaming

_LOG_HEAD = 2000
_LOG_TAIL = 4000
_LOG_CODE_HEAD = 6000
_LOG_CODE_TAIL = 2000

# Last line of a traceback / error report, e.g. "ValueError: bad value",
# "ModuleNotFoundError: No module named 'x'", "SyntaxError: invalid syntax".
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Fault)\b.*")


def _format_timeout(timeout: Optional[float]) -> str:
    return proc_lifecycle.fmt_seconds(timeout)


@contextmanager
def _temp_dir(prefix: str) -> Iterator[str]:
    """TemporaryDirectory that never turns a successful run into an exception
    because a child still holds a file (WinError 32) during cleanup."""
    if sys.version_info >= (3, 10):
        with tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True) as d:
            yield d
        return
    d = tempfile.mkdtemp(prefix=prefix)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _error_summary(stderr: str, returncode: int) -> str:
    """Short failure summary: the last exception line, not a copy of stderr."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _EXC_LINE_RE.match(ln):
            return f"exit code {returncode}: {ln[:300]}"
    if lines:
        return f"exit code {returncode}: {lines[-1][:300]}"
    return f"exit code {returncode}"


class CodeExecTool:
    """Execute Python code supplied directly by the LLM."""

    def __init__(self, session_id: str, workspace_dir: Optional[str] = None):
        self.session_id = session_id
        self.workspace_dir: Optional[Path] = None
        if workspace_dir:
            try:
                self.workspace_dir = Path(workspace_dir).expanduser().resolve()
            except Exception:
                self.workspace_dir = Path(workspace_dir)
        self.max_output_size = getattr(config, "PYTHON_EXECUTOR_MAX_OUTPUT_SIZE", 10 * 1024 * 1024)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        code: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Write *code* to a temp dir and run it (cwd = workspace when set)."""
        exec_timeout = self._resolve_timeout(timeout)
        start_time = time.time()
        if not isinstance(code, str):
            code = "" if code is None else str(code)
        script_name = self._make_script_name(code)

        with _temp_dir("code_exec_") as tmpdir:
            script_path = (Path(tmpdir) / script_name).resolve()
            script_path.write_text(code, encoding="utf-8")

            workspace = self.workspace_dir if (self.workspace_dir and self.workspace_dir.is_dir()) else None
            cwd = str(workspace) if workspace else str(Path(tmpdir).resolve())
            env = self._build_env(workspace)

            print(f"\n[CODE_EXEC] Script: {script_path}")
            print(f"[CODE_EXEC] Cwd: {cwd}")
            print(f"[CODE_EXEC] Timeout: {_format_timeout(exec_timeout)}")
            await self._log(self._format_start(code, script_name, exec_timeout, cwd))

            try:
                result = await run_streaming(
                    program=sys.executable,
                    args=["-u", str(script_path)],
                    cwd=cwd,
                    timeout=exec_timeout,
                    max_output_size=self.max_output_size,
                    env=env,
                    display_command=self._display_command(script_name, code),
                    tool="code_exec",
                )
            except Exception as e:
                execution_time = time.time() - start_time
                msg = f"{type(e).__name__}: {e}"
                print(f"[CODE_EXEC] ERROR: {msg}")
                await self._log(self._format_error("ERROR", msg, execution_time))
                return self._result_dict(
                    success=False, script_name=script_name, executed=False,
                    stdout="", stderr="", returncode=-1,
                    execution_time=execution_time, error=f"failed to start python: {msg}",
                    cwd=cwd,
                )

            execution_time = time.time() - start_time
            notes = list(result.notes)
            if result.stopped:
                msg = f"stopped: {result.stop_reason} (process tree killed)"
                print(f"[CODE_EXEC] STOPPED: {msg}")
                stderr = (result.stderr.rstrip("\n") + "\n" if result.stderr.strip() else "") + f"[{msg}]"
                await self._log(self._format_result(False, -1, execution_time, result.stdout, stderr))
                return self._result_dict(
                    success=False, script_name=script_name, executed=True,
                    stdout=result.stdout, stderr=stderr, returncode=-1,
                    execution_time=execution_time, error=msg, cwd=cwd,
                    notes=notes,
                )
            if result.timed_out:
                if result.idle_killed:
                    msg = (f"no output for too long; killed after {execution_time:.0f}s "
                           f"(process tree killed)")
                else:
                    msg = f"timed out after {_format_timeout(exec_timeout)} (process tree killed)"
                print(f"[CODE_EXEC] TIMEOUT: {msg}")
                stderr = (result.stderr.rstrip("\n") + "\n" if result.stderr.strip() else "") + f"[{msg}]"
                await self._log(self._format_result(False, -1, execution_time, result.stdout, stderr))
                notes.append(
                    "For long jobs, raise `timeout` (max "
                    f"{_format_timeout(self._hard_cap())}) or start the script in the background "
                    "with process_monitor and poll it."
                )
                return self._result_dict(
                    success=False, script_name=script_name, executed=True,
                    stdout=result.stdout, stderr=stderr, returncode=-1,
                    execution_time=execution_time, error=msg, cwd=cwd,
                    timed_out=True, notes=notes,
                )

            success = result.returncode == 0
            print(f"[CODE_EXEC] returncode={result.returncode}, time={execution_time:.2f}s")
            await self._log(self._format_result(success, result.returncode, execution_time,
                                                result.stdout, result.stderr))
            return self._result_dict(
                success=success, script_name=script_name, executed=True,
                stdout=result.stdout, stderr=result.stderr, returncode=result.returncode,
                execution_time=execution_time,
                error=None if success else _error_summary(result.stderr, result.returncode),
                cwd=cwd, notes=notes,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_timeout() -> float:
        return float(getattr(
            config, "CODE_EXEC_DEFAULT_TIMEOUT_SECONDS",
            getattr(config, "PYTHON_EXECUTOR_TIMEOUT", 300),
        ) or 300)

    @staticmethod
    def _hard_cap() -> float:
        return float(getattr(config, "CODE_EXEC_HARD_CAP_SECONDS", 1800) or 1800)

    @classmethod
    def _resolve_timeout(cls, timeout: Any) -> float:
        """None/invalid/<=0 -> default (never unlimited); clamp to the hard cap."""
        return proc_lifecycle.coerce_timeout(timeout, cls._default_timeout(), cls._hard_cap())

    @staticmethod
    def _display_command(script_name: str, code: str) -> str:
        """Registry label: script name + first meaningful code line."""
        first = ""
        for line in (code or "").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                first = line[:80]
                break
        return f"python {script_name}" + (f"  # {first}" if first else "")

    @staticmethod
    def _build_env(workspace: Optional[Path]) -> Dict[str, str]:
        env = proc_lifecycle.child_env(str(workspace) if workspace else None)
        if workspace:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(workspace) + (os.pathsep + existing if existing else "")
        return env

    def _make_script_name(self, code: str) -> str:
        """Generate a human-readable timestamped script filename."""
        func_matches = re.findall(r"^\s*def\s+(\w+)", code, re.MULTILINE)
        class_matches = re.findall(r"^\s*class\s+(\w+)", code, re.MULTILINE)
        ts = datetime.now().strftime("%H%M%S")
        if func_matches:
            return f"{func_matches[0]}_{ts}.py"
        if class_matches:
            return f"{class_matches[0].lower()}_{ts}.py"
        return f"exec_{ts}.py"

    # -- prompts.log (bounded head+tail, written off the event loop) --------

    @staticmethod
    async def _log(record: str) -> None:
        try:
            await asyncio.to_thread(log_to_prompts_file, record)
        except Exception as e:
            print(f"[CODE_EXEC] prompts.log write failed: {e}")

    def _format_start(self, code: str, script_name: str, timeout: Optional[float], cwd: str) -> str:
        sep = "=" * 80
        return "\n".join([
            "\n\n" + sep,
            "TOOL EXECUTION: code_exec",
            sep,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session: {self.session_id}",
            f"Script: {script_name}  Timeout: {_format_timeout(timeout)}  Cwd: {cwd}",
            "",
            "CODE:",
            "-" * 80,
            proc_lifecycle.head_tail_text(code, _LOG_CODE_HEAD, _LOG_CODE_TAIL),
            "-" * 80,
        ])

    @staticmethod
    def _format_result(
        success: bool,
        returncode: int,
        execution_time: float,
        stdout: str,
        stderr: str,
    ) -> str:
        lines: List[str] = [
            "",
            f"RESULT: {'SUCCESS' if success else 'FAILED'}",
            f"Return Code: {returncode}",
            f"Execution Time: {execution_time:.2f}s",
        ]
        if stdout:
            lines += ["STDOUT:", proc_lifecycle.head_tail_text(stdout, _LOG_HEAD, _LOG_TAIL)]
        if stderr:
            lines += ["STDERR:", proc_lifecycle.head_tail_text(stderr, _LOG_HEAD, _LOG_TAIL)]
        lines.append("=" * 80)
        return "\n".join(lines)

    @staticmethod
    def _format_error(kind: str, message: str, execution_time: float) -> str:
        return "\n".join([
            "",
            f"{kind}: {message}",
            f"Execution Time: {execution_time:.2f}s",
            "=" * 80,
        ])

    def _result_dict(
        self,
        success: bool,
        script_name: str,
        executed: bool,
        stdout: str,
        stderr: str,
        returncode: int,
        execution_time: float,
        error: Optional[str],
        cwd: Optional[str] = None,
        timed_out: bool = False,
        notes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Status keys first so they survive any downstream truncation.
        result: Dict[str, Any] = {
            "success": success,
            "returncode": returncode,
            "error": error,
            "timed_out": timed_out,
            "execution_mode": "code_exec",
            "script_name": script_name,
            "executed": executed,
            "execution_time": round(execution_time, 2),
            "cwd": cwd,
            "stdout": stdout,
            "stderr": stderr,
        }
        if notes:
            result["notes"] = notes
        return result
