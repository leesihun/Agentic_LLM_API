"""
Asynchronous subprocess execution with incremental output streaming.

Shared helper used by code_exec. Thin compatibility wrapper over
backend.utils.proc_lifecycle.run_process, which owns the hardened lifecycle:
new process group/session, stdin=DEVNULL, non-interactive env, tree kill on
timeout AND on cancellation, bounded wait/drains (a background child holding
the pipes can no longer hang the call), and head+tail output capture.

Two optional kill triggers (beyond normal exit):
  - wall-clock timeout  : process runs longer than `timeout` seconds total.
                          Disabled when `timeout` is None or <= 0 (callers
                          are expected to always pass a finite timeout).
  - idle-output timeout : process produces no stdout/stderr for
                          `idle_timeout` seconds (catches silent hangs).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.utils import proc_lifecycle


@dataclass
class StreamResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    idle_killed: bool = False   # True when killed for output silence, not wall-clock
    pid: Optional[int] = None
    stdout_omitted: int = 0     # bytes dropped from the middle (head+tail kept)
    stderr_omitted: int = 0
    drain_incomplete: bool = False
    duration: float = 0.0
    notes: List[str] = field(default_factory=list)


async def run_streaming(
    program: str,
    args: List[str],
    cwd: str,
    timeout: Optional[float],
    max_output_size: int,
    idle_timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    head_bytes: Optional[int] = None,
    tail_bytes: Optional[int] = None,
) -> StreamResult:
    """Run a program asynchronously, streaming both stdout and stderr.

    Killed (whole process tree) when either the wall-clock `timeout` expires
    OR `idle_timeout` seconds pass with no new output (whichever fires first).
    Returns whatever was streamed up to that point (timed_out=True). Output is
    kept head+tail, each part bounded by `max_output_size`.
    """
    cap = max(1, int(max_output_size or 1))
    head = min(cap, head_bytes or proc_lifecycle.DEFAULT_HEAD_BYTES)
    tail = min(cap, tail_bytes or proc_lifecycle.DEFAULT_TAIL_BYTES)
    res = await proc_lifecycle.run_process(
        [program, *args],
        cwd=cwd,
        env=env if env is not None else proc_lifecycle.child_env(),
        timeout=timeout,
        idle_timeout=idle_timeout,
        head_bytes=head,
        tail_bytes=tail,
    )
    return StreamResult(
        stdout=res.stdout,
        stderr=res.stderr,
        returncode=-1 if res.timed_out else res.returncode,
        timed_out=res.timed_out,
        idle_killed=res.idle_killed,
        pid=res.pid,
        stdout_omitted=res.stdout_omitted,
        stderr_omitted=res.stderr_omitted,
        drain_incomplete=res.drain_incomplete,
        duration=res.duration,
        notes=list(res.notes),
    )
