"""
Shared path-resolution helpers for file_ops tools.

When a relative path doesn't resolve, the tool should return enough information
for the model to self-correct on the next turn — not just "file not found".

This module provides:
- candidate_roots():  ordered list of (label, Path) roots to try
- near_matches():     basenames within Levenshtein <= max_distance of a target
- deepest_ancestor(): deepest existing directory along a non-resolving path
- build_failure_report(): assembled structured failure dict

and the primitives that keep every filesystem tool BOUNDED (a tool call that
never returns used to freeze the whole turn):
- Deadline:     cooperative wall-clock budget + cancel flag (also cancelled by
                proc_lifecycle.kill_processes() on a user stop / supervisor stop)
- run_bounded:  run a callable in a daemon thread and give up after a timeout
                (a syscall stuck on a dead network mount cannot be interrupted;
                the abandoned thread dies with the process)
- scan_entries: os.scandir with an entry cap and deadline checks
- HEAVY_DIRS:   directories never descended into unless explicitly targeted
"""
import contextvars
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config


IS_WINDOWS = os.name == "nt"

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules",
    ".venv", "venv", "env", "dist", "dist-web", "build",
    ".pytest_cache", ".mypy_cache", "data",
}

# Never descended into by recursive navigation (tree / ** search / python grep)
# unless the caller's path or pattern names them explicitly.
HEAVY_DIRS = frozenset({
    "node_modules", ".git", ".hg", ".svn", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".gradle", ".idea",
})


# ======================================================================
# Bounded-work primitives
# ======================================================================

class Deadline:
    """Wall-clock budget + cancel flag, checked cooperatively inside loops.

    Registers a cancel hook with proc_lifecycle (when importable) so that
    kill_processes() for this tool call / session — user @stop, supervisor
    verdict — makes the loop stop within a few entries instead of running on
    in an abandoned thread. Use as a context manager (or call close())."""

    def __init__(self, seconds: float, *, hook: bool = True):
        self.seconds = max(0.1, float(seconds))
        self._end = time.monotonic() + self.seconds
        self._cancelled = threading.Event()
        self._hook: Optional[int] = None
        if hook:
            try:
                from backend.utils import proc_lifecycle
                self._hook = proc_lifecycle.register_cancel_hook(self.cancel)
            except Exception:
                self._hook = None

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def expired(self) -> bool:
        return self._cancelled.is_set() or time.monotonic() >= self._end

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def reason(self) -> str:
        if self._cancelled.is_set():
            return "stopped (cancelled)"
        return f"time limit {self.seconds:g}s reached"

    def close(self) -> None:
        hook, self._hook = self._hook, None
        if hook is not None:
            try:
                from backend.utils import proc_lifecycle
                proc_lifecycle.unregister_cancel_hook(hook)
            except Exception:
                pass

    def __enter__(self) -> "Deadline":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def run_bounded(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float,
    cancel: Optional[Deadline] = None,
    name: str = "op",
    **kwargs: Any,
) -> Tuple[bool, Any]:
    """Run fn(*args, **kwargs) in a daemon thread; wait at most *timeout*.

    Returns (True, value) when it finished (re-raising its exception), or
    (False, None) after cancelling *cancel* if it is still running — e.g.
    blocked in a syscall on a hung mount. The call context (session / tool
    call id) is propagated into the thread."""
    box: List[Tuple[bool, Any]] = []
    ctx = contextvars.copy_context()

    def target() -> None:
        try:
            box.append((True, ctx.run(fn, *args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box.append((False, exc))

    thread = threading.Thread(target=target, name=f"bounded-{name}", daemon=True)
    thread.start()
    thread.join(max(0.1, float(timeout)))
    if thread.is_alive() or not box:
        if cancel is not None:
            cancel.cancel()
        return False, None
    ok, value = box[0]
    if not ok:
        raise value
    return True, value


def scan_entries(
    path: Any,
    deadline: Optional[Deadline],
    limit: int,
) -> Tuple[List["os.DirEntry[str]"], Optional[str], Optional[str]]:
    """os.scandir(path) capped at *limit* entries with deadline checks.

    Returns (entries, truncated_reason_or_None, os_error_message_or_None)."""
    entries: List["os.DirEntry[str]"] = []
    reason: Optional[str] = None
    error: Optional[str] = None
    try:
        with os.scandir(path) as it:
            for entry in it:
                if len(entries) >= limit:
                    reason = f"entry limit ({limit}) reached"
                    break
                entries.append(entry)
                if deadline is not None and (len(entries) & 127) == 0 and deadline.expired:
                    reason = deadline.reason()
                    break
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return entries, reason, error


def entry_is_dir(entry: "os.DirEntry[str]") -> bool:
    try:
        return entry.is_dir()
    except OSError:
        return False


def entry_is_real_dir(entry: "os.DirEntry[str]") -> bool:
    """Directory that is not a symlink (os.walk(followlinks=False) rule)."""
    try:
        return entry.is_dir(follow_symlinks=False) and not entry.is_symlink()
    except OSError:
        return False


def realpath_key(path: str) -> str:
    try:
        return os.path.normcase(os.path.realpath(path))
    except OSError:
        return os.path.normcase(path)


# ======================================================================
# Path resolution / failure reports
# ======================================================================

def candidate_roots(
    workspace_dir: Optional[Path],
    username: Optional[str] = None,
) -> List[Tuple[str, Path]]:
    """Ordered list of roots a relative path is tried against.

    The order matches the resolution order in FileReaderTool._resolve_path:
    workspace -> user uploads -> cwd. Each entry is (label, Path).
    """
    roots: List[Tuple[str, Path]] = []
    if workspace_dir:
        roots.append(("workspace", Path(workspace_dir).resolve()))
    if username:
        try:
            roots.append(("uploads", (config.UPLOAD_DIR / username).resolve()))
        except Exception:
            pass
    default_root = workspace_dir or getattr(config, "AGENT_DEFAULT_WORKSPACE", None) or Path.cwd()
    roots.append(("default", Path(default_root).resolve()))
    return roots


def _bounded_levenshtein(a: str, b: str, max_distance: int) -> int:
    """Levenshtein with early exit. Returns max_distance+1 when exceeded."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_distance:
        return max_distance + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        row_min = curr[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_distance:
            return max_distance + 1
        prev = curr
    return prev[lb]


def near_matches(
    basename: str,
    search_root: Path,
    max_distance: int = 2,
    max_depth: int = 3,
    cap: int = 10,
    time_budget: float = 4.0,
    max_visited: int = 20000,
) -> List[str]:
    """Return up to *cap* file paths whose basename is within Levenshtein
    *max_distance* of *basename*. Walks at most *max_depth* levels below
    *search_root*, pruning VCS / build / cache directories BEFORE descending,
    never following symlinks, and giving up after *time_budget* seconds or
    *max_visited* directory entries (this runs on every "file not found", often
    rooted at a huge home directory — it must stay cheap).

    Returned paths are relative to *search_root* when possible, otherwise
    absolute. Case-insensitive comparison.
    """
    try:
        if not search_root.is_dir():
            return []
    except OSError:
        return []
    target = basename.casefold()
    root_str = str(search_root)
    matches: List[Tuple[int, str]] = []
    visited = 0
    deadline = Deadline(time_budget, hook=False)
    stack: List[Tuple[str, int]] = [(root_str, 0)]

    while stack and len(matches) < cap * 4 and visited < max_visited and not deadline.expired:
        current, level = stack.pop()
        entries, _reason, _err = scan_entries(current, deadline, 5000)
        subdirs: List[str] = []
        for entry in entries:
            visited += 1
            name = entry.name
            if level < max_depth and name not in _SKIP_DIRS and entry_is_real_dir(entry):
                subdirs.append(entry.path)
                continue
            dist = _bounded_levenshtein(target, name.casefold(), max_distance)
            if dist <= max_distance:
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                try:
                    rel = os.path.relpath(entry.path, root_str)
                except ValueError:
                    rel = entry.path
                matches.append((dist, rel))
        # Visit shallow directories first-ish: push in reverse so sorted order pops first.
        for sub in sorted(subdirs, reverse=True):
            stack.append((sub, level + 1))

    matches.sort(key=lambda pair: (pair[0], pair[1]))
    return [path for _, path in matches[:cap]]


def deepest_ancestor(requested: Path) -> Tuple[Optional[Path], List[str]]:
    """Walk up *requested* until an existing directory is found. Return
    (ancestor, top_entries). ancestor is None if no part of the path exists
    (rare — root usually exists). top_entries is up to 10 sorted entries
    of that directory, dirs suffixed with '/'."""
    current = requested
    try:
        while current and not current.exists():
            parent = current.parent
            if parent == current:
                return None, []
            current = parent
        if not current.is_dir():
            return current, []
    except OSError:
        return None, []
    # Bounded listing: a directory with millions of entries must not stall the
    # failure report. The first 2000 entries (dirs first, then by name) are
    # plenty to show 10.
    entries, _reason, _err = scan_entries(current, Deadline(2.0, hook=False), 2000)
    if _err and not entries:
        return current, []
    ordered = sorted(entries, key=lambda e: (not entry_is_dir(e), e.name))
    formatted = [f"{e.name}/" if entry_is_dir(e) else e.name for e in ordered[:10]]
    return current, formatted


def build_failure_report(
    requested: str,
    attempted: List[Path],
    workspace_dir: Optional[Path],
    error: str = "file not found",
) -> Dict[str, Any]:
    """Assemble the structured failure dict that gets returned to the agent.

    Includes attempted_paths, near_matches (scanned from workspace or cwd),
    and parent_listing of the deepest existing ancestor of the last attempt.
    """
    attempted_strs = [str(p) for p in attempted]

    requested_basename = Path(requested).name
    if workspace_dir:
        search_root = Path(workspace_dir).resolve()
    else:
        search_root = Path(
            getattr(config, "AGENT_DEFAULT_WORKSPACE", None) or Path.cwd()
        ).resolve()
    near = near_matches(requested_basename, search_root) if requested_basename else []

    ancestor: Optional[Path] = None
    entries: List[str] = []
    for candidate in reversed(attempted):
        ancestor, entries = deepest_ancestor(candidate)
        if ancestor is not None:
            break

    report: Dict[str, Any] = {
        "success": False,
        "error": error,
        "requested": requested,
        "attempted_paths": attempted_strs,
    }
    if near:
        report["near_matches"] = near
    if ancestor is not None:
        report["parent_listing"] = {
            "path": str(ancestor),
            "entries": entries,
        }
    return report
