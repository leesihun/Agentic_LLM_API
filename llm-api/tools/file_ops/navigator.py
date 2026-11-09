"""
File Navigator Tool
List directory contents and search for files using glob patterns.

Every operation is BOUNDED. The previous implementation could hang a whole turn
(rglob/os.walk/Path.glob over a huge or looping tree, a broken symlink crashing
`list`, a dead network mount blocking a syscall forever). Now:

- all walking is a pruned, iterative os.scandir traversal (no recursion, no
  symlink following, heavy directories such as node_modules/.git are listed but
  not descended into unless explicitly targeted);
- each operation has a wall-clock Deadline, an entry cap and a depth cap, and
  returns PARTIAL results with `truncated: true`, `reason` and a `hint`;
- the whole operation additionally runs under run_bounded(): if a syscall is
  stuck beyond deadline + grace, the call returns a structured error instead of
  blocking (the stuck daemon thread is abandoned);
- user stop / supervisor stop cancels the Deadline cooperatively through
  proc_lifecycle.kill_processes().
"""
import fnmatch
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import config
from tools.file_ops._pathing import (
    HEAVY_DIRS,
    Deadline,
    build_failure_report,
    entry_is_dir,
    entry_is_real_dir,
    realpath_key,
    run_bounded,
    scan_entries,
)


def _cfg(name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


_IS_WINDOWS = os.name == "nt"
_MAGIC_RE = re.compile(r"[*?\[]")
_GLOB_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _glob_regex(segment: str) -> "re.Pattern[str]":
    compiled = _GLOB_CACHE.get(segment)
    if compiled is None:
        flags = re.IGNORECASE if _IS_WINDOWS else 0
        try:
            compiled = re.compile(fnmatch.translate(segment), flags)
        except re.error:
            compiled = re.compile(re.escape(segment), flags)
        if len(_GLOB_CACHE) > 256:
            _GLOB_CACHE.clear()
        _GLOB_CACHE[segment] = compiled
    return compiled


def _iso(ts: float) -> Optional[str]:
    try:
        return datetime.fromtimestamp(ts).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _entry_stat(entry: "os.DirEntry[str]") -> Tuple[int, Optional[str], bool]:
    """(size, modified_iso, broken) — never raises (broken symlinks, races,
    permission errors all degrade to size 0 / no mtime)."""
    try:
        st = entry.stat()
        return st.st_size, _iso(st.st_mtime), False
    except OSError:
        pass
    try:
        st = entry.stat(follow_symlinks=False)
        return st.st_size, _iso(st.st_mtime), True
    except OSError:
        return 0, None, True


class FileNavigatorTool:
    """List and search files on local filesystem."""

    def __init__(self, username: str = None, session_id: str = None,
                 workspace_dir: Optional[Path] = None):
        self.username = username
        self.session_id = session_id
        # Default base for relative paths: workspace_dir if set,
        # else AGENT_DEFAULT_WORKSPACE (typically $HOME / /home/<user>).
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        default_base = getattr(config, "AGENT_DEFAULT_WORKSPACE", None) or Path.cwd()
        self.workspace = self.workspace_dir or Path(default_base)

    def _resolve_base_path(self, path: Optional[str]) -> Path:
        """Resolve base path for list/find. Defaults to the session workspace
        (workspace_dir if set, otherwise server CWD)."""
        if not path:
            return self.workspace.resolve()
        target = Path(path).expanduser()
        if target.is_absolute():
            return target.resolve()
        return (self.workspace / target).resolve()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def navigate(
        self,
        operation: str = "list",
        path: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List directory contents or search for files.

        Args:
            operation: "list" to list directory, "search" to search with glob, "tree" to show directory tree
            path: Directory path for list operation (relative to scratch or absolute)
            pattern: Glob pattern for search operation (e.g. "*.csv", "**/*.py")

        Results may carry `truncated: true` + `reason` + `hint` when a limit
        (time, entries, depth) cut the walk short; the entries returned are
        still valid.
        """
        if operation == "list":
            run = lambda dl: self._list_directory(path, dl)  # noqa: E731
        elif operation == "search":
            run = lambda dl: self._search_files(pattern or "*", path, dl)  # noqa: E731
        elif operation == "tree":
            run = lambda dl: self._tree_directory(path, dl)  # noqa: E731
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}. Use 'list', 'search', or 'tree'.",
            }

        budget = float(_cfg("NAVIGATOR_TIMEOUT_SECONDS", 25))
        grace = float(_cfg("NAVIGATOR_HARD_TIMEOUT_GRACE_SECONDS", 5))
        deadline = Deadline(budget)
        try:
            finished, result = run_bounded(
                run, deadline, timeout=budget + grace, cancel=deadline, name=f"navigator-{operation}",
            )
        finally:
            deadline.close()
        if not finished:
            return {
                "success": False,
                "error": (
                    f"file_navigator {operation} did not finish within {budget + grace:g}s: "
                    "the filesystem is not responding (dead network mount or very large tree)"
                ),
                "truncated": True,
                "reason": "hard timeout",
                "hint": (
                    "Try a narrower path (a specific subdirectory), or check whether the "
                    "mount is available with shell_exec (e.g. `ls <dir>` with a timeout)."
                ),
            }
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_dir(self, path: Optional[str], target: Path) -> Optional[Dict[str, Any]]:
        if not target.exists():
            return build_failure_report(
                requested=path or str(target),
                attempted=[target],
                workspace_dir=self.workspace_dir,
                error="path not found",
            )
        if not target.is_dir():
            return build_failure_report(
                requested=path or str(target),
                attempted=[target],
                workspace_dir=self.workspace_dir,
                error=f"not a directory: {target}",
            )
        return None

    @staticmethod
    def _mark_truncated(result: Dict[str, Any], reason: Optional[str], hint: str) -> None:
        result["truncated"] = bool(reason)
        if reason:
            result["reason"] = reason
            result["hint"] = hint

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def _list_directory(self, path: Optional[str], deadline: Deadline) -> Dict[str, Any]:
        target = self._resolve_base_path(path)
        failure = self._check_dir(path, target)
        if failure is not None:
            return failure

        limit = int(_cfg("NAVIGATOR_MAX_LIST_ENTRIES", 5000))
        raw, reason, error = scan_entries(target, deadline, limit)
        if error and not raw:
            return {"success": False, "error": f"cannot list {target}: {error}", "path": str(target)}
        if reason and reason.startswith("entry limit"):
            reason += "; the listing is an arbitrary subset in directory order, not the alphabetical head"
        raw.sort(key=lambda e: e.name)

        entries: List[Dict[str, Any]] = []
        for entry in raw:
            if (len(entries) & 63) == 63 and deadline.expired:
                reason = reason or deadline.reason()
                break
            size, modified, broken = _entry_stat(entry)
            item: Dict[str, Any] = {
                "name": entry.name,
                "path": entry.path,
                "size": size,
                "modified": modified,
                "is_dir": entry_is_dir(entry),
            }
            try:
                if entry.is_symlink():
                    item["is_symlink"] = True
                    if broken:
                        item["broken_symlink"] = True
            except OSError:
                pass
            entries.append(item)

        result: Dict[str, Any] = {
            "success": True,
            "files": entries,
            "path": str(target),
            "count": len(entries),
        }
        if error:
            result["errors"] = [error]
        self._mark_truncated(
            result, reason,
            "Directory is large; showing a partial listing. Use pattern search "
            "(operation='search', pattern='*.ext') or shell_exec `ls | head` to narrow down.",
        )
        return result

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def _search_files(self, pattern: str, path: Optional[str], deadline: Deadline) -> Dict[str, Any]:
        root = self._resolve_base_path(path)
        failure = self._check_dir(path, root)
        if failure is not None:
            return failure

        raw_pattern = (pattern or "*").strip() or "*"
        if _IS_WINDOWS:
            raw_pattern = raw_pattern.replace("\\", "/")
        if raw_pattern.startswith("/") or (_IS_WINDOWS and re.match(r"^[A-Za-z]:", raw_pattern)):
            return {
                "success": False,
                "error": f"pattern must be relative to the search path, got absolute pattern: {pattern}",
                "hint": "Pass the directory in `path` and a relative glob such as '**/*.py' in `pattern`.",
            }
        segs = [s for s in raw_pattern.split("/") if s and s != "."]
        if not segs:
            segs = ["*"]
        if segs[-1] == "**":
            segs[-1:] = ["**", "*"]  # "dir/**" == "dir/**/*": every file below dir (recursive)

        max_results = int(_cfg("NAVIGATOR_MAX_SEARCH_RESULTS", 2000))
        max_visited = int(_cfg("NAVIGATOR_MAX_VISITED_ENTRIES", 200000))
        max_depth = int(_cfg("NAVIGATOR_MAX_SEARCH_DEPTH", 20))
        allowed_heavy: Set[str] = set(segs) | set(root.parts)

        results: List[Dict[str, Any]] = []
        seen_paths: Set[str] = set()
        seen_states: Set[Tuple[str, int]] = set()
        skipped: List[str] = []
        errors: List[str] = []
        reason: Optional[str] = None
        visited = 0

        def add_file(full: str, entry: Optional["os.DirEntry[str]"] = None) -> None:
            key = os.path.normcase(full)
            if key in seen_paths:
                return
            seen_paths.add(key)
            try:
                st = entry.stat() if entry is not None else os.stat(full)
                size, modified = st.st_size, _iso(st.st_mtime)
            except OSError:
                size, modified = 0, None
            results.append({
                "name": os.path.basename(full),
                "path": full,
                "size": size,
                "modified": modified,
            })

        # (directory, segment index, depth)
        stack: List[Tuple[str, int, int]] = [(str(root), 0, 0)]
        while stack:
            if deadline.expired:
                reason = deadline.reason()
                break
            if len(results) >= max_results:
                reason = f"result limit ({max_results}) reached"
                break
            if visited >= max_visited:
                reason = f"scan limit ({max_visited} entries) reached"
                break
            directory, idx, depth = stack.pop()
            state = (os.path.normcase(directory), idx)
            if state in seen_states:
                continue
            seen_states.add(state)
            seg = segs[idx]
            last = idx == len(segs) - 1

            if seg == "**":
                # Zero directories: move on to the next segment in this directory.
                stack.append((directory, idx + 1, depth))
                if depth >= max_depth:
                    reason = reason or f"depth limit ({max_depth}) reached"
                    continue
                entries, scan_reason, err = scan_entries(directory, deadline, max_visited - visited)
                visited += len(entries)
                if err:
                    errors.append(err)
                if scan_reason and not reason:
                    reason = scan_reason
                subdirs: List[str] = []
                for entry in entries:
                    if not entry_is_real_dir(entry):
                        continue
                    if entry.name in HEAVY_DIRS and entry.name not in allowed_heavy:
                        if len(skipped) < 50:
                            skipped.append(entry.path)
                        continue
                    subdirs.append(entry.path)
                for sub in sorted(subdirs, reverse=True):
                    stack.append((sub, idx, depth + 1))
                continue

            if not _MAGIC_RE.search(seg):
                child = os.path.join(directory, seg)
                try:
                    if last:
                        if os.path.isfile(child):
                            add_file(child)
                    elif os.path.isdir(child):
                        stack.append((child, idx + 1, depth))
                except OSError as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                visited += 1
                continue

            matcher = _glob_regex(seg)
            entries, scan_reason, err = scan_entries(directory, deadline, max_visited - visited)
            visited += len(entries)
            if err:
                errors.append(err)
            if scan_reason and not reason:
                reason = scan_reason
            next_dirs: List[str] = []
            for entry in entries:
                if not matcher.match(entry.name):
                    continue
                if last:
                    try:
                        if entry.is_file():
                            add_file(entry.path, entry)
                    except OSError:
                        pass
                    if len(results) >= max_results:
                        reason = f"result limit ({max_results}) reached"
                        break
                elif entry_is_dir(entry):
                    next_dirs.append(entry.path)
            for sub in sorted(next_dirs, reverse=True):
                stack.append((sub, idx + 1, depth + 1))

        results.sort(key=lambda r: r["path"])
        result: Dict[str, Any] = {
            "success": True,
            "files": results,
            "root": str(root),
            "pattern": pattern,
            "count": len(results),
        }
        if skipped:
            result["skipped_dirs"] = skipped
            result["note"] = (
                "Not searched (heavy directories): " + ", ".join(sorted({os.path.basename(s) for s in skipped}))
                + ". Pass one of them as `path` to search inside it."
            )
        if errors:
            result["errors"] = errors[:20]
        self._mark_truncated(
            result, reason,
            "Results are partial. Narrow the search with a more specific `path` "
            "(e.g. 'src') or pattern (e.g. 'src/**/*.py'), or use grep/shell_exec.",
        )
        return result

    # ------------------------------------------------------------------
    # tree
    # ------------------------------------------------------------------

    def _tree_directory(self, path: Optional[str], deadline: Deadline) -> Dict[str, Any]:
        root = self._resolve_base_path(path)
        failure = self._check_dir(path, root)
        if failure is not None:
            return failure

        max_entries = int(_cfg("NAVIGATOR_MAX_TREE_ENTRIES", 5000))
        max_depth = int(_cfg("NAVIGATOR_MAX_TREE_DEPTH", 8))
        allowed_heavy: Set[str] = set(root.parts)

        tree: List[Dict[str, Any]] = []
        errors: List[str] = []
        skipped: List[str] = []
        reason: Optional[str] = None
        depth_limited = False
        root_str = str(root)
        visited_real: Set[str] = {realpath_key(root_str)}

        # (directory, relative prefix, depth of its children); DFS in os.walk order.
        stack: List[Tuple[str, str, int]] = [(root_str, "", 0)]
        while stack and reason is None:
            if deadline.expired:
                reason = deadline.reason()
                break
            directory, prefix, level = stack.pop()
            entries, scan_reason, err = scan_entries(directory, deadline, max(1, max_entries * 2))
            if err:
                errors.append(err)
            if scan_reason and not reason:
                reason = scan_reason
            dirs = sorted((e for e in entries if entry_is_dir(e)), key=lambda e: e.name)
            files = sorted((e for e in entries if not entry_is_dir(e)), key=lambda e: e.name)

            descend: List[Tuple[str, str, int]] = []
            for entry, is_dir in [(e, True) for e in dirs] + [(e, False) for e in files]:
                if len(tree) >= max_entries:
                    reason = f"entry limit ({max_entries}) reached"
                    break
                rel = prefix + entry.name
                tree.append({
                    "name": entry.name,
                    "path": entry.path,
                    "relative_path": rel,
                    "depth": level,
                    "is_dir": is_dir,
                })
                if not is_dir or not entry_is_real_dir(entry):
                    continue
                if entry.name in HEAVY_DIRS and entry.name not in allowed_heavy:
                    if len(skipped) < 50:
                        skipped.append(rel)
                    continue
                if level + 1 >= max_depth:
                    depth_limited = True
                    continue
                key = realpath_key(entry.path)
                if key in visited_real:
                    continue  # junction / bind-mount loop
                visited_real.add(key)
                descend.append((entry.path, rel + os.sep, level + 1))
            for item in reversed(descend):
                stack.append(item)

        if reason is None and depth_limited:
            reason = f"depth limit ({max_depth}) reached; deeper levels not shown"

        result: Dict[str, Any] = {
            "success": True,
            "root": root_str,
            "entries": tree,
            "count": len(tree),
            "errors": errors[:20],
        }
        if skipped:
            result["skipped_dirs"] = skipped
            result["note"] = (
                "Listed but not expanded (heavy directories): "
                + ", ".join(sorted({os.path.basename(s.rstrip(os.sep)) for s in skipped}))
                + ". Run tree/list with that directory as `path` to look inside."
            )
        self._mark_truncated(
            result, reason,
            "Tree is partial. Run operation='tree' or 'list' on a specific subdirectory, "
            "or use operation='search' with a pattern.",
        )
        return result
