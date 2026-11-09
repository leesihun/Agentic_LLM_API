"""
File Reader Tool
Read file contents from local filesystem.

Bounded work: only regular files are opened (FIFOs / devices can block forever
or stream endlessly), lines are pulled in fixed-size chunks (a newline-less
multi-GB file never becomes one string), skipping to `offset` and counting
total lines stop at a scan budget, and the whole call runs under a wall-clock
Deadline plus run_bounded() so a dead network mount cannot hang the turn.
"""
import stat as stat_mod
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import config
from tools.file_ops._pathing import Deadline, build_failure_report, candidate_roots, run_bounded


TEXT_EXTENSIONS = {
    '.txt', '.md', '.json', '.csv', '.py', '.js', '.ts', '.jsx', '.tsx',
    '.html', '.css', '.xml', '.yaml', '.yml', '.log', '.ini', '.cfg',
    '.toml', '.sh', '.bat', '.ps1', '.sql', '.r', '.java', '.cpp', '.c',
    '.h', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.lua',
    '.pl', '.m', '.tex', '.rst', '.org', '.env', '.gitignore', '.dockerfile',
}

MAX_READ_BYTES = 50 * 1024  # 50KB cap
_LINE_CHUNK = 64 * 1024  # max characters pulled per readline


class FileReaderTool:
    """Read files from local filesystem."""

    def __init__(self, username: str = None, session_id: str = None,
                 workspace_dir: Optional[Path] = None):
        self.username = username
        self.session_id = session_id
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None

    def _resolve_path(self, path: str) -> Tuple[Path, List[Path]]:
        """Resolve a relative path against workspace_dir, then user uploads,
        then the server CWD. Absolute paths are used directly.

        Returns (chosen, attempted). *chosen* is the first candidate that
        exists, or the final fallback if none exist. *attempted* lists every
        candidate inspected, in order — used to build actionable error reports.
        """
        target = Path(path).expanduser()
        if target.is_absolute():
            resolved = target.resolve()
            return resolved, [resolved]

        attempted: List[Path] = []
        for _label, root in candidate_roots(self.workspace_dir, self.username):
            candidate = (root / target).resolve()
            attempted.append(candidate)
            if candidate.exists():
                return candidate, attempted
        # No candidate existed - return the last one as the "chosen" fallback
        # so callers can still pattern-match against it.
        return attempted[-1], attempted

    def read(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Read file contents.

        Args:
            path: Absolute path or path relative to current working directory
            offset: Start reading from this line number (1-based)
            limit: Maximum number of lines to return
        """
        budget = float(getattr(config, "READER_TIMEOUT_SECONDS", 30) or 30)
        grace = float(getattr(config, "READER_HARD_TIMEOUT_GRACE_SECONDS", 5) or 5)
        deadline = Deadline(budget)
        try:
            finished, result = run_bounded(
                self._read_inner, path, offset, limit, deadline,
                timeout=budget + grace, cancel=deadline, name="file_reader",
            )
        finally:
            deadline.close()
        if not finished:
            return {
                "success": False,
                "error": (
                    f"file_reader did not finish within {budget + grace:g}s: the file or "
                    "filesystem is not responding (dead network mount?)"
                ),
                "path": path,
                "reason": "hard timeout",
            }
        return result

    def _read_inner(
        self,
        path: str,
        offset: Optional[int],
        limit: Optional[int],
        deadline: Deadline,
    ) -> Dict[str, Any]:
        resolved, attempted = self._resolve_path(path)

        if not resolved.exists():
            return build_failure_report(
                requested=path,
                attempted=attempted,
                workspace_dir=self.workspace_dir,
                error="file not found",
            )
        if resolved.is_dir():
            return build_failure_report(
                requested=path,
                attempted=attempted,
                workspace_dir=self.workspace_dir,
                error=f"not a file (is a directory): {resolved}",
            )

        try:
            st = resolved.stat()
        except OSError as e:
            return {"success": False, "error": f"cannot stat file: {e}", "path": str(resolved)}
        if not stat_mod.S_ISREG(st.st_mode):
            # FIFOs, /dev/zero, sockets, devices: opening them can block forever
            # or yield an endless stream.
            return {
                "success": False,
                "error": f"not a regular file (device, pipe or socket): {resolved}",
                "path": str(resolved),
            }
        file_size = st.st_size
        suffix = resolved.suffix.lower()

        if suffix not in TEXT_EXTENSIONS and suffix != '':
            return {
                "success": False,
                "error": f"Unsupported file type: {suffix}. Only text files are supported.",
                "path": str(resolved),
                "size": file_size,
            }

        try:
            return self._read_bounded(resolved, file_size, offset, limit, deadline)
        except Exception as e:
            return {"success": False, "error": str(e), "path": str(resolved)}

    @staticmethod
    def _next_line(f: Any, deadline: Deadline, max_scan: int, scanned: int) -> Optional[Tuple[str, int]]:
        """Pull one logical line. Returns (kept_text, chars_consumed) or None at
        EOF. Lines longer than one chunk are consumed in chunks, keeping only
        about MAX_READ_BYTES of text."""
        chunk = f.readline(_LINE_CHUNK)
        if not chunk:
            return None
        consumed = len(chunk)
        kept = chunk
        while not chunk.endswith('\n') and len(chunk) == _LINE_CHUNK:
            chunk = f.readline(_LINE_CHUNK)
            if not chunk:
                break
            consumed += len(chunk)
            if len(kept) < MAX_READ_BYTES:
                kept += chunk
            if scanned + consumed > max_scan or deadline.expired:
                break
        return kept, consumed

    def _read_bounded(
        self,
        resolved: Path,
        file_size: int,
        offset: Optional[int],
        limit: Optional[int],
        deadline: Deadline,
    ) -> Dict[str, Any]:
        """Line-oriented read with bounded memory and bounded scan work.

        Skipping to `offset` and counting total lines stop at the scan
        byte/time budget (total_lines_exact=False when that happens)."""
        start_line = max(1, offset or 1)
        requested_limit = None if limit is None else max(0, limit)
        max_scan = int(getattr(config, "READER_MAX_SCAN_BYTES", 512 * 1024 * 1024) or 512 * 1024 * 1024)
        count_budget = float(getattr(config, "READER_COUNT_LINES_SECONDS", 5) or 5)

        total_lines = 0
        returned_lines = 0
        collected_parts: List[str] = []
        collected_bytes = 0
        truncated = False
        exact = True
        scan_reason: Optional[str] = None
        scanned = 0
        window_full_at: Optional[float] = None

        with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
            while True:
                got = self._next_line(f, deadline, max_scan, scanned)
                if got is None:
                    break
                line, consumed = got
                scanned += consumed

                if deadline.expired or scanned > max_scan:
                    scan_reason = deadline.reason() if deadline.expired else f"scan limit ({max_scan} bytes) reached"
                    exact = False
                    break

                total_lines += 1

                if total_lines >= start_line and not (
                    requested_limit is not None and returned_lines >= requested_limit
                ):
                    line_bytes = line.encode('utf-8', errors='ignore')
                    remaining = MAX_READ_BYTES - collected_bytes
                    if remaining <= 0:
                        truncated = True
                    else:
                        if len(line_bytes) <= remaining:
                            collected_parts.append(line)
                            collected_bytes += len(line_bytes)
                        else:
                            collected_parts.append(line_bytes[:remaining].decode('utf-8', errors='ignore'))
                            collected_bytes = MAX_READ_BYTES
                            truncated = True
                        returned_lines += 1

                # Once the requested window is full, only keep counting lines
                # for a short, bounded time.
                window_full = (
                    (requested_limit is not None and returned_lines >= requested_limit)
                    or collected_bytes >= MAX_READ_BYTES
                )
                if window_full:
                    now = time.monotonic()
                    if window_full_at is None:
                        window_full_at = now
                    elif now - window_full_at > count_budget:
                        exact = False
                        scan_reason = f"line counting stopped after {count_budget:g}s"
                        break

        if total_lines < start_line and not exact:
            return {
                "success": False,
                "error": f"offset {start_line} not reached: {scan_reason}",
                "path": str(resolved),
                "size": file_size,
            }

        result: Dict[str, Any] = {
            "success": True,
            "content": ''.join(collected_parts),
            "path": str(resolved),
            "size": file_size,
            "total_lines": total_lines,
            "lines_returned": returned_lines,
            "truncated": truncated,
        }
        if not exact:
            result["total_lines_exact"] = False
            result["reason"] = scan_reason
            result["hint"] = "total_lines is a lower bound (file too large to count fully); use offset/limit to page."
        elif truncated:
            result["reason"] = f"output limited to {MAX_READ_BYTES // 1024}KB"
            result["hint"] = "Use offset/limit to read the rest in pages."
        return result
