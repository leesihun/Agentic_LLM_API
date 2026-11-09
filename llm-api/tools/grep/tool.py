"""
Grep Tool
Content search using ripgrep (rg) with a pure-Python fallback.
Mirrors OpenClaude's GrepTool parameter surface exactly.

Bounded work (a grep call must never hang a turn):
- ripgrep runs as a tracked child in its own process group (registered in
  proc_lifecycle so a user stop / supervisor stop kills the tree), its output is
  streamed with a byte cap, and it is killed early once enough lines for
  offset + head_limit have been produced or the wall-clock deadline passes;
- the Python fallback uses a pruned os.scandir walk (heavy directories skipped,
  no symlink following), a file cap, per-line matching and a Deadline;
- the whole call additionally runs under run_bounded() as a last-resort guard
  (a filesystem hang is abandoned with its daemon thread). That guard CANNOT
  save us from catastrophic regex backtracking: CPython's `re` holds the GIL
  for the whole match, freezing the entire process. So the Python fallback
  rejects patterns with nested unbounded repeats / ambiguous alternation under
  a repeat up front (_regex_risk) and caps the text length handed to `re`;
  ripgrep (no backtracking) is immune;
- timeouts and stops return PARTIAL results with truncated/reason/hint.
"""
import bisect
import fnmatch
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from backend.utils import proc_lifecycle
from tools.file_ops._pathing import (
    HEAVY_DIRS,
    Deadline,
    build_failure_report,
    entry_is_real_dir,
    run_bounded,
    scan_entries,
)


_RG_BIN: Optional[str] = shutil.which("rg")

_VCS_DIRS = (".git", ".svn", ".hg", ".bzr")
_PY_MAX_FILE_BYTES = 10 * 1024 * 1024
_PY_MAX_LINE_CHARS = 20000          # per-line match window (single-line mode)
_PY_MAX_MULTILINE_CHARS = 2 * 1024 * 1024

try:  # Python 3.11+
    from re import _constants as _sre_constants, _parser as _sre_parse
except ImportError:  # Python 3.10
    import sre_constants as _sre_constants
    import sre_parse as _sre_parse

_REPEAT_OPS = (_sre_constants.MAX_REPEAT, _sre_constants.MIN_REPEAT)
_UNBOUNDED_MIN = 100   # {n,m} with m >= this counts as unbounded


def _first_key(sub: Any) -> Any:
    """Comparable key for the first item a branch alternative can match."""
    try:
        items = list(sub)
        if not items:
            return None
        op, av = items[0]
        if op == _sre_constants.LITERAL:
            return ("L", av)
        return ("O", str(op))
    except Exception:
        return None


def _regex_risk(pattern: str, flags: int) -> Optional[str]:
    """Reason string when *pattern* can backtrack catastrophically (nested
    unbounded repeats, or overlapping alternatives under an unbounded repeat);
    None when it looks safe or cannot be analysed."""
    try:
        tree = _sre_parse.parse(pattern, flags)
    except Exception:
        return None

    def has_repeat(sub: Any) -> bool:
        for op, av in sub:
            if op in _REPEAT_OPS:
                if av[1] > 1:
                    return True
                if has_repeat(av[2]):
                    return True
            elif op == _sre_constants.SUBPATTERN:
                if has_repeat(av[-1]):
                    return True
            elif op == _sre_constants.BRANCH:
                if any(has_repeat(b) for b in av[1]):
                    return True
        return False

    def walk(sub: Any, in_unbounded: bool) -> Optional[str]:
        for op, av in sub:
            if op in _REPEAT_OPS:
                unbounded = av[1] >= _UNBOUNDED_MIN or av[1] == _sre_constants.MAXREPEAT
                body = av[2]
                if unbounded and has_repeat(body):
                    return "nested repeat (e.g. (a+)+)"
                hit = walk(body, in_unbounded or unbounded)
                if hit:
                    return hit
            elif op == _sre_constants.SUBPATTERN:
                hit = walk(av[-1], in_unbounded)
                if hit:
                    return hit
            elif op == _sre_constants.BRANCH:
                alts = av[1]
                if in_unbounded:
                    keys = [_first_key(a) for a in alts]   # None == empty alternative
                    if len(set(keys)) < len(keys) or None in keys:
                        return "overlapping alternatives under a repeat (e.g. (a|aa)+)"
                for a in alts:
                    hit = walk(a, in_unbounded)
                    if hit:
                        return hit
            elif op in (_sre_constants.ASSERT, _sre_constants.ASSERT_NOT):
                hit = walk(av[1], in_unbounded)
                if hit:
                    return hit
        return None

    return walk(tree, False)


def _cfg(name: str, default: Any) -> Any:
    value = getattr(config, name, default)
    return default if value is None else value


def _expand_braces(pattern: str) -> List[str]:
    """'*.{ts,tsx}' -> ['*.ts', '*.tsx'] (single nesting level repeated)."""
    match = re.search(r"\{([^{}]*)\}", pattern)
    if not match:
        return [pattern]
    out: List[str] = []
    for alt in match.group(1).split(","):
        out.extend(_expand_braces(pattern[:match.start()] + alt + pattern[match.end():]))
    return out[:64]


def _empty(error: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "success": error is None,
        "results": "",
        "num_matches": 0,
        "truncated": False,
    }
    if error is not None:
        result["error"] = error
    result.update(extra)
    return result


_BENIGN_RG_MARKERS = (
    "os error", "permission denied", "io error", "access is denied", "no such file",
    "being used by another process", "too many levels", "not a directory",
)


def _rg_stderr_is_benign(stderr: str) -> bool:
    """True when every stderr line is a per-entry read problem (the search itself
    ran); False for regex/flag errors or an empty/unknown stderr."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    if not lines:
        return False
    return all(any(m in ln.lower() for m in _BENIGN_RG_MARKERS) for ln in lines)


def _heavy_note(names: List[str]) -> Dict[str, Any]:
    return {
        "note": (
            "Heavy/VCS directories are excluded at any depth: " + ", ".join(names)
            + ". To search inside one, pass it as `path` or name it in `glob`."
        )
    }


class GrepTool:
    """
    Fast content search. Uses ripgrep when available, pure-Python otherwise.
    Results default to files_with_matches (paths only). Use output_mode='content'
    to see matching lines with context.
    """

    def __init__(self, session_id: Optional[str] = None, username: Optional[str] = None,
                 workspace_dir: Optional[Any] = None):
        self.session_id = session_id
        self.username = username
        self.workspace_dir: Optional[Path] = None
        if workspace_dir:
            try:
                self.workspace_dir = Path(workspace_dir).expanduser().resolve()
            except Exception:
                self.workspace_dir = Path(workspace_dir)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def search(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: str = "files_with_matches",
        context: int = 0,
        before: int = 0,
        after: int = 0,
        case_insensitive: bool = False,
        file_type: Optional[str] = None,
        head_limit: int = 250,
        offset: int = 0,
        multiline: bool = False,
    ) -> Dict[str, Any]:
        """
        Search file contents for pattern.

        Args:
            pattern:          Regex to search for.
            path:             Directory or file to search (default: workspace / cwd).
            glob:             Glob filter, e.g. "*.py" or "*.{ts,tsx}".
            output_mode:      "files_with_matches" | "content" | "count".
            context:          Lines of context around each match (rg -C).
            before:           Lines before each match (rg -B).
            after:            Lines after each match (rg -A).
            case_insensitive: Case-insensitive search (rg -i).
            file_type:        Ripgrep file type filter, e.g. "py", "js".
            head_limit:       Cap output at N lines/entries (default 250; 0=unlimited).
            offset:           Skip first N entries before applying head_limit.
            multiline:        Match across newlines (rg -U --multiline-dotall).
        """
        def _int(value: Any, default: int = 0) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default

        context, before, after = _int(context), _int(before), _int(after)
        offset = _int(offset)
        head_limit = 250 if head_limit is None else _int(head_limit, 250)
        output_mode = output_mode or "files_with_matches"
        if not isinstance(pattern, str) or pattern == "":
            return _empty("pattern must be a non-empty string")

        budget = float(_cfg("GREP_TIMEOUT_SECONDS", 30))
        grace = float(_cfg("GREP_HARD_TIMEOUT_GRACE_SECONDS", 5))
        deadline = Deadline(budget)
        try:
            finished, result = run_bounded(
                self._search_inner, pattern, path, glob, output_mode, context, before, after,
                case_insensitive, file_type, head_limit, offset, multiline, deadline,
                timeout=budget + grace, cancel=deadline, name="grep",
            )
        finally:
            deadline.close()
        if not finished:
            return _empty(
                f"grep did not finish within {budget + grace:g}s (filesystem not responding "
                "or pathological regex)",
                truncated=True,
                reason="hard timeout",
                hint="Narrow `path`/`glob`, or simplify the regex (avoid nested quantifiers).",
            )
        return result

    def _search_inner(
        self,
        pattern: str,
        path: Optional[str],
        glob: Optional[str],
        output_mode: str,
        context: int,
        before: int,
        after: int,
        case_insensitive: bool,
        file_type: Optional[str],
        head_limit: int,
        offset: int,
        multiline: bool,
        deadline: Deadline,
    ) -> Dict[str, Any]:
        default_base = self.workspace_dir or getattr(config, "AGENT_DEFAULT_WORKSPACE", None) or Path.cwd()
        base = Path(default_base).resolve()
        if path:
            candidate = Path(path).expanduser()
            if not candidate.is_absolute() and self.workspace_dir:
                candidate = self.workspace_dir / candidate
            search_path = candidate.resolve()
        else:
            search_path = base

        if path is not None and not search_path.exists():
            report = build_failure_report(
                requested=path,
                attempted=[search_path],
                workspace_dir=self.workspace_dir,
                error="search path not found",
            )
            report["results"] = ""
            report["num_matches"] = 0
            report["truncated"] = False
            return report

        rel_base = str(self.workspace_dir) if self.workspace_dir else str(Path.cwd())

        if _RG_BIN:
            return self._rg_search(
                pattern, search_path, glob, output_mode, context, before, after,
                case_insensitive, file_type, head_limit, offset, multiline,
                deadline, rel_base,
            )
        return self._py_search(
            pattern, search_path, glob, output_mode, context, before, after,
            case_insensitive, head_limit, offset, multiline, deadline, rel_base,
        )

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _relativize(line: str, rel_base: str) -> str:
        prefix = rel_base.rstrip("\\/") + os.sep
        if line.startswith(prefix):
            return line[len(prefix):]
        if os.name == "nt" and line.lower().startswith(prefix.lower()):
            return line[len(prefix):]
        return line

    @staticmethod
    def _finish(
        lines: List[str],
        head_limit: int,
        offset: int,
        engine: str,
        incomplete_reason: Optional[str] = None,
        stopped: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply offset/head_limit and build the result dict. *incomplete_reason*
        means the search itself did not run to completion (time/stop/caps)."""
        if offset:
            lines = lines[offset:]
        truncated = False
        reason: Optional[str] = None
        if head_limit and len(lines) > head_limit:
            lines = lines[:head_limit]
            truncated = True
            reason = f"head_limit ({head_limit}) reached"
        result: Dict[str, Any] = {
            "success": True,
            "results": "\n".join(lines),
            "num_matches": len(lines),
            "truncated": truncated,
            "engine": engine,
        }
        if incomplete_reason:
            result["truncated"] = True
            result["reason"] = incomplete_reason
            result["hint"] = (
                "Search did not finish; results are partial. Narrow `path` or `glob`, "
                "use a more specific pattern, or raise the limits via offset/head_limit."
            )
            if stopped or incomplete_reason.startswith(("time limit", "stopped")):
                result["success"] = False
                result["error"] = (
                    incomplete_reason if incomplete_reason.startswith("stopped")
                    else f"grep timed out: {incomplete_reason} (partial results returned)"
                )
        elif truncated:
            result["reason"] = reason
            result["hint"] = (
                "More matches exist. Use offset to page, raise head_limit, or narrow "
                "`path`/`glob`/pattern."
            )
        if extra:
            result.update(extra)
        return result

    # ------------------------------------------------------------------ #
    # Ripgrep backend
    # ------------------------------------------------------------------ #

    def _rg_search(
        self,
        pattern: str,
        search_path: Path,
        glob: Optional[str],
        output_mode: str,
        context: int,
        before: int,
        after: int,
        case_insensitive: bool,
        file_type: Optional[str],
        head_limit: int,
        offset: int,
        multiline: bool,
        deadline: Deadline,
        rel_base: str,
    ) -> Dict[str, Any]:
        args: List[str] = [_RG_BIN, "--no-config", "--hidden"]

        # Exclude VCS dirs, and heavy build/cache dirs unless explicitly targeted
        # (path inside one, or named in the glob filter).
        explicit = set(search_path.parts) | set(re.findall(r"[\w.\-]+", glob or ""))
        excluded_names: List[str] = []
        for name in _VCS_DIRS:
            args += ["--glob", f"!{name}"]
        for name in sorted(HEAVY_DIRS):
            if name not in _VCS_DIRS and name not in explicit:
                args += ["--glob", f"!{name}/"]   # trailing slash: directories only (not files named e.g. `build`)
                excluded_names.append(name)

        args += ["--max-columns", "500", "--max-filesize", "50M"]

        if multiline:
            args += ["-U", "--multiline-dotall"]

        if case_insensitive:
            args.append("-i")

        # Output mode flags
        if output_mode == "files_with_matches":
            args.append("-l")
        elif output_mode == "count":
            args.append("-c")
        else:  # content
            args.append("-n")  # line numbers
            if context:
                args += ["-C", str(context)]
            elif before or after:
                if before:
                    args += ["-B", str(before)]
                if after:
                    args += ["-A", str(after)]

        if file_type:
            args += ["--type", file_type]
        if glob:
            args += ["--glob", glob]

        # Pattern (prefix with -e if it starts with a dash)
        if pattern.startswith("-"):
            args += ["-e", pattern]
        else:
            args.append(pattern)

        args += ["--", str(search_path)]

        needed = (offset + head_limit + 1) if head_limit else 0   # lines needed to prove truncation
        out_cap = int(_cfg("GREP_MAX_OUTPUT_BYTES", 16 * 1024 * 1024))

        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=proc_lifecycle.child_env(),
                # rg anchors `--glob a/b*` to its cwd: run from the search root so
                # slash-containing globs behave like the Python fallback.
                cwd=str(search_path if search_path.is_dir() else search_path.parent),
                **proc_lifecycle.spawn_kwargs(),
            )
        except Exception as exc:
            return _empty(f"rg error: {exc}")

        proc_lifecycle.track(proc)
        entry = None
        try:
            entry = proc_lifecycle.register_process(
                proc,
                command=f"rg {pattern}  # {search_path}",
                tool="grep",
                cwd=str(search_path),
                kind="popen",
                background=False,
            )
        except Exception:
            entry = None

        out_buf = bytearray()
        err_buf = bytearray()
        state = {"lines": 0, "capped": False}
        enough = threading.Event()

        def pump_out() -> None:
            stream = proc.stdout
            try:
                while True:
                    chunk = stream.read1(65536)
                    if not chunk:
                        break
                    if entry is not None:
                        entry.note_output(chunk)
                    if len(out_buf) < out_cap:
                        out_buf.extend(chunk[: out_cap - len(out_buf)])
                        if len(out_buf) >= out_cap:
                            state["capped"] = True
                            enough.set()
                    state["lines"] += chunk.count(b"\n")
                    if needed and state["lines"] >= needed:
                        enough.set()
            except Exception:
                pass

        def pump_err() -> None:
            stream = proc.stderr
            try:
                while True:
                    chunk = stream.read1(8192)
                    if not chunk:
                        break
                    if len(err_buf) < 16384:
                        err_buf.extend(chunk[: 16384 - len(err_buf)])
            except Exception:
                pass

        threads = [
            threading.Thread(target=pump_out, name="grep-rg-out", daemon=True),
            threading.Thread(target=pump_err, name="grep-rg-err", daemon=True),
        ]
        for t in threads:
            t.start()

        killed = False
        incomplete: Optional[str] = None
        stopped = False
        try:
            while proc.poll() is None:
                if enough.is_set():
                    killed = True
                    break
                stop_reason = getattr(entry, "stop_reason", "") if entry is not None else ""
                if stop_reason or deadline.expired:
                    killed = True
                    stopped = bool(stop_reason) or deadline.cancelled
                    if stopped:
                        incomplete = f"stopped: {stop_reason or 'cancelled'}"
                    else:
                        incomplete = deadline.reason()
                    break
                enough.wait(0.05)
            if killed:
                proc_lifecycle.kill_tree_sync(proc, term_grace=0.5, wait_grace=2.0)
            for t in threads:
                t.join(timeout=2.0)
        finally:
            if entry is not None:
                proc_lifecycle.unregister_process(entry)
            if proc.poll() is not None:
                proc_lifecycle.untrack(proc)

        data = bytes(out_buf)
        text = data.decode("utf-8", errors="replace")
        raw_lines = text.splitlines()
        if killed or state["capped"]:
            # The last line may be cut mid-way by the kill / byte cap.
            if raw_lines and not data.endswith(b"\n"):
                raw_lines.pop()
        if state["capped"] and not incomplete and not (needed and len(raw_lines) >= needed):
            incomplete = f"output byte cap ({out_cap}) reached"

        rc = proc.returncode
        if rc is not None and rc >= 2 and not killed and not raw_lines:
            msg = bytes(err_buf).decode("utf-8", errors="replace").strip()
            if not _rg_stderr_is_benign(msg):
                return _empty(f"rg failed (exit {rc}): {msg[:500] or 'unknown error'}", engine="ripgrep")
            # Search ran; only some entries were unreadable -> a normal "no matches" + warnings.

        relativized = [self._relativize(line, rel_base) for line in raw_lines]
        extra: Dict[str, Any] = {}
        if rc == 2 and not killed and err_buf:
            extra["warnings"] = bytes(err_buf).decode("utf-8", errors="replace").strip()[:500]
        if excluded_names:
            extra.update(_heavy_note(excluded_names))
        return self._finish(
            relativized, head_limit, offset, "ripgrep",
            incomplete_reason=incomplete, stopped=stopped, extra=extra or None,
        )

    # ------------------------------------------------------------------ #
    # Pure-Python fallback
    # ------------------------------------------------------------------ #

    def _py_search(
        self,
        pattern: str,
        search_path: Path,
        glob: Optional[str],
        output_mode: str,
        context: int,
        before: int,
        after: int,
        case_insensitive: bool,
        head_limit: int,
        offset: int,
        multiline: bool,
        deadline: Deadline,
        rel_base: str,
    ) -> Dict[str, Any]:
        flags = re.IGNORECASE if case_insensitive else 0
        if multiline:
            flags |= re.DOTALL

        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return _empty(f"Invalid regex: {exc}")
        risk = _regex_risk(pattern, flags)
        if risk:
            return _empty(
                f"Refused regex with catastrophic-backtracking risk: {risk}. The built-in "
                "(non-ripgrep) engine cannot interrupt a runaway match.",
                engine="python-fallback",
                hint="Rewrite without nested quantifiers / ambiguous alternation "
                     "(e.g. use `a+$` instead of `(a+)+$`), or use a simpler literal pattern.",
            )

        files, collect_reason, heavy_skipped = self._collect_files(search_path, glob, deadline)

        output_lines: List[str] = []
        needed = (offset + head_limit + 1) if head_limit else 0
        incomplete: Optional[str] = collect_reason
        ctx_before = max(context, before)
        ctx_after = max(context, after)

        for fpath in files:
            if deadline.expired:
                incomplete = deadline.reason()
                break
            if needed and len(output_lines) >= needed:
                break
            try:
                with open(fpath, "rb") as fh:
                    head = fh.read(8192)
                    if b"\0" in head:
                        continue  # binary
                    data = head + fh.read(_PY_MAX_FILE_BYTES)
                text = data.decode("utf-8", errors="replace")
            except Exception:
                continue

            rel = self._relativize(fpath, rel_base)

            if multiline:
                if len(text) > _PY_MAX_MULTILINE_CHARS:
                    text = text[:_PY_MAX_MULTILINE_CHARS]
                if output_mode == "files_with_matches":
                    if compiled.search(text):
                        output_lines.append(rel)
                    continue
                if output_mode == "count":
                    count = len(compiled.findall(text))
                    if count:
                        output_lines.append(f"{rel}: {count}")
                    continue
                # content: report the lines each multiline match touches
                lines = text.splitlines()
                starts: List[int] = []
                pos = 0
                for ln in lines:
                    starts.append(pos)
                    pos += len(ln) + 1
                matched_indices = set()
                for m in compiled.finditer(text):
                    first = bisect.bisect_right(starts, m.start()) - 1
                    last = bisect.bisect_right(starts, max(m.start(), m.end() - 1)) - 1
                    for idx in range(max(0, first), min(len(lines), last + 1)):
                        matched_indices.add(idx)
                    if deadline.expired:
                        break
                self._emit_content(output_lines, rel, lines, matched_indices, ctx_before, ctx_after)
                continue

            # Single-line mode: match per line (same semantics as ripgrep).
            if output_mode == "files_with_matches":
                for line in text.splitlines():
                    if compiled.search(line[:_PY_MAX_LINE_CHARS]):
                        output_lines.append(rel)
                        break
                continue
            lines = text.splitlines()
            if output_mode == "count":
                count = sum(1 for line in lines if compiled.search(line[:_PY_MAX_LINE_CHARS]))
                if count:
                    output_lines.append(f"{rel}: {count}")
                continue
            matched_indices = {
                i for i, line in enumerate(lines) if compiled.search(line[:_PY_MAX_LINE_CHARS])
            }
            self._emit_content(output_lines, rel, lines, matched_indices, ctx_before, ctx_after)

        return self._finish(
            output_lines, head_limit, offset, "python-fallback",
            incomplete_reason=incomplete if (incomplete and not (needed and len(output_lines) >= needed)) else None,
            extra=_heavy_note(sorted(heavy_skipped)) if heavy_skipped else None,
        )

    @staticmethod
    def _emit_content(
        out: List[str],
        rel: str,
        lines: List[str],
        matched_indices: "set[int]",
        ctx_before: int,
        ctx_after: int,
    ) -> None:
        if not matched_indices:
            return
        shown = set()
        for idx in sorted(matched_indices):
            start = max(0, idx - ctx_before)
            end = min(len(lines), idx + ctx_after + 1)
            for j in range(start, end):
                if j not in shown:
                    out.append(f"{rel}:{j+1}: {lines[j]}")
                    shown.add(j)

    def _collect_files(
        self,
        root: Path,
        glob_pattern: Optional[str],
        deadline: Deadline,
    ) -> Tuple[List[str], Optional[str], "set[str]"]:
        """Pruned, bounded walk. Returns (paths sorted by mtime desc then path,
        incomplete_reason_or_None, names of heavy directories that were skipped)."""
        heavy_skipped: "set[str]" = set()
        max_files = int(_cfg("GREP_MAX_FILES", 50000))
        reason: Optional[str] = None
        collected: List[Tuple[float, str]] = []

        if root.is_file():
            return [str(root)], None, heavy_skipped

        globs = _expand_braces(glob_pattern) if glob_pattern else []

        def matches(name: str, rel_posix: str) -> bool:
            if not globs:
                return True
            for g in globs:
                if "/**/" in g and (fnmatch.fnmatch(rel_posix, g.replace("/**/", "/"))):
                    return True   # `a/**/b` also matches zero directories (a/b), like ripgrep
                if fnmatch.fnmatch(name, g) or fnmatch.fnmatch(rel_posix, g):
                    return True
                if g.startswith("**/") and (
                    fnmatch.fnmatch(name, g[3:]) or fnmatch.fnmatch(rel_posix, g[3:])
                ):
                    return True
            return False

        explicit = set(root.parts) | set(re.findall(r"[\w.\-]+", glob_pattern or ""))
        skip = {d for d in (HEAVY_DIRS | set(_VCS_DIRS)) if d not in explicit}
        root_str = str(root)
        stack: List[str] = [root_str]
        while stack:
            if deadline.expired:
                reason = deadline.reason()
                break
            current = stack.pop()
            entries, scan_reason, _err = scan_entries(current, deadline, 100000)
            if scan_reason and not reason:
                reason = scan_reason
            subdirs: List[str] = []
            for entry in entries:
                name = entry.name
                if entry_is_real_dir(entry):
                    if name not in skip:
                        subdirs.append(entry.path)
                    elif name in HEAVY_DIRS:
                        heavy_skipped.add(name)
                    continue
                try:
                    if not entry.is_file():
                        continue
                    st = entry.stat()
                except OSError:
                    continue
                if st.st_size > _PY_MAX_FILE_BYTES:
                    continue
                if globs:
                    rel = os.path.relpath(entry.path, root_str).replace(os.sep, "/")
                    if not matches(name, rel):
                        continue
                collected.append((st.st_mtime, entry.path))
                if len(collected) >= max_files:
                    reason = f"file limit ({max_files}) reached"
                    break
            if reason and reason.startswith("file limit"):
                break
            for sub in sorted(subdirs, reverse=True):
                stack.append(sub)

        collected.sort(key=lambda item: (-item[0], item[1]))
        return [p for _m, p in collected], reason, heavy_skipped
