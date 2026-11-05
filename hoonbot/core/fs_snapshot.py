"""Periodic filesystem-hierarchy snapshot for Hoonbot self-awareness.

A pure-Python (no LLM) directory walk that gives the bot durable "spatial
awareness" of its own deployment: what files / skills / configs exist, where
data lives, and what changed since last time. Run from the heartbeat every few
ticks, it writes a compact map to ``data/filesystem_map.md`` and records drift
(added / removed / modified) against the previous scan.

Kept OUT of memory.md on purpose — memory.md is injected into every session
(capped at MEMORY_MAX_CHARS), so a full tree there would eat that budget every
turn. Instead the heartbeat surfaces a one-line digest via
build_per_turn_context, and the agent can file_reader the full map on demand at
the path in Session Variables.

Runtime data dirs (hoonbot/data, llm-api/data, messenger/data) are not walked:
their churn (sessions, logs, memory backups) would drown the drift signal.
Writes are atomic (tmp + os.replace) so a crash never leaves a torn state file.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config

# core/ -> hoonbot/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOONBOT_DIR = Path(__file__).resolve().parents[1]

_DATA_DIR = Path(config.DATA_DIR)
MAP_FILE = _DATA_DIR / "filesystem_map.md"
_STATE_FILE = _DATA_DIR / "filesystem_snapshot.json"

# Directories never worth walking (noise / churn / huge).
_EXCLUDE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "dist-web",
    "build", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "offline_models",
    "models", "llamacpp", ".cache",
}

# Absolute runtime-data directories skipped during the walk (both the resolved
# and the plain absolute spelling, so a symlinked checkout still matches).
_EXCLUDE_PATHS = {
    os.path.normcase(form)
    for p in (
        _DATA_DIR,
        _HOONBOT_DIR / "data",
        _REPO_ROOT / "llm-api" / "data",
        _REPO_ROOT / "messenger" / "data",
    )
    for form in (str(p.resolve()), os.path.abspath(str(p)))
}

# Roots to scan. The repo gives project structure; hoonbot's own runtime dirs
# (skills/prompts) are where the bot's editable behavior lives — drift there is
# the most useful signal.
_ROOTS = [
    _REPO_ROOT,
    _HOONBOT_DIR / "skills",
    _HOONBOT_DIR / "prompts",
]

_MAX_FILES = 4000
_MAX_DEPTH = 4
_RECENT_WINDOW_SECONDS = 86400  # 24h


def _walk(root: Path) -> dict[str, float]:
    """Return {relpath_from_repo_root: mtime} for files under *root*, bounded."""
    out: dict[str, float] = {}
    if not root.exists():
        return out
    root_depth = len(root.resolve().parts)
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs in place so os.walk doesn't descend into them.
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDE_DIRS
            and not d.startswith(".")
            and os.path.normcase(os.path.abspath(os.path.join(dirpath, d))) not in _EXCLUDE_PATHS
        ]
        depth = len(Path(dirpath).resolve().parts) - root_depth
        if depth >= _MAX_DEPTH:
            dirnames[:] = []
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            try:
                rel = p.resolve().relative_to(_REPO_ROOT).as_posix()
            except ValueError:
                rel = p.resolve().as_posix()
            out[rel] = mtime
            if len(out) >= _MAX_FILES:
                return out
    return out


def _collect() -> dict[str, float]:
    files: dict[str, float] = {}
    for root in _ROOTS:
        for rel, mtime in _walk(root).items():
            files[rel] = mtime
        if len(files) >= _MAX_FILES:
            break
    return files


def _load_previous() -> dict[str, Any]:
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _diff(prev: dict[str, float], cur: dict[str, float]) -> tuple[list[str], list[str], list[str]]:
    prev_keys = set(prev)
    cur_keys = set(cur)
    added = sorted(cur_keys - prev_keys)
    removed = sorted(prev_keys - cur_keys)
    modified = sorted(
        k for k in (cur_keys & prev_keys)
        if abs(float(cur[k]) - float(prev.get(k, 0))) > 1.0
    )
    return added, removed, modified


def _dir_summary(files: dict[str, float]) -> list[tuple[str, int]]:
    """Top-level dir -> file count, for a compact tree."""
    counts: dict[str, int] = {}
    for rel in files:
        top = rel.split("/", 1)[0] if "/" in rel else "."
        counts[top] = counts.get(top, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _recent(files: dict[str, float], limit: int = 15) -> list[str]:
    now = time.time()
    recent = [(m, rel) for rel, m in files.items() if now - m <= _RECENT_WINDOW_SECONDS]
    recent.sort(reverse=True)
    return [rel for _, rel in recent[:limit]]


def _fmt_drift(added: list[str], removed: list[str], modified: list[str]) -> str:
    def _sample(items: list[str], n: int = 5) -> str:
        if not items:
            return "none"
        head = ", ".join(items[:n])
        return head + (f", +{len(items) - n} more" if len(items) > n else "")
    return (
        f"Added ({len(added)}): {_sample(added)}\n"
        f"Removed ({len(removed)}): {_sample(removed)}\n"
        f"Modified ({len(modified)}): {_sample(modified)}"
    )


def run_snapshot() -> str:
    """Scan, write the map + state, and return a one-line digest.

    Safe to call from the heartbeat; never raises (returns "" on failure).
    """
    try:
        cur = _collect()
    except Exception as exc:  # pragma: no cover - defensive
        return f"Filesystem: scan failed ({type(exc).__name__})"

    prev_state = _load_previous()
    prev_files = prev_state.get("files", {}) if isinstance(prev_state, dict) else {}
    added, removed, modified = _diff(prev_files, cur)

    dir_counts = _dir_summary(cur)
    n_files = len(cur)
    n_dirs = len(dir_counts)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- one-line digest for ambient context ----
    drift_bits = []
    if prev_files:
        drift_bits.append(f"+{len(added)}/-{len(removed)}/~{len(modified)} since last scan")
        sample = (added or modified or removed)
        if sample:
            drift_bits.append(f"e.g. {sample[0]}")
    digest = f"Filesystem: {n_files} files / {n_dirs} top-level dirs" + (
        f"; {'; '.join(drift_bits)}" if drift_bits else ""
    )

    # ---- full map ----
    lines = [
        "# Filesystem Map",
        "_Auto-generated by Hoonbot. Read on demand; do not treat as memory._",
        "",
        f"Generated: {ts}",
        f"Repo root: {_REPO_ROOT.as_posix()}",
        f"Files: {n_files} across {n_dirs} top-level dirs",
        "",
        "## Drift since last scan",
        (_fmt_drift(added, removed, modified) if prev_files else "(first scan — no prior snapshot)"),
        "",
        "## Top-level layout (file counts)",
    ]
    for top, count in dir_counts:
        lines.append(f"- {top}/  ({count} files)")
    lines.append("")
    lines.append("## Recently modified (24h)")
    recent = _recent(cur)
    if recent:
        lines.extend(f"- {r}" for r in recent)
    else:
        lines.append("(none)")

    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(MAP_FILE, "\n".join(lines) + "\n")
        _atomic_write_text(
            _STATE_FILE,
            json.dumps({"generated_at": ts, "digest": digest, "files": cur}, ensure_ascii=False),
        )
    except Exception:
        pass  # snapshot is best-effort; never break the heartbeat

    return digest


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via a same-dir tmp file + os.replace."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                # Windows: a concurrent reader can block the rename briefly.
                if attempt == 9:
                    raise
                time.sleep(0.1)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


_digest_cache: dict[str, Any] = {"key": None, "digest": ""}


def get_digest() -> str:
    """Return the last digest cheaply (no walk) for ambient context. '' if none."""
    try:
        st = _STATE_FILE.stat()
        key = (st.st_mtime_ns, st.st_size)
        if _digest_cache["key"] == key:
            return _digest_cache["digest"]
        state = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        digest = str(state.get("digest", "")) if isinstance(state, dict) else ""
        _digest_cache["key"] = key
        _digest_cache["digest"] = digest
        return digest
    except Exception:
        return ""
