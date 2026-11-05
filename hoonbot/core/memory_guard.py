"""
memory.md size guard (contract C10).

memory.md is injected into every new LLM session, so an unbounded file
eventually overflows the context window and every turn fails. This module keeps
the ON-DISK file bounded:

    memory_size()           -> chars in memory.md (0 if missing)
    memory_fingerprint()    -> sha1 of memory.md content
    enforce_memory_cap(reason="", force=False) -> bool   (never raises)
    register_on_condensed(cb)

enforce_memory_cap():
  1. asks the LLM for a condensed rewrite (tool-less, ephemeral) targeting
     <= MEMORY_SOFT_CHARS,
  2. validates it (non-empty, fits under MEMORY_MAX_CHARS with room for the
     guard note, not suspiciously short, keeps "## Reminders" when the
     original had one),
  3. falls back to a deterministic trim (headings + Reminders + newest lines)
     when the LLM path fails and the file is over the hard cap / forced,
  4. re-checks that memory.md did not change meanwhile, byte-copies it to
     data/memory_backups/ (only now — aborted/failed attempts make no
     backup; plain and pre-trim backups each keep the newest
     MEMORY_BACKUP_KEEP copies, reset.py's ``-reset`` copies are never
     rotated), then atomically replaces it (tmp + re-stat + os.replace).
Calls are serialized by a module-level asyncio.Lock.

Callers: hoonbot.py startup (background), every heartbeat tick, and the
webhook post-turn hook (size > soft cap; force=True after a context overflow).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Callable

import config

logger = logging.getLogger(__name__)

MEMORY_MAX_CHARS = int(getattr(config, "MEMORY_MAX_CHARS", 12000))
MEMORY_SOFT_CHARS = min(int(getattr(config, "MEMORY_SOFT_CHARS", 8000)), MEMORY_MAX_CHARS)
MEMORY_BACKUP_KEEP = max(1, int(getattr(config, "MEMORY_BACKUP_KEEP", 10)))
_MIN_INTERVAL = float(getattr(config, "MEMORY_CONDENSE_MIN_INTERVAL_SECONDS", 600))
_CONDENSE_TIMEOUT = float(getattr(config, "MEMORY_CONDENSE_TIMEOUT_SECONDS", 600))
_FAILURE_COOLDOWN = 300.0
# Deterministic-trim target: leaves room for the guard note line so the
# result lands under the soft cap (and is not re-condensed next startup).
_TRIM_TARGET = max(200, MEMORY_SOFT_CHARS - 300)
# Inputs larger than this are pre-trimmed deterministically before the LLM call
# so the condense request itself cannot overflow the context window.
_LLM_INPUT_MAX_CHARS = 120_000
# Room kept free in an accepted LLM rewrite for the guard note prepended on
# commit, so the note never pushes the file over MEMORY_MAX_CHARS.
_NOTE_RESERVE = 400

MEMORY_FILE = os.path.join(config.DATA_DIR, "memory.md")
BACKUP_DIR = os.path.join(config.DATA_DIR, "memory_backups")

_REMINDERS_RE = re.compile(r"^#{1,6}\s*reminders\b", re.IGNORECASE | re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
_GUARD_NOTE_RE = re.compile(r"^<!-- memory_guard:.*?-->\s*\n?", re.MULTILINE)

_lock: asyncio.Lock | None = None
_callbacks: list[Callable[..., Any]] = []
_last_rewrite_at: float = 0.0
_last_written_fp: str = ""
_last_failure_at: float = 0.0


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _stat_key() -> tuple | None:
    try:
        st = os.stat(MEMORY_FILE)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read() -> str:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def memory_size() -> int:
    """Number of characters in memory.md (0 if missing/unreadable)."""
    try:
        return len(_read())
    except Exception:
        return 0


def memory_fingerprint() -> str:
    """sha1 hex digest of memory.md content ('' content hashes too)."""
    try:
        text = _read()
    except Exception:
        text = ""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def register_on_condensed(cb: Callable[..., Any]) -> None:
    """Register a callback run after memory.md was rewritten.

    The callback may be sync or async and may take no arguments or one info
    dict ({"reason", "before", "after", "method", "backup"}).
    """
    if cb not in _callbacks:
        _callbacks.append(cb)


def _takes_no_args(cb: Callable[..., Any]) -> bool:
    try:
        inspect.signature(cb).bind()
        return True
    except TypeError:
        return False
    except ValueError:  # no signature available (some builtins)
        return True


def _get_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


# ---------------------------------------------------------------------------
# Backup / write helpers
# ---------------------------------------------------------------------------

# memory-<ts>.md (before an LLM condense) / memory-<ts>-pretrim.md (before a
# lossy deterministic trim) / memory-<ts>-reset.md (reset.py, never rotated).
_BACKUP_NAME_RE = re.compile(r"^memory-(\d{8}-\d{6}-\d{6})(?:-([a-z]+))?\.md$")
_ROTATED_KINDS = ("", "pretrim")


class _ConcurrentWrite(Exception):
    """memory.md changed after it was read — the rewrite must not land."""


def _backup(kind: str = "") -> str:
    """Byte-copy the current memory.md into BACKUP_DIR (raw bytes, so invalid
    UTF-8 / CRLF survive) and rotate. Plain and ``pretrim`` backups rotate in
    separate buckets so a churn of plain backups can never evict the copy
    taken before a lossy trim; other suffixes (``-reset``) are never touched.
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)
    suffix = f"-{kind}" if kind else ""
    name = f"memory-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}{suffix}.md"
    path = os.path.join(BACKUP_DIR, name)
    shutil.copy2(MEMORY_FILE, path)
    try:
        buckets: dict[str, list[str]] = {}
        for p in os.listdir(BACKUP_DIR):
            m = _BACKUP_NAME_RE.match(p)
            if m and (m.group(2) or "") in _ROTATED_KINDS:
                buckets.setdefault(m.group(2) or "", []).append(p)
        for names in buckets.values():
            for old in sorted(names)[:-MEMORY_BACKUP_KEEP]:
                try:
                    os.remove(os.path.join(BACKUP_DIR, old))
                except OSError:
                    pass
    except OSError:
        pass
    return path


def _atomic_write(text: str, expected: tuple | None = None) -> None:
    """Atomically replace memory.md with *text*. With *expected* (a
    ``_stat_key()``), re-stat right before ``os.replace`` and raise
    ``_ConcurrentWrite`` instead of clobbering a write that landed after the
    caller's last check."""
    directory = os.path.dirname(MEMORY_FILE) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".memory.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        # mkstemp creates 0600; keep memory.md's existing permissions (POSIX).
        try:
            os.chmod(tmp, os.stat(MEMORY_FILE).st_mode & 0o7777)
        except OSError:
            pass
        last_exc: Exception | None = None
        # Windows: a reader holding the file open makes os.replace fail
        # transiently with PermissionError.
        for _ in range(10):
            if expected is not None and _stat_key() != expected:
                raise _ConcurrentWrite("memory.md changed just before commit")
            try:
                os.replace(tmp, MEMORY_FILE)
                return
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.1)
        raise last_exc or OSError("os.replace failed")
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Deterministic trim
# ---------------------------------------------------------------------------

def _reminders_span(lines: list[str]) -> tuple[int, int] | None:
    for i, line in enumerate(lines):
        if _REMINDERS_RE.match(line):
            level = len(line) - len(line.lstrip("#"))
            end = len(lines)
            for j in range(i + 1, len(lines)):
                m = _HEADING_RE.match(lines[j])
                if m and len(m.group(1)) <= level:
                    end = j
                    break
            return i, end
    return None


def _hard_cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    return text[:cut] if cut > 0 else text[:limit]


def _split_line(line: str, piece: int) -> list[str]:
    """Split one overlong line into <= *piece*-char parts, preferring sentence
    then word boundaries in the back half of each window."""
    parts: list[str] = []
    while len(line) > piece:
        cut = line.rfind(". ", piece // 2, piece)
        if cut > 0:
            cut += 1  # keep the period with its sentence
        else:
            cut = line.rfind(" ", piece // 2, piece)
            if cut <= 0:
                cut = piece
        parts.append(line[:cut].rstrip())
        line = line[cut:].lstrip()
    if line:
        parts.append(line)
    return parts


def _deterministic_trim(text: str, target: int, limit: int | None = None) -> str:
    """Keep headings, the Reminders section and the newest lines up to *target*.

    Lines keep their original order. The result never exceeds *limit*
    (default ``MEMORY_MAX_CHARS - 300``, leaving room for the guard note);
    when it would, the OLDEST unprotected lines are dropped first, so the
    newest content and the Reminders section survive. Lines too long to fit
    the budget whole (e.g. a memory.md written as one huge line) are split
    first, so the trim keeps their newest part instead of dropping them.
    """
    cap = (MEMORY_MAX_CHARS - 300) if limit is None else int(limit)
    text = _GUARD_NOTE_RE.sub("", text)
    split_over = max(400, target // 2)
    piece = max(200, target // 8)
    lines: list[str] = []
    for raw_line in text.splitlines():
        if len(raw_line) > split_over:
            lines.extend(_split_line(raw_line, piece))
        else:
            lines.append(raw_line)
    n = len(lines)
    keep = [False] * n
    protected = [False] * n
    used = 0

    # 1. Reminders section (open obligations), capped at half the budget.
    span = _reminders_span(lines)
    if span:
        start, end = span
        rem_budget = max(200, target // 2)
        for i in range(start, end):
            cost = len(lines[i]) + 1
            if used + cost > rem_budget and i != start:
                break
            keep[i] = True
            protected[i] = True
            used += cost

    # 2. Top-level structure (level <= 2 headings), cheap and orientation-preserving.
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) <= 2 and not keep[i]:
            cost = len(line) + 1
            if used + cost > target:
                break
            keep[i] = True
            protected[i] = True
            used += cost

    # 3. Newest content: fill from the bottom, skipping lines that don't fit.
    for i in range(n - 1, -1, -1):
        if keep[i]:
            continue
        line = lines[i]
        if not line.strip():
            continue
        cost = len(line) + 1
        if used + cost > target:
            continue
        keep[i] = True
        used += cost

    # 4. Emit in original order (a blank line before each heading).
    def _emit() -> str:
        out: list[str] = []
        for i in range(n):
            if keep[i]:
                m = _HEADING_RE.match(lines[i])
                if m and out and out[-1].strip():
                    out.append("")
                out.append(lines[i])
        return "\n".join(out).strip() + "\n"

    result = _emit()
    # 5. Over the cap (target near/above it, or separator blank lines): drop
    #    the OLDEST unprotected lines — never the tail, which holds the newest
    #    content (and often the Reminders section).
    while len(result) > cap:
        excess = len(result) - cap
        freed = 0
        for i in range(n):
            if keep[i] and not protected[i]:
                keep[i] = False
                freed += len(lines[i]) + 1
                if freed >= excess:
                    break
        if not freed:
            break  # only protected lines left
        result = _emit()
    return _hard_cut(result, cap)


def _with_note(text: str, method: str, backup: str) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    note = (
        f"<!-- memory_guard: condensed {stamp} via {method}; "
        f"previous version: {os.path.basename(backup) if backup else 'no backup'} "
        f"in data/memory_backups -->\n"
    )
    body = _GUARD_NOTE_RE.sub("", text).lstrip("\n")
    return note + body


# ---------------------------------------------------------------------------
# LLM condense
# ---------------------------------------------------------------------------

_CONDENSE_SYSTEM = """You condense a bot's long-term memory file (markdown).
Output ONLY the new memory.md content as plain markdown. No preamble, no code fences, no commentary, no tool calls.

Hard limits:
- Target about {target} characters. Never exceed {soft} characters.
- Keep the same language(s) as the input. Keep dates on entries that have them.

Keep (as concise bullets):
- Durable facts about the user, people, projects, machines, paths, IDs, URLs, credentials references, and decisions.
- User preferences and standing instructions.
- The entire "## Reminders" section with every still-open reminder (heading text unchanged).

Drop or merge:
- Live status, metrics, counters, health checks, timestamps of routine runs, logs, command output, transcripts, and anything superseded by a newer entry.
- Duplicates and near-duplicates (keep the newest wording).
- Rules that tell the bot to save or display everything in full: replace them with a single short bullet under "User preferences" describing what the user actually wants to see.
"""


def _clean_llm_output(raw: Any) -> str:
    if isinstance(raw, dict):
        try:
            raw = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raw = raw.get("content") or ""
    text = str(raw or "")
    text = _THINK_RE.sub("", text)
    # Unterminated <think> (reasoning parser missing): keep what follows </think>
    # or drop everything if it never closed.
    low = text.lower()
    if "<think>" in low:
        idx = low.rfind("</think>")
        text = text[idx + len("</think>"):] if idx >= 0 else ""
    elif "</think>" in low:
        # Opening tag consumed by the chat template: reasoning precedes it.
        text = text[low.rfind("</think>") + len("</think>"):]
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text + "\n" if text else ""


def _validate(new: str, original: str) -> str | None:
    """Return an error string, or None when *new* is acceptable."""
    if not new.strip():
        return "empty output"
    # Leave room for the guard note prepended on commit: a rewrite that only
    # fits without it would be tail-cut afterwards (dropping a trailing
    # Reminders section after this check already passed).
    max_len = MEMORY_MAX_CHARS - _NOTE_RESERVE
    if len(new) > max_len:
        return f"output too long ({len(new)} > {max_len})"
    if len(new) >= len(original):
        return "output not shorter than the original"
    # Floor scales with the input (a quarter of it, capped at a quarter of
    # the soft cap), so a one-paragraph non-answer can't replace the memory.
    min_len = max(200, min(len(original) // 4, MEMORY_SOFT_CHARS // 4))
    if len(new) < min_len:
        return f"output suspiciously short ({len(new)} < {min_len} chars)"
    low = new.lower()
    if "<tool_call>" in low or "<think>" in low:
        return "output contains tool-call / reasoning markup"
    if _REMINDERS_RE.search(original) and not _REMINDERS_RE.search(new):
        return "output dropped the Reminders section"
    return None


async def _llm_condense(text: str) -> str:
    from core import llm_api  # lazy: avoid import cycles at module load

    source = text
    if len(source) > _LLM_INPUT_MAX_CHARS:
        # limit= keeps the default MEMORY_MAX_CHARS-sized cap from cutting the
        # 120k pre-trim down to the oldest ~12k chars.
        source = _deterministic_trim(source, _LLM_INPUT_MAX_CHARS, limit=_LLM_INPUT_MAX_CHARS)
    target = int(MEMORY_SOFT_CHARS * 0.75)
    messages = [
        {"role": "system", "content": _CONDENSE_SYSTEM.format(target=target, soft=MEMORY_SOFT_CHARS)},
        {
            "role": "user",
            "content": (
                f"Current memory.md ({len(text):,} chars). Rewrite it to about "
                f"{target:,} chars (max {MEMORY_SOFT_CHARS:,}).\n\n"
                f"<memory>\n{source}\n</memory>"
            ),
        },
    ]
    payload = {"model": getattr(config, "LLM_MODEL", ""), "messages": json.dumps(messages, ensure_ascii=False)}
    headers = {"Authorization": f"Bearer {getattr(config, 'LLM_API_KEY', '')}"}
    kwargs: dict = {
        "disable_tools": True,
        "ephemeral": True,
        "reasoning_effort": "high",
    }
    chat_json = getattr(llm_api, "chat_json", None)
    call = chat_json if callable(chat_json) else llm_api.chat
    if "kind" in inspect.signature(call).parameters:
        kwargs["kind"] = "heartbeat"  # keep condensation off the interactive pool
    raw = await call(payload, headers, timeout=_CONDENSE_TIMEOUT, **kwargs)
    if isinstance(raw, dict):
        # A cut-off or fallback reply must never replace memory.md: with
        # finish_reason "length" the rewrite is silently missing its tail, and
        # a fallback_used reply is llm-api's canned "no answer" notice.
        if raw.get("error"):
            err = raw.get("error")
            raise ValueError(f"LLM error: {str(err.get('message') if isinstance(err, dict) else err)[:200]}")
        diag = raw.get("x_diagnostics") if isinstance(raw.get("x_diagnostics"), dict) else {}
        finish = ""
        try:
            finish = str(raw["choices"][0].get("finish_reason") or "")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        finish = str(diag.get("finish_reason") or finish or "")
        if finish in ("length", "error", "interrupted"):
            raise ValueError(f"condense reply incomplete (finish_reason={finish})")
        if diag.get("fallback_used"):
            raise ValueError("condense reply is a fallback notice, not a rewrite")
    return _clean_llm_output(raw)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

async def _run_callbacks(info: dict) -> None:
    try:
        from core import context as _context
        _context.invalidate_caches()
    except Exception:
        pass
    for cb in list(_callbacks):
        try:
            # Decide arity up front: a TypeError raised INSIDE cb() must not
            # trigger a second call with the info dict.
            result = cb() if _takes_no_args(cb) else cb(info)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"[MemoryGuard] on_condensed callback failed: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def enforce_memory_cap(reason: str = "", force: bool = False) -> bool:
    """Condense memory.md when it exceeds the soft cap. Returns True if rewritten.

    Never raises (except CancelledError). With force=True (context overflow)
    the soft-cap/hysteresis checks are skipped, but a file that is already
    small (<= soft/2) is left alone — the overflow was not memory's fault.
    """
    global _last_rewrite_at, _last_written_fp, _last_failure_at
    tag = f"[MemoryGuard]{'[' + reason + ']' if reason else ''}"
    try:
        lock = _get_lock()
        if lock.locked() and not force:
            return False
        async with lock:
            # Stat BEFORE reading: a write landing between the two then shows
            # up as a changed stat at commit time instead of being overwritten.
            stat_before = _stat_key()
            original = _read()
            size = len(original)
            now = time.time()
            if size == 0:
                return False
            if force:
                if size <= MEMORY_SOFT_CHARS // 2:
                    return False
            else:
                if size <= MEMORY_SOFT_CHARS:
                    return False
                if size <= MEMORY_MAX_CHARS:
                    if now - _last_rewrite_at < _MIN_INTERVAL:
                        return False
                    if now - _last_failure_at < _FAILURE_COOLDOWN:
                        return False
                fp = hashlib.sha1(original.encode("utf-8", errors="replace")).hexdigest()
                if fp == _last_written_fp and size <= MEMORY_MAX_CHARS:
                    return False

            logger.info(f"{tag} memory.md is {size:,} chars (soft {MEMORY_SOFT_CHARS:,} / max {MEMORY_MAX_CHARS:,}) — condensing")
            if "\ufffd" in original:
                logger.warning(
                    f"{tag} memory.md contains undecodable (non-UTF-8) bytes — they are "
                    "lost in the rewrite; the byte-exact original is kept in the backup"
                )

            method = "llm"
            new_text = ""
            try:
                new_text = await _llm_condense(original)
                err = _validate(new_text, original)
                if err:
                    raise ValueError(err)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"{tag} LLM condense failed: {exc}")
                # A deterministic trim only helps when it actually cuts
                # something (> soft cap); below that it would just churn the
                # file and invalidate every cached room session.
                if size > MEMORY_MAX_CHARS or (force and size > MEMORY_SOFT_CHARS):
                    method = "deterministic"
                    new_text = _deterministic_trim(original, _TRIM_TARGET)
                else:
                    _last_failure_at = time.time()
                    return False

            # Someone (the agent) wrote memory.md while we were condensing.
            if _stat_key() != stat_before:
                # Stat paired with the re-read, so a later write is caught at
                # commit time too.
                stat_before = _stat_key()
                fresh = _read()
                if len(fresh) <= MEMORY_SOFT_CHARS or not (force or len(fresh) > MEMORY_MAX_CHARS):
                    logger.info(f"{tag} memory.md changed during condense — aborting this round")
                    # Cool down: otherwise the next post-turn check relaunches
                    # a multi-minute condense immediately, in a loop.
                    _last_failure_at = time.time()
                    return False
                method = "deterministic"
                new_text = _deterministic_trim(fresh, _TRIM_TARGET)
                original = fresh

            # Backup only now — after validation and the concurrent-write
            # check — so failed/aborted attempts never churn the backup ring.
            try:
                backup = await asyncio.to_thread(
                    _backup, "pretrim" if method == "deterministic" else ""
                )
            except Exception as exc:
                logger.warning(f"{tag} backup failed: {exc}")
                if len(original) <= MEMORY_MAX_CHARS:
                    _last_failure_at = time.time()
                    return False
                backup = ""

            new_text = _with_note(new_text, method, backup)
            if len(new_text) > MEMORY_MAX_CHARS:
                # Never tail-cut: that would silently drop a trailing Reminders
                # section. The deterministic trim protects it and fits the note.
                method = "deterministic"
                new_text = _with_note(_deterministic_trim(original, _TRIM_TARGET), method, backup)
            if len(new_text) > MEMORY_MAX_CHARS:
                new_text = _hard_cut(new_text, MEMORY_MAX_CHARS)

            try:
                await asyncio.to_thread(_atomic_write, new_text, stat_before)
            except _ConcurrentWrite:
                logger.info(f"{tag} memory.md changed just before commit — aborting this round")
                _last_failure_at = time.time()
                return False
            _last_rewrite_at = time.time()
            _last_written_fp = hashlib.sha1(new_text.encode("utf-8", errors="replace")).hexdigest()
            logger.info(
                f"{tag} memory.md condensed {len(original):,} -> {len(new_text):,} chars "
                f"via {method}; backup={backup or 'none'}"
            )
            info = {
                "reason": reason,
                "before": len(original),
                "after": len(new_text),
                "method": method,
                "backup": backup,
            }
        await _run_callbacks(info)
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(f"{tag} enforce_memory_cap failed: {exc}", exc_info=True)
        return False
