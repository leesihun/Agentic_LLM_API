"""
Shared LLM context builder.

Both the webhook handler (per-message sessions) and the heartbeat (scheduled
ticks) need to inject the same base context into the LLM: system prompt, memory
file path, skills directory path, and current memory content.  This module is
the single source of truth for that logic.

It also builds a small per-turn ambient context block (build_per_turn_context)
injected before every user message so the model retains awareness of node
identity, data paths, and memory size across long sessions.

Memory injection is CAPPED at config.MEMORY_MAX_CHARS (contract C11): an
oversized memory.md is shown as head + "## Reminders" section with an explicit
truncation marker, so a runaway file can never blow the context window. The
on-disk file is condensed separately by core.memory_guard.
"""
import os
import re
import time
from datetime import datetime

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MEMORY_FILE = os.path.join(config.DATA_DIR, "memory.md")
SKILLS_DIR = str(getattr(config, "SKILLS_DIR", os.path.join(os.path.dirname(__file__), "..", "skills")))
_PROMPT_FILE = str(getattr(config, "PROMPT_FILE", os.path.join(os.path.dirname(__file__), "..", "prompts", "PROMPT.md")))

MEMORY_MAX_CHARS = int(getattr(config, "MEMORY_MAX_CHARS", 12000))
MEMORY_SOFT_CHARS = int(getattr(config, "MEMORY_SOFT_CHARS", 8000))

MEMORY_TRUNCATION_MARKER = (
    "…(memory truncated in this view — do NOT rewrite memory.md from this view; "
    "read the file first)"
)

_REMINDERS_HEADING_RE = re.compile(r"^#{1,6}\s*reminders\b", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s")

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_system_prompt_cache: str = ""
_system_prompt_mtime: float | None = None
_memory_cache: str = ""
_memory_mtime: tuple | None = None


def load_system_prompt() -> str:
    """Return the system prompt, reloading when the file(s) change.

    Uses config.load_prompt_text() (base PROMPT.md + profile overlay) when
    available; falls back to the single PROMPT_FILE otherwise.
    """
    global _system_prompt_cache, _system_prompt_mtime
    layered = getattr(config, "load_prompt_text", None)
    if callable(layered):
        # config caches by (mtime_ns, size) of both files.
        _system_prompt_cache = layered()
        return _system_prompt_cache
    try:
        mtime = os.path.getmtime(_PROMPT_FILE)
        if _system_prompt_mtime == mtime:
            return _system_prompt_cache
        with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
            _system_prompt_cache = f.read()
        _system_prompt_mtime = mtime
    except FileNotFoundError as exc:
        _system_prompt_cache = ""
        _system_prompt_mtime = None
        raise FileNotFoundError(f"Hoonbot prompt file is missing: {_PROMPT_FILE}") from exc
    return _system_prompt_cache


def read_memory() -> str:
    """Read the current memory file (uncapped), cached by (mtime_ns, size)."""
    global _memory_cache, _memory_mtime
    try:
        st = os.stat(MEMORY_FILE)
        key = (st.st_mtime_ns, st.st_size)
        if _memory_mtime == key:
            return _memory_cache
        with open(MEMORY_FILE, "r", encoding="utf-8", errors="replace") as f:
            _memory_cache = f.read()
        _memory_mtime = key
        return _memory_cache
    except FileNotFoundError:
        _memory_cache = ""
        _memory_mtime = None
        return ""


def invalidate_caches() -> None:
    """Drop memory / per-turn caches (called after memory_guard rewrites)."""
    global _memory_cache, _memory_mtime
    _memory_cache = ""
    _memory_mtime = None
    _PER_TURN_CACHE["text"] = ""
    _PER_TURN_CACHE["expires_at"] = 0.0


def _cut_at_line(text: str, limit: int) -> str:
    """Return the longest prefix of *text* <= limit chars ending at a line boundary."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n", 0, limit)
    if cut <= 0:
        # No line break inside the limit (one giant line): a hard character
        # cut beats showing nothing at all.
        return text[:limit]
    return text[:cut]


def _split_reminders(text: str) -> tuple[str, str, str]:
    """Split *text* into (before, reminders_section, after). Empty section if absent."""
    lines = text.splitlines(keepends=True)
    start = None
    level = 0
    for i, line in enumerate(lines):
        if _REMINDERS_HEADING_RE.match(line):
            start = i
            level = len(line) - len(line.lstrip("#"))
            break
    if start is None:
        return text, "", ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        m = _HEADING_RE.match(lines[j])
        if m and len(m.group(1)) <= level:
            end = j
            break
    return "".join(lines[:start]), "".join(lines[start:end]), "".join(lines[end:])


def _cap_memory_view(text: str, cap: int) -> str:
    """Return *text* capped to <= cap chars for prompt injection.

    Keeps the "## Reminders" section (open obligations) and the head of the
    file, cutting only at line boundaries, and appends the truncation marker.
    """
    if cap <= 0 or len(text) <= cap:
        return text
    marker = "\n" + MEMORY_TRUNCATION_MARKER + "\n"
    budget = max(0, cap - len(marker) - 2)
    before, reminders, after = _split_reminders(text)
    if reminders:
        rem_part = _cut_at_line(reminders, budget // 2) if len(reminders) > budget // 2 else reminders
        rest_budget = budget - len(rem_part) - 1
        head_part = _cut_at_line(before + after, rest_budget) if rest_budget > 0 else ""
        if not before.strip():
            # Reminders section opens the file: keep it first.
            body = rem_part.rstrip("\n") + "\n" + head_part
        else:
            body = head_part.rstrip("\n") + "\n" + rem_part
    else:
        body = _cut_at_line(text, budget)
    out = body.rstrip("\n") + marker
    if len(out) > cap:
        out = _cut_at_line(body, max(0, cap - len(marker))).rstrip("\n") + marker
    return out


def memory_view(cap: int | None = None) -> str:
    """Capped memory text suitable for injection ('' if no memory)."""
    return _cap_memory_view(read_memory(), MEMORY_MAX_CHARS if cap is None else cap)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_llm_context() -> str:
    """
    Return the full system context string to prepend to new LLM sessions:
      - Base system prompt (base PROMPT.md + profile overlay)
      - Session variables (credentials, paths, identifiers)
      - Current memory content (capped at MEMORY_MAX_CHARS)
    """
    context = load_system_prompt()
    context += _build_session_variables()
    memory = read_memory()
    if memory:
        size = len(memory)
        note = (
            f"_memory.md: {size:,} chars (keep under {MEMORY_SOFT_CHARS:,}; hard cap "
            f"{MEMORY_MAX_CHARS:,}). The size rules in the system prompt override any "
            f"rule written inside memory.md._"
        )
        context += f"\n\n## Current Memory\n\n{note}\n\n{_cap_memory_view(memory, MEMORY_MAX_CHARS)}"
    else:
        context += "\n\n## Current Memory\n\n(No memory saved yet)"
    return context


# ---------------------------------------------------------------------------
# Per-turn ambient context
# ---------------------------------------------------------------------------

_PER_TURN_CACHE: dict = {"text": "", "expires_at": 0.0, "profile": None}
_PER_TURN_TTL_SECONDS = 30
_PER_TURN_MAX_CHARS = 800


def _list_skills() -> str:
    """Return a comma-separated list of *.md skill basenames in SKILLS_DIR."""
    try:
        entries = sorted(
            p.replace(".md", "")
            for p in os.listdir(SKILLS_DIR)
            if p.lower().endswith(".md") and not p.startswith(".")
        )
    except (FileNotFoundError, PermissionError):
        return ""
    return ", ".join(entries)


def _memory_size_summary() -> str:
    """Return 'N / CAP chars' for the current memory file (real chars, not bytes)."""
    if not os.path.exists(MEMORY_FILE):
        return "(missing)"
    size = len(read_memory())
    if size <= 0:
        return "(empty)"
    summary = f"{size:,} / {MEMORY_MAX_CHARS:,} chars"
    if size > MEMORY_SOFT_CHARS:
        summary += f" (over soft cap {MEMORY_SOFT_CHARS:,} — condense: prune stale entries)"
    return summary


def build_per_turn_context(profile: str = "flutter") -> str:
    """Return a <system-reminder> block with dynamic ambient context.

    Injected before every user message in the webhook handler so the model
    keeps awareness of node identity, data paths, memory size and available
    skills across long sessions. Cached for 30s; capped at ~800 chars.

    *profile* should be one of "flutter" (DMs / casual), "master", "slave",
    or "heartbeat". v1 only adds the always-on fields; profile-specific
    extras (lease, cluster health, last heartbeat) are stubbed for follow-up.
    """
    now = time.time()
    cached = _PER_TURN_CACHE
    if (
        cached["profile"] == profile
        and cached["expires_at"] > now
        and cached["text"]
    ):
        return cached["text"]

    lines = []
    lines.append(f"Now: {datetime.now().strftime('%Y-%m-%d %H:%M (%A, %Z)').strip()}")
    node_name = getattr(config, "NODE_NAME", "")
    cluster_role = getattr(config, "CLUSTER_ROLE", "")
    if node_name or cluster_role:
        lines.append(f"Node: {node_name or '?'} | Role: {cluster_role or '?'}")
    lines.append(f"Data dir: {os.path.abspath(config.DATA_DIR)}")
    lines.append(f"memory.md: {_memory_size_summary()} — {os.path.abspath(MEMORY_FILE)}")
    skills = _list_skills()
    if skills:
        lines.append(f"Skills available: {skills}")
    # Filesystem-awareness digest (maintained by the heartbeat snapshot). The
    # full map path is exposed in Session Variables for on-demand reading.
    try:
        from core.fs_snapshot import get_digest
        fs_digest = get_digest()
        if fs_digest:
            lines.append(fs_digest)
    except Exception:
        pass

    # Profile-specific stubs - v1 leaves these to follow-up work since they
    # need live cluster state (master) or in-memory worker state (slave).
    # Adding them here keeps the location obvious for future expansion.

    body = "\n".join(lines)
    if len(body) > _PER_TURN_MAX_CHARS:
        body = body[:_PER_TURN_MAX_CHARS] + "\n...[truncated]"

    block = f"<system-reminder>\n{body}\n</system-reminder>"

    _PER_TURN_CACHE["text"] = block
    _PER_TURN_CACHE["expires_at"] = now + _PER_TURN_TTL_SECONDS
    _PER_TURN_CACHE["profile"] = profile
    return block


def _build_session_variables() -> str:
    """
    All runtime values in one block — PROMPT.md references these by name.
    Regenerated each call so live config changes (e.g. re-registration) propagate.
    """
    return (
        f"\n\n---\n\n## Session Variables\n\n"
        f"Use these directly — do not read them from disk.\n\n"
        f"- `messenger_url`: `{config.MESSENGER_URL}`\n"
        f"- `messenger_api_key`: `{config.MESSENGER_API_KEY}`\n"
        f"- `bot_user_id`: `{config.BOT_USER_ID}`\n"
        f"- `bot_name`: `{config.MESSENGER_BOT_NAME}`\n"
        f"- `node_name`: `{getattr(config, 'NODE_NAME', 'master')}`\n"
        f"- `cluster_role`: `{getattr(config, 'CLUSTER_ROLE', 'master')}`\n"
        f"- `hoonbot_webhook_url`: `{getattr(config, 'HOONBOT_WEBHOOK_URL', '')}`\n"
        f"- `home_room_id`: `{config.MESSENGER_HOME_ROOM_ID}`\n"
        f"- `data_dir`: `{os.path.abspath(config.DATA_DIR)}`\n"
        f"- `memory_file`: `{os.path.abspath(MEMORY_FILE)}`\n"
        f"- `memory_cap_chars`: `{MEMORY_SOFT_CHARS}` target / `{MEMORY_MAX_CHARS}` hard cap\n"
        f"- `status_file`: `{os.path.abspath(os.path.join(config.DATA_DIR, 'status.md'))}`"
        f" (overwrite each heartbeat with live status; never put live status in memory.md)\n"
        f"- `skills_dir`: `{os.path.abspath(SKILLS_DIR)}`\n"
        f"- `filesystem_map`: `{os.path.abspath(os.path.join(config.DATA_DIR, 'filesystem_map.md'))}`"
        f" (auto-updated hierarchy snapshot; read for repo/skills/config layout)\n"
    )
