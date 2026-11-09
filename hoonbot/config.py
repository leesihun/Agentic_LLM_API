"""
Hoonbot configuration — single source of truth.

This is the single source of truth for all Hoonbot settings. Edit values
directly in this file. Variables that should be runtime-overridable use
``os.environ.get``; the rest are plain Python constants.

This module deliberately has NO external file dependency for static
settings — it does NOT read ``settings.txt`` or any other config file.
The only files it does read are runtime credential blobs written by the
setup scripts (``data/.llm_key``, ``data/.llm_model``).

Conventions:
    * Constants are uppercase module-level attributes.
    * Anything that callers in ``hoonbot.py`` / ``core/`` / ``handlers/``
      read from this module is preserved by name. Renames are propagated
      to all read sites or simply not done.
"""

import importlib.util
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).parent
DATA_DIR = str(_BASE_DIR / "data")
PROMPTS_DIR = _BASE_DIR / "prompts"


def prompt_path(relative_path: str) -> Path:
    return PROMPTS_DIR / relative_path


def read_prompt(relative_path: str) -> str:
    return prompt_path(relative_path).read_text(encoding="utf-8")


def _load_cluster_config():
    path = _BASE_DIR.parent / "cluster_config.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_hoonbot_cluster_config", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CLUSTER = _load_cluster_config()


def _read_credential_file(name: str) -> str:
    """Read a runtime credential file from ``data/``. Returns "" if absent.

    These files are written by the setup scripts (``scripts/setup_credentials.py``)
    and the Messenger bot registration step. They are intentionally NOT inline
    constants because they contain secrets and per-install values.
    """
    path = _BASE_DIR / "data" / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
HOONBOT_PORT = int(getattr(_CLUSTER, "HOONBOT_PORT", 10001))
HOONBOT_HOST = getattr(_CLUSTER, "HOONBOT_BIND_HOST", "0.0.0.0")
MESSENGER_PORT = int(getattr(_CLUSTER, "MESSENGER_PORT", 10006))
LLM_API_PORT = int(getattr(_CLUSTER, "LLM_API_PORT", 10002))



# ---------------------------------------------------------------------------
# External Services — Messenger
# ---------------------------------------------------------------------------
MESSENGER_URL = os.environ.get(
    "MESSENGER_URL",
    getattr(_CLUSTER, "MESSENGER_URL", f"http://127.0.0.1:{MESSENGER_PORT}"),
).rstrip("/")
MESSENGER_API_KEY = ""  # populated at runtime after bot registration


# ---------------------------------------------------------------------------
# External Services — LLM API
# ---------------------------------------------------------------------------
# LLM_API_URL is the one legitimate runtime override: it lets ops point
# Hoonbot at a remote LLM API host without editing this file.
_LLM_API_URL_OVERRIDE = os.environ.get("LLM_API_URL", "").strip().rstrip("/")
_LLM_API_LOCAL_URL = getattr(_CLUSTER, "HOONBOT_LLM_API_URL", f"http://127.0.0.1:{LLM_API_PORT}").rstrip("/")

LLM_API_URL = _LLM_API_URL_OVERRIDE or _LLM_API_LOCAL_URL

# Ordered candidate list for autofind in hoonbot.py: explicit override first,
# then the local URL.
LLM_API_CANDIDATES: list[str] = []
if _LLM_API_URL_OVERRIDE:
    LLM_API_CANDIDATES.append(_LLM_API_URL_OVERRIDE)
if _LLM_API_LOCAL_URL not in LLM_API_CANDIDATES:
    LLM_API_CANDIDATES.append(_LLM_API_LOCAL_URL)

# Credentials used by scripts/setup_credentials.py to obtain an API token.
LLM_API_USERNAME = "admin"
LLM_API_PASSWORD = "administrator"

# Runtime credential files (created by setup scripts).
LLM_API_KEY = _read_credential_file(".llm_key")
LLM_MODEL = _read_credential_file(".llm_model")


# ---------------------------------------------------------------------------
# Bot Identity
# ---------------------------------------------------------------------------
# Bot name + home room — edit in the root cluster_config.py EDIT HERE block.
MESSENGER_BOT_NAME = getattr(_CLUSTER, "BOT_NAME", "Bot")
# When MESSENGER_HOME_ROOM_NAME is set, the room is resolved by name at
# startup and MESSENGER_HOME_ROOM_ID is overwritten with the resolved id.
# Set MESSENGER_HOME_ROOM_NAME = "" to fall back to the numeric id below.
MESSENGER_HOME_ROOM_NAME = getattr(_CLUSTER, "BOT_HOME_ROOM_NAME", "Heartbeat")
MESSENGER_HOME_ROOM_ID = 1
# Populated at startup from /api/bots/me — used for context injection.
BOT_USER_ID: int = 0


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------
# Heartbeat settings — edit in the root cluster_config.py EDIT HERE block.
HEARTBEAT_ENABLED = bool(getattr(_CLUSTER, "HEARTBEAT_ENABLED", True))
HEARTBEAT_INTERVAL_SECONDS = int(getattr(_CLUSTER, "HEARTBEAT_INTERVAL_SECONDS", 1200))
# If heartbeat hits LLM connectivity errors, pause heartbeat LLM calls
# for this many seconds.
HEARTBEAT_LLM_COOLDOWN_SECONDS = 600
# Active hours window (24h HH:MM). Heartbeat only runs in this window.
HEARTBEAT_ACTIVE_START = getattr(_CLUSTER, "HEARTBEAT_ACTIVE_START", "00:00")
HEARTBEAT_ACTIVE_END = getattr(_CLUSTER, "HEARTBEAT_ACTIVE_END", "23:59")


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------
# Debounce window for rapid messages (seconds). Each room has a FIFO queue and a
# single worker: the worker waits this long after the latest message to gather a
# burst, then answers up to BATCH_MAX_MESSAGES queued messages in ONE turn as
# separately labelled blocks (each message stays distinguishable — they are not
# blindly concatenated). Messages that arrive while a turn is running are
# handled per FOLLOWUP_POLICY.
DEBOUNCE_SECONDS = 1.5
# What happens to a message that arrives while the room's turn is still running
# — edit in the root cluster_config.py EDIT HERE block:
#   "preempt"   (default) after the debounce, STOP the running turn (its tools are
#               killed, its draft bubble keeps the partial text) and answer the
#               new message(s) right away together with the interrupted one(s).
#               A turn that is already writing its answer / delegating to a
#               cluster node / compacting is never preempted: the message waits
#               like "queue" (⏳ reaction).
#   "queue"     answered in the NEXT turn, in order (⏳ reaction).
#   "interrupt" a follow-up from the SAME sender cancels and merges into the
#               running turn; other senders still queue.
FOLLOWUP_POLICY = str(getattr(_CLUSTER, "FOLLOWUP_POLICY", "preempt")).strip().lower()
if FOLLOWUP_POLICY not in {"preempt", "queue", "interrupt"}:
    import sys as _sys
    print(f"[hoonbot config] WARNING: unknown FOLLOWUP_POLICY {FOLLOWUP_POLICY!r} "
          f"(expected preempt | queue | interrupt) - using 'queue'", file=_sys.stderr)
    FOLLOWUP_POLICY = "queue"
# Max queued (not yet answered) messages per room; beyond this the oldest
# pending ones are dropped with a warning.
ROOM_QUEUE_MAX = 50
# Max queued messages answered together in one turn (the rest go to the next).
BATCH_MAX_MESSAGES = 20
# Hard wall-clock cap (seconds) for NON-streaming LLM API calls that don't pass
# an explicit timeout (core/llm_api.chat). Streaming calls (interactive turns,
# heartbeat tasks, cluster tasks) are not capped by wall clock — they are
# guarded by LLM_STREAM_IDLE_TIMEOUT_SECONDS instead, while llm-api's SSE
# keepalive pings (~15s) keep a healthy-but-slow stream alive. Server-side,
# llm-api's STREAM_FIRST_BYTE_TIMEOUT_SECONDS / STREAM_IDLE_TIMEOUT_SECONDS
# stop a stalled vLLM stream. 3600s fits a cold 1M-token prefill plus a long
# agentic run.
LLM_TIMEOUT_SECONDS = 3600
# Streaming idle watchdog (seconds): if NO bytes (not even a keepalive ping)
# arrive from the LLM API for this long, the stream is aborted with an error.
LLM_STREAM_IDLE_TIMEOUT_SECONDS = 900
# Max concurrent HTTP connections per LLM API client pool (interactive,
# heartbeat and cluster each get their own pool).
LLM_POOL_MAX_CONNECTIONS = 100
# Max startup retry attempts for Messenger registration / webhook setup.
STARTUP_RETRY_ATTEMPTS = 6
# Base delay between startup retries (seconds, doubles each attempt).
STARTUP_RETRY_DELAY = 1.0
# How many recent messages to scan per room on startup catch-up.
CATCHUP_MESSAGE_LIMIT = 20
# Max message length before auto-splitting into multiple Messenger sends.
MAX_MESSAGE_LENGTH = 2000
# Session max age in days. Older sessions start fresh. 0 = never expire.
SESSION_MAX_AGE_DAYS = 1
# Threshold (number of messages) at which the agent flushes memory.md.
MEMORY_FLUSH_THRESHOLD = 30
# memory.md size caps (characters, not bytes) — edit in the root
# cluster_config.py EDIT HERE block. memory.md is re-sent cold with every new
# session, heartbeat task and cluster task, so it stays SMALL even with a 1M
# context window:
#   MEMORY_SOFT_CHARS  above this, core/memory_guard condenses the file
#                      (LLM rewrite, tool-less; deterministic trim fallback)
#                      after backing it up to data/memory_backups/.
#   MEMORY_MAX_CHARS   hard cap: the copy injected into prompts is never larger
#                      (cut at line boundaries, Reminders section kept).
MEMORY_MAX_CHARS = max(1000, int(getattr(_CLUSTER, "MEMORY_MAX_CHARS", 12000)))
MEMORY_SOFT_CHARS = max(500, min(int(getattr(_CLUSTER, "MEMORY_SOFT_CHARS", 8000)), MEMORY_MAX_CHARS))
# How many memory.md backups memory_guard keeps in data/memory_backups/.
MEMORY_BACKUP_KEEP = 10
# Minimum seconds between two NON-forced condensations while memory.md is
# between the soft and hard caps (stops a condense-per-turn loop when the
# model keeps adding a little). Above the hard cap it always condenses.
MEMORY_CONDENSE_MIN_INTERVAL_SECONDS = 600
# Timeout (seconds) for the tool-less LLM condensation call.
MEMORY_CONDENSE_TIMEOUT_SECONDS = 600
# Conversation-history token budget, kept in step with llm-api's
# MAX_CONVERSATION_TOKENS. Sized for the 1M-token window but well below it: a
# cold prefill of hundreds of thousands of tokens is slow, so hot history
# stays ~300k. (Cluster delegation uses CLUSTER_DELEGATION_TRANSCRIPT_TOKENS.)
MAX_CONVERSATION_TOKENS = 300_000
# Token budget for the room transcript forwarded with a cluster delegation
# (@node). Kept much smaller than MAX_CONVERSATION_TOKENS — every slave pays a
# cold prefill for it on top of its own prompt + memory.
CLUSTER_DELEGATION_TRANSCRIPT_TOKENS = max(
    1000, int(getattr(_CLUSTER, "CLUSTER_DELEGATION_TRANSCRIPT_TOKENS", 32000))
)
# Stream LLM responses with live tool-status updates.
STREAMING_ENABLED = True


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------
# Optional shared secret for the inbound /webhook endpoint. Empty disables
# the check.
WEBHOOK_INCOMING_SECRET = ""


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------
CLUSTER_ENABLED = bool(getattr(_CLUSTER, "CLUSTER_ENABLED", True))
CLUSTER_ROLE = getattr(_CLUSTER, "NODE_ROLE", "master")
NODE_NAME = getattr(_CLUSTER, "NODE_NAME", "master")
NODE_IP = getattr(_CLUSTER, "NODE_IP", "127.0.0.1")
NODE_CAPABILITIES = list(getattr(_CLUSTER, "NODE_CAPABILITIES", []))
NODE_TAGS = list(getattr(_CLUSTER, "NODE_TAGS", []))
CLUSTER_TOKEN = getattr(_CLUSTER, "CLUSTER_TOKEN", "")
CLUSTER_MASTER_API_URL = getattr(_CLUSTER, "CLUSTER_MASTER_API_URL", f"http://127.0.0.1:{LLM_API_PORT}").rstrip("/")
CLUSTER_SLAVE_POLL_INTERVAL_SECONDS = float(getattr(_CLUSTER, "CLUSTER_SLAVE_POLL_INTERVAL_SECONDS", 3.0))
CLUSTER_SLAVE_MAX_CONCURRENT_TASKS = max(1, int(getattr(_CLUSTER, "CLUSTER_SLAVE_MAX_CONCURRENT_TASKS", 4)))
CLUSTER_ENABLE_DELEGATION = CLUSTER_ENABLED and CLUSTER_ROLE == "master"
HOONBOT_WEBHOOK_URL = os.environ.get(
    "HOONBOT_WEBHOOK_URL",
    getattr(_CLUSTER, "HOONBOT_WEBHOOK_URL", f"http://127.0.0.1:{HOONBOT_PORT}/webhook"),
).rstrip("/")


# ---------------------------------------------------------------------------
# Prompt, heartbeat, and skill profiles
# ---------------------------------------------------------------------------
_DEFAULT_PROMPT = _BASE_DIR / "prompts" / "PROMPT.md"
_DEFAULT_HEARTBEAT = _BASE_DIR / "prompts" / "HEARTBEAT.md"
_PROFILE_PROMPT = Path(getattr(_CLUSTER, "PROMPT_FILE", _DEFAULT_PROMPT))
_PROFILE_HEARTBEAT = Path(getattr(_CLUSTER, "HEARTBEAT_FILE", _DEFAULT_HEARTBEAT))
_PROFILE_SKILLS = Path(getattr(_CLUSTER, "SKILLS_DIR", _BASE_DIR / "skills"))

# PROMPT_FILE / HEARTBEAT_FILE keep their historical meaning (the most specific
# existing file: the profile file, else the base file). The prompt the model
# actually gets is LAYERED: base prompts/PROMPT.md first, then the profile file
# appended after a separator as an overlay — see load_prompt_text() /
# load_heartbeat_text(). Without layering the master/slave profiles silently
# lost every base rule (memory discipline, "never end silently", Messenger API
# conventions).
PROMPT_FILE = str(_PROFILE_PROMPT if _PROFILE_PROMPT.exists() else _DEFAULT_PROMPT)
HEARTBEAT_FILE = str(_PROFILE_HEARTBEAT if _PROFILE_HEARTBEAT.exists() else _DEFAULT_HEARTBEAT)
SKILLS_DIR = str(_PROFILE_SKILLS if _PROFILE_SKILLS.exists() else _BASE_DIR / "skills")
PROMPT_PROFILE = getattr(_CLUSTER, "PROMPT_PROFILE", CLUSTER_ROLE)
HEARTBEAT_PROFILE = getattr(_CLUSTER, "HEARTBEAT_PROFILE", CLUSTER_ROLE)
SKILLS_PROFILE = getattr(_CLUSTER, "SKILLS_PROFILE", CLUSTER_ROLE)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    except Exception:
        return False


PROMPT_BASE_FILE = str(_DEFAULT_PROMPT)
PROMPT_OVERLAY_FILE = (
    str(_PROFILE_PROMPT) if not _same_file(_PROFILE_PROMPT, _DEFAULT_PROMPT) else ""
)
HEARTBEAT_BASE_FILE = str(_DEFAULT_HEARTBEAT)
HEARTBEAT_OVERLAY_FILE = (
    str(_PROFILE_HEARTBEAT) if not _same_file(_PROFILE_HEARTBEAT, _DEFAULT_HEARTBEAT) else ""
)

_OVERLAY_SEPARATOR = (
    "\n\n---\n\n"
    "# Profile Overlay: {profile}\n\n"
    "The profile rules below extend the base instructions above. Where they "
    "conflict, the profile rules win.\n\n"
)
_CHECKLIST_MARKER_RE = None  # compiled lazily (keeps import cheap)
_layered_cache: dict[str, tuple[tuple, str]] = {}


def _file_key(path: str) -> tuple | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _checklist_start(text: str) -> int:
    """Offset of a '# Heartbeat Checklist' heading in *text*, or -1."""
    global _CHECKLIST_MARKER_RE
    if _CHECKLIST_MARKER_RE is None:
        import re
        _CHECKLIST_MARKER_RE = re.compile(
            r"^#+\s*Heartbeat Checklist\s*$", re.MULTILINE | re.IGNORECASE
        )
    m = _CHECKLIST_MARKER_RE.search(text)
    return m.start() if m else -1


def _load_layered(kind: str, base: str, overlay: str, profile: str) -> str:
    """Return base + separator + overlay, mtime-cached per file.

    Missing overlay → base only. Missing base → overlay only. Both missing →
    FileNotFoundError. For heartbeat files, when BOTH carry a
    '# Heartbeat Checklist' section, the base checklist is dropped so the
    planner sees exactly one checklist (the profile's); the base rules above
    it are kept.
    """
    base_key = _file_key(base) if base else None
    overlay_key = _file_key(overlay) if overlay else None
    if base_key is None and overlay_key is None:
        missing = overlay or base
        raise FileNotFoundError(f"Hoonbot {kind} prompt file is missing: {missing}")
    cache_key = (base, base_key, overlay, overlay_key, profile)
    cached = _layered_cache.get(kind)
    if cached and cached[0] == cache_key:
        return cached[1]

    base_text = _read_text(base) if base_key is not None else ""
    overlay_text = _read_text(overlay) if overlay_key is not None else ""
    if not overlay_text.strip():
        text = base_text
    elif not base_text.strip():
        text = overlay_text
    else:
        if kind == "heartbeat":
            base_cut = _checklist_start(base_text)
            if base_cut >= 0 and _checklist_start(overlay_text) >= 0:
                base_text = base_text[:base_cut]
        text = (
            base_text.rstrip()
            + _OVERLAY_SEPARATOR.format(profile=profile or "profile")
            + overlay_text.lstrip()
        )
    _layered_cache[kind] = (cache_key, text)
    return text


def load_prompt_text() -> str:
    """Layered system prompt: base PROMPT.md + profile PROMPT.md overlay."""
    return _load_layered("prompt", PROMPT_BASE_FILE, PROMPT_OVERLAY_FILE, str(PROMPT_PROFILE))


def load_heartbeat_text() -> str:
    """Layered heartbeat prompt: base HEARTBEAT.md + profile HEARTBEAT.md overlay."""
    return _load_layered("heartbeat", HEARTBEAT_BASE_FILE, HEARTBEAT_OVERLAY_FILE, str(HEARTBEAT_PROFILE))
