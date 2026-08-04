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
# Debounce window for rapid messages (seconds). Messages within this window
# are combined into one LLM call.
DEBOUNCE_SECONDS = 1.5
# LLM request timeout (seconds) for the non-streaming cluster-task path
# (core/cluster_worker._execute_task → core/llm_api.chat). The cluster call is
# NON-streaming, so this acts as a hard wall-clock cap on a delegated task: if a
# turn hasn't returned within this window (e.g. vLLM stalled), the call raises,
# the slave marks the task failed, and its concurrency slot is released instead
# of being held forever. 1200s = 20 min. Interactive/streaming paths are
# unaffected (they use the shared client's read=None + SSE keepalive pings); the
# server-side llm-api → vLLM idle timeout (STREAM_READ_TIMEOUT) is the deeper
# stall killer that also stops the churn inside llm-api itself.
LLM_TIMEOUT_SECONDS = 1200
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
# Token budget for the room-history snapshot forwarded with a cluster
# delegation (@node). Kept in step with llm-api's MAX_CONVERSATION_TOKENS so
# the recent-context limit is uniform across the stack.
MAX_CONVERSATION_TOKENS = 200_000
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

PROMPT_FILE = str(_PROFILE_PROMPT if _PROFILE_PROMPT.exists() else _DEFAULT_PROMPT)
HEARTBEAT_FILE = str(_PROFILE_HEARTBEAT if _PROFILE_HEARTBEAT.exists() else _DEFAULT_HEARTBEAT)
SKILLS_DIR = str(_PROFILE_SKILLS if _PROFILE_SKILLS.exists() else _BASE_DIR / "skills")
PROMPT_PROFILE = getattr(_CLUSTER, "PROMPT_PROFILE", CLUSTER_ROLE)
HEARTBEAT_PROFILE = getattr(_CLUSTER, "HEARTBEAT_PROFILE", CLUSTER_ROLE)
SKILLS_PROFILE = getattr(_CLUSTER, "SKILLS_PROFILE", CLUSTER_ROLE)
