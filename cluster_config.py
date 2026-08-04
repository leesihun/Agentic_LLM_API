"""
================================================================================
  THE MAIN CONFIG FILE FOR THE WHOLE STACK — edit the "EDIT HERE" block below.
================================================================================

One file, at the repo root, controls every machine and every service
(llm-api, hoonbot, messenger). To set up a node you only edit the EDIT HERE
block: pick master/slave, set this machine's IP, point at your llama.cpp
server, and (for a real deployment) change the secrets.

The three apps each load this file at startup and read their settings from it,
so you do NOT need to touch llm-api/config.py, hoonbot/config.py, or
messenger/config.py for normal setup — those hold advanced per-service tuning.

Every value in EDIT HERE can also be overridden by an environment variable of
the same name, which is how the Start-Master / Start-Slave launchers select the
role. Editing the file is all most deployments need.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse


ROOT_DIR = Path(__file__).resolve().parent


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            ▼  EDIT HERE  ▼                                 ║
# ║   These are the only settings most deployments need. Set them once per     ║
# ║   machine. Each value can also be overridden by an env var of the same     ║
# ║   name. Everything further down derives from these and rarely changes.     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── This machine ───────────────────────────────────────────────────────────
# ROLE: "master" runs all three services (messenger + llm-api + hoonbot).
#       "slave" runs llm-api + hoonbot only and takes delegated tasks.
#       The Start-Master / Start-Slave launchers set this for you; this is the
#       default used when you start a single service by hand.
ROLE = "master" # slave

# NAME: unique handle for this machine — used for routing, logs, and the
#       Messenger @mention. "" picks a default ("master" or "slave-01").
NAME = "137"   # 136 135

# THIS_NODE_IP:   LAN IP other nodes use to reach THIS machine.
# MASTER_NODE_IP: LAN IP of the MASTER machine. On the master, leave it equal
#                 to THIS_NODE_IP. On a slave, set it to the master's IP.
# Single-machine / local testing: leave both as 127.0.0.1.
THIS_NODE_IP   = "10.228.69.135" # 137ip -> 135 136ip -> 134 135ip -> 133
MASTER_NODE_IP = "10.228.69.135"

# ─── LLM backend (vLLM) — where llm-api loads the model server from ──────────
# Each node talks to its own local vLLM server. Change the host/port if your
# vLLM server listens elsewhere. (vLLM is NOT part of this repo — start it
# separately.)
VLLM_SERVER_URL = "http://10.228.69.135:10000"
# VLLM_MODEL: the model name sent in every request. This MUST match the name
# vLLM serves it under (vLLM's --served-model-name, or the model path if that
# flag is omitted). A wrong value — including the placeholder "default" — makes
# vLLM reply 404 "model does not exist". Discover the exact name with:
#     curl http://127.0.0.1:10000/v1/models
VLLM_MODEL = "GLM-5.2"

# ─── Service ports ──────────────────────────────────────────────────────────
MESSENGER_PORT = 10006
HOONBOT_PORT   = 10001
LLM_API_PORT   = 10002

# ─── Shared cluster secret — every node must use the SAME value ─────────────
# CHANGE THIS for any real (non-loopback) deployment.
CLUSTER_SECRET = "change-me-cluster-token"

# ─── LLM API service (llm-api/) ─────────────────────────────────────────────
LLM_API_ADMIN_USERNAME = "admin"
LLM_API_ADMIN_PASSWORD = "administrator"
TAVILY_API_KEY         = "your-secret-key-change-in-production"  # web_search tool
# RAG embedding + reranker model locations and device. These paths are
# deployment-specific — point them at where your models are staged.
RAG_EMBEDDING_MODEL  = "/scratch/LLM_models/offline_models/bge-m3"
RAG_RERANKER_MODEL   = "/scratch/LLM_models/offline_models/mmarco-mMiniLMv2-L12-H384-v1"
RAG_EMBEDDING_DEVICE = "cuda"   # "cuda" or "cpu"

# ─── Hoonbot service (hoonbot/) ─────────────────────────────────────────────
BOT_NAME                   = "Bot"
BOT_HOME_ROOM_NAME         = "Heartbeat"   # room for heartbeat output
HEARTBEAT_ENABLED          = True
HEARTBEAT_INTERVAL_SECONDS = 3600
HEARTBEAT_ACTIVE_START     = "00:00"       # 24h HH:MM — only run in this window
HEARTBEAT_ACTIVE_END       = "23:59"

# ─── Messenger service (messenger/) ─────────────────────────────────────────
# Token gating the embedded /claude and /opencode terminals. CHANGE in prod.
MESSENGER_TERMINAL_TOKEN = "leesihun"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            ▲  EDIT HERE  ▲                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _url(host: str, port: int, scheme: str = "http") -> str:
    return f"{scheme}://{host}:{port}"


def _port_of(url: str, default: int) -> int:
    try:
        port = urlparse(url).port
        return int(port) if port else default
    except (ValueError, TypeError):
        return default


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _is_loopback_host(host: str) -> bool:
    if host in {"localhost", "::1"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_advertised_url(name: str, value: str, allow_loopback: bool = False) -> tuple[bool, str]:
    """
    Validate the cluster policy for URLs shared across nodes.

    Inter-node advertised URLs should be IP-style: http://<ip>:<port>. Loopback
    is allowed only for local/single-machine development or explicitly local
    service URLs.
    """
    parsed = urlparse(value)
    host = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not host or not parsed.port:
        return False, f"{name} must look like http://<ip>:<port>"
    lowered = host.lower()
    if lowered == "localhost":
        return False, f"{name} must use an IP address, not localhost"
    if "cloudflare" in lowered or lowered.endswith("trycloudflare.com") or lowered == "aihoonbot.com":
        return False, f"{name} must not use Cloudflare/tunnel hostnames"
    if _is_loopback_host(host):
        if allow_loopback:
            return True, ""
        return False, f"{name} must not advertise loopback outside this node"
    if not _is_ip_host(host):
        return False, f"{name} must use an IP address, not DNS"
    return True, ""


def require_valid_advertised_urls() -> None:
    checks = [
        ("MASTER_LLM_API_URL", MASTER_LLM_API_URL, NODE_ROLE == "master"),
        ("MESSENGER_URL", MESSENGER_URL, NODE_ROLE == "master"),
        ("HOONBOT_WEBHOOK_URL", HOONBOT_WEBHOOK_URL, NODE_ROLE == "master"),
        ("ADVERTISED_LLM_API_URL", ADVERTISED_LLM_API_URL, NODE_ROLE == "master"),
    ]
    errors = []
    for name, value, allow_loopback in checks:
        ok, message = validate_advertised_url(name, value, allow_loopback=allow_loopback)
        if not ok:
            errors.append(message)
    if errors:
        raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# Node identity
# ---------------------------------------------------------------------------

NODE_ROLE = _env("CLUSTER_ROLE", ROLE).lower()
if NODE_ROLE not in {"master", "slave"}:
    raise ValueError("CLUSTER_ROLE must be 'master' or 'slave'")

NODE_NAME = _env("NODE_NAME", NAME or ("master" if NODE_ROLE == "master" else "slave-01"))
NODE_IP = _env("NODE_IP", THIS_NODE_IP)
MASTER_IP = _env("MASTER_IP", NODE_IP if NODE_ROLE == "master" else MASTER_NODE_IP)


# ---------------------------------------------------------------------------
# Ports and URLs
# ---------------------------------------------------------------------------

MESSENGER_PORT = int(_env("MESSENGER_PORT", str(MESSENGER_PORT)))
HOONBOT_PORT = int(_env("HOONBOT_PORT", str(HOONBOT_PORT)))
LLM_API_PORT = int(_env("LLM_API_PORT", str(LLM_API_PORT)))
VLLM_PORT = int(_env("VLLM_PORT", str(_port_of(VLLM_SERVER_URL, 10000))))
LLM_API_BIND_HOST = _env("LLM_API_BIND_HOST", "0.0.0.0")
HOONBOT_BIND_HOST = _env("HOONBOT_BIND_HOST", "0.0.0.0")

MESSENGER_URL = _env("MESSENGER_URL", _url(MASTER_IP, MESSENGER_PORT))
LOCAL_HOONBOT_URL = _env("LOCAL_HOONBOT_URL", _url("127.0.0.1", HOONBOT_PORT))
HOONBOT_WEBHOOK_URL = _env("HOONBOT_WEBHOOK_URL", f"{_url(NODE_IP, HOONBOT_PORT)}/webhook")

LOCAL_LLM_API_URL = _env("LOCAL_LLM_API_URL", _url("127.0.0.1", LLM_API_PORT))
MASTER_LLM_API_URL = _env("MASTER_LLM_API_URL", _url(MASTER_IP, LLM_API_PORT))
ADVERTISED_LLM_API_URL = _env("ADVERTISED_LLM_API_URL", _url(NODE_IP, LLM_API_PORT))
HOONBOT_LLM_API_URL = _env("HOONBOT_LLM_API_URL", LOCAL_LLM_API_URL)

LOCAL_VLLM_URL = _env("LOCAL_VLLM_URL", VLLM_SERVER_URL)
VLLM_MODEL = _env("VLLM_MODEL", VLLM_MODEL)


# ---------------------------------------------------------------------------
# Cluster control plane
# ---------------------------------------------------------------------------

CLUSTER_ENABLED = _env("CLUSTER_ENABLED", "true").lower() == "true"
CLUSTER_MASTER_API_URL = _env("CLUSTER_MASTER_API_URL", MASTER_LLM_API_URL)
CLUSTER_TOKEN = _env("CLUSTER_TOKEN", CLUSTER_SECRET)
CLUSTER_NODE_STALE_SECONDS = int(_env("CLUSTER_NODE_STALE_SECONDS", "90"))
CLUSTER_TASK_LEASE_SECONDS = int(_env("CLUSTER_TASK_LEASE_SECONDS", "900"))
CLUSTER_SLAVE_POLL_INTERVAL_SECONDS = float(_env("CLUSTER_SLAVE_POLL_INTERVAL_SECONDS", "3"))
# Max cluster tasks a slave executes concurrently (vLLM batches the requests).
# Higher = more headroom so a few slow/hung slots can't starve the whole slave;
# hung slots now self-release via the LLM timeout (see hoonbot LLM_TIMEOUT_SECONDS).
CLUSTER_SLAVE_MAX_CONCURRENT_TASKS = int(_env("CLUSTER_SLAVE_MAX_CONCURRENT_TASKS", "20"))

if NODE_ROLE == "master":
    _default_caps = ["orchestrator", "messenger", "llm-api"]
    _default_tags = ["master", "control-plane"]
else:
    _default_caps = ["agent", "llm-api", "local-model"]
    _default_tags = ["slave", "worker"]

NODE_CAPABILITIES = _csv("NODE_CAPABILITIES", _default_caps)
NODE_TAGS = _csv("NODE_TAGS", _default_tags)


# ---------------------------------------------------------------------------
# Hoonbot prompt and skill profiles
# ---------------------------------------------------------------------------

HOONBOT_PROMPTS_DIR = ROOT_DIR / "hoonbot" / "prompts"
HOONBOT_SKILLS_DIR = ROOT_DIR / "hoonbot" / "skills"
HOONBOT_SKILL_PROFILES_DIR = ROOT_DIR / "hoonbot" / "profiles"
PROMPT_PROFILE = _env("PROMPT_PROFILE", NODE_ROLE)
HEARTBEAT_PROFILE = _env("HEARTBEAT_PROFILE", NODE_ROLE)
SKILLS_PROFILE = _env("SKILLS_PROFILE", NODE_ROLE)

PROMPT_FILE = HOONBOT_PROMPTS_DIR / PROMPT_PROFILE / "PROMPT.md"
HEARTBEAT_FILE = HOONBOT_PROMPTS_DIR / HEARTBEAT_PROFILE / "HEARTBEAT.md"
_PROFILE_SKILLS_DIR = HOONBOT_SKILL_PROFILES_DIR / SKILLS_PROFILE / "skills"
_PROFILE_SKILL_FILES = [
    path for path in _PROFILE_SKILLS_DIR.glob("*.md")
    if path.name.lower() != "readme.md"
] if _PROFILE_SKILLS_DIR.exists() else []
SKILLS_DIR = _PROFILE_SKILLS_DIR if _PROFILE_SKILL_FILES else HOONBOT_SKILLS_DIR


def node_payload() -> dict:
    return {
        "node_name": NODE_NAME,
        "role": NODE_ROLE,
        "ip": NODE_IP,
        "api_url": ADVERTISED_LLM_API_URL,
        "local_llm_api_url": LOCAL_LLM_API_URL,
        "capabilities": NODE_CAPABILITIES,
        "tags": NODE_TAGS,
        "prompt_profile": PROMPT_PROFILE,
        "heartbeat_profile": HEARTBEAT_PROFILE,
        "skills_profile": SKILLS_PROFILE,
    }
