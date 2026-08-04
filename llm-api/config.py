"""
LLM API Configuration
All settings are configurable here.
Single server, vLLM backend, native tool calling.
"""
import importlib.util
import os
import sys
from pathlib import Path
from typing import Literal, Optional

# ============================================================================
# Paths
# ============================================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PROMPTS_DIR = APP_DIR / "prompts"

# Default base directory for agent tools when a session has no explicit
# workspace. Lets the agent operate on the whole user dir instead of being
# pinned to APP_DIR. Override via env AGENT_DEFAULT_WORKSPACE.
# Internal API paths (DATA_DIR, UPLOAD_DIR, SCRATCH_DIR) are unaffected -
# they always resolve under APP_DIR.
def _resolve_agent_default_workspace() -> Path:
    override = os.environ.get("AGENT_DEFAULT_WORKSPACE")
    if override:
        try:
            return Path(override).expanduser().resolve()
        except Exception:
            pass
    if sys.platform != "win32":
        for candidate in (Path("/home/leesihun"), Path.home(), Path("/home")):
            if candidate.is_dir():
                return candidate.resolve()
    return Path.home().resolve()


AGENT_DEFAULT_WORKSPACE: Path = _resolve_agent_default_workspace()


def prompt_path(relative_path: str) -> Path:
    return PROMPTS_DIR / relative_path


def read_prompt(relative_path: str) -> str:
    return prompt_path(relative_path).read_text(encoding="utf-8")


def _load_cluster_config():
    path = APP_DIR.parent / "cluster_config.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_llm_api_cluster_config", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CLUSTER = _load_cluster_config()

# ============================================================================
# Server Settings
# ============================================================================
SERVER_HOST = getattr(_CLUSTER, "LLM_API_BIND_HOST", "0.0.0.0")
SERVER_PORT = int(getattr(_CLUSTER, "LLM_API_PORT", 10002))
SERVER_WORKERS = 2
LOG_LEVEL = "INFO"

# ============================================================================
# vLLM Backend
# ============================================================================
VLLM_HOST = os.environ.get("VLLM_HOST", getattr(_CLUSTER, "LOCAL_VLLM_URL", "http://127.0.0.1:10000"))
VLLM_MODEL = os.environ.get("VLLM_MODEL", getattr(_CLUSTER, "VLLM_MODEL", "default"))
OPENCODE_MODEL: str = "llama.cpp/MiniMax"  # "provider/model" format (e.g., "llama.cpp/default", "opencode/minimax-m2.5-free")

# ============================================================================
# Cluster Settings
# ============================================================================
CLUSTER_ENABLED = bool(getattr(_CLUSTER, "CLUSTER_ENABLED", True))
CLUSTER_ROLE = getattr(_CLUSTER, "NODE_ROLE", "master")
NODE_NAME = getattr(_CLUSTER, "NODE_NAME", "master")
CLUSTER_TOKEN = getattr(_CLUSTER, "CLUSTER_TOKEN", "")
ADVERTISED_LLM_API_URL = getattr(_CLUSTER, "ADVERTISED_LLM_API_URL", f"http://127.0.0.1:{SERVER_PORT}")
CLUSTER_NODE_STALE_SECONDS = int(getattr(_CLUSTER, "CLUSTER_NODE_STALE_SECONDS", 90))
CLUSTER_TASK_LEASE_SECONDS = int(getattr(_CLUSTER, "CLUSTER_TASK_LEASE_SECONDS", 900))
CLUSTER_DIR = DATA_DIR / "cluster"
# Retention for the cluster task store (data/cluster/tasks, /artifacts). Terminal
# tasks age off their completed_at; tasks that never reached a terminal state
# (no matching node ever leased them) age off created_at instead, so the queue
# can't grow unbounded. 0 disables cleanup.
CLUSTER_TASKS_CLEANUP_DAYS = int(getattr(_CLUSTER, "CLUSTER_TASKS_CLEANUP_DAYS", 14))

# ============================================================================
# Model Parameters (Default LLM Inference Settings)
# ============================================================================
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
# GLM-5.2's published sampling floor is min_p=0.01. The previous 0.1 was ~10x
# too aggressive and clipped valid tail tokens (worse with temperature=1.0).
DEFAULT_MIN_P = 0.01
DEFAULT_MAX_TOKENS = 8192
DEFAULT_REPETITION_PENALTY = 1.0   # vLLM's field is `repetition_penalty` (1.0 = no penalty)

# Per-model temperature overrides (substring match on model name, lowercase).
# Reasoning-trained models have published optimal settings — using a generic
# default kills their performance. MiniMax explicitly recommends temperature=1.0
# for M2; Qwen3-Thinking and DeepSeek-R1 land around 0.6-0.7. GLM-5.2's official
# recommendation is temperature=1.0, top_p=0.95 (tune only one of the two).
MODEL_TEMPERATURE_OVERRIDES = {
    "glm": 1.0,
    "minimax": 1.0,
    "qwen3": 0.7,
    "deepseek": 0.6,
}

# Extra chat-template kwargs forwarded to vLLM as a top-level `chat_template_kwargs`
# body field (vLLM passes them to the served model's chat template). For GLM-5.2
# this selects the thinking mode via `reasoning_effort`:
#   "max"  (default) — deepest chain-of-thought, best for complex/agentic work
#   "high"           — balanced reasoning depth vs latency
# and {"enable_thinking": False} disables reasoning entirely (fast replies).
# Set to {} / None to send nothing and let the server default decide.
CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "max"}

# ============================================================================
# vLLM Performance Tuning
# ============================================================================
# Prefix caching is a vLLM *server* flag (`--enable-prefix-caching`), not a
# per-request field, so there is no cache_prompt knob here. The agent still
# keeps its system-prompt prefix byte-stable (see prompt_assembly) so vLLM's
# automatic prefix cache hits across iterations.
VLLM_CONNECTION_POOL_SIZE = 20

# vLLM streaming idle timeout (seconds). This is an INTER-CHUNK (httpx read)
# timeout, not a total-turn cap: it only fires when vLLM sends NO bytes for this
# long, so a healthy turn that keeps streaming tokens — however long — is never
# cut off, while a genuine stall (GPU wedged / stream hung) is killed so the
# agent loop errors out instead of hanging forever and pinning a cluster slot.
# Must exceed worst-case time-to-first-token under load (requests queued behind
# others on a busy vLLM). None = never time out (old behaviour). 1200s = 20 min.
STREAM_READ_TIMEOUT = 1200
# Finite connect timeout so a dead vLLM fails fast rather than hanging.
STREAM_CONNECT_TIMEOUT = 10.0

# ============================================================================
# Logging Settings (before Agent — agent log target references PROMPTS_LOG_PATH)
# ============================================================================
LOG_DIR = DATA_DIR / "logs"
PROMPTS_LOG_PATH = LOG_DIR / "prompts.log"
PROMPTS_LOG_MAX_LINES = 10_000

# ============================================================================
# Agent Settings
# ============================================================================
AGENT_MAX_ITERATIONS = 100
# Goal mode (request `mode="goal"`): an extensive, goal-focused run. Raises the
# iteration budget and enforces a plan->execute->verify done-gate (see
# backend/agent/loop.py). AGENT_GOAL_MAX_COMPLETION_BLOCKS caps how many times the
# loop may refuse to let the agent stop with an incomplete todo list before it is
# allowed to finish regardless (AGENT_GOAL_MAX_ITERATIONS is the absolute backstop).
AGENT_GOAL_MAX_ITERATIONS = 250
AGENT_GOAL_MAX_COMPLETION_BLOCKS = 3
AGENT_GOAL_MODE_PROMPT = "agent/goal_mode.txt"
# Wall-clock cap on a single subagent (`agent` tool) invocation. Prevents a stuck
# sub-loop from blocking the parent's tool call indefinitely. AGENT_MAX_ITERATIONS
# only bounds iteration count; this bounds total time.
SUBAGENT_TIMEOUT_SECONDS = 1800
# Hard upper bound on the `timeout` argument the LLM may pass to shell_exec.
# The model can still request lower values; this only clamps obvious mistakes
# (e.g. timeout=86400) that would let a runaway command wedge the agent loop.
SHELL_EXEC_HARD_CAP_SECONDS = 3600
AGENT_TOOL_LOOP_MAX_TOKENS = 8192
AGENT_SYSTEM_PROMPT = "system.txt"
AGENT_DYNAMIC_CONTEXT_MAX_CHARS = 12000
AGENT_REPO_DOC_CONTEXT_MAX_CHARS = 6000
AGENT_MEMO_MAX_CHARS = 2000
AGENT_FILE_PREVIEW_MAX_CHARS = 120
AGENT_OLD_TOOL_RESULT_SUMMARY_MAX_CHARS = 4000  # MiniMax M2 / Qwen3 handle long context fine; aggressive compaction was erasing useful detail
AGENT_COMPACTION_WARM_WINDOW = 10  # multi-file edit flows need more headroom than 5

# Proactive compaction: when vLLM reports REAL prompt-token usage (via
# stream_options.include_usage) and the last call's prompt crossed
# MODEL_CONTEXT_WINDOW * AGENT_COMPACT_AT_TOKEN_FRACTION, the agent summarizes
# the older half BEFORE the next send — avoiding the reactive 400-then-resend.
# MODEL_CONTEXT_WINDOW = 0 disables proactive compaction (reactive still works).
# Set this to your served model's max context length (see vLLM --max-model-len).
MODEL_CONTEXT_WINDOW = int(os.environ.get("MODEL_CONTEXT_WINDOW", getattr(_CLUSTER, "MODEL_CONTEXT_WINDOW", 0)))
AGENT_COMPACT_AT_TOKEN_FRACTION = 0.75

# Auto-compact: also triggered reactively when vLLM returns a context-overflow
# error. We summarize the older half of the conversation via a single LLM call
# and replace it with a compact system message, then retry.
AGENT_AUTOCOMPACT_ENABLED = True
AGENT_AUTOCOMPACT_MAX_RETRIES = 2          # number of summarize-and-retry attempts per LLM call
AGENT_AUTOCOMPACT_SUMMARY_MAX_TOKENS = 1500 # cap on the summary the LLM is allowed to emit
AGENT_AUTOCOMPACT_PER_MSG_CHARS = 1500     # truncate each old message to this before summarizing
AGENT_AUTOCOMPACT_KEEP_RECENT = 8          # always keep this many most-recent non-system msgs verbatim
AGENT_LOG_VERBOSITY: Literal["off", "summary", "debug"] = "summary"
AGENT_LOG_ASYNC = True
AGENT_LOG_PATH = PROMPTS_LOG_PATH

# ----------------------------------------------------------------------------
# Agent: workspace awareness
# ----------------------------------------------------------------------------
# AGENT_DEFAULT_WORKSPACE is computed near the top of this file by
# _resolve_agent_default_workspace(). Callers (chat.py, hoonbot) may pass a
# per-request `workspace` field that overrides it for that session.
# Inline attached-file contents into the dynamic context when total size
# (across all attached files) is <= this budget. Above the budget, only the
# metadata preview is included.
AGENT_ATTACHED_FILE_INLINE_BUDGET = 8000

# ----------------------------------------------------------------------------
# Agent: anti-spiral / reflection
# ----------------------------------------------------------------------------
AGENT_STUCK_REPEAT_THRESHOLD = 3          # same signature N times within window
AGENT_STUCK_REPEAT_WINDOW = 6
AGENT_STUCK_COOLDOWN_ITERATIONS = 4
AGENT_CONSECUTIVE_FAILURE_THRESHOLD = 2   # all-failed iterations in a row before reflection nudge
AGENT_GOAL_REMINDER_ITERATIONS: tuple = (10, 25, 50)

# ----------------------------------------------------------------------------
# Agent: user-input fidelity
# ----------------------------------------------------------------------------
AGENT_TAIL_GOAL_REMINDER_ENABLED = True
AGENT_TAIL_GOAL_MIN_TURNS = 4          # only inject when conversation has at least this many msgs
AGENT_TAIL_GOAL_MAX_CHARS = 1500       # cap on echoed user message length

# ----------------------------------------------------------------------------
# Agent: turn boundary / previous-context isolation
# ----------------------------------------------------------------------------
AGENT_TURN_BOUNDARY_MARKER_ENABLED = True

# ============================================================================
# Database Settings
# ============================================================================
DATABASE_PATH = str(DATA_DIR / "app.db")

# ============================================================================
# Authentication Settings
# ============================================================================
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "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")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7

# Admin credentials — edit in the root cluster_config.py EDIT HERE block.
DEFAULT_ADMIN_USERNAME = getattr(_CLUSTER, "LLM_API_ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = getattr(_CLUSTER, "LLM_API_ADMIN_PASSWORD", "administrator")

# ============================================================================
# File Storage Settings
# ============================================================================
UPLOAD_DIR = DATA_DIR / "uploads"
SCRATCH_DIR = DATA_DIR / "scratch"
MAX_FILE_SIZE_MB = 100
IMAGE_SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]
IMAGE_MAX_SIZE_MB = 20
IMAGE_MAX_DIMENSION = 4096  # resize if either side exceeds this (saves context tokens)
# Whether the served vLLM model can consume images. When True (default),
# attached images are embedded as OpenAI image_url content parts on the user
# turn. Set False for a text-only model so it isn't sent image parts it can't
# decode (vLLM 400s on image_url for text-only models); images then appear as
# metadata only. Override via env MODEL_SUPPORTS_VISION or cluster_config.
MODEL_SUPPORTS_VISION = str(
    os.environ.get("MODEL_SUPPORTS_VISION", getattr(_CLUSTER, "MODEL_SUPPORTS_VISION", "true"))
).strip().lower() in ("1", "true", "yes", "on")

# ============================================================================
# Stop Signal Settings
# ============================================================================
STOP_FILE = DATA_DIR / "STOP"

# ============================================================================
# Tools Settings
# ============================================================================
AVAILABLE_TOOLS = [
    "websearch",
    "code_exec",        # Direct Python execution — write and run the code yourself
    "rag",              # FAISS vector search over uploaded documents
    "file_reader",      # Read any file (prefer over shell_exec cat/head/tail)
    "file_edit",        # Surgical exact-string replacement (single-line / small changes)
    "apply_patch",      # V4A context-anchored patch (for multi-line / .ps1 / .sh edits)
    "file_writer",      # Create new files or complete rewrites only
    "file_navigator",   # Discover files by name/glob (prefer over shell_exec find)
    "grep",             # ripgrep content search (prefer over shell_exec grep)
    "shell_exec",       # Shell commands (use for git, package managers, build tools)
    "shell_lint",       # Lint shell scripts before running (PSScriptAnalyzer / shellcheck)
    "process_monitor",  # Background process lifecycle
    "memo",             # Persistent cross-session key-value memory
    "todo_write",       # Session task checklist (3+ step tasks)
    "agent",            # Spawn explore/general subagent in fresh context
]
# Truncated tool results include their disk path in the truncation marker;
# file_reader handles retrieval.

TOOL_PARAMETERS = {
    "code_exec": {
        "timeout": 864000,
    },
    "rag": {
        "temperature": 0.2,
        "max_tokens": 30000,
    },
    "shell_exec": {
        "timeout": 864000,
    },
}

# ============================================================================
# Microcompaction: Tool Result Budgets (chars)
# ============================================================================
TOOL_RESULT_BUDGET = {
    "websearch": 2000,
    "code_exec": 8000,
    "rag": 3000,
    "file_reader": 8000,   # was 4000; .ps1/.sh scripts can be 300-800 lines
    "file_edit": 1500,     # was 500; failed-edit diffs need to be readable
    "apply_patch": 2000,   # V4A patch tool
    "file_writer": 500,
    "file_navigator": 2000,
    "grep": 6000,          # was 4000; multi-file context windows
    "shell_exec": 3000,
    "shell_lint": 4000,    # new shell linting tool
    "process_monitor": 3000,
    "memo": 1000,
    "todo_write": 300,
    "agent": 6000,
}
TOOL_RESULT_DEFAULT_BUDGET = 3000
TOOL_RESULTS_DIR = DATA_DIR / "tool_results"

# ============================================================================
# Web Search Tool Settings
# ============================================================================
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", getattr(_CLUSTER, "TAVILY_API_KEY", "your-secret-key-change-in-production"))
TAVILY_SEARCH_DEPTH = "advanced"
TAVILY_INCLUDE_DOMAINS = []
TAVILY_EXCLUDE_DOMAINS = []
WEBSEARCH_MAX_RESULTS = 5

# ============================================================================
# code_exec subprocess caps
# ============================================================================
PYTHON_EXECUTOR_MODE = "native"     # "native" = ephemeral subprocess; "opencode" = remote opencode server
PYTHON_EXECUTOR_TIMEOUT = 300       # code_exec default; tool timeout=0 disables wall-clock kill
PYTHON_EXECUTOR_MAX_OUTPUT_SIZE = 1024 * 1024 * 10
OPENCODE_SERVER_HOST = "127.0.0.1"
OPENCODE_SERVER_PORT = 37254

# ============================================================================
# RAG Tool Settings
# ============================================================================
RAG_DOCUMENTS_DIR = DATA_DIR / "rag_documents"
RAG_INDEX_DIR = DATA_DIR / "rag_indices"
RAG_METADATA_DIR = DATA_DIR / "rag_metadata"

# RAG model paths/device — edit in the root cluster_config.py EDIT HERE block.
RAG_EMBEDDING_MODEL = getattr(_CLUSTER, "RAG_EMBEDDING_MODEL", "/scratch0/LLM_models/offline_models/bge-m3")
RAG_EMBEDDING_DEVICE = getattr(_CLUSTER, "RAG_EMBEDDING_DEVICE", "cuda")
RAG_EMBEDDING_BATCH_SIZE = 16

RAG_INDEX_TYPE = "Flat"
RAG_SIMILARITY_METRIC = "cosine"

RAG_CHUNK_SIZE = 512
RAG_CHUNK_OVERLAP = 50
RAG_CHUNKING_STRATEGY = "semantic"
RAG_MAX_RESULTS = 10
RAG_MIN_SCORE_THRESHOLD = 0.5
RAG_CONTEXT_WINDOW = 1

RAG_USE_HYBRID_SEARCH = True
RAG_HYBRID_ALPHA = 0.5

RAG_USE_RERANKING = True
RAG_RERANKER_MODEL = getattr(_CLUSTER, "RAG_RERANKER_MODEL", "/scratch0/LLM_models/offline_models/mmarco-mMiniLMv2-L12-H384-v1")
RAG_RERANKING_TOP_K = 20
# Preload RAG models by default on non-Windows platforms. Windows local
# bring-up often runs without the offline embedding/reranker assets staged.
RAG_PRELOAD_MODELS = os.name != "nt"

RAG_QUERY_PREFIX = ""

RAG_SUPPORTED_FORMATS = [".txt", ".pdf", ".docx", ".xlsx", ".xls", ".md", ".json", ".csv"]

# ============================================================================
# Process Monitor Tool Settings
# ============================================================================
PROCESS_MONITOR_MAX_BUFFER_LINES = 5000
PROCESS_MONITOR_MAX_PER_SESSION = 20

# ============================================================================
# Shell Exec Tool Settings
# ============================================================================
# True = kill the process when shell_exec timeout is reached (prevents orphan processes).
# False = legacy behaviour: return partial output, leave process running.
SHELL_EXEC_KILL_ON_TIMEOUT = True

# ============================================================================
# Memo Tool Settings
# ============================================================================
MEMO_DIR = DATA_DIR / "memory"
MEMO_MAX_ENTRIES = 100
MEMO_MAX_VALUE_LENGTH = 1000

# ============================================================================
# Background Jobs Settings
# ============================================================================
JOBS_DIR = DATA_DIR / "jobs"
JOBS_CLEANUP_DAYS = 30

# ============================================================================
# Session Settings
# ============================================================================
MAX_CONVERSATION_HISTORY = 200
# Hard cap on the recent conversation fed to the LLM, in estimated tokens.
# Applied alongside MAX_CONVERSATION_HISTORY (whichever is more restrictive) so
# the hot context never exceeds this budget regardless of message count.
MAX_CONVERSATION_TOKENS = 200_000
SESSIONS_DIR = DATA_DIR / "sessions"
SESSION_CLEANUP_DAYS = 7

# ============================================================================
# Cleanup Settings (data retention — 2-week rolling window)
# ============================================================================
SCRATCH_CLEANUP_DAYS = 14       # data/scratch/{session_id}/ dirs
TOOL_RESULTS_CLEANUP_DAYS = 14  # data/tool_results/{session_id}/ dirs
LOG_ROTATION_DAYS = 14          # rotate data/logs/prompts.log after N days

# ============================================================================
# LLM File Write Policy
# ============================================================================
LLM_GENERATED_DIR = DATA_DIR / "llm_generated"   # dedicated dir for LLM absolute-path writes
LLM_FILE_RETENTION_DAYS = 3                        # auto-delete files older than N days (0 = disabled)
ALLOWED_WRITE_DIRS: list[Path] = []  # empty = allow all absolute paths

# ============================================================================
# Streaming Settings
# ============================================================================
# vLLM HTTP client timeouts. read=None → never time out waiting for the next
# token, so long agentic turns and slow prefill on a near-full context window
# are never cut off mid-stream. Connect stays finite so a dead vLLM backend
# still fails fast (the backend also pre-checks /v1/models with a 3s timeout
# before every stream, see VllmBackend._select_available_host).
STREAM_CONNECT_TIMEOUT = 10.0
STREAM_READ_TIMEOUT = None

# ============================================================================
# CORS Settings
# ============================================================================
CORS_ORIGINS = [
    "http://127.0.0.1:10002",
    "*",
]

# ============================================================================
# Ensure directories exist
# ============================================================================
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
RAG_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
RAG_METADATA_DIR.mkdir(parents=True, exist_ok=True)
TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MEMO_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
LLM_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
