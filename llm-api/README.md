# LLM API

A self-hosted, OpenAI-compatible LLM API server that wraps **vLLM** with a full agentic loop, JWT auth, RAG, and 10 built-in tools.

## Quick Start

```bash
# 1. Edit config (point at your vLLM server and model)
nano config.py

# 2. Start vLLM separately (example: adjust model path)
#    --enable-auto-tool-choice + --tool-call-parser are REQUIRED for tool calls
#    to stream as structured deltas instead of raw text. On a reasoning model
#    (GLM, Qwen3-Thinking, DeepSeek-R1) also pass --reasoning-parser so the
#    <think>...</think> chain is lifted into reasoning_content instead of
#    leaking into the visible answer. Match every parser to the served family.
# Match BOTH parsers to the served family. GLM emits <arg_key>/<arg_value> XML
# (NOT Hermes JSON), so a hermes tool parser fails to parse the call and leaks
# the raw <tool_call> markup into the answer. GLM-5.2's tool parser is glm47.
# --reasoning-effort max = deepest thinking (GLM-5.2 default); config.py already
# sends chat_template_kwargs={"reasoning_effort":"max"} per request.
vllm serve /path/to/model --port 10000 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser glm45          # GLM-5.2 (GLM-4.5/4.6 use --tool-call-parser glm45)
# Qwen3-Thinking:  --tool-call-parser hermes     --reasoning-parser qwen3
# DeepSeek-R1:     --tool-call-parser deepseek_v3 --reasoning-parser deepseek_r1

# 3. Build/install dependencies and start the API
./start.sh --build
# -> http://127.0.0.1:10002
# -> Swagger UI: http://127.0.0.1:10002/docs
```

On Windows:

```powershell
.\start.ps1 -Build
```

## Configuration

**All settings live in `config.py`** — edit it directly. Key values:

Runtime prompt templates live under `prompts/`: `system.txt`, agent templates
under `prompts/agent/`, and tool templates under `prompts/tools/`.

| Setting | Default | Purpose |
|---|---|---|
| `SERVER_PORT` | `10002` | API listen port |
| `VLLM_HOST` | `http://127.0.0.1:10000` | vLLM server URL |
| `AVAILABLE_TOOLS` | (list) | Tools exposed to the LLM |
| `AGENT_MAX_ITERATIONS` | `30` | Max tool-call iterations per request |
| `JWT_SECRET_KEY` | env or hardcoded | Set via `JWT_SECRET_KEY` env var in prod |
| `RAG_EMBEDDING_MODEL` | (path) | Path to SentenceTransformer model dir |
| `AGENT_LOG_VERBOSITY` | `"summary"` | `"off"` / `"summary"` / `"debug"` |

## Directory Layout

```
llm-api/
├── config.py          All settings — edit this, not env vars
├── run_backend.py     Entry point (uvicorn)
├── start.sh           Linux build-and-launch script
├── start.ps1          Windows build-and-launch script
├── deps/
│   └── requirements.txt
├── backend/
│   ├── agent/         Agentic loop (tool dispatch, compaction, streaming)
│   ├── api/           FastAPI app + routes (auth, chat, sessions, jobs, RAG)
│   ├── core/          llm_backend (httpx→vLLM), database (SQLite), job store
│   ├── models/        Pydantic schemas
│   └── utils/         Auth (JWT), file handler, logging helpers
├── tools/             10 built-in tools (websearch, RAG, code_exec, shell, ...)
├── prompts/           system.txt + agent/tool prompt templates
├── scripts/           Dev helpers (clear_data.py, create_user.py, etc.)
└── data/              Runtime data (SQLite, uploads, sessions, logs) — gitignored
```

## API (OpenAI-Compatible)

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/` | GET | No | Status |
| `/health` | GET | No | Server + vLLM health |
| `/api/auth/login` | POST | No | Returns JWT |
| `/api/auth/signup` | POST | No | Create account |
| `/v1/models` | GET | Optional | List models |
| `/v1/chat/completions` | POST | Optional | Chat + streaming (OpenAI-compatible) |
| `/api/chat/sessions` | GET | Yes | List sessions |
| `/api/tools/rag/collections` | GET | Yes | List RAG collections |
| `/api/rag/upload/stream` | POST | Yes | Upload docs (SSE progress) |
| `/api/jobs` | POST | Yes | Submit background job |
| `/docs` | GET | No | Swagger UI |

Default admin credentials: `admin` / `administrator` (change in `config.py`).

## Dev Commands

```bash
# Clear logs, scratch, sessions
python3 scripts/clear_data.py

# Clear RAG indices (required when switching embedding models)
python3 scripts/clear_rag_data.py --all

# Create a user via the API (server must be running)
python3 scripts/create_user.py alice secret123

# Create a user directly in the DB (no server needed)
python3 scripts/create_user.py alice secret123 --direct

# Halt inference without killing the server
python3 scripts/stop_inference.py stop    # creates data/STOP
python3 scripts/stop_inference.py clear   # removes data/STOP
```

## Architecture Notes

- **Single `while` loop** in `backend/agent/` — no sub-agents, no chains.
- **Parallel tool execution** — `asyncio.gather` runs all tool calls concurrently.
- **Prompt caching** — system prompt cached at module import; `cache_prompt=True` sent to vLLM.
- **Microcompaction** — old iterations are compressed; oversize results spill to `data/tool_results/`.
- **Session slot pinning** — `id_slot = hash(session_id) % VLLM_SLOTS` for stable KV cache hits.
- **Workers > 1** — multiple processes share no state; use `workers=1` for development.
