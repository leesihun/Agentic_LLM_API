# Hoonbot

A Python/FastAPI AI bot that bridges **Huni Messenger** and the **LLM API**, giving Messenger rooms an AI assistant with full tool access (file ops, web search, code execution, RAG, shell, memory).

## Quick Start

```bash
# Build/install dependencies and start (Messenger + LLM API must already be running)
./start.sh --build
# -> http://127.0.0.1:10001
# -> Health: http://127.0.0.1:10001/health
```

On Windows:

```powershell
.\start.ps1 -Build
```

## Prerequisites

| Service | Port | How to start |
|---|---|---|
| Huni Messenger | 10006 | `cd ../messenger && ./start.sh` |
| LLM API | 10002 | `cd ../llm-api && ./start.sh` |

## Configuration

**All settings live in `config.py`** — edit it directly.

Prompt files are app-local under `prompts/`: the default files are
`prompts/PROMPT.md` and `prompts/HEARTBEAT.md`, and cluster role profiles live
in `prompts/master/` and `prompts/slave/`.

| Setting | Default | Purpose |
|---|---|---|
| `HOONBOT_PORT` | `10001` | Bot webhook listener port |
| `MESSENGER_URL` | `http://127.0.0.1:10006` | Messenger server URL |
| `LLM_API_URL` | `http://127.0.0.1:10002` | LLM API URL (`LLM_API_URL` env var overrides) |
| `LLM_API_USERNAME` | `admin` | Used by setup to obtain a token |
| `LLM_API_PASSWORD` | `administrator` | Used by setup to obtain a token |
| `MESSENGER_BOT_NAME` | `Bot` | Display name in Messenger |
| `MESSENGER_HOME_ROOM_NAME` | `Heartbeat` | Room for heartbeat output |
| `HEARTBEAT_ENABLED` | `True` | Proactive heartbeat task loop |
| `HEARTBEAT_INTERVAL_SECONDS` | `3000` | Time between heartbeat ticks |
| `DEBOUNCE_SECONDS` | `1.5` | Combine rapid messages in this window |
| `LLM_TIMEOUT_SECONDS` | `None` | LLM request read timeout; `None` = no timeout (long tool-heavy turns never cut off) |
| `STREAMING_ENABLED` | `True` | Stream LLM responses with live tool status |
| `SESSION_MAX_AGE_DAYS` | `1` | Sessions older than this reset |

Runtime credentials are stored in `data/` (never commit these):

| File | Contents |
|---|---|
| `data/.llm_key` | Bearer token for LLM API (written by `scripts/setup_credentials.py`) |
| `data/.llm_model` | Model name used for requests |
| `data/.apikey` | Messenger bot API key (written at first startup) |

## Directory Layout

```
hoonbot/
├── config.py               All settings — single source of truth
├── hoonbot.py              Entry point (FastAPI + startup sequence)
├── start.sh                Linux build-and-launch script
├── start.ps1               Windows build-and-launch script
├── deps/
│   └── requirements.txt
├── core/
│   ├── context.py          System prompt + memory assembler
│   ├── heartbeat.py        Background proactive task loop
│   ├── llm_api.py          Async HTTP client for LLM API calls
│   ├── messenger.py        Messenger REST API client
│   └── retry.py            Exponential backoff helper
├── handlers/
│   ├── health.py           GET /health
│   └── webhook.py          POST /webhook (message processing pipeline)
├── prompts/
│   ├── PROMPT.md           System prompt (bot identity + memory instructions)
│   ├── HEARTBEAT.md        Proactive task checklist (edit to customise)
│   ├── master/             Master prompt + heartbeat profile
│   └── slave/              Slave prompt + heartbeat profile
├── skills/                 Markdown docs for LLM-executed skills
├── scripts/
│   ├── setup_credentials.py  One-time LLM token + model setup
│   ├── reset.py              Clear memory / session data
│   └── test_llm.py           Verify LLM API connectivity
├── docs/
│   └── ARCHITECTURE.md
└── data/                   Runtime state — gitignored
    ├── memory.md           Persistent LLM memory (Markdown)
    ├── room_sessions.json  Room → session_id map
    ├── .llm_key            LLM API bearer token
    ├── .llm_model          LLM model name
    └── .apikey             Messenger bot API key
```

## Message Flow

```
Messenger UI
    │  webhook POST /webhook
    ▼
handlers/webhook.py
    ├── validate (text/image/file, not bot, @mention if non-home room)
    ├── mark_read
    ├── debounce (combine rapid messages)
    └── process_message()
            │  POST /v1/chat/completions
            ▼
        LLM API (port 10002) — full agentic loop with tools
            │  SSE stream or JSON
            ▼
        Hoonbot → sends threaded reply via Messenger API
```

## Dev Commands

```bash
# Verify LLM API connectivity
python3 scripts/test_llm.py

# Clear Hoonbot's persistent memory
python3 scripts/reset.py --memory

# List memory contents
python3 scripts/reset.py --list-memory

# Re-run credential setup (new LLM API token)
python3 scripts/setup_credentials.py
```

## Gotchas

- **Memory path must be absolute** — injected into system prompt so the LLM can use file_reader/file_writer tools. `config.DATA_DIR` is resolved at import time; don't move `data/` after startup without restarting.
- **Bot API key survives restarts** — stored in `data/.apikey`. Delete it to force re-registration with Messenger.
- **Streaming requires LLM API SSE support** — if the LLM API doesn't return `tool_status` events, tool-status messages won't appear, but the final reply still sends.
- **Heartbeat first tick is delayed** — runs after one full interval at startup, never immediately.
