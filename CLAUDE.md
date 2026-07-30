# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Shape

Three independent services live side-by-side in this repo, plus a shared cluster control plane:

- **`llm-api/`** (port 10002, Python/FastAPI) — OpenAI-compatible API wrapping a separately-run `vLLM` server, with an agentic loop, JWT auth, RAG, and tool calling.
- **`hoonbot/`** (port 10001, Python/FastAPI) — Bridges Messenger ↔ LLM API; handles webhooks, debouncing, persistent memory, heartbeat loop, and (on master) cluster delegation.
- **`messenger/`** (port 10006, Node/Express + React/Vite) — Real-time chat UI with Socket.IO, file uploads, and embedded `claude` / `opencode` terminals over WebSocket. Master-only.
- **`cluster_config.py`** at the repo root — single source of truth for node role, name, IPs, ports, prompt/heartbeat/skill profiles, and cluster token. Each app's local `config.py` imports it via `_load_cluster_config()` and exposes only what that app needs.

`vLLM` is **not** in this repo. It must be running before `llm-api` starts (default `http://127.0.0.1:10000`).

## Common Commands

### Whole-stack (cluster)

**Windows** — double-click `Start-Master.cmd` or `Start-Slave.cmd`. These call `setup-and-start-master.ps1` / `setup-and-start-slave.ps1`, which create a `.venv`, install Python deps, build the Messenger bundle (master only), and launch all services. Later runs start instantly with no npm/build. `-Rebuild` redoes the one-time setup.

```powershell
# Direct PowerShell (after first run, or with -Rebuild to redo setup):
.\setup-and-start-master.ps1          # master: venv + messenger build + launch
.\setup-and-start-master.ps1 -Rebuild # force redo setup
.\setup-and-start-slave.ps1           # slave: venv + launch
# Fast restart (no setup, uses existing venv set via $env:PYTHON):
.\start-master.ps1
.\start-slave.ps1
```

**Linux** — `./start-master.sh` / `./start-slave.sh`. Pass `--build` on first run to stage offline deps and build the Messenger bundle; later runs start directly.

```bash
./start-master.sh --build   # first run: stage deps (OFFLINE_DEPS_DIR) + launch
./start-master.sh           # fast restart
./start-slave.sh --build
./start-slave.sh
```

On Linux, `OFFLINE_DEPS_DIR` is the airgap contract for staged Messenger runtime assets. Supported bundle paths: `messenger/node_modules`, `messenger/server/dist`, `messenger/client/dist-web`. Linux skips Python package installation and assumes the target server's Python environment is already provisioned. If `OFFLINE_DEPS_DIR` is unset, the scripts auto-detect nearby bundle directories such as `./llm_api_fast_airgap`, `./offline_deps`, `../llm_api_fast_airgap`, or `$HOME/llm_api_fast_airgap`. They refuse online npm fallback and fail fast instead of contacting `registry.npmjs.org`.

The root `start-*` / `setup-and-start-*` entry points are thin wrappers over shared role-parameterized scripts in `scripts/`: `start-node.ps1`, `setup-node.ps1`, `start-node.sh`, and `install-node.sh` (the latter is called internally by `start-node.sh --build`; not user-facing).

### Per-service (run individually)

```powershell
cd llm-api    ; .\start.ps1 -Build      # vLLM must already be running
cd messenger  ; .\start.ps1 -Build      # first time: npm install + build:web + bundle server, then run
cd messenger  ; .\start.ps1             # later: runs prebuilt server/dist/server.cjs (no npm); -Dev for tsx watch
cd hoonbot    ; .\start.ps1 -Build      # depends on messenger + llm-api being up
```

For Linux Messenger production, `./start.sh --prod` now runs the prebuilt server bundle directly (`server/dist/server.cjs`) instead of `npm run start`. On Windows, `messenger\start.ps1` runs that same bundle by default (npm/Vite/tsx only on `-Build`).

### Validation (no repo-wide test suite)

```powershell
python -m py_compile <file.py>                  # syntax check Python edits
cd messenger ; npm.cmd run typecheck            # TS check (server + client)
cd messenger ; npm.cmd run build:web            # rebuild served bundle after UI/shared-type changes
cd hoonbot   ; python scripts\test_llm.py       # verify LLM API connectivity from hoonbot
cd llm-api   ; python scripts\clear_data.py     # wipe logs/scratch/sessions
cd llm-api   ; python scripts\clear_rag_data.py --all   # required when switching embedding models
```

`messenger/config.py --export powershell` (or `bash`) prints the env-var exports the start scripts use; `--ensure-dirs` creates runtime dirs.

## Cluster Architecture

The cluster is a master + N slaves over plain HTTP on the LAN. Roles are picked by `CLUSTER_ROLE` env var (set by the start scripts).

- **Master** runs all three services. Slaves run only `llm-api` + `hoonbot` (in worker mode).
- The master `llm-api` exposes a control plane under `/api/cluster/*` (register, heartbeat, claim/complete tasks). Backed by `llm-api/backend/core/cluster_store.py`.
- Slave `hoonbot` polls the master via `hoonbot/core/cluster_client.py` and `cluster_worker.py` to claim delegated tasks.
- Master `hoonbot` parses Messenger directives `@<node-name> task`, `@tag:<tag> task`, `@role:<role> task`, `@all-slaves task` and submits them to `/api/cluster/tasks`. Reserved directives: `bot`, `clear`, `compact`, `goal`, plus the bot's own name.
- `cluster_config.validate_advertised_url()` enforces a hard rule: inter-node URLs must be `http://<ip>:<port>`. Loopback, `localhost`, DNS names, and Cloudflare/`trycloudflare.com`/`aihoonbot.com` hostnames are rejected for advertised URLs. `MASTER_LLM_API_URL`, `MESSENGER_URL`, `HOONBOT_WEBHOOK_URL`, and `ADVERTISED_LLM_API_URL` are validated on master start.
- Each node has `NODE_NAME`, `NODE_CAPABILITIES`, and `NODE_TAGS`. `NODE_NAME` doubles as the routing handle, log name, and Messenger mention name — keep it unique per machine.
- Prompt/heartbeat/skill profiles resolve from `cluster_config` to `hoonbot/prompts/<profile>/PROMPT.md`, `.../HEARTBEAT.md`, and `hoonbot/profiles/<profile>/skills/`. Defaults are `master` and `slave`. Hoonbot falls back to top-level `prompts/PROMPT.md` / `HEARTBEAT.md` / `skills/` if a profile file is absent.

## Per-Service Architecture Notes

**llm-api** — single `while` loop in `backend/agent/`, no sub-agents or chains. Tool calls use capability-based segmented parallel execution: consecutive `is_concurrency_safe` calls (reads, web/RAG, subagent fan-out) run via `asyncio.gather`; mutating tools (`shell_exec`, `file_edit`, `apply_patch`, `file_writer`, `code_exec`, `memo`, `todo_write`, `process_monitor`) are serial barriers. Metadata flags live in `llm-api/tools/schemas.py` (`TOOL_METADATA`). The system prompt is cached at module import; vLLM prefix caching is a server-side flag (`--enable-prefix-caching`), not a per-request field. Microcompaction compresses old iterations and spills oversize tool results to `data/tool_results/`. Workers > 1 share no state — use `SERVER_WORKERS = 1` during development. Default admin: `admin` / `administrator`.

**vLLM tool-call parser (critical):** vLLM only streams tool-call deltas when launched with `--enable-auto-tool-choice --tool-call-parser <parser>`. `<parser>` must match the served model family (GLM-5.2→`glm47`, GLM-4.5/4.6→`glm45`, Qwen3→`hermes`, Llama 3.x→`llama3_json`, Mistral→`mistral`). Without this — or with a mismatch like `hermes` on a GLM model (GLM emits `<arg_key>/<arg_value>` XML, not Hermes JSON) — tool calls arrive as raw text and mid-output dispatch never fires. As a defense-in-depth safety net, `backend/core/llm_backend.py` (`_extract_leaked_tool_calls` / `_parse_leaked_tool_call`) peels leaked `<tool_call>...</tool_call>` markup back out of the `content` stream: cleanly-parsed blocks (GLM XML or Hermes JSON) are re-emitted as real tool-call events so they still execute, and anything unparseable is suppressed instead of shown to the user. Prefer fixing the launch flag; this is the backstop.

**GLM-5.2 thinking mode:** GLM-5.2 has three modes selected by `reasoning_effort` (`"max"` = default/deepest, `"high"` = balanced) or `enable_thinking:false` (non-thinking). `config.CHAT_TEMPLATE_KWARGS` (default `{"reasoning_effort": "max"}`) is forwarded on every request as a top-level `chat_template_kwargs` body field. Official GLM-5.2 sampling is `temperature=1.0, top_p=0.95, min_p=0.01` — set via `MODEL_TEMPERATURE_OVERRIDES["glm"]=1.0` and the `DEFAULT_TOP_P`/`DEFAULT_MIN_P` defaults.

**vLLM reasoning parser (critical for reasoning models):** GLM-4.5/4.6/5.x, Qwen3-Thinking, and DeepSeek-R1 emit a `<think>...</think>` chain before the answer. Launch vLLM with `--reasoning-parser <parser>` (GLM→`glm45`, Qwen3→`qwen3`, DeepSeek-R1→`deepseek_r1`) so that chain is lifted into the `reasoning_content` delta field. Without it the `<think>` block streams inline in `content` — the model's raw thinking (full of draft Python) leaks straight into the user's answer, and the hoonbot heartbeat's non-streaming planner/summarizer parsing chokes on it. As a defense-in-depth safety net, `backend/core/llm_backend.py::_split_inline_reasoning` peels model-emitted inline `<think>...</think>` back out of the content stream and re-emits it as `ReasoningEvent` (preserved in history, never shown), so a missing/mismatched reasoning parser degrades gracefully instead of leaking. Prefer fixing the launch flag; the splitter is a backstop.

The `AgentLoop` class (`backend/agent/loop.py`) is composed from five mixins — each in its own file under `backend/agent/`:

| Mixin | File | Responsibility |
|---|---|---|
| `LoggingMixin` | `logging_helpers.py` | Per-tool and per-iteration logging |
| `CompactionMixin` | `compaction.py` | Microcompaction, auto-compact on context overflow, anti-spiral state |
| `DispatchMixin` | `tool_dispatch.py` | In-process tool execution, parallel via `asyncio.gather` |
| `FormattingMixin` | `result_formatting.py` | Tool result formatting before feeding back to LLM |
| `PromptMixin` | `prompt_assembly.py` | System prompt, dynamic context (git, env, RAG collections) |

Add new agent capabilities by extending the appropriate mixin or adding a new one. Tool schemas live in `llm-api/tools/schemas.py`; each tool's implementation is under `llm-api/tools/<name>/`.

**RAG package** (`llm-api/tools/rag/`): one config-driven `RAGTool` in `tool.py` — hybrid BM25 search, cross-encoder reranking, and chunking strategy are runtime gates on `config.RAG_USE_HYBRID_SEARCH` / `RAG_USE_RERANKING` / `RAG_CHUNKING_STRATEGY` (no separate "enhanced" class). Supporting modules: `chunking.py` (fixed/sentence/semantic/recursive), `retrieval.py` (RRF fusion + reranker), `readers.py` (document ingestion; parallel PyMuPDF for big PDFs). Embedding model, reranker, FAISS, and BM25 instances are process-level singletons in `tool.py`.

**Available agent tools:** `websearch`, `code_exec`, `rag`, `file_reader`, `file_edit`, `apply_patch`, `file_writer`, `file_navigator`, `grep`, `shell_exec`, `shell_lint`, `process_monitor`, `memo`, `todo_write`, `agent`. Tool parameters are in `config.TOOL_PARAMETERS`; the enabled subset per-request is `config.AVAILABLE_TOOLS`.

**Per-session workspace:** The `AgentLoop` accepts a `workspace_dir` parameter. When set, `file_reader`, `file_writer`, `file_navigator`, `file_editor`, `apply_patch`, `grep`, and `shell_exec` all treat it as the project root and relative paths resolve against it.

**Data layout** (`llm-api/data/`, never committed):
- `sessions/{id}.jsonl` — full conversation history; `.recent.json` — last N messages for fast startup
- `jobs/{id}.json` — async background job state (RAG uploads via `/jobs/*` routes + `backend/core/job_store.py`)
- `tool_results/{id}.json` — spilled oversize tool outputs from microcompaction
- `logs/`, `scratch/` — agent logs and temp files

**Swagger UI** is served at `http://localhost:10002/docs` (ReDoc at `/redoc`) in development.

**hoonbot** — FastAPI webhook receiver. `handlers/webhook.py` validates → `mark_read` → debounces (`DEBOUNCE_SECONDS=1.5`) → calls LLM API. Inline directives handled in `process_message`: `@stop`, `@clear`, `@compact`, and `/goal` (also `@goal`). `/goal <task>` turns on sticky *goal mode* for the room (persisted in `data/room_goal_mode.json`) and forwards `mode="goal"` to the LLM API on every turn until `/goal off` (or `@clear`) clears it. Shared HTTP plumbing lives in `core/llm_api.py` (`get_client()`, `chat()`, `stream_chat()` for `/v1/chat/completions`) and `core/cluster_http.py` (headers/URL/JSON helpers for the master's `/api/cluster/*`) — use these instead of ad-hoc `httpx` scaffolding. Memory lives at `hoonbot/data/memory.md` and is injected into the system prompt with an absolute path so the LLM can use file_reader/file_writer tools — **don't move `data/` after startup without restarting**. Heartbeat first tick is delayed one full interval (never immediate). Bot API key is persisted at `hoonbot/data/.apikey`; deleting forces re-registration with Messenger. Skills are plain Markdown files in `hoonbot/skills/`; the agent discovers them via `file_navigator` — no loader code needed.

**messenger** — `sql.js` runs the SQLite DB **in memory**, auto-saved to `data/messenger.db` every 5s — unclean shutdown can lose up to 5s. No TLS built-in; expects LAN/VPN. Terminal WebSocket (`/claude`, `/opencode`) must be registered on the HTTP server **before** Socket.IO attaches (order in `server/src/index.ts`); it gates access via `SECRET_TOKEN` (default `leesihun` — change in prod). Terminal executable paths are configurable via `CLAUDE_CMD`, `OPENCODE_CMD`, and `WORKSPACE_DIR` env vars (set in `messenger/config.py`). Web watchers (`server/src/services/web-poller.ts`) poll URLs on interval, hash responses, and post diffs to rooms when content changes. **All message writes** (create/edit/delete/react — REST bot API, socket handlers, web poller alike) go through `server/src/services/messages.ts`, which owns the DB insert, room broadcast, and webhook dispatch; the Socket.IO instance is shared via `server/src/services/io.ts`. The bot API is split per resource under `server/src/routes/api/` (mounted at `/api` with `apiKeyAuth`).

**Run vs build:** `messenger/start.ps1` runs the prebuilt `server/dist/server.cjs` directly by default — npm/Vite/tsx only run on `-Build` (`-Dev` runs the tsx watch dev server). The bundle (`server/dist`, `client/dist-web`) is gitignored, so `setup-and-start-master.ps1` builds it once if missing. **Desktop app (`build-portable.mjs` at the messenger root → `npm run build:portable`)** is a **thin client**, not an embedded server: `electron/main.ts` loads the master URL (`http://<master-ip>:10006`, baked from `cluster_config.MESSENGER_URL` into `app-config.json`, editable via `electron/setup.html` / Ctrl+,). No server/web bundle/node-pty ships inside the `.exe`; the web client targets `window.location.origin` (`client/src/services/api.ts`), so loading the master origin routes axios + Socket.IO + terminals to the master.

## Configuration Conventions

- **`/cluster_config.py` is the single user-facing config surface.** Its top-of-file `EDIT HERE` block holds every commonly-changed setting for all three services: role, node name, LAN IPs, the vLLM backend URL, ports, cluster secret, llm-api admin creds + Tavily key + RAG model paths, hoonbot bot name + heartbeat, and the Messenger terminal token. Each value is also overridable by an env var of the same name (that's how the launchers set `CLUSTER_ROLE`).
- Each app's `config.py` holds **advanced per-service tuning** and reads the common knobs from `cluster_config.py` via `_load_cluster_config()` + `getattr(_CLUSTER, NAME, default)`. When adding a new commonly-changed setting, expose it in the `cluster_config.py` `EDIT HERE` block and read it in the app config with a `getattr` fallback — don't hardcode deployment-specific values in app configs.
- **`VLLM_MODEL`** in `cluster_config.py` must match vLLM's `--served-model-name`. Discover via `curl http://127.0.0.1:10000/v1/models`. vLLM returns 404 on `/v1/chat/completions` if the `model` field doesn't match.
- **`response_format` / `guided_json`** can be passed to `/v1/chat/completions` (JSON body or form) to enable vLLM guided/structured decoding. `AgentLoop` accepts them as constructor args and forwards them to every LLM call in the session.
- **`mode="goal"`** on `/v1/chat/completions` (JSON body or form) opts the request into *goal mode* — an extensive, goal-focused run: `AgentLoop` raises the iteration budget to `AGENT_GOAL_MAX_ITERATIONS`, injects the `prompts/agent/goal_mode.txt` preamble, and enforces a done-gate that refuses to finish while `todo_write` items remain incomplete (capped by `AGENT_GOAL_MAX_COMPLETION_BLOCKS`). The param is per-request/stateless; hoonbot provides the sticky Messenger UX via the `/goal` directive.
- Do NOT introduce hidden `.env` runtime requirements. Legitimate runtime env overrides are limited and explicit: e.g. `LLM_API_URL` for hoonbot, `VLLM_HOST` for llm-api, the `MESSENGER_*_DIR` paths for messenger. Don't add new env-var knobs casually.
- Never commit: `data/` (any service's runtime dir), `*.gguf`, `models/`, `llamacpp/`, `offline_models/`, `.env*`, `.apikey`, `.llm_key`, `.llm_model`, `node_modules/`, `dist*/`.

## Style (from AGENTS.md)

- 4-space indent for Python, 2-space for TS/TSX.
- `snake_case` for Python; `camelCase` vars/funcs and `PascalCase` components for TS/React.
- Commits: small, scoped subjects like `messenger: rebuild web attachment flow`.
- PRs should list verification commands run, mention any port/config changes, and include screenshots for visible Messenger UI changes.
