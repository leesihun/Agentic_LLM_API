# Hoonbot Architecture

## Overview

Hoonbot is a tool-driven personal AI assistant that bridges Huni Messenger (chat frontend) and LLM_API_fast (LLM backend). It has no custom databases or complex state management — just a memory file, a skills directory, and pure tool usage through the LLM agent.

```
┌─────────────────────┐
│  Huni Messenger     │ (Chat UI on port 10006)
│   (TypeScript)      │
└──────────┬──────────┘
           │ webhooks + REST API
           │
┌──────────▼──────────┐
│     Hoonbot         │ (FastAPI on port 10001)
│    (Python)         │
│                     │
│  • Webhook handler  │
│  • Messenger client │
│  • Heartbeat loop   │
│  • Session manager  │
│  • Prompt builder   │
└──────────┬──────────┘
           │ /v1/chat/completions (streaming SSE)
           │
┌──────────▼──────────┐
│ LLM_API_fast        │ (Agent system on port 10002)
│ (Python)            │
│                     │
│ • websearch         │
│ • file_reader/writer│
│ • file_navigator    │
│ • python_coder      │
│ • shell_exec        │
│ • rag / memo        │
│ • process_monitor   │
└─────────────────────┘
```

## File Organization

```
Hoonbot/
├── hoonbot.py              # Entry point, FastAPI server + lifespan startup
├── config.py               # Single Hoonbot runtime config surface
├── prompts/
│   ├── PROMPT.md           # Base system prompt (always loaded first)
│   ├── HEARTBEAT.md        # Base heartbeat prompt + checklist
│   ├── master/             # Master overlay prompt + heartbeat profile
│   └── slave/              # Slave overlay prompt + heartbeat profile
├── ARCHITECTURE.md         # This file
├── setup.py                # One-time: configure LLM API credentials
├── reset.py                # Utility: clear/view memory
├── test_llm.py             # Dev utility: test LLM connectivity
├── requirements.txt        # fastapi, uvicorn, httpx
│
├── handlers/
│   ├── webhook.py          # Message processing, session management, streaming
│   └── health.py           # GET /health endpoint
│
├── core/
│   ├── context.py          # System prompt + capped memory view + per-turn context
│   ├── memory_guard.py     # memory.md size cap: backup + condense
│   ├── fs_snapshot.py      # Filesystem map / drift digest
│   ├── messenger.py        # Messenger REST API client (persistent httpx pool)
│   ├── heartbeat.py        # Background heartbeat loop
│   └── retry.py            # Exponential backoff helper
│
├── skills/                 # Skill docs — loaded by agent via file_reader
│   └── diagnose_system.md  # Full system health check procedure
│
└── data/
    ├── memory.md           # Persistent LLM memory (Markdown, durable facts only)
    ├── memory_backups/     # Pre-condense / pre-reset copies (last 10)
    ├── status.md           # Live status, overwritten by the heartbeat
    ├── room_sessions.json  # Room → session ID mapping (auto-created)
    ├── room_queue.json     # Room → unanswered (queued / in-flight) msg_ids, replayed on start
    ├── .apikey             # Messenger bot API key (auto-created)
    ├── .llm_key            # LLM API bearer token (created by setup.py)
    └── .llm_model          # LLM model name (created by setup.py)
```

## Core Components

### 1. Entry Point: `hoonbot.py`

FastAPI app with async lifespan startup:

1. LLM health check (non-fatal if unreachable), then a background memory size check (`memory_guard.enforce_memory_cap("startup")`; never blocks startup)
2. Bot registration — restores key from `data/.apikey` or registers new
3. Webhook subscription — `new_message`, `message_edited`, `message_deleted`
4. Catch-up — replays, in id order, every human message after the bot's last post in each room plus every msg_id the previous run left queued or in flight (`data/room_queue.json`, fetched by id via `messenger.get_message`), through the per-room queue (`handlers.webhook.enqueue_catchup_message`). Attachments are rebuilt the same way as for a live webhook, so attachment-only messages replay too. Persisted messages skip room gating (they were already accepted) and have a stale ⏳ from the previous run cleared; deleted or vanished ones are dropped. The msg_id LRU dedupe keeps a replay from racing a live webhook. Without `enqueue_catchup_message`, the last missed text message is answered directly.
5. Heartbeat loop — background task

On shutdown the lifespan cancels catch-up and calls `handlers.webhook.shutdown_room_workers()`. That stops new turns and lets a turn that is already writing its final answer (or submitting a cluster delegation) finish. Any other running turn is cancelled, and its draft is settled with a "⏹️ 재시작으로 중단됨" marker. The queue state is then flushed so the next start replays the unanswered messages.

### 2. Webhook Handler: `handlers/webhook.py`

**Session management:** Per-room sessions in `data/room_sessions.json`. Sessions auto-expire after `SESSION_MAX_AGE_DAYS`. On 404 from LLM API, clears and starts fresh.

**Message flow:**
1. Validate (text/image/file, not bot, @mention in non-home rooms)
2. Mark read (fire-and-forget)
3. `@stop` / `@status` / `@clear` run immediately; other messages enter the per-room FIFO queue
4. Debounce (`DEBOUNCE_SECONDS`), then the room worker takes up to `BATCH_MAX_MESSAGES` queued messages as one labelled batch; messages arriving mid-turn are handled by `FOLLOWUP_POLICY` (default `"preempt"`, see **Follow-up policy** below)
5. `process_message()` — builds context and calls LLM

**New session context injection** (`core/context.py::build_llm_context`):
- Layered system prompt from `config.load_prompt_text()`: base `prompts/PROMPT.md`, then the profile `PROMPT.md` (`prompts/<profile>/`) appended after a separator. Profile rules win on conflict; a missing profile file means base only.
- Session variables (Messenger URL/key, paths, `memory_cap_chars`, `status_file`, skills dir)
- Current memory content, capped at `MEMORY_MAX_CHARS` (see Memory)

**Existing session:** sends only the user message with `session_id` (LLM API retains history).

**Memory flush:** At message #`MEMORY_FLUSH_THRESHOLD` (default 30) in a session, or when memory.md is over the soft cap, `prompts/webhook/memory_flush_hint.txt` is injected (formatted with `{memory_chars}` / `{memory_cap}`), nudging the agent to save durable facts with targeted `file_edit`/`apply_patch` edits and prune stale entries.

**Follow-up policy** (`FOLLOWUP_POLICY`, set in `cluster_config.py`):
- `preempt` (default): a human message that arrives while a turn is running does not wait. Once its burst has been quiet for `DEBOUNCE_SECONDS`, the room worker cancels the running turn if it is still in its working phase. The turn's draft bubble is settled by edit with the partial text plus `⏹️ 새 메시지가 와서 여기서 중단했어요 — 이어서 답할게요`, tools still running in it show as stopped, and hoonbot calls `POST /api/chat/sessions/{id}/stop` (`llm_api.stop_session`) so llm-api kills the tools' processes and releases the session lease. The worker then waits (bounded) for the lease to clear (`llm_api.get_activity`) and starts the new turn at once. The new batch carries the interrupted message(s) again, sorted by message id and flagged `status="interrupted-earlier-request"`, with a line telling the model not to repeat work already done. The room's llm-api session is reused, never reset, so its history already holds the interrupted partial turn and tool ledger. Interrupted messages stay in `data/room_queue.json` until a turn that includes them ends.
- A turn is not preemptable once it is finalizing (writing the answer), delegating to the cluster, or running an in-queue directive such as `@compact`. The follow-up then falls back to `queue` behaviour: it waits with the ⏳ reaction and runs after the current turn. A directive at the head of the queue (`@compact`, `/goal`, a queued `@clear`) never preempts, so order is kept. Only messages that arrived after the running turn started can preempt it (overflow beyond `BATCH_MAX_MESSAGES` that was already queued cannot), and a message preempted `_PREEMPT_MAX_PER_BATCH` (3) times is no longer preempted: its turn runs to completion and newer messages queue behind it. Messages deleted while their turn is running are not re-queued after a preempt (edits are carried over). A message whose turn never reached llm-api is re-queued without the `interrupted-earlier-request` flag. An unknown `FOLLOWUP_POLICY` value falls back to `queue` with a warning.
- `queue`: the follow-up always waits for the next turn. `interrupt`: a same-sender follow-up cancels the running turn and merges with it.
- `@stop`, `@status` and `@clear` keep their immediate handling under every policy.

**`@status`** (also `/status`): runs outside the queue like `@stop`, is read-only and never preempts or cancels. It replies with whether a turn is running and for how long (working / finalizing / delegating), the number of queued messages, goal mode, and what llm-api reports from `GET /api/chat/sessions/{id}/activity` (agent turn age and iteration, running tools with detail and elapsed time, live processes with their idle time).

**Queue persistence:** the msg_ids of each room's in-flight batch and waiting queue are written to `data/room_queue.json` atomically. Writes are debounced (0.5 s) on enqueue, dequeue, delete and turn end, and happen immediately during shutdown. A batch leaves the file when its turn ends (answered, errored or `@stop`ped); it stays only when shutdown cut the turn short before its final answer was being written.

**Streaming:** Text streams into one threaded draft bubble that is edited in place and settled into the final reply. Tool activity shows in the typing indicator and an in-bubble checklist. While a tool runs, llm-api sends `tool_status` events with `status: "progress"` (plus `detail`, `elapsed` and `processes`); the typing indicator then names the longest-running tool with its elapsed time (`file_navigator 1분 20초째 사용 중 (list /srv/data)`), and its worklog line reads `⏳ label — detail · 1분 20초` with up to two indented process lines (`↳ pid 1234 · python train.py · 출력 없음 45초`). Progress updates reach the bubble at most every 4 s on top of the existing edit limiter, so many tools cannot cause an edit storm. A call killed by the supervisor or a hard timeout shows `❌ label · 실패 — stopped: <reason>`; a supervisor review entry shows as `🛡️ 진행 점검`. Creating the draft uses `messenger.send_message_returning_id`, which returns `None` only when the message was definitely not created; a later flush retries, up to 3 times. It raises `MaybeDeliveredError` when the send may have been stored. In that case the bubble is looked up among the newest room messages instead of a second one being posted; if it cannot be found, only the final reply is sent. When llm-api already ended the text with its inline cut notice (`[출력이 길이 제한으로 잘렸습니다]` / `[응답 생성 중 연결 오류로 출력이 중단되었습니다]`), hoonbot does not add its own length/error notice.

### 3. Heartbeat: `core/heartbeat.py`

Background loop that reads the profile heartbeat file (`config.HEARTBEAT_FILE`; `config.load_heartbeat_text()` gives the layered base + profile text) from `prompts/`, runs `memory_guard.enforce_memory_cap("heartbeat")` each tick, builds full context (same as new session), and runs through the agent non-streaming. Posts results to home room. Features:
- Active-hours window (configurable start/end times)
- LLM cooldown on connection errors (avoids hammering dead server)
- First tick after one full interval (never immediate)

### 4. Skills System

Markdown files in `skills/` — the agent loads them via `file_reader`/`file_navigator` at runtime. No loader code needed; the skills directory path is injected into every new session context.

To add a skill: drop a `.md` file in the profile skills directory (`config.SKILLS_DIR`). The agent finds it with `file_navigator` (skill names are also listed in the per-turn context); do not add skill references to `memory.md`.

### 5. Memory: `data/memory.md`

Single curated Markdown file of **durable facts** (preferences, decisions, stable paths/IDs, `## Reminders`). Read by the agent with `file_reader` and edited in place with `file_edit`/`apply_patch`. The absolute path is provided so the LLM knows where to find it. Live status (health numbers, training progress) goes to `data/status.md`, overwritten each heartbeat.

Size is bounded at two levels:

- **Injection cap** (`core/context.py`): the copy injected into a new session is cut to `MEMORY_MAX_CHARS` (default 12,000) at line boundaries, keeping the head and the `## Reminders` section, followed by a truncation marker telling the agent to read the file before rewriting it. The per-turn reminder shows `memory.md: N / 12,000 chars`.
- **On-disk guard** (`core/memory_guard.py`): `enforce_memory_cap(reason, force)` runs at startup, every heartbeat tick, after a turn when the file is over `MEMORY_SOFT_CHARS` (default 8,000), and with `force=True` after a context overflow. It backs the file up to `data/memory_backups/` (last 10 kept), asks the LLM (tool-less, ephemeral) for a condensed rewrite, validates it (non-empty, under the cap, `## Reminders` kept), and swaps it in atomically only if the file did not change meanwhile. If the LLM path fails and the file is over the hard cap (or the call is forced), a deterministic trim keeps headings, Reminders and the newest lines. Callbacks registered with `register_on_condensed()` notify the webhook, which keeps every room conversation: it bumps its sessions epoch and flags each cached room session `memory_refresh` in `data/room_sessions.json`. The next turn on a flagged session prepends a `<system-reminder>` carrying the current (capped) memory.md and marks the first-turn copy stale; the flag is cleared after that turn unless another condense landed meanwhile, and a session created by a turn that was in flight across a condense is saved already flagged.

Restore a backup by copying it over `data/memory.md` (with Hoonbot stopped).

### 6. Messenger Client: `core/messenger.py`

Persistent `httpx.AsyncClient` wrapping the Messenger REST API. Auto-splits long messages at `MAX_MESSAGE_LENGTH` (2000 chars). All methods are best-effort with logging.

## Configuration

All Hoonbot runtime settings live in `config.py`. Key settings:

| Setting | Default | Purpose |
|---------|---------|---------|
| `HOONBOT_PORT` | 10001 | Bot server port (cluster_config) |
| `MESSENGER_BOT_NAME` | `BOT_NAME` from cluster_config | Display name |
| `MESSENGER_HOME_ROOM_ID` | 1 | Fallback room for heartbeat output (name lookup first) |
| `STREAMING_ENABLED` | True | Stream LLM responses |
| `DEBOUNCE_SECONDS` | 1.5 | Burst-gathering window before a queued batch runs |
| `FOLLOWUP_POLICY` | preempt | Mid-turn messages: `preempt` (cancel the running turn, merge), `queue` or `interrupt` (cluster_config) |
| `ROOM_QUEUE_MAX` / `BATCH_MAX_MESSAGES` | 50 / 20 | Per-room queue bound / messages per turn |
| `LLM_TIMEOUT_SECONDS` | 3600 | Non-streaming wall-clock cap (s) |
| `LLM_STREAM_IDLE_TIMEOUT_SECONDS` | 900 | Streaming idle watchdog (s) |
| `SESSION_MAX_AGE_DAYS` | 1 | Auto-expire sessions |
| `MEMORY_FLUSH_THRESHOLD` | 30 | Messages before memory flush hint |
| `MEMORY_SOFT_CHARS` / `MEMORY_MAX_CHARS` | 8000 / 12000 | memory.md target / hard cap (cluster_config) |
| `MAX_CONVERSATION_TOKENS` | 300000 | Conversation budget (in step with llm-api) |
| `HEARTBEAT_INTERVAL_SECONDS` | 3600 | Seconds between heartbeats (cluster_config) |
| `HEARTBEAT_ACTIVE_START/END` | 00:00/23:59 | Active hours window |

## Design Decisions

1. **No custom systems** — use LLM_API_fast's built-in tools
2. **Single memory file** — simple, inspectable, LLM-managed, size-capped with backups
3. **Skills as docs** — agent reads and executes them, no special loader
4. **Absolute paths** — injected into prompt so LLM knows exact locations
5. **Minimal code** — 3 dependencies, easy to debug and extend
6. **Heartbeat for scheduling** — user puts tasks in memory, heartbeat picks them up; live results go to `data/status.md`, not memory

---

**Last Updated:** 2026-11-05
