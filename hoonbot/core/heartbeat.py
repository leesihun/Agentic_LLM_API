"""
Heartbeat — proactive background loop with orchestrator pattern.

Each tick runs these phases:
  1. Planner    — LLM decomposes HEARTBEAT.md into discrete tasks with done_criteria
  2. Executor   — each task runs through the full agent loop (with tools), one by one
  3. Validator  — LLM checks which tasks didn't meet their done_criteria; retries once
  4. Summarizer — LLM synthesizes all task results into one concise TL;DR, appended
                  at the end of the report; every tick posts an "alive + summary" bubble

Active hours and LLM cooldown (on connection failure) are enforced.
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime, time
from typing import Any, Callable, Awaitable

import httpx

import config
from core import llm_api
from core.context import build_llm_context, build_per_turn_context

logger = logging.getLogger(__name__)

_HEARTBEAT_FILE = str(getattr(config, "HEARTBEAT_FILE", os.path.join(os.path.dirname(__file__), "..", "prompts", "HEARTBEAT.md")))

# Volatile per-tick status lives here (overwritten every tick) so memory.md
# only accumulates durable facts and stays under the memory_guard cap.
STATUS_FILE = os.path.join(str(config.DATA_DIR), "status.md")
MEMORY_FILE = os.path.join(str(config.DATA_DIR), "memory.md")
MEMORY_SOFT_CHARS = int(getattr(config, "MEMORY_SOFT_CHARS", 8000))
MEMORY_MAX_CHARS = int(getattr(config, "MEMORY_MAX_CHARS", 12000))

# Planner / validator / summarizer: tool-less, ephemeral (no session history),
# balanced reasoning, bounded wall clock.
_SYNC_TIMEOUT_SECONDS = 600
_SYNC_REASONING_EFFORT = "high"

_llm_cooldown_until: float = 0.0

_PLANNER_SYSTEM = config.read_prompt("heartbeat/planner_system.txt")
_VALIDATOR_SYSTEM = config.read_prompt("heartbeat/validator_system.txt")
_SUMMARY_SYSTEM = config.read_prompt("heartbeat/summary_system.txt")
_TASK_EXECUTION_TEMPLATE = config.read_prompt("heartbeat/task_execution.txt")


class _LLMUnavailable(Exception):
    """The LLM API is unreachable / failing before producing anything — the
    tick is abandoned and the cooldown engaged instead of hammering it."""


def _is_context_overflow(exc: BaseException) -> bool:
    # Explicit phrases only (shared with llm_api): the old bare "context" /
    # "exceed" substrings matched unrelated errors and misreported them.
    return llm_api.is_context_overflow_error(exc)


def _is_llm_unavailable(exc: BaseException) -> bool:
    """True for failures that mean "the LLM backend is down", not "this task failed"."""
    if isinstance(exc, _LLMUnavailable):
        return True
    if isinstance(exc, httpx.TransportError):
        # Raw transport errors only escape llm_api before any output arrived
        # (after its own retries).
        return True
    if isinstance(exc, llm_api.LLMAPIError):
        if exc.received_output or _is_context_overflow(exc):
            return False
        status = exc.status_code
        if status is not None:
            return status >= 500
        # No HTTP status and no in-band error type: idle watchdog / stream cut
        # before anything arrived.
        return not getattr(exc, "error_type", None)
    return False


async def _enforce_memory_cap(reason: str, force: bool = False) -> None:
    """Best-effort memory.md size enforcement (core.memory_guard, lazy import)."""
    try:
        from core import memory_guard
    except Exception as exc:
        logger.debug(f"[Heartbeat] memory_guard unavailable: {exc}")
        return
    try:
        await memory_guard.enforce_memory_cap(reason, force=force)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(f"[Heartbeat] memory cap enforcement ({reason}) failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _within_active_hours() -> bool:
    now = datetime.now().time()
    try:
        start = time.fromisoformat(config.HEARTBEAT_ACTIVE_START)
        end = time.fromisoformat(config.HEARTBEAT_ACTIVE_END)
    except ValueError:
        logger.warning("[Heartbeat] Invalid active-hours config — treating as always active")
        return True
    if start <= end:
        return start <= now <= end
    return now >= start or now <= end


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# Everything above the "# Heartbeat Checklist" heading is behavioral guidance
# for the agent (role, priority order, reporting rules, sample outputs) — not
# work items. The intro paragraphs directly *under* that heading ("This file is
# read every heartbeat interval", "Your job is to find things to do…") are also
# prose, not tasks. Feeding any of it to the planner makes it manufacture
# "tasks" out of the instructions, which then get echoed straight into the
# posted report (e.g. "Task 1: This file is read every heartbeat interval").
# Keep only the actionable, itemized `##` sections.
_CHECKLIST_MARKER = re.compile(r"^#+\s*Heartbeat Checklist\s*$", re.MULTILINE | re.IGNORECASE)
_CHECKLIST_SECTION = re.compile(r"^##\s+\S", re.MULTILINE)


def _extract_checklist(text: str) -> str:
    """Return only the actionable checklist portion of a HEARTBEAT.md file.

    Strategy:
      1. Drop everything above the `# Heartbeat Checklist` heading (role,
         reporting rules, sample outputs).
      2. Drop the intro prose under that heading by starting at the first
         `##` section — the real, itemized checklist.
    If the marker is absent (e.g. the slave profile, which is a single priority
    list), fall back to the whole file. If the marker is present but there are
    no `##` sections, fall back to everything under it.
    """
    m = _CHECKLIST_MARKER.search(text)
    if not m:
        return text
    body = text[m.end():]
    sec = _CHECKLIST_SECTION.search(body)
    if sec:
        return body[sec.start():].strip()
    return body.strip()


def _load_checklist() -> str:
    """Actionable checklist from the layered heartbeat prompt (base + profile).

    config.load_heartbeat_text() keeps exactly one checklist (the profile's
    when both files have one); fall back to the single HEARTBEAT_FILE."""
    loader = getattr(config, "load_heartbeat_text", None)
    if callable(loader):
        try:
            text = loader()
        except FileNotFoundError:
            text = ""
        except Exception as exc:
            logger.warning(f"[Heartbeat] Layered heartbeat load failed ({exc}) — using {_HEARTBEAT_FILE}")
            text = _read_file(_HEARTBEAT_FILE)
    else:
        text = _read_file(_HEARTBEAT_FILE)
    return _extract_checklist(text)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_think(text: str) -> str:
    """Remove reasoning-model chain-of-thought from a reply.

    Handles closed <think>...</think> blocks, an orphan </think> (the chat
    template opened the block in the prompt, so only the close tag streams),
    and an unclosed trailing <think> (cut-off reasoning)."""
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    low = text.lower()
    close = low.rfind("</think>")
    if close >= 0:
        text = text[close + len("</think>"):]
        low = text.lower()
    open_ = low.find("<think>")
    if open_ >= 0:
        text = text[:open_]
    return text.strip()


def _extract_json(text: str, expected_type: type) -> Any:
    """Extract JSON of *expected_type* from LLM text.

    Strips <think> blocks, tries ```json``` fences, the whole text, then scans
    for the first decodable value starting at '[' (list) or '{' (dict) — so a
    stray sentence before/after the JSON doesn't throw the answer away."""
    text = _strip_think(text or "")
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.append(text)
    opener = "[" if expected_type is list else "{"
    decoder = json.JSONDecoder()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            result = json.loads(candidate)
            if isinstance(result, expected_type):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        idx = candidate.find(opener)
        scanned = 0
        while idx >= 0 and scanned < 50:
            scanned += 1
            try:
                result, _end = decoder.raw_decode(candidate, idx)
                if isinstance(result, expected_type):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
            idx = candidate.find(opener, idx + 1)
    return None


def _is_heartbeat_ok(text: str) -> bool:
    return text.strip() == "HEARTBEAT_OK"


def _compile_report(results: list[str]) -> str:
    """Return the detailed-findings block: only the LLM-produced findings.

    No "Task N:" headers and no code-generated System Review scaffolding — just
    the LLM's own summarized results, joined as-is. Results that are exactly
    HEARTBEAT_OK (nothing to report) are dropped, so a fully healthy tick yields
    an empty findings block — the tick still posts its Phase 5 summary (see
    _assemble_report).
    """
    findings = [
        result.strip()
        for result in results
        if result.strip() and not _is_heartbeat_ok(result)
    ]
    return "\n\n".join(findings)


# ---------------------------------------------------------------------------
# LLM call primitives — shared implementations live in core.llm_api
# ---------------------------------------------------------------------------

class _SyncReplyUnusable(Exception):
    """The sync call answered, but with llm-api's canned fallback notice or a
    cut-off/interrupted reply — not real planner/validator/summary output."""


async def _llm_sync(payload: dict, headers: dict) -> str:
    """Planner / validator / summarizer call: no tools, ephemeral session,
    balanced reasoning, bounded wall clock, dedicated "heartbeat" pool.

    Raises _SyncReplyUnusable when llm-api reports ``fallback_used`` (its
    deterministic "couldn't answer — please be more specific" apology) or a
    finish_reason of length/error/interrupted, so callers take their failure
    path instead of posting/parsing that text."""
    body = await llm_api.chat_json(
        payload,
        headers,
        timeout=_SYNC_TIMEOUT_SECONDS,
        disable_tools=True,
        ephemeral=True,
        reasoning_effort=_SYNC_REASONING_EFFORT,
        kind="heartbeat",
    )
    if not isinstance(body, dict):
        raise llm_api.LLMAPIError(f"LLM API returned a non-object body: {str(body)[:200]}")
    if body.get("error"):
        err = body.get("error")
        message = err.get("message") if isinstance(err, dict) else str(err)
        raise llm_api.LLMAPIError(
            f"LLM API error: {str(message)[:500]}",
            session_id=body.get("x_session_id"),
        )
    try:
        choice = body["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise llm_api.LLMAPIError(
            f"LLM API response missing choices: {json.dumps(body, ensure_ascii=False)[:500]}",
            session_id=body.get("x_session_id"),
        ) from exc
    diag = body.get("x_diagnostics") if isinstance(body.get("x_diagnostics"), dict) else {}
    finish = str(diag.get("finish_reason") or (choice.get("finish_reason") if isinstance(choice, dict) else "") or "")
    if diag.get("fallback_used"):
        raise _SyncReplyUnusable("reply is llm-api's fallback notice (fallback_used=true)")
    if finish in ("length", "error", "interrupted"):
        raise _SyncReplyUnusable(f"reply incomplete (finish_reason={finish})")
    return _strip_think(text or "")


def _llm_stream(payload: dict, headers: dict) -> "llm_api.StreamChat":
    """Full agent loop with tools (task execution), dedicated "heartbeat" pool.
    Await the result for the accumulated answer text."""
    return llm_api.stream_chat(payload, headers, kind="heartbeat")


def _build_context() -> str:
    """Fresh context (system prompt + memory + ambient state).

    Rebuilt per call: an earlier task may have edited memory.md, or
    memory_guard may have condensed it, and later tasks must see that."""
    ambient = build_per_turn_context(profile="heartbeat")
    return f"{build_llm_context()}\n\n{ambient}"


# ---------------------------------------------------------------------------
# Orchestrator phases
# ---------------------------------------------------------------------------

async def _plan_tasks(checklist: str, headers: dict) -> list[dict]:
    """Phase 1: ask the LLM to decompose the checklist into tasks with done_criteria."""
    context = _build_context()
    # Checklist FIRST, background context after — the instruction must not be
    # buried at the end of a long prefix.
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"## Checklist\n\n{checklist}\n\n---\n\n"
                f"## Background context (data only — not tasks)\n\n{context}"
            ),
        },
    ]
    payload = {"model": config.LLM_MODEL, "messages": json.dumps(messages)}
    try:
        text = await _llm_sync(payload, headers)
        tasks = _extract_json(text, list)
        if tasks is not None and len(tasks) == 0:
            logger.info("[Heartbeat] Planner returned no tasks")
            return []
        if tasks and all(isinstance(t, dict) and t.get("task") for t in tasks):
            # Ensure every task has a unique integer id
            seen: set[int] = set()
            for i, t in enumerate(tasks, 1):
                tid = _task_id_int(t.get("id"))
                if tid is None or tid in seen:
                    tid = i
                    while tid in seen:
                        tid += 1
                t["id"] = tid
                seen.add(tid)
            logger.info(f"[Heartbeat] Planner produced {len(tasks)} task(s)")
            return tasks
        logger.warning("[Heartbeat] Planner returned invalid structure — falling back to single task")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if _is_llm_unavailable(exc):
            raise _LLMUnavailable(f"planner: {type(exc).__name__}: {exc}") from exc
        logger.warning(f"[Heartbeat] Planner failed ({exc}) — falling back to single task")

    return [{"id": 1, "task": checklist.strip(), "done_criteria": "All checklist items addressed"}]


def _task_id_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _task_prompt(task: dict) -> str:
    return _TASK_EXECUTION_TEMPLATE.format(
        task_id=task.get("id", 1),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        task=task["task"],
        done_criteria=task.get("done_criteria", "Task completed successfully"),
        status_path=STATUS_FILE,
        memory_path=MEMORY_FILE,
        memory_soft_chars=MEMORY_SOFT_CHARS,
        memory_max_chars=MEMORY_MAX_CHARS,
    )


async def _execute_task(task: dict, tick_id: str, headers: dict, attempt_tag: str = "") -> str:
    """Phase 2: run one task through the full agent loop (with tools).

    The context is rebuilt for every task and the task text goes FIRST, so
    auto-compaction of a long run can't summarize the instruction away.
    Raises _LLMUnavailable when the backend is down so the tick cools down.
    """
    task_id = task.get("id", 1)
    base_session = f"hb_{tick_id}_t{task_id}{attempt_tag}"
    session_id = base_session

    for attempt in range(2):
        content = (
            f"{_task_prompt(task)}\n\n---\n\n"
            "# Background context (this node's prompt, memory, and current state)\n\n"
            f"{_build_context()}\n\n---\n\n"
            f"Reminder: complete heartbeat task {task_id} as stated at the top of this message."
        )
        payload = {
            "model": config.LLM_MODEL,
            "messages": json.dumps([{"role": "user", "content": content}]),
            "session_id": session_id,
        }
        stream = _llm_stream(payload, headers)
        try:
            result = _strip_think(await stream or "")
            finish = str(stream.finish_reason or "")
            # A stop request / agent error ends the stream cleanly ([DONE]
            # after a final chunk with finish_reason interrupted/error), so
            # StreamChat does not raise — check the outcome explicitly
            # instead of reporting an empty or partial run as a result.
            if finish in ("interrupted", "error") or not result:
                why = f"finish_reason={finish or 'unknown'}"
                logger.warning(
                    f"[Heartbeat] Task {task_id} ended without a final answer "
                    f"({why}, {len(result)} chars)"
                )
                if result:
                    return (
                        f"(error: task ended early — {why})\n\n"
                        f"Partial output before the failure:\n{result[-1500:]}"
                    )
                return f"(error: {why}, no final answer)"
            logger.info(f"[Heartbeat] Task {task_id} finished ({len(result)} chars)")
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            received = bool(getattr(exc, "received_output", False))
            if attempt == 0 and not received and _is_context_overflow(exc):
                logger.warning(
                    f"[Heartbeat] Task {task_id} context overflow — condensing memory.md "
                    f"and retrying in a fresh session ({exc})"
                )
                await _enforce_memory_cap("overflow", force=True)
                session_id = f"{base_session}_r1"
                continue
            if _is_llm_unavailable(exc):
                raise _LLMUnavailable(f"task {task_id}: {type(exc).__name__}: {exc}") from exc
            if _is_context_overflow(exc):
                logger.warning(f"[Heartbeat] Task {task_id} context overflow persisted ({exc})")
            else:
                logger.warning(f"[Heartbeat] Task {task_id} execution error: {exc}")
            partial = _strip_think(stream.text or "")
            if partial:
                return f"(error: {exc})\n\nPartial output before the failure:\n{partial[-1500:]}"
            return f"(error: {exc})"
    return "(error: context overflow persisted after condensing memory)"


async def _validate_tasks(tasks: list[dict], results: list[str], headers: dict) -> list[int]:
    """Phase 3: return IDs of tasks that did not meet their done_criteria."""
    items = []
    for task, result in zip(tasks, results):
        items.append(
            f"Task {task.get('id', '?')}: {task['task']}\n"
            f"Done criteria: {task.get('done_criteria', 'N/A')}\n"
            f"Result: {result[:600]}"
        )
    messages = [
        {"role": "system", "content": _VALIDATOR_SYSTEM},
        {"role": "user", "content": "\n\n---\n\n".join(items)},
    ]
    payload = {"model": config.LLM_MODEL, "messages": json.dumps(messages)}
    try:
        text = await _llm_sync(payload, headers)
        incomplete = _extract_json(text, list)
        if isinstance(incomplete, list):
            ids = (_task_id_int(i) for i in incomplete)
            return [i for i in ids if i is not None]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if _is_llm_unavailable(exc):
            raise _LLMUnavailable(f"validator: {type(exc).__name__}: {exc}") from exc
        logger.warning(f"[Heartbeat] Validator failed ({type(exc).__name__}: {exc}) — assuming all complete")
    return []


async def _summarize(tasks: list[dict], results: list[str], headers: dict) -> str:
    """Phase 5: synthesize all task results into one concise TL;DR summary.

    No tools — a single planner/validator-style sync call. Returns "" on
    failure so the detailed findings still post without a summary.
    """
    items = []
    for task, result in zip(tasks, results):
        items.append(
            f"Task {task.get('id', '?')}: {task['task']}\n"
            f"Result: {result[:1200]}"
        )
    messages = [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": "\n\n---\n\n".join(items)},
    ]
    payload = {"model": config.LLM_MODEL, "messages": json.dumps(messages)}
    try:
        text = await _llm_sync(payload, headers)
        return text.strip()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(f"[Heartbeat] Summarizer failed ({exc}) — posting findings without summary")
        return ""


def _assemble_report(findings: str, summary: str) -> str:
    """Combine detailed findings and the tick summary into the posted report.

    The summary is always appended at the end under a `Summary` header. When
    there are no detailed findings (a fully healthy tick), the report is just
    the summary — so every tick still posts an "alive + summary" bubble.
    """
    summary = summary.strip()
    findings = findings.strip()
    summary_block = f"**Summary**\n\n{summary}" if summary else ""
    if findings and summary_block:
        return f"{findings}\n\n---\n\n{summary_block}"
    return findings or summary_block


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------

async def _run_once(send_fn: Callable[[int, str], Awaitable[None]]) -> None:
    """Execute one orchestrated heartbeat tick: plan → execute → validate → report."""
    global _llm_cooldown_until

    loop = asyncio.get_running_loop()
    if loop.time() < _llm_cooldown_until:
        remaining = int(_llm_cooldown_until - loop.time())
        logger.info(f"[Heartbeat] LLM cooldown active ({remaining}s remaining) — skipping tick")
        return

    # Keep memory.md under its cap before it is injected into any prompt.
    await _enforce_memory_cap("heartbeat")

    checklist = _load_checklist()
    if not checklist.strip():
        logger.info("[Heartbeat] HEARTBEAT.md is empty or missing — skipping tick")
        return

    if not config.LLM_API_KEY or not config.LLM_MODEL:
        logger.warning("[Heartbeat] LLM not configured — skipping tick")
        return

    tick_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}

    logger.info(f"[Heartbeat] Starting orchestrated tick (model={config.LLM_MODEL})")

    full_text = ""
    cancelled = False
    try:
        # Phase 1 — Plan
        tasks = await _plan_tasks(checklist, headers)

        # Phase 2 — Execute each task (context rebuilt per task)
        results: list[str] = []
        for task in tasks:
            results.append(await _execute_task(task, tick_id, headers))

        if tasks:
            # Phase 3 — Validate; retry incomplete tasks once
            incomplete_ids = set(await _validate_tasks(tasks, results, headers))
            if incomplete_ids:
                logger.info(f"[Heartbeat] Retrying {len(incomplete_ids)} incomplete task(s): {sorted(incomplete_ids)}")
                for i, task in enumerate(tasks):
                    if _task_id_int(task.get("id")) in incomplete_ids:
                        results[i] = await _execute_task(task, tick_id, headers, attempt_tag="_retry")

            # Phase 4 — Compile detailed findings (HEARTBEAT_OK results dropped)
            findings = _compile_report(results)

            # Phase 5 — Summarize the whole tick and append it at the end
            summary = await _summarize(tasks, results, headers)
            full_text = _assemble_report(findings, summary)

    except _LLMUnavailable as exc:
        _llm_cooldown_until = loop.time() + config.HEARTBEAT_LLM_COOLDOWN_SECONDS
        logger.warning(
            f"[Heartbeat] LLM unavailable ({exc}). Cooling down for "
            f"{config.HEARTBEAT_LLM_COOLDOWN_SECONDS}s."
        )
        return
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        _llm_cooldown_until = loop.time() + config.HEARTBEAT_LLM_COOLDOWN_SECONDS
        logger.warning(f"[Heartbeat] LLM unreachable ({type(exc).__name__}: {exc}). Cooling down.")
        return
    except asyncio.CancelledError:
        # Shutdown: don't start an LLM-backed memory compaction while being
        # cancelled (awaiting in a cancelled task's finally delays/eats it).
        cancelled = True
        raise
    except Exception as exc:
        logger.error(f"[Heartbeat] Unexpected error ({type(exc).__name__}): {exc}", exc_info=True)
        return
    finally:
        # Tasks may have appended to memory.md; trim before the next turn uses it.
        if not cancelled:
            await _enforce_memory_cap("heartbeat")

    # Every tick posts an "alive + summary" bubble. If both findings and the
    # summary came back empty (e.g. all tasks HEARTBEAT_OK and the summarizer
    # failed), fall back to a minimal heartbeat line rather than staying silent.
    reply = _strip_think(full_text) or "**Summary**\n\nHEARTBEAT_OK — all systems nominal."
    if config.MESSENGER_HOME_ROOM_ID < 0:
        logger.warning("[Heartbeat] Home room not resolved — skipping post")
    else:
        logger.info(f"[Heartbeat] Posting report ({len(reply)} chars) to room {config.MESSENGER_HOME_ROOM_ID}")
        await send_fn(config.MESSENGER_HOME_ROOM_ID, f"**Autonomous agent**\n\n{reply}")


# ---------------------------------------------------------------------------
# Loop entry point
# ---------------------------------------------------------------------------

async def run_heartbeat_loop(send_fn: Callable[[int, str], Awaitable[None]]) -> None:
    """
    Heartbeat background loop.

    Sleeps HEARTBEAT_INTERVAL_SECONDS between ticks (first tick after one full
    interval so startup is never blocked). Active-hours and LLM-cooldown checks
    are applied on every tick.
    """
    if not config.HEARTBEAT_ENABLED:
        logger.info("[Heartbeat] Disabled by config (HOONBOT_HEARTBEAT_ENABLED=false)")
        return

    logger.info(
        f"[Heartbeat] Loop started — interval={config.HEARTBEAT_INTERVAL_SECONDS}s, "
        f"active hours={config.HEARTBEAT_ACTIVE_START}–{config.HEARTBEAT_ACTIVE_END}"
    )

    snapshot_every = max(1, int(getattr(config, "HEARTBEAT_FS_SNAPSHOT_EVERY_TICKS", 1)))
    tick = 0

    while True:
        await asyncio.sleep(config.HEARTBEAT_INTERVAL_SECONDS)
        tick += 1

        # Refresh the filesystem-awareness snapshot (cheap, no LLM). Runs even
        # outside active hours so the map/digest stays current for the next tick.
        if tick % snapshot_every == 0:
            try:
                from core import fs_snapshot
                digest = await asyncio.to_thread(fs_snapshot.run_snapshot)
                if digest:
                    logger.info(f"[Heartbeat] {digest}")
            except Exception as exc:
                logger.warning(f"[Heartbeat] Filesystem snapshot failed: {exc}")

        if not _within_active_hours():
            logger.debug("[Heartbeat] Outside active hours — skipping tick")
            continue

        try:
            # No wall-clock deadline on a tick. A heartbeat may run long,
            # tool-heavy tasks and must never be cut off mid-task — a cancelled
            # tick would abandon the in-flight llm-api turn and waste the work.
            # Liveness is instead provided by the LLM call itself (SSE keepalive
            # pings + a finite connect timeout in the shared client).
            await _run_once(send_fn)
        except Exception as exc:
            logger.error(f"[Heartbeat] Unhandled loop error: {exc}", exc_info=True)
