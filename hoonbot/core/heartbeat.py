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

_llm_cooldown_until: float = 0.0

_PLANNER_SYSTEM = config.read_prompt("heartbeat/planner_system.txt")
_VALIDATOR_SYSTEM = config.read_prompt("heartbeat/validator_system.txt")
_SUMMARY_SYSTEM = config.read_prompt("heartbeat/summary_system.txt")
_TASK_EXECUTION_TEMPLATE = config.read_prompt("heartbeat/task_execution.txt")

_OVERFLOW_KEYWORDS = (
    # vLLM: "This model's maximum context length is N tokens. However, you
    # requested M ... Please reduce the length of the messages."
    "context",
    "exceed",
    "too large",
    "too long",
    "maximum context length",
    "maximum model length",
    "please reduce the length",
    "longer than the maximum",
    "reduce the length of the messages",
)


def _is_context_overflow(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in _OVERFLOW_KEYWORDS)


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


def _extract_json(text: str, expected_type: type) -> Any:
    """Extract JSON from LLM text, handling optional ```json``` fences."""
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1)
    try:
        result = json.loads(text.strip())
        if isinstance(result, expected_type):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
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

_llm_stream = llm_api.stream_chat   # full agent loop with tools (task execution)
_llm_sync = llm_api.chat            # planner/validator calls, no tools


# ---------------------------------------------------------------------------
# Orchestrator phases
# ---------------------------------------------------------------------------

async def _plan_tasks(checklist: str, context: str, headers: dict) -> list[dict]:
    """Phase 1: ask the LLM to decompose the checklist into tasks with done_criteria."""
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": f"{context}\n\n## Checklist\n\n{checklist}"},
    ]
    payload = {"model": config.LLM_MODEL, "messages": json.dumps(messages)}
    try:
        text = await _llm_sync(payload, headers)
        tasks = _extract_json(text, list)
        if tasks and all(isinstance(t, dict) and "task" in t for t in tasks):
            # Ensure every task has a numeric id
            for i, t in enumerate(tasks, 1):
                t.setdefault("id", i)
            logger.info(f"[Heartbeat] Planner produced {len(tasks)} task(s)")
            return tasks
        logger.warning("[Heartbeat] Planner returned invalid structure — falling back to single task")
    except Exception as exc:
        logger.warning(f"[Heartbeat] Planner failed ({exc}) — falling back to single task")

    return [{"id": 1, "task": checklist.strip(), "done_criteria": "All checklist items addressed"}]


async def _execute_task(task: dict, context: str, tick_id: str, headers: dict) -> str:
    """Phase 2: run one task through the full agent loop (with tools)."""
    task_id = task.get("id", 1)
    session_id = f"hb_{tick_id}_t{task_id}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt = _TASK_EXECUTION_TEMPLATE.format(
        task_id=task_id,
        timestamp=timestamp,
        task=task["task"],
        done_criteria=task.get("done_criteria", "Task completed successfully"),
    )
    messages = [{"role": "user", "content": f"{context}\n\n---\n\n{prompt}"}]
    payload = {
        "model": config.LLM_MODEL,
        "messages": json.dumps(messages),
        "stream": "true",
        "session_id": session_id,
    }
    try:
        result = await _llm_stream(payload, headers)
        logger.info(f"[Heartbeat] Task {task_id} finished ({len(result)} chars)")
        return result.strip() or "(no output)"
    except Exception as exc:
        if _is_context_overflow(exc):
            logger.warning(
                f"[Heartbeat] Task {task_id} context overflow — "
                f"increase interval or shrink memory.md ({exc})"
            )
        else:
            logger.warning(f"[Heartbeat] Task {task_id} execution error: {exc}")
        return f"(error: {exc})"


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
            return [int(i) for i in incomplete if str(i).isdigit() or isinstance(i, int)]
    except Exception as exc:
        logger.warning(f"[Heartbeat] Validator failed ({exc}) — assuming all complete")
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

    loop_time = asyncio.get_event_loop().time()
    if loop_time < _llm_cooldown_until:
        remaining = int(_llm_cooldown_until - loop_time)
        logger.info(f"[Heartbeat] LLM cooldown active ({remaining}s remaining) — skipping tick")
        return

    checklist = _extract_checklist(_read_file(_HEARTBEAT_FILE))
    if not checklist.strip():
        logger.info("[Heartbeat] HEARTBEAT.md is empty or missing — skipping tick")
        return

    if not config.LLM_API_KEY or not config.LLM_MODEL:
        logger.warning("[Heartbeat] LLM not configured — skipping tick")
        return

    ambient = build_per_turn_context(profile="heartbeat")
    context = f"{build_llm_context()}\n\n{ambient}"
    tick_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}

    logger.info(f"[Heartbeat] Starting orchestrated tick (model={config.LLM_MODEL})")

    full_text = ""
    try:
        # Phase 1 — Plan
        tasks = await _plan_tasks(checklist, context, headers)

        # Phase 2 — Execute each task
        results: list[str] = []
        for task in tasks:
            result = await _execute_task(task, context, tick_id, headers)
            results.append(result)

        # Phase 3 — Validate; retry incomplete tasks once
        incomplete_ids = await _validate_tasks(tasks, results, headers)
        if incomplete_ids:
            logger.info(f"[Heartbeat] Retrying {len(incomplete_ids)} incomplete task(s): {incomplete_ids}")
            for i, task in enumerate(tasks):
                if task.get("id") in incomplete_ids:
                    results[i] = await _execute_task(task, context, tick_id, headers)

        # Phase 4 — Compile detailed findings (HEARTBEAT_OK results dropped)
        findings = _compile_report(results)

        # Phase 5 — Summarize the whole tick and append it at the end
        summary = await _summarize(tasks, results, headers)
        full_text = _assemble_report(findings, summary)

    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        _llm_cooldown_until = asyncio.get_event_loop().time() + config.HEARTBEAT_LLM_COOLDOWN_SECONDS
        logger.warning(f"[Heartbeat] LLM unreachable ({type(exc).__name__}: {exc}). Cooling down.")
        return
    except httpx.TimeoutException as exc:
        if not full_text:
            _llm_cooldown_until = asyncio.get_event_loop().time() + config.HEARTBEAT_LLM_COOLDOWN_SECONDS
            logger.warning(f"[Heartbeat] LLM timeout with no response. Cooling down.")
            return
        logger.warning(f"[Heartbeat] Stream timeout after partial response — using what we have")
    except Exception as exc:
        logger.error(f"[Heartbeat] Unexpected error ({type(exc).__name__}): {exc}", exc_info=True)
        return

    # Every tick posts an "alive + summary" bubble. If both findings and the
    # summary came back empty (e.g. all tasks HEARTBEAT_OK and the summarizer
    # failed), fall back to a minimal heartbeat line rather than staying silent.
    reply = full_text.strip() or "**Summary**\n\nHEARTBEAT_OK — all systems nominal."
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
