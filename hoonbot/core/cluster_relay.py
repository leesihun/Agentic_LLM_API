"""Master-side watcher that relays finished cluster task results to Messenger.

When a Messenger user issues an `@<node>` directive, `cluster_client` queues a
task on the master and posts "Queued cluster task X". The slave executes it and
writes the result back into the master's cluster store — but nothing delivered
that result to the user. This loop closes the gap: it watches for tasks that
reach a terminal state and posts their result (or error) back to the originating
room, keyed by `metadata.room_id` saved at submission time.

Master-only. No-op unless cluster + delegation are enabled.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

import config
from core.cluster_http import get_json

logger = logging.getLogger(__name__)

_TERMINAL = {"completed", "failed", "cancelled"}
_RELAYED_FILE = Path(config.DATA_DIR) / "relayed_tasks.json"
_RELAYED_CAP = 2000


class _RelayState:
    """Ordered set of relayed task ids plus per-task chunk progress.

    Insertion order is preserved (dict keys), so trimming to the cap drops the
    OLDEST ids — a plain ``set`` trimmed by ``list(set)[-N:]`` dropped arbitrary
    ids and could re-relay old results. ``progress`` maps task_id -> index of
    the next chunk to send, so a crash mid-delivery resumes instead of
    re-posting chunks the room already has.

    On-disk format: ``{"relayed": [...], "progress": {task_id: next_chunk}}``;
    the legacy plain-list format is still accepted on load.
    """

    def __init__(self) -> None:
        self.relayed: dict[str, None] = {}
        self.progress: dict[str, int] = {}

    def __contains__(self, task_id: object) -> bool:
        return task_id in self.relayed

    def __len__(self) -> int:
        return len(self.relayed)

    def add(self, task_id: str) -> None:
        self.relayed.pop(task_id, None)
        self.relayed[task_id] = None
        self.progress.pop(task_id, None)
        while len(self.relayed) > _RELAYED_CAP:
            oldest = next(iter(self.relayed))
            del self.relayed[oldest]

    def to_json(self) -> dict[str, Any]:
        return {"relayed": list(self.relayed), "progress": dict(self.progress)}


def _load_relayed() -> _RelayState:
    state = _RelayState()
    try:
        data = json.loads(_RELAYED_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return state
    except Exception as exc:  # corrupt file — start clean rather than crash
        logger.warning("[Relay] Could not read %s (%s); starting empty", _RELAYED_FILE, exc)
        return state
    if isinstance(data, list):  # legacy format
        ids, progress = data, {}
    elif isinstance(data, dict):
        ids = data.get("relayed") or []
        progress = data.get("progress") or {}
    else:
        ids, progress = [], {}
    for task_id in ids:
        state.add(str(task_id))
    if isinstance(progress, dict):
        for task_id, idx in progress.items():
            try:
                if str(task_id) not in state:
                    state.progress[str(task_id)] = max(0, int(idx))
            except (TypeError, ValueError):
                continue
    return state


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + os.replace so a crash never leaves a torn file.

    On Windows os.replace can fail transiently with PermissionError when an
    AV scanner / indexer holds the target open; retry briefly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                break
            time.sleep(0.05 * (attempt + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise PermissionError(f"could not replace {path}")


def _save_relayed(state: _RelayState) -> None:
    try:
        _atomic_write_text(_RELAYED_FILE, json.dumps(state.to_json(), ensure_ascii=False))
    except Exception as exc:
        logger.warning("[Relay] Could not persist relayed state: %s", exc)


def _chunk_message(message: str) -> list[str]:
    """Split *message* into Messenger-sized parts labelled ``(i/n)``."""
    from core import messenger  # lazy: keeps this module importable standalone

    limit = int(getattr(config, "MAX_MESSAGE_LENGTH", 2000))
    # Leave room for the "(i/n) " prefix so labelled parts stay under the limit
    # (send_message would otherwise re-split them without labels).
    parts = messenger._split_message(message, max(200, limit - 20))
    parts = [p for p in parts if p.strip()] or [message]
    if len(parts) == 1:
        return parts
    n = len(parts)
    return [f"({i}/{n}) {part}" for i, part in enumerate(parts, 1)]


async def _list_terminal_tasks(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    data = await get_json(client, "/api/cluster/tasks", params={"include_completed": "true"})
    tasks = data.get("tasks") or []
    return [t for t in tasks if t.get("status") in _TERMINAL]


async def _load_full_task(client: httpx.AsyncClient, task_id: str) -> dict[str, Any] | None:
    """Fetch the un-truncated task (the list endpoint strips result to 300 chars)."""
    try:
        data = await get_json(client, f"/api/cluster/tasks/{task_id}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    return data.get("task")


def _format_message(task: dict[str, Any]) -> str | None:
    """Build the Messenger message for a finished task, or None to skip it."""
    meta = task.get("metadata") or {}
    directive = meta.get("directive") or task.get("target_node") or "cluster"
    node = task.get("leased_by") or task.get("target_node") or "?"
    status = task.get("status")
    if status == "completed":
        result = (task.get("result") or "").strip()
        if not result:
            if task.get("artifacts"):
                return f"**@{directive} → {node}**\n\n(no text output — see the attached artifacts)"
            # A "completed" task with nothing to show is not a success the
            # user can act on — say so instead of a bare "(no output)".
            return (
                f"**@{directive} → {node} failed**\n\n"
                "The task was reported completed but returned no output."
            )
        return f"**@{directive} → {node}**\n\n{result}"
    if status == "failed":
        err = (task.get("error") or "Task failed with no error detail").strip()
        return f"**@{directive} → {node} failed**\n\n{err}"
    if status == "cancelled":
        return f"**@{directive} → {node} cancelled**"
    return None


async def _send_artifacts(task: dict[str, Any], room_id: int, send_file_fn) -> None:
    """Post the task's stored artifacts to the room. Failures are logged, not retried."""
    for artifact in task.get("artifacts") or []:
        path = str(artifact.get("path") or "")
        if not path or not Path(path).is_file():
            logger.warning("[Relay] Artifact missing on disk for task %s: %s", task.get("task_id"), path)
            continue
        try:
            await send_file_fn(room_id, path)
        except Exception as exc:
            logger.warning("[Relay] Failed to send artifact %s for task %s: %s", path, task.get("task_id"), exc)


async def run_cluster_relay_loop(send_fn, send_file_fn=None) -> None:
    """Poll for terminal Messenger-sourced tasks and relay their results.

    *send_fn* is an async ``(room_id: int, content: str) -> None`` — typically
    ``messenger.send_message``. *send_file_fn* is an optional async
    ``(room_id: int, file_path: str) -> Any`` (``messenger.send_file``) used to
    deliver artifacts slaves uploaded to the master's cluster store.
    """
    if not getattr(config, "CLUSTER_ENABLED", False):
        logger.info("[Relay] Cluster disabled — relay loop not started")
        return
    if getattr(config, "CLUSTER_ROLE", "master") != "master":
        return
    if not getattr(config, "CLUSTER_ENABLE_DELEGATION", False):
        logger.info("[Relay] Delegation disabled — relay loop not started")
        return

    interval = float(getattr(config, "CLUSTER_RELAY_POLL_INTERVAL_SECONDS", 5))
    relayed = _load_relayed()
    seeded = _RELAYED_FILE.exists()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=None),
        trust_env=False,
    ) as client:
        # First run: seed with whatever is already terminal so we don't spam the
        # room with a backlog of old results on initial startup.
        if not seeded:
            try:
                for task in await _list_terminal_tasks(client):
                    relayed.add(task["task_id"])
                _save_relayed(relayed)
                logger.info("[Relay] Seeded %d existing terminal task(s) as already-relayed", len(relayed))
            except Exception as exc:
                logger.warning("[Relay] Seed failed (%s) — will relay on next pass", exc)

        logger.info("[Relay] Watching for finished cluster tasks (interval=%ss)", interval)
        while True:
            await asyncio.sleep(interval)
            try:
                terminal = await _list_terminal_tasks(client)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[Relay] Poll failed: %s", exc)
                continue

            new_relays = False
            for summary in terminal:
                task_id = summary.get("task_id")
                if not task_id or task_id in relayed:
                    continue
                meta = summary.get("metadata") or {}
                room_id = meta.get("room_id")
                # Only relay Messenger-originated tasks that carry a room.
                if summary.get("source") != "messenger" or room_id is None:
                    relayed.add(task_id)  # not ours to deliver — mark and move on
                    new_relays = True
                    continue
                try:
                    full = await _load_full_task(client, task_id)
                    if not full:
                        relayed.add(task_id)
                        new_relays = True
                        continue
                    message = _format_message(full)
                    if message:
                        parts = _chunk_message(message)
                        start_at = min(relayed.progress.get(task_id, 0), len(parts))
                        for idx in range(start_at, len(parts)):
                            await send_fn(int(room_id), parts[idx])
                            relayed.progress[task_id] = idx + 1
                            if len(parts) > 1:
                                # Persist per chunk so a crash resumes at the
                                # next part instead of re-posting the whole set.
                                _save_relayed(relayed)
                        logger.info(
                            "[Relay] Delivered task %s to room %s (%d part(s))",
                            task_id, room_id, len(parts),
                        )
                    if send_file_fn:
                        await _send_artifacts(full, int(room_id), send_file_fn)
                    relayed.add(task_id)
                    new_relays = True
                except asyncio.CancelledError:
                    raise
                except httpx.HTTPStatusError as exc:
                    if (exc.request.url.path.endswith("/api/send-message")
                            and exc.response.status_code in (400, 403)):
                        # Permanent Messenger rejection (bot no longer a member
                        # of the room / invalid content): retrying every pass
                        # would spin forever. Give up on this task.
                        logger.warning("[Relay] Messenger rejected task %s for room %s (HTTP %s) — giving up",
                                       task_id, room_id, exc.response.status_code)
                        relayed.add(task_id)
                        new_relays = True
                    else:
                        logger.warning("[Relay] Failed to relay task %s: %s", task_id, exc)
                        if relayed.progress.get(task_id):
                            _save_relayed(relayed)
                except Exception as exc:
                    # Leave it unmarked so we retry next pass (resuming at the
                    # saved chunk index).
                    logger.warning("[Relay] Failed to relay task %s: %s", task_id, exc)
                    if relayed.progress.get(task_id):
                        _save_relayed(relayed)

            if new_relays:
                _save_relayed(relayed)
