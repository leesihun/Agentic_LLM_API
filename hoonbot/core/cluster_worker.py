"""Slave worker loop for master-owned cluster tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

import config
from core import llm_api
from core.cluster_http import master_url, post_json
from core.context import build_llm_context

logger = logging.getLogger(__name__)

_TASK_EXECUTION_TEMPLATE = config.read_prompt("cluster/task_execution.txt")

# Keep in sync with MAX_ARTIFACTS_PER_TASK / MAX_ARTIFACT_BYTES in
# llm-api/backend/core/cluster_store.py — the master rejects excess uploads.
_ARTIFACT_LIMIT = 10
_ARTIFACT_MAX_BYTES = 50 * 1024 * 1024


def _node_payload() -> dict[str, Any]:
    disk = shutil.disk_usage(".")
    return {
        "node_name": config.NODE_NAME,
        "role": config.CLUSTER_ROLE,
        "ip": config.NODE_IP,
        "api_url": getattr(config, "LLM_API_URL", ""),
        "capabilities": config.NODE_CAPABILITIES,
        "tags": config.NODE_TAGS,
        "prompt_profile": getattr(config, "PROMPT_PROFILE", "slave"),
        "heartbeat_profile": getattr(config, "HEARTBEAT_PROFILE", "slave"),
        "skills_profile": getattr(config, "SKILLS_PROFILE", "slave"),
        "model": config.LLM_MODEL,
        "disk_free_gb": round(disk.free / 1e9, 1),
    }


async def _register(client: httpx.AsyncClient) -> None:
    await post_json(client, "/api/cluster/register", _node_payload())
    logger.info("[Cluster] Registered node '%s' with master %s", config.NODE_NAME, config.CLUSTER_MASTER_API_URL)


async def _heartbeat(client: httpx.AsyncClient) -> None:
    payload = _node_payload()
    payload["heartbeat_at_monotonic"] = time.monotonic()
    await post_json(client, "/api/cluster/heartbeat", payload)


async def _lease(client: httpx.AsyncClient) -> dict[str, Any] | None:
    result = await post_json(client, "/api/cluster/tasks/lease", {
        "node_name": config.NODE_NAME,
        "capabilities": config.NODE_CAPABILITIES,
        "tags": config.NODE_TAGS,
    })
    return result.get("task")


async def _task_event(client: httpx.AsyncClient, task_id: str, message: str, event_type: str = "event") -> None:
    await post_json(client, f"/api/cluster/tasks/{task_id}/events", {
        "node_name": config.NODE_NAME,
        "type": event_type,
        "message": message,
    })


def _lease_renew_interval(task: dict[str, Any]) -> float:
    """Renew at half the lease duration (floor 30s). Deriving the interval from
    the leased task keeps the slave correct even if the master changes
    CLUSTER_TASK_LEASE_SECONDS; falls back to 300s if the timestamps are absent."""
    try:
        leased = datetime.fromisoformat(task["leased_at"])
        expires = datetime.fromisoformat(task["lease_expires_at"])
        duration = (expires - leased).total_seconds()
        if duration > 0:
            return max(30.0, duration / 2)
    except (KeyError, TypeError, ValueError):
        pass
    return 300.0


async def _renew_lease_loop(client: httpx.AsyncClient, task: dict[str, Any]) -> None:
    """Keep a running task's lease fresh so the master never re-queues it (which
    would duplicate execution) while it runs — however long that takes."""
    task_id = task["task_id"]
    interval = _lease_renew_interval(task)
    while True:
        await asyncio.sleep(interval)
        try:
            await post_json(client, f"/api/cluster/tasks/{task_id}/renew", {
                "node_name": config.NODE_NAME,
            })
        except Exception as exc:
            logger.debug("[Cluster] Lease renew failed for %s: %s", task_id, exc)


async def _complete(client: httpx.AsyncClient, task_id: str, result: str | None = None, error: str | None = None) -> None:
    await post_json(client, f"/api/cluster/tasks/{task_id}/complete", {
        "node_name": config.NODE_NAME,
        "status": "failed" if error else "completed",
        "result": result,
        "error": error,
    })


def _outbox_dir(task_id: str) -> Path:
    return Path(config.DATA_DIR) / "cluster_outbox" / task_id


async def _upload_artifacts(client: httpx.AsyncClient, task_id: str) -> int:
    """Upload every file the agent left in the task outbox to the master."""
    outbox = _outbox_dir(task_id)
    files = sorted(p for p in outbox.rglob("*") if p.is_file()) if outbox.is_dir() else []
    if len(files) > _ARTIFACT_LIMIT:
        await _task_event(client, task_id, f"Outbox has {len(files)} files; uploading first {_ARTIFACT_LIMIT}")
        files = files[:_ARTIFACT_LIMIT]

    headers = {"x-cluster-token": config.CLUSTER_TOKEN} if config.CLUSTER_TOKEN else {}
    uploaded = 0
    for path in files:
        try:
            if path.stat().st_size > _ARTIFACT_MAX_BYTES:
                await _task_event(client, task_id, f"Skipped oversize artifact {path.name}")
                continue
            with open(path, "rb") as f:
                resp = await client.post(
                    master_url(f"/api/cluster/tasks/{task_id}/artifacts"),
                    headers=headers,
                    data={"node_name": config.NODE_NAME},
                    files={"file": (path.name, f)},
                )
            resp.raise_for_status()
            uploaded += 1
        except Exception as exc:
            logger.warning("[Cluster] Artifact upload failed for %s: %s", path, exc)
            try:
                await _task_event(client, task_id, f"Artifact upload failed: {path.name} ({exc})")
            except Exception:
                pass
    return uploaded


async def _execute_task(client: httpx.AsyncClient, task: dict[str, Any]) -> None:
    task_id = task["task_id"]
    prompt = task.get("prompt") or ""
    await _task_event(client, task_id, "Starting local LLM execution", "started")

    if not config.LLM_API_KEY:
        await _complete(client, task_id, error="LLM_API_KEY is not configured on slave")
        return
    if not config.LLM_MODEL:
        await _complete(client, task_id, error="LLM_MODEL is not configured on slave")
        return

    outbox = _outbox_dir(task_id)
    outbox.mkdir(parents=True, exist_ok=True)

    context = build_llm_context()
    task_prompt = _TASK_EXECUTION_TEMPLATE.format(
        task_id=task_id,
        node_name=config.NODE_NAME,
        prompt=prompt,
        outbox_dir=str(outbox.resolve()),
    )
    messages = [{
        "role": "user",
        "content": f"{context}\n\n---\n\n{task_prompt}",
    }]
    payload = {
        "model": config.LLM_MODEL,
        "messages": json.dumps(messages),
        "session_id": f"cluster_{task_id}",
    }
    renew = asyncio.create_task(_renew_lease_loop(client, task))
    try:
        result = await llm_api.chat(
            payload,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        # Ship outbox files before completing so the relay sees them on delivery.
        uploaded = await _upload_artifacts(client, task_id)
        await _complete(client, task_id, result=result)
        logger.info("[Cluster] Completed task %s (%s chars, %s artifact(s))", task_id, len(result), uploaded)
    except Exception as exc:
        logger.error("[Cluster] Task %s failed: %s", task_id, exc, exc_info=True)
        await _complete(client, task_id, error=f"{type(exc).__name__}: {exc}")
    finally:
        renew.cancel()
        try:
            await renew
        except asyncio.CancelledError:
            pass
        shutil.rmtree(outbox, ignore_errors=True)


async def run_slave_worker_loop() -> None:
    if not config.CLUSTER_ENABLED:
        logger.info("[Cluster] Disabled by config")
        return
    if config.CLUSTER_ROLE != "slave":
        logger.info("[Cluster] Worker loop skipped on role=%s", config.CLUSTER_ROLE)
        return

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None), trust_env=False) as client:
        while True:
            try:
                await _register(client)
                break
            except Exception as exc:
                logger.warning("[Cluster] Register failed: %s", exc)
                await asyncio.sleep(config.CLUSTER_SLAVE_POLL_INTERVAL_SECONDS)

        running: set[asyncio.Task[None]] = set()

        def _on_task_done(done: asyncio.Task[None]) -> None:
            running.discard(done)
            if not done.cancelled() and done.exception() is not None:
                logger.error("[Cluster] Task runner crashed: %s", done.exception(), exc_info=done.exception())

        try:
            while True:
                try:
                    await _heartbeat(client)
                    # Fill free execution slots, then sleep; running tasks
                    # proceed in the background so the loop keeps polling.
                    while len(running) < config.CLUSTER_SLAVE_MAX_CONCURRENT_TASKS:
                        task = await _lease(client)
                        if not task:
                            break
                        runner = asyncio.create_task(
                            _execute_task(client, task),
                            name=f"cluster-task-{task['task_id']}",
                        )
                        running.add(runner)
                        runner.add_done_callback(_on_task_done)
                    await asyncio.sleep(config.CLUSTER_SLAVE_POLL_INTERVAL_SECONDS)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[Cluster] Worker loop error: %s", exc)
                    await asyncio.sleep(config.CLUSTER_SLAVE_POLL_INTERVAL_SECONDS)
        finally:
            for runner in running:
                runner.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
