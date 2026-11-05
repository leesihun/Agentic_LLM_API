"""Master cluster registry and task lease APIs.

The store is synchronous and file-lock based, so every call runs in a worker
thread (asyncio.to_thread) — the master event loop also serves every chat
stream and must never block on cluster file I/O.

Error mapping (shared by every route):
    ValueError          → 400
    FileNotFoundError   → 404 "Task not found"
    LeaseConflictError  → 409 (stale worker: task re-leased / finished elsewhere)
    store lock timeout  → 503 + Retry-After (transient; slaves retry)
    other OSError       → 503 + Retry-After (transient file I/O)
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

import config
from backend.core.cluster_store import (
    MAX_ARTIFACT_BYTES,
    LeaseConflictError,
    StoreBusyError,
    cluster_store,
)


router = APIRouter(prefix="/api/cluster", tags=["cluster"])

_UPLOAD_CHUNK = 1024 * 1024


def _check_cluster_enabled() -> None:
    if not getattr(config, "CLUSTER_ENABLED", True):
        raise HTTPException(status_code=404, detail="Cluster APIs are disabled")
    if getattr(config, "CLUSTER_ROLE", "master") != "master":
        raise HTTPException(status_code=404, detail="Cluster APIs are master-only on this node")


def _check_token(request: Request) -> None:
    token = getattr(config, "CLUSTER_TOKEN", "")
    if not token:
        return
    supplied = request.headers.get("x-cluster-token", "")
    if supplied != token:
        raise HTTPException(status_code=401, detail="Invalid cluster token")


def _auth(request: Request) -> None:
    _check_cluster_enabled()
    _check_token(request)


async def _json(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return payload


async def _call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a cluster_store call off the event loop and map store errors."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except LeaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc) or "lease conflict")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except StoreBusyError:
        raise HTTPException(
            status_code=503,
            detail="Cluster store busy; retry",
            headers={"Retry-After": "1"},
        )
    except OSError as exc:
        # Transient file I/O failure (EMFILE/EIO, Windows sharing violation
        # outlasting the replace retries) — retryable, never a 404.
        raise HTTPException(
            status_code=503,
            detail=f"Cluster store I/O error; retry ({type(exc).__name__})",
            headers={"Retry-After": "1"},
        )


@router.post("/register")
async def register_node(request: Request):
    _auth(request)
    payload = await _json(request)
    node = await _call(cluster_store.register_node, payload)
    return {"ok": True, "node": node}


@router.post("/heartbeat")
async def heartbeat_node(request: Request):
    _auth(request)
    payload = await _json(request)
    node_name = str(payload.get("node_name") or "").strip()
    node = await _call(cluster_store.heartbeat_node, node_name, payload)
    response: dict[str, Any] = {"ok": True, "node": node}
    # Optional: a slave that lists the tasks it is running gets their leases
    # extended, and learns which ones it no longer holds (re-leased elsewhere
    # or cancelled) so it can abort the duplicate run.
    active = payload.get("active_task_ids")
    if isinstance(active, list) and active:
        response["leases"] = await _call(cluster_store.reconcile_node_leases, node_name, active)
    return response


@router.get("/nodes")
async def list_nodes(request: Request):
    _auth(request)
    return {"nodes": await _call(cluster_store.list_nodes)}


@router.get("/status")
async def cluster_status(request: Request):
    _auth(request)
    return await _call(cluster_store.status)


@router.post("/tasks")
async def create_task(request: Request):
    _auth(request)
    payload = await _json(request)
    task = await _call(cluster_store.create_task, payload)
    return {"ok": True, "task": task}


@router.get("/tasks")
async def list_tasks(request: Request, include_completed: bool = True):
    _auth(request)
    return {"tasks": await _call(cluster_store.list_tasks, include_completed=include_completed)}


@router.post("/tasks/lease")
async def lease_task(request: Request):
    _auth(request)
    payload = await _json(request)
    node_name = str(payload.get("node_name") or "").strip()
    capabilities = list(payload.get("capabilities") or [])
    tags = list(payload.get("tags") or [])
    task = await _call(cluster_store.lease_task, node_name, capabilities, tags)
    return {"ok": True, "task": task}


@router.post("/tasks/{task_id}/renew")
async def renew_task_lease(task_id: str, request: Request):
    """Extend the caller's lease. 409 when the caller no longer holds the task
    (re-leased to another node / finished elsewhere) — the caller must stop."""
    _auth(request)
    payload = await _json(request)
    node_name = str(payload.get("node_name") or "").strip()
    task = await _call(cluster_store.renew_lease, node_name, task_id, payload.get("lease_epoch"))
    return {"ok": True, "task": task}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    _auth(request)
    task = await _call(cluster_store.load_task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    events = await _call(cluster_store.load_events, task_id)
    return {"task": task, "events": events}


@router.post("/tasks/{task_id}/events")
async def append_task_event(task_id: str, request: Request):
    _auth(request)
    if not await _call(cluster_store.load_task, task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    payload = await _json(request)
    event = await _call(cluster_store.append_event, task_id, payload)
    return {"ok": True, "event": event}


@router.post("/tasks/{task_id}/artifacts")
async def upload_task_artifact(
    task_id: str,
    request: Request,
    file: UploadFile = File(...),
    node_name: str = Form(""),
):
    _auth(request)
    # Read at most MAX+1 bytes so an oversize upload is rejected without
    # buffering the whole thing in memory.
    chunks: list[bytes] = []
    total = 0
    while total <= MAX_ARTIFACT_BYTES:
        chunk = await file.read(min(_UPLOAD_CHUNK, MAX_ARTIFACT_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > MAX_ARTIFACT_BYTES:
        raise HTTPException(status_code=400, detail=f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    content = b"".join(chunks)
    artifact = await _call(cluster_store.save_artifact, task_id, node_name, file.filename or "", content)
    return {"ok": True, "artifact": artifact}


@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, request: Request):
    """Record a terminal result. Idempotent for a retried identical report from
    the holder; 409 when the caller no longer holds the task or the task is
    already terminal with a different outcome (never downgrades a result)."""
    _auth(request)
    payload = await _json(request)
    task = await _call(cluster_store.complete_task, task_id, payload)
    return {"ok": True, "task": task}
