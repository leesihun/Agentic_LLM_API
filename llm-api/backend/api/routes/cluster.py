"""Master cluster registry and task lease APIs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

import config
from backend.core.cluster_store import cluster_store


router = APIRouter(prefix="/api/cluster", tags=["cluster"])


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


@router.post("/register")
async def register_node(request: Request):
    _auth(request)
    payload = await request.json()
    try:
        node = cluster_store.register_node(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "node": node}


@router.post("/heartbeat")
async def heartbeat_node(request: Request):
    _auth(request)
    payload = await request.json()
    node_name = str(payload.get("node_name") or "").strip()
    try:
        node = cluster_store.heartbeat_node(node_name, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "node": node}


@router.get("/nodes")
async def list_nodes(request: Request):
    _auth(request)
    return {"nodes": cluster_store.list_nodes()}


@router.get("/status")
async def cluster_status(request: Request):
    _auth(request)
    return cluster_store.status()


@router.post("/tasks")
async def create_task(request: Request):
    _auth(request)
    payload: dict[str, Any] = await request.json()
    try:
        task = cluster_store.create_task(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "task": task}


@router.get("/tasks")
async def list_tasks(request: Request, include_completed: bool = True):
    _auth(request)
    return {"tasks": cluster_store.list_tasks(include_completed=include_completed)}


@router.post("/tasks/lease")
async def lease_task(request: Request):
    _auth(request)
    payload = await request.json()
    node_name = str(payload.get("node_name") or "").strip()
    capabilities = list(payload.get("capabilities") or [])
    tags = list(payload.get("tags") or [])
    try:
        task = cluster_store.lease_task(node_name, capabilities, tags)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "task": task}


@router.post("/tasks/{task_id}/renew")
async def renew_task_lease(task_id: str, request: Request):
    _auth(request)
    payload = await request.json()
    node_name = str(payload.get("node_name") or "").strip()
    try:
        task = cluster_store.renew_lease(node_name, task_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True, "task": task}


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    _auth(request)
    task = cluster_store.load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task": task, "events": cluster_store.load_events(task_id)}


@router.post("/tasks/{task_id}/events")
async def append_task_event(task_id: str, request: Request):
    _auth(request)
    if not cluster_store.load_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")
    payload = await request.json()
    event = cluster_store.append_event(task_id, payload)
    return {"ok": True, "event": event}


@router.post("/tasks/{task_id}/artifacts")
async def upload_task_artifact(
    task_id: str,
    request: Request,
    file: UploadFile = File(...),
    node_name: str = Form(""),
):
    _auth(request)
    content = await file.read()
    try:
        artifact = cluster_store.save_artifact(task_id, node_name, file.filename or "", content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "artifact": artifact}


@router.post("/tasks/{task_id}/complete")
async def complete_task(task_id: str, request: Request):
    _auth(request)
    payload = await request.json()
    try:
        task = cluster_store.complete_task(task_id, payload)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Task not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "task": task}
