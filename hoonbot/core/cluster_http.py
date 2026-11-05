"""Shared HTTP helpers for talking to the master's /api/cluster endpoints."""
from __future__ import annotations

from typing import Any

import httpx

import config
from core.retry import with_retry


def cluster_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if getattr(config, "CLUSTER_TOKEN", ""):
        headers["x-cluster-token"] = config.CLUSTER_TOKEN
    return headers


def master_url(path: str) -> str:
    return f"{config.CLUSTER_MASTER_API_URL}{path}"


def cluster_timeout() -> httpx.Timeout:
    """Timeouts for control-plane calls (register/lease/renew/complete/events).

    Finite read so a hung master can't wedge the worker loop forever; generous
    write for artifact uploads (up to 50 MB over the LAN)."""
    return httpx.Timeout(connect=10.0, read=120.0, write=300.0, pool=None)


def status_code_of(exc: BaseException) -> int | None:
    """HTTP status of an httpx.HTTPStatusError, else None."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def is_conflict(exc: BaseException) -> bool:
    """409 from the master — e.g. the task's lease is held by another node."""
    return status_code_of(exc) == 409


async def post_json(client: httpx.AsyncClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = await client.post(master_url(path), headers=cluster_headers(), json=payload)
    resp.raise_for_status()
    return resp.json()


async def get_json(client: httpx.AsyncClient, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    resp = await client.get(master_url(path), params=params, headers=cluster_headers())
    resp.raise_for_status()
    return resp.json()


async def post_json_retry(
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.0,
    label: str = "",
) -> dict[str, Any]:
    """post_json with exponential backoff on transport errors and 5xx.

    4xx responses (409 lease conflict, 404 unknown task, 401 bad token) are
    raised immediately — retrying them cannot succeed."""
    return await with_retry(
        post_json, client, path, payload,
        max_attempts=max_attempts, base_delay=base_delay,
        label=label or f"cluster POST {path}",
    )
