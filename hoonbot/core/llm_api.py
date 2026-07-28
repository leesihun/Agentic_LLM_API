"""Shared HTTP client + chat-call helpers for Hoonbot -> LLM API requests."""
import json
import logging
from typing import Optional

import httpx

import config

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None

# Connection went stale between requests (server restarted / keepalive dropped)
# — safe to retry the call once on a fresh connection.
STALE_ERRORS = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)


def get_client() -> httpx.AsyncClient:
    """Return the shared LLM API client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            # read=None → never time out mid-stream, so long agentic turns and
            # slow tools (OpenCode startup, multi-minute shell/code_exec, big
            # RAG ingests) are never cut off. The llm-api SSE stream emits ~15s
            # keepalive pings, which keep a healthy-but-slow stream flowing;
            # connect stays finite so a genuinely dead llm-api server still
            # fails fast instead of hanging the caller forever.
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=60.0),
            trust_env=False,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    return _client


async def close_client() -> None:
    """Close the shared LLM API client."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


async def chat(payload: dict, headers: dict, timeout: Optional[float] = None) -> str:
    """Non-streaming /v1/chat/completions call; returns the response text."""
    kwargs: dict = {"data": {**payload, "stream": "false"}, "headers": headers}
    if timeout is not None:
        kwargs["timeout"] = timeout
    resp = await get_client().post(
        f"{config.LLM_API_URL}/v1/chat/completions", **kwargs,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def stream_chat(payload: dict, headers: dict) -> str:
    """Stream a /v1/chat/completions call; return the accumulated text.

    Retries once on a stale keepalive connection (see STALE_ERRORS)."""

    async def _once() -> str:
        buf = ""
        async with get_client().stream(
            "POST",
            f"{config.LLM_API_URL}/v1/chat/completions",
            data=payload,
            headers=headers,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "tool_status" in event:
                    ts = event["tool_status"]
                    logger.debug(f"[LLM] Tool {ts.get('tool_name')}: {ts.get('status')}")
                    continue
                choices = event.get("choices", [])
                if choices:
                    chunk = choices[0].get("delta", {}).get("content", "")
                    if chunk:
                        buf += chunk
        return buf

    try:
        return await _once()
    except STALE_ERRORS as exc:
        logger.warning(f"[LLM] Stale connection ({type(exc).__name__}) — retrying once")
        return await _once()
