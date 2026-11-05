"""Retry helper with exponential backoff for async functions."""
import asyncio
import logging
import random
from typing import Tuple, Type

import httpx

logger = logging.getLogger(__name__)

RETRYABLE = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    # Mid-request resets / stale keep-alive sockets (server restarted, idle
    # connection dropped by the peer) — transient, worth another attempt.
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

# Failures that prove the request never reached the server (no connection was
# established / no pooled connection was obtained). The only transport errors a
# non-idempotent call (send a message, toggle a reaction, upload a file) may
# retry: after a ReadError / ReadTimeout / RemoteProtocolError the server may
# already have processed the request, and a resend would duplicate the message
# (or, for the reaction toggle, flip it back off).
CONNECT_PHASE_RETRYABLE = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

# Gateway/unavailable statuses: the upstream did not handle the request, so a
# non-idempotent call may retry them. (A bare 500 may follow a partial write.)
_NON_IDEMPOTENT_RETRY_STATUSES = frozenset({502, 503, 504})


def _is_retryable(exc: BaseException, retryable: tuple, idempotent: bool = True) -> bool:
    """Check if an exception is worth retrying (includes 5xx HTTP errors;
    only 502/503/504 for non-idempotent calls)."""
    if isinstance(exc, retryable):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if idempotent:
            return status >= 500
        return status in _NON_IDEMPOTENT_RETRY_STATUSES
    return False


async def with_retry(
    coro_fn,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    retryable: Tuple[Type[BaseException], ...] = RETRYABLE,
    label: str = "",
    idempotent: bool = True,
    **kwargs,
):
    """
    Call an async function with exponential backoff on transient failures.
    Also retries HTTP 5xx errors.

    ``idempotent=False`` is for calls that must not run twice (message sends,
    reaction toggles, uploads): only connect-phase errors
    (``CONNECT_PHASE_RETRYABLE``, intersected with *retryable*) and HTTP
    502/503/504 are retried — never an error that can occur after the request
    reached the server.

    Usage:
        result = await with_retry(some_async_fn, arg1, arg2, label="LLM chat")
    """
    if not idempotent:
        retryable = tuple(
            exc_type for exc_type in CONNECT_PHASE_RETRYABLE
            if any(issubclass(exc_type, allowed) for allowed in retryable)
        )
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc, retryable, idempotent):
                raise
            last_exc = exc
            if attempt == max_attempts:
                break
            # Exponential backoff with a little jitter so concurrent callers
            # that failed together don't retry in lockstep.
            delay = base_delay * (2 ** (attempt - 1)) * (1.0 + random.uniform(0.0, 0.25))
            logger.warning(
                f"[Retry] {label or coro_fn.__name__} attempt {attempt}/{max_attempts} "
                f"failed ({type(exc).__name__}), retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    logger.error(f"[Retry] {label or coro_fn.__name__} failed after {max_attempts} attempts")
    raise last_exc
