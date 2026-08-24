"""Shared Redis client construction for gateway/.

One factory, used by every Redis-backed gateway module (`rate_limit.py`,
`quota.py`) and by tests — CLAUDE.md §5.1: no explicit timeout on a
network call is a bug, and a single factory means that requirement can't
accidentally be forgotten at one call site while present at another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis

_DEFAULT_TIMEOUT_S = 2.0


def make_redis_client(
    url: str, *, socket_timeout_s: float = _DEFAULT_TIMEOUT_S
) -> redis.Redis[bytes]:
    import redis

    return redis.Redis.from_url(
        url, socket_timeout=socket_timeout_s, socket_connect_timeout=socket_timeout_s
    )


__all__ = ["make_redis_client"]
