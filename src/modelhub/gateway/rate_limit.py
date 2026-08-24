"""Redis-backed fixed-window rate limiting.

The increment-and-maybe-set-expiry sequence is a real INCR-then-EXPIRE
race if done as two separate round trips (a process could crash, or two
requests could interleave, between the two calls, leaving a key that
increments forever without ever expiring) — so it runs as a single
server-side Lua script (`EVAL`), atomic by construction, not "atomic in
the common case." `redis_client` is expected to have been built with
`gateway/redis_support.py::make_redis_client`, which sets an explicit
socket timeout (CLAUDE.md §5.1: no timeout on a network call is a bug) —
this module does not construct its own client, so it cannot itself
guarantee that; callers and tests share the one factory instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode, GatewayError, Stage

# KEYS[1] = the counter key. ARGV[1] = window_seconds.
# Returns the post-increment count. Sets the TTL only on the request that
# creates the key (count == 1 immediately after INCR) — a later request
# in the same window must never reset the window's remaining time.
_INCR_WITH_WINDOW_LUA = """
local count = redis.call("INCR", KEYS[1])
if count == 1 then
    redis.call("EXPIRE", KEYS[1], ARGV[1])
end
return count
"""


class RedisLike(Protocol):
    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object: ...


class RateLimitConfig(ModelHubBaseConfig):
    requests_per_window: int
    window_seconds: int


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    current_count: int
    limit: int


def check_rate_limit(
    redis_client: RedisLike, principal_key: str, config: RateLimitConfig
) -> RateLimitDecision:
    """Increment `principal_key`'s counter for the current fixed window
    and report whether this request is within `config.requests_per_window`.

    Fixed-window (not sliding-window/token-bucket): simpler, and its
    known weakness — up to 2x the configured rate at a window boundary —
    is an accepted trade-off for this round's scope, not an oversight;
    a sliding-window limiter would need to be revisited if that boundary
    burst is ever actually a problem in practice.
    """
    redis_key = f"modelhub:ratelimit:{principal_key}"
    count = redis_client.eval(_INCR_WITH_WINDOW_LUA, 1, redis_key, config.window_seconds)
    current_count = int(count)  # type: ignore[call-overload]
    return RateLimitDecision(
        allowed=current_count <= config.requests_per_window,
        current_count=current_count,
        limit=config.requests_per_window,
    )


def enforce_rate_limit(
    redis_client: RedisLike, principal_key: str, config: RateLimitConfig
) -> RateLimitDecision:
    """Like `check_rate_limit`, but raises `GatewayError` instead of
    returning an `allowed=False` decision the caller might forget to
    check — use this at the actual request-handling call site."""
    decision = check_rate_limit(redis_client, principal_key, config)
    if not decision.allowed:
        raise GatewayError(
            f"rate limit exceeded: {decision.current_count}/{decision.limit} "
            f"requests in the current {config.window_seconds}s window",
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            stage=Stage.SERVE,
            context={"principal_key": principal_key, "current_count": decision.current_count},
            retryable=True,
        )
    return decision


__all__ = [
    "RateLimitConfig",
    "RateLimitDecision",
    "check_rate_limit",
    "enforce_rate_limit",
]
