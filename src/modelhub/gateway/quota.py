"""Redis-backed per-tenant daily token quota tracking.

Same atomicity concern as `rate_limit.py`: recording usage and checking
the new total against the daily limit must happen as one atomic
server-side operation, or two concurrent requests near the limit could
both read "still under quota" before either one's increment lands,
letting the tenant burst past the configured cap. `record_and_check`
therefore does the INCRBY-and-set-expiry in a single Lua script.

CLAUDE.md §1.1: going over quota must hard-deny the request, not degrade
into a "best-effort" allow — `enforce_quota` raises, it does not just
warn.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode, GatewayError, Stage

_SECONDS_PER_DAY = 24 * 60 * 60

# KEYS[1] = the tenant-day counter key. ARGV[1] = tokens_used this request.
# Returns the post-increment total for today. Sets a 1-day TTL only when
# this call creates the key, so the counter resets daily without a
# separate cron job and without ever losing the remainder of today's
# window on a later call.
_INCRBY_WITH_DAILY_EXPIRY_LUA = f"""
local total = redis.call("INCRBY", KEYS[1], ARGV[1])
if tonumber(ARGV[1]) == total then
    redis.call("EXPIRE", KEYS[1], {_SECONDS_PER_DAY})
end
return total
"""


class RedisLike(Protocol):
    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object: ...


class QuotaConfig(ModelHubBaseConfig):
    daily_token_quota_by_tier: dict[str, int]


@dataclass(frozen=True)
class QuotaStatus:
    allowed: bool
    tokens_used_today: int
    daily_limit: int


def _quota_key(tenant_id: str, *, today: str) -> str:
    return f"modelhub:quota:{tenant_id}:{today}"


def record_and_check_quota(
    redis_client: RedisLike,
    tenant_id: str,
    tier: str,
    tokens_used: int,
    config: QuotaConfig,
) -> QuotaStatus:
    if tokens_used <= 0:
        raise ValueError(f"tokens_used must be positive, got {tokens_used}")
    if tier not in config.daily_token_quota_by_tier:
        raise ValueError(f"no configured quota for tier {tier!r}")

    daily_limit = config.daily_token_quota_by_tier[tier]
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    key = _quota_key(tenant_id, today=today)
    total = redis_client.eval(_INCRBY_WITH_DAILY_EXPIRY_LUA, 1, key, tokens_used)
    tokens_used_today = int(total)  # type: ignore[call-overload]

    return QuotaStatus(
        allowed=tokens_used_today <= daily_limit,
        tokens_used_today=tokens_used_today,
        daily_limit=daily_limit,
    )


def enforce_quota(
    redis_client: RedisLike,
    tenant_id: str,
    tier: str,
    tokens_used: int,
    config: QuotaConfig,
) -> QuotaStatus:
    status = record_and_check_quota(redis_client, tenant_id, tier, tokens_used, config)
    if not status.allowed:
        raise GatewayError(
            f"daily token quota exceeded: {status.tokens_used_today}/{status.daily_limit} "
            f"tokens used today",
            code=ErrorCode.QUOTA_EXCEEDED,
            stage=Stage.SERVE,
            context={
                "tenant_id": tenant_id,
                "tier": tier,
                "tokens_used_today": status.tokens_used_today,
                "daily_limit": status.daily_limit,
            },
            retryable=False,  # retryable only after the daily window resets — not "soon"
        )
    return status


__all__ = [
    "QuotaConfig",
    "QuotaStatus",
    "enforce_quota",
    "record_and_check_quota",
]
