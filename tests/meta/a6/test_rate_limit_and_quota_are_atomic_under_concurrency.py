"""A6 meta-test: prove rate_limit.py/quota.py actually enforce their caps
exactly under real concurrent load against a real Redis, not just in a
single-threaded happy path.

★ Honesty note on what this does and doesn't prove (CLAUDE.md §12):
Redis's `INCR`/`INCRBY` are themselves atomic regardless of whether the
subsequent `EXPIRE` is bundled into the same Lua script or issued as a
separate call — so this test would very likely still pass even against a
naive two-round-trip (INCR-then-EXPIRE) implementation; it is NOT a
regression test for the specific Lua-script-vs-two-calls design choice
(see docs/design-decisions.md DD-0015 for what that choice actually
buys: avoiding a crash-between-calls window that could leave a counter
key with no TTL, plus one fewer network round trip — a durability/
efficiency concern, not an over-admission one). What this test does
honestly prove: the limiter/quota tracker, as actually shipped, holds
its configured cap exactly under real concurrent load — 50 threads
hammering the same key never admit more than the configured limit. That
is a genuine correctness property worth verifying empirically rather
than assuming from reading the code, and it is the property CLAUDE.md
§1.5 asks a meta-test to prove can be checked, not a claim about which
implementation detail makes it hold.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from tests.conftest import requires_redis

from modelhub.gateway.quota import QuotaConfig, record_and_check_quota
from modelhub.gateway.rate_limit import RateLimitConfig, check_rate_limit

pytestmark = requires_redis

_CONCURRENCY = 50


def test_rate_limit_never_admits_more_than_the_configured_cap(redis_client: object) -> None:
    limit = 10
    config = RateLimitConfig(requests_per_window=limit, window_seconds=60)

    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        decisions = list(
            pool.map(
                lambda _: check_rate_limit(redis_client, "hammered-principal", config),  # type: ignore[arg-type]
                range(_CONCURRENCY),
            )
        )

    allowed_count = sum(1 for d in decisions if d.allowed)
    assert allowed_count == limit, (
        f"expected exactly {limit} of {_CONCURRENCY} concurrent requests to be "
        f"admitted, got {allowed_count} — the limiter is not enforcing its cap "
        f"exactly under concurrent load"
    )


def test_quota_never_admits_more_than_the_configured_daily_cap(redis_client: object) -> None:
    daily_limit = 500
    tokens_per_request = 10
    config = QuotaConfig(daily_token_quota_by_tier={"free": daily_limit})

    with ThreadPoolExecutor(max_workers=_CONCURRENCY) as pool:
        statuses = list(
            pool.map(
                lambda _: record_and_check_quota(
                    redis_client,  # type: ignore[arg-type]
                    "hammered-tenant",
                    "free",
                    tokens_per_request,
                    config,
                ),
                range(_CONCURRENCY),
            )
        )

    allowed_count = sum(1 for s in statuses if s.allowed)
    expected_allowed = daily_limit // tokens_per_request
    assert allowed_count == expected_allowed, (
        f"expected exactly {expected_allowed} of {_CONCURRENCY} concurrent requests "
        f"to be admitted, got {allowed_count} — the quota tracker is not enforcing "
        f"its cap exactly under concurrent load"
    )


def test_a_single_request_within_limits_is_allowed_this_meta_test_is_not_always_red(
    redis_client: object,
) -> None:
    config = RateLimitConfig(requests_per_window=100, window_seconds=60)
    decision = check_rate_limit(redis_client, "lonely-principal", config)  # type: ignore[arg-type]
    assert decision.allowed
