"""OpenAI-compatible API gateway: auth, rate limit, quota, routing, billing.

Public API: `authenticate`/`AuthConfig`/`Principal`/`hash_api_key`
(auth.py); `enforce_rate_limit`/`check_rate_limit`/`RateLimitConfig`
(rate_limit.py, Redis-backed); `enforce_quota`/`record_and_check_quota`/
`QuotaConfig` (quota.py, Redis-backed); `route_by_schema_length`/
`RoutingConfig`/`estimate_token_count` (routing.py); `CircuitBreaker`/
`CircuitBreakerConfig`/`CircuitState` (circuit_breaker.py, in-process);
`compute_request_cost`/`BillingConfig`/`RequestCost` (billing.py);
`make_redis_client` (redis_support.py, shared client factory with an
explicit socket timeout — CLAUDE.md §5.1).

Every module here raises `GatewayError` (a `ModelHubError` subclass) on a
refused request rather than returning a bare bool the caller might not
check — CLAUDE.md §1.1's "silent success" anti-pattern applies just as
much to a gateway silently letting a request through as it does to a
model producing a fabricated result.
"""

from modelhub.gateway.auth import AuthConfig, Principal, authenticate, hash_api_key
from modelhub.gateway.billing import BillingConfig, RequestCost, compute_request_cost
from modelhub.gateway.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from modelhub.gateway.quota import (
    QuotaConfig,
    QuotaStatus,
    enforce_quota,
    record_and_check_quota,
)
from modelhub.gateway.rate_limit import (
    RateLimitConfig,
    RateLimitDecision,
    check_rate_limit,
    enforce_rate_limit,
)
from modelhub.gateway.redis_support import make_redis_client
from modelhub.gateway.routing import RoutingConfig, estimate_token_count, route_by_schema_length

__all__ = [
    "AuthConfig",
    "BillingConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "Principal",
    "QuotaConfig",
    "QuotaStatus",
    "RateLimitConfig",
    "RateLimitDecision",
    "RequestCost",
    "RoutingConfig",
    "authenticate",
    "check_rate_limit",
    "compute_request_cost",
    "enforce_quota",
    "enforce_rate_limit",
    "estimate_token_count",
    "hash_api_key",
    "make_redis_client",
    "record_and_check_quota",
    "route_by_schema_length",
]
