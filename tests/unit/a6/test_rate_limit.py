"""Unit tests for gateway/rate_limit.py, against a real local Redis
(tests/conftest.py::redis_client) — the atomicity claim in the module
docstring is only meaningful if exercised against the real server, not a
Python dict standing in for one."""

from __future__ import annotations

import pytest
from tests.conftest import requires_redis

from modelhub.common.errors import ErrorCode, GatewayError
from modelhub.gateway.rate_limit import RateLimitConfig, check_rate_limit, enforce_rate_limit

pytestmark = requires_redis


def test_requests_within_limit_are_allowed(redis_client: object) -> None:
    config = RateLimitConfig(requests_per_window=3, window_seconds=60)
    for _ in range(3):
        decision = check_rate_limit(redis_client, "alice", config)  # type: ignore[arg-type]
        assert decision.allowed


def test_request_over_limit_is_denied(redis_client: object) -> None:
    config = RateLimitConfig(requests_per_window=3, window_seconds=60)
    for _ in range(3):
        check_rate_limit(redis_client, "alice", config)  # type: ignore[arg-type]
    decision = check_rate_limit(redis_client, "alice", config)  # type: ignore[arg-type]
    assert not decision.allowed
    assert decision.current_count == 4
    assert decision.limit == 3


def test_different_principals_have_independent_limits(redis_client: object) -> None:
    config = RateLimitConfig(requests_per_window=1, window_seconds=60)
    alice = check_rate_limit(redis_client, "alice", config)  # type: ignore[arg-type]
    bob = check_rate_limit(redis_client, "bob", config)  # type: ignore[arg-type]
    assert alice.allowed
    assert bob.allowed


def test_enforce_rate_limit_raises_gateway_error_when_exceeded(redis_client: object) -> None:
    config = RateLimitConfig(requests_per_window=1, window_seconds=60)
    enforce_rate_limit(redis_client, "alice", config)  # type: ignore[arg-type]
    with pytest.raises(GatewayError) as exc_info:
        enforce_rate_limit(redis_client, "alice", config)  # type: ignore[arg-type]
    assert exc_info.value.code is ErrorCode.RATE_LIMIT_EXCEEDED
    assert exc_info.value.retryable is True


def test_counter_key_actually_has_a_ttl_set(redis_client: object) -> None:
    config = RateLimitConfig(requests_per_window=5, window_seconds=42)
    check_rate_limit(redis_client, "alice", config)  # type: ignore[arg-type]
    ttl = redis_client.ttl("modelhub:ratelimit:alice")  # type: ignore[attr-defined]
    assert 0 < ttl <= 42
