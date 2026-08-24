"""Unit tests for gateway/quota.py, against a real local Redis."""

from __future__ import annotations

import pytest
from tests.conftest import requires_redis

from modelhub.common.errors import ErrorCode, GatewayError
from modelhub.gateway.quota import QuotaConfig, enforce_quota, record_and_check_quota

pytestmark = requires_redis


def _config() -> QuotaConfig:
    return QuotaConfig(daily_token_quota_by_tier={"free": 1000, "standard": 100000})


def test_usage_within_quota_is_allowed(redis_client: object) -> None:
    status = record_and_check_quota(redis_client, "alice", "free", 400, _config())  # type: ignore[arg-type]
    assert status.allowed
    assert status.tokens_used_today == 400
    assert status.daily_limit == 1000


def test_usage_accumulates_across_calls(redis_client: object) -> None:
    config = _config()
    record_and_check_quota(redis_client, "alice", "free", 400, config)  # type: ignore[arg-type]
    status = record_and_check_quota(redis_client, "alice", "free", 400, config)  # type: ignore[arg-type]
    assert status.tokens_used_today == 800
    assert status.allowed


def test_usage_over_quota_is_denied(redis_client: object) -> None:
    config = _config()
    record_and_check_quota(redis_client, "alice", "free", 900, config)  # type: ignore[arg-type]
    status = record_and_check_quota(redis_client, "alice", "free", 200, config)  # type: ignore[arg-type]
    assert not status.allowed
    assert status.tokens_used_today == 1100


def test_different_tenants_have_independent_quotas(redis_client: object) -> None:
    config = _config()
    record_and_check_quota(redis_client, "alice", "free", 900, config)  # type: ignore[arg-type]
    status = record_and_check_quota(redis_client, "bob", "free", 10, config)  # type: ignore[arg-type]
    assert status.tokens_used_today == 10
    assert status.allowed


def test_enforce_quota_raises_gateway_error_when_exceeded(redis_client: object) -> None:
    config = _config()
    with pytest.raises(GatewayError) as exc_info:
        enforce_quota(redis_client, "alice", "free", 1500, config)  # type: ignore[arg-type]
    assert exc_info.value.code is ErrorCode.QUOTA_EXCEEDED
    assert exc_info.value.retryable is False


def test_unknown_tier_raises_value_error(redis_client: object) -> None:
    with pytest.raises(ValueError, match="tier"):
        record_and_check_quota(redis_client, "alice", "nonexistent-tier", 10, _config())  # type: ignore[arg-type]


def test_non_positive_tokens_used_raises_value_error(redis_client: object) -> None:
    with pytest.raises(ValueError, match="positive"):
        record_and_check_quota(redis_client, "alice", "free", 0, _config())  # type: ignore[arg-type]
