"""A6 smoke test: one simulated request through the full gateway chain —
auth -> circuit breaker -> rate limit -> routing -> (fake backend call) ->
quota -> billing — wired the way a real request handler would call these
modules in order, against real Redis."""

from __future__ import annotations

import pytest
from tests.conftest import requires_redis

from modelhub.gateway.auth import AuthConfig, authenticate, hash_api_key
from modelhub.gateway.billing import BillingConfig, compute_request_cost
from modelhub.gateway.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from modelhub.gateway.quota import QuotaConfig, enforce_quota
from modelhub.gateway.rate_limit import RateLimitConfig, enforce_rate_limit
from modelhub.gateway.routing import RoutingConfig, route_by_schema_length

pytestmark = [pytest.mark.smoke, requires_redis]


def test_gateway_pipeline_smoke(redis_client: object) -> None:
    auth_config = AuthConfig.model_validate(
        {
            "principals_by_key_hash": {
                hash_api_key("sk-test-key"): {"tenant_id": "smoke-tenant", "tier": "standard"}
            }
        }
    )
    rate_limit_config = RateLimitConfig(requests_per_window=10, window_seconds=60)
    routing_config = RoutingConfig(
        schema_token_threshold=100,
        threshold_source="estimate",
        short_schema_model_id="arctic-text2sql-r1-7b",
        long_schema_model_id="qwen3.5-9b",
    )
    quota_config = QuotaConfig(daily_token_quota_by_tier={"standard": 1_000_000})
    billing_config = BillingConfig(
        price_per_1k_prompt_tokens_usd={"arctic-text2sql-r1-7b": 0.0002},
        price_per_1k_completion_tokens_usd={"arctic-text2sql-r1-7b": 0.0006},
    )
    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=5, failure_window_seconds=30, cooldown_seconds=15)
    )

    # 1. auth
    principal = authenticate("sk-test-key", auth_config)
    assert principal.tenant_id == "smoke-tenant"

    # 2. circuit breaker guard
    breaker.guard()  # must not raise

    # 3. rate limit
    enforce_rate_limit(redis_client, principal.tenant_id, rate_limit_config)  # type: ignore[arg-type]

    # 4. routing (short schema -> Arctic)
    schema_text = "CREATE TABLE students (id INTEGER, name TEXT);"
    model_id = route_by_schema_length(schema_text, routing_config)
    assert model_id == "arctic-text2sql-r1-7b"

    # 5. simulated backend call outcome
    prompt_tokens, completion_tokens = 120, 40
    breaker.record_success()

    # 6. quota
    quota_status = enforce_quota(
        redis_client,  # type: ignore[arg-type]
        principal.tenant_id,
        principal.tier,
        prompt_tokens + completion_tokens,
        quota_config,
    )
    assert quota_status.allowed

    # 7. billing
    cost = compute_request_cost(model_id, prompt_tokens, completion_tokens, billing_config)
    assert cost.total_cost_usd > 0
