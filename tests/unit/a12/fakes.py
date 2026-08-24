"""Shared builders for A12 tests — CLAUDE.md §1.4: plain construction
helpers around real project types."""

from __future__ import annotations

from modelhub.gateway.app import GatewayDeps
from modelhub.gateway.auth import AuthConfig, Principal, hash_api_key
from modelhub.gateway.billing import BillingConfig
from modelhub.gateway.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from modelhub.gateway.quota import QuotaConfig
from modelhub.gateway.rate_limit import RateLimitConfig
from modelhub.gateway.routing import RoutingConfig

DEMO_API_KEY = "demo-key-not-a-real-secret"
DEMO_TENANT_ID = "demo-tenant"
SHORT_MODEL_ID = "arctic-text2sql-r1-7b"
LONG_MODEL_ID = "qwen3.5-9b"

_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=5, failure_window_seconds=30.0, cooldown_seconds=15.0
)


def demo_auth_config() -> AuthConfig:
    return AuthConfig(
        principals_by_key_hash={
            hash_api_key(DEMO_API_KEY): Principal(tenant_id=DEMO_TENANT_ID, tier="standard")
        }
    )


def demo_gateway_deps(*, redis_client: object, model_clients: dict[str, object]) -> GatewayDeps:
    return GatewayDeps(
        auth_config=demo_auth_config(),
        rate_limit_config=RateLimitConfig(requests_per_window=60, window_seconds=60),
        quota_config=QuotaConfig(
            daily_token_quota_by_tier={
                "free": 100_000,
                "standard": 1_000_000,
                "premium": 10_000_000,
            }
        ),
        billing_config=BillingConfig(
            price_per_1k_prompt_tokens_usd={SHORT_MODEL_ID: 0.0002, LONG_MODEL_ID: 0.00025},
            price_per_1k_completion_tokens_usd={SHORT_MODEL_ID: 0.0006, LONG_MODEL_ID: 0.00075},
        ),
        routing_config=RoutingConfig(
            schema_token_threshold=1500,
            threshold_source="estimate",
            short_schema_model_id=SHORT_MODEL_ID,
            long_schema_model_id=LONG_MODEL_ID,
        ),
        redis_client=redis_client,
        model_clients=model_clients,
        circuit_breakers={
            SHORT_MODEL_ID: CircuitBreaker(_BREAKER_CONFIG),
            LONG_MODEL_ID: CircuitBreaker(_BREAKER_CONFIG),
        },
    )
