"""Unit tests for gateway/app.py — the real FastAPI wiring of A6's six
gateway concerns, driven through Starlette's TestClient (in-process, real
ASGI request handling, real Redis) rather than a live uvicorn process."""

from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.conftest import requires_redis
from tests.unit.a4.fakes import FakeModelClient, fixed_response_client
from tests.unit.a12.fakes import DEMO_API_KEY, LONG_MODEL_ID, SHORT_MODEL_ID, demo_gateway_deps

from modelhub.gateway.app import create_app
from modelhub.gateway.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from modelhub.monitor.exporter import GatewayMetrics

SHORT_SCHEMA = "CREATE TABLE t (id INTEGER)"  # well under the 1500-token threshold
LONG_SCHEMA = "x" * 4 * 2000  # ~2000 tokens at the routing module's 4-chars/token estimate

_DEFAULT_CLIENT_MAP = {
    SHORT_MODEL_ID: fixed_response_client("SELECT 1", model_id=SHORT_MODEL_ID),
    LONG_MODEL_ID: fixed_response_client("SELECT 1", model_id=LONG_MODEL_ID),
}


def _post(
    tc: TestClient, *, schema_text: str = SHORT_SCHEMA, api_key: str = DEMO_API_KEY
) -> httpx.Response:
    return tc.post(
        "/v1/completions",
        json={"schema_text": schema_text, "prompt": "SELECT 1"},
        headers={"x-api-key": api_key},
    )


def _app(redis_client: object, model_clients: dict[str, object] = _DEFAULT_CLIENT_MAP) -> FastAPI:
    return create_app(demo_gateway_deps(redis_client=redis_client, model_clients=model_clients))


@requires_redis
def test_valid_request_routes_to_short_schema_model(redis_client: object) -> None:
    with TestClient(_app(redis_client)) as tc:
        response = _post(tc)
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == SHORT_MODEL_ID
    assert body["text"] == "SELECT 1"


@requires_redis
def test_long_schema_routes_to_the_other_model(redis_client: object) -> None:
    with TestClient(_app(redis_client)) as tc:
        response = _post(tc, schema_text=LONG_SCHEMA)
    assert response.status_code == 200
    assert response.json()["model_id"] == LONG_MODEL_ID


@requires_redis
def test_invalid_api_key_is_401(redis_client: object) -> None:
    with TestClient(_app(redis_client)) as tc:
        response = _post(tc, api_key="not-the-real-key")
    assert response.status_code == 401


@requires_redis
def test_rate_limit_exceeded_is_429(redis_client: object) -> None:
    deps = demo_gateway_deps(redis_client=redis_client, model_clients=_DEFAULT_CLIENT_MAP)
    deps.rate_limit_config = deps.rate_limit_config.model_copy(update={"requests_per_window": 1})
    app = create_app(deps)
    with TestClient(app) as tc:
        first = _post(tc)
        second = _post(tc)
    assert first.status_code == 200
    assert second.status_code == 429
    assert "Retry-After" in second.headers


@requires_redis
def test_upstream_failure_is_502_and_opens_the_circuit_breaker(redis_client: object) -> None:
    def _raise(_prompt: str) -> None:
        raise RuntimeError("simulated upstream failure")

    client_map = {
        SHORT_MODEL_ID: FakeModelClient(_raise),
        LONG_MODEL_ID: fixed_response_client("SELECT 1", model_id=LONG_MODEL_ID),
    }
    deps = demo_gateway_deps(redis_client=redis_client, model_clients=client_map)
    deps.circuit_breakers[SHORT_MODEL_ID] = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=1, failure_window_seconds=30.0, cooldown_seconds=15.0
        )
    )
    app = create_app(deps)
    with TestClient(app) as tc:
        first = _post(tc)
        second = _post(tc)
    assert first.status_code == 502
    assert second.status_code == 503


@requires_redis
def test_quota_exceeded_is_429(redis_client: object) -> None:
    deps = demo_gateway_deps(redis_client=redis_client, model_clients=_DEFAULT_CLIENT_MAP)
    # a request always costs >= 1 token in this app's accounting (see
    # app.py's `max(prompt_tokens + completion_tokens, 1)`), so a 0
    # daily limit guarantees the very first request is already over.
    deps.quota_config = deps.quota_config.model_copy(
        update={"daily_token_quota_by_tier": {"free": 0, "standard": 0, "premium": 0}}
    )
    app = create_app(deps)
    with TestClient(app) as tc:
        response = _post(tc)
    assert response.status_code == 429


@requires_redis
def test_billing_cost_reflects_real_token_counts(redis_client: object) -> None:
    with TestClient(_app(redis_client)) as tc:
        response = _post(tc)
    assert response.status_code == 200
    assert response.json()["cost_usd"] >= 0.0


@requires_redis
def test_metrics_endpoint_reflects_a_real_recorded_request(redis_client: object) -> None:
    deps = demo_gateway_deps(redis_client=redis_client, model_clients=_DEFAULT_CLIENT_MAP)
    deps.metrics = GatewayMetrics()
    app = create_app(deps)
    with TestClient(app) as tc:
        _post(tc)
        metrics_response = tc.get("/metrics")
    assert metrics_response.status_code == 200
    body = metrics_response.text
    assert "modelhub_gateway_requests_total" in body
    assert f'model_id="{SHORT_MODEL_ID}"' in body
    assert 'outcome="SUCCESS"' in body


@requires_redis
def test_metrics_endpoint_is_404_when_not_configured(redis_client: object) -> None:
    with TestClient(_app(redis_client)) as tc:
        response = tc.get("/metrics")
    assert response.status_code == 404
