"""Unit tests for monitor/exporter.py, against a real
`prometheus_client.CollectorRegistry` and real exposition-format
rendering — not a stub metrics backend."""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from modelhub.gateway.circuit_breaker import CircuitState
from modelhub.monitor.exporter import GatewayMetrics


def test_each_instance_uses_its_own_registry_by_default() -> None:
    a = GatewayMetrics()
    b = GatewayMetrics()
    assert a.registry is not b.registry


def test_record_request_increments_counter_and_observes_latency() -> None:
    metrics = GatewayMetrics()
    metrics.record_request(model_id="arctic-text2sql-r1-7b", outcome="success", latency_s=0.25)
    rendered = metrics.render().decode("utf-8")
    assert (
        'modelhub_gateway_requests_total{model_id="arctic-text2sql-r1-7b",outcome="success"} 1.0'
        in rendered
    )
    assert "modelhub_gateway_request_latency_seconds" in rendered


def test_record_rate_limit_denied() -> None:
    metrics = GatewayMetrics()
    metrics.record_rate_limit_denied("alice")
    metrics.record_rate_limit_denied("alice")
    rendered = metrics.render().decode("utf-8")
    assert 'modelhub_gateway_rate_limit_denied_total{principal_key="alice"} 2.0' in rendered


def test_record_quota_denied() -> None:
    metrics = GatewayMetrics()
    metrics.record_quota_denied("tenant-1")
    rendered = metrics.render().decode("utf-8")
    assert 'modelhub_gateway_quota_denied_total{tenant_id="tenant-1"} 1.0' in rendered


def test_set_circuit_breaker_state_maps_enum_to_numeric_gauge() -> None:
    metrics = GatewayMetrics()
    metrics.set_circuit_breaker_state("arctic", CircuitState.OPEN)
    rendered = metrics.render().decode("utf-8")
    assert 'modelhub_gateway_circuit_breaker_state{backend="arctic"} 2.0' in rendered


def test_set_mfu_and_mbu_gauges() -> None:
    metrics = GatewayMetrics()
    metrics.set_mfu("qwen3.5-9b", "decode", 0.12)
    metrics.set_mbu("qwen3.5-9b", "decode", 0.85)
    rendered = metrics.render().decode("utf-8")
    assert 'modelhub_serve_mfu_ratio{model_id="qwen3.5-9b",stage="decode"} 0.12' in rendered
    assert 'modelhub_serve_mbu_ratio{model_id="qwen3.5-9b",stage="decode"} 0.85' in rendered


def test_set_prefix_cache_hit_rate() -> None:
    metrics = GatewayMetrics()
    metrics.set_prefix_cache_hit_rate("arctic-text2sql-r1-7b", 0.73)
    rendered = metrics.render().decode("utf-8")
    assert 'modelhub_serve_prefix_cache_hit_rate{model_id="arctic-text2sql-r1-7b"} 0.73' in rendered


def test_accepts_an_externally_provided_registry() -> None:
    registry = CollectorRegistry()
    metrics = GatewayMetrics(registry)
    assert metrics.registry is registry
