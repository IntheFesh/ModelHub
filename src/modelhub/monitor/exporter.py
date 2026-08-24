"""Prometheus metrics exporter (PLAN.md's platform stack: "...+ Redis +
Prometheus/Grafana").

Wraps a dedicated `CollectorRegistry` — never `prometheus_client`'s
global default registry. A shared global registry makes tests running in
the same process step on each other's metric state (a `Counter` can't be
re-registered under the same name), and makes constructing more than one
exporter instance in a single process (e.g. one per served model
backend) impossible.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from modelhub.gateway.circuit_breaker import CircuitState

_CIRCUIT_STATE_VALUE = {
    CircuitState.CLOSED: 0,
    CircuitState.HALF_OPEN: 1,
    CircuitState.OPEN: 2,
}


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()

        self.requests_total = Counter(
            "modelhub_gateway_requests_total",
            "Total requests handled by the gateway, by model and outcome.",
            ["model_id", "outcome"],
            registry=self.registry,
        )
        self.request_latency_seconds = Histogram(
            "modelhub_gateway_request_latency_seconds",
            "Gateway request latency in seconds, by model.",
            ["model_id"],
            registry=self.registry,
        )
        self.rate_limit_denied_total = Counter(
            "modelhub_gateway_rate_limit_denied_total",
            "Requests denied by the rate limiter, by principal.",
            ["principal_key"],
            registry=self.registry,
        )
        self.quota_denied_total = Counter(
            "modelhub_gateway_quota_denied_total",
            "Requests denied by quota enforcement, by tenant.",
            ["tenant_id"],
            registry=self.registry,
        )
        self.circuit_breaker_state = Gauge(
            "modelhub_gateway_circuit_breaker_state",
            "Circuit breaker state by backend: 0=CLOSED 1=HALF_OPEN 2=OPEN.",
            ["backend"],
            registry=self.registry,
        )
        self.mfu_ratio = Gauge(
            "modelhub_serve_mfu_ratio",
            "Model FLOPs Utilization, by model and stage (prefill/decode).",
            ["model_id", "stage"],
            registry=self.registry,
        )
        self.mbu_ratio = Gauge(
            "modelhub_serve_mbu_ratio",
            "Model Bandwidth Utilization, by model and stage (prefill/decode).",
            ["model_id", "stage"],
            registry=self.registry,
        )
        self.prefix_cache_hit_rate = Gauge(
            "modelhub_serve_prefix_cache_hit_rate",
            "vLLM prefix cache hit rate, by model.",
            ["model_id"],
            registry=self.registry,
        )

    def record_request(self, *, model_id: str, outcome: str, latency_s: float) -> None:
        self.requests_total.labels(model_id=model_id, outcome=outcome).inc()
        self.request_latency_seconds.labels(model_id=model_id).observe(latency_s)

    def record_rate_limit_denied(self, principal_key: str) -> None:
        self.rate_limit_denied_total.labels(principal_key=principal_key).inc()

    def record_quota_denied(self, tenant_id: str) -> None:
        self.quota_denied_total.labels(tenant_id=tenant_id).inc()

    def set_circuit_breaker_state(self, backend: str, state: CircuitState) -> None:
        self.circuit_breaker_state.labels(backend=backend).set(_CIRCUIT_STATE_VALUE[state])

    def set_mfu(self, model_id: str, stage: str, value: float) -> None:
        self.mfu_ratio.labels(model_id=model_id, stage=stage).set(value)

    def set_mbu(self, model_id: str, stage: str, value: float) -> None:
        self.mbu_ratio.labels(model_id=model_id, stage=stage).set(value)

    def set_prefix_cache_hit_rate(self, model_id: str, hit_rate: float) -> None:
        self.prefix_cache_hit_rate.labels(model_id=model_id).set(hit_rate)

    def render(self) -> bytes:
        """Prometheus text exposition format, for a `/metrics` endpoint."""
        return generate_latest(self.registry)


__all__ = ["GatewayMetrics"]
