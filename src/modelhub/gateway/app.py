"""A real FastAPI app wiring gateway/'s six concerns (auth, rate limit,
quota, circuit breaker, routing, billing — A6) into one OpenAI-completions
-shaped endpoint.

A6 built the six concerns as standalone modules but never an app object
proving they compose — this module is that composition, added here in
A12 because `make demo` (PLAN.md's "起服务 → 发请求") needs one real place
they all actually run together, and a real deployment eventually needs
exactly this same wiring anyway. Nothing here is demo-only: the model
clients this app calls through are injected (`GatewayDeps.model_clients`),
so a real deployment supplies real `HttpModelClient`s pointed at vLLM and
`make demo` supplies the stub server in `scripts/demo_stub_model_server.py`
— the app itself does not know or care which.

Request handling order, and why:
  1. auth (`authenticate`) — reject unknown keys before spending anything
  2. rate limit (`enforce_rate_limit`) — cheap, Redis-atomic, before routing
  3. routing (`route_by_schema_length`) — picks the target model
  4. circuit breaker (`CircuitBreaker.guard`) — refuse fast if that
     specific backend is already known-broken, before calling it
  5. the actual model call
  6. quota (`enforce_quota`) — CLAUDE.md/PLAN.md's "配额...超额 429" reads
     naturally as a pre-request check, but `quota.py`'s only entry point
     (`enforce_quota`) computes usage from THIS request's own token count,
     which isn't known until after generation — so quota is necessarily
     checked post-generation here. An over-quota request still discards
     its (already-computed) completion and returns 429 rather than
     silently under-enforcing by returning it anyway; the generation
     compute already spent is an accepted cost of the current quota.py
     API shape, not a bug — see docs/design-decisions.md.
  7. billing (`compute_request_cost`) — only reached on a response that's
     actually being returned to the caller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from fastapi import FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel

from modelhub.common.errors import ErrorCode, GatewayError
from modelhub.eval.model_client import GenerationResult, ModelClient
from modelhub.gateway.auth import AuthConfig, Principal, authenticate
from modelhub.gateway.billing import BillingConfig, compute_request_cost
from modelhub.gateway.circuit_breaker import CircuitBreaker
from modelhub.gateway.quota import QuotaConfig, enforce_quota
from modelhub.gateway.rate_limit import RateLimitConfig, enforce_rate_limit
from modelhub.gateway.routing import RoutingConfig, route_by_schema_length
from modelhub.monitor.exporter import GatewayMetrics


class RedisLike(Protocol):
    def eval(self, script: str, numkeys: int, *keys_and_args: object) -> object: ...


# GatewayError.code -> HTTP status. A code with no entry here is a real
# programming error (a GatewayError this app doesn't know how to map),
# not something to default to a generic 500 silently — see the
# `completions` handler's final `except GatewayError` branch.
_ERROR_CODE_STATUS: dict[ErrorCode, int] = {
    ErrorCode.AUTH_INVALID_API_KEY: 401,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.CIRCUIT_BREAKER_OPEN: 503,
    ErrorCode.QUOTA_EXCEEDED: 429,
}


@dataclass
class GatewayDeps:
    auth_config: AuthConfig
    rate_limit_config: RateLimitConfig
    quota_config: QuotaConfig
    billing_config: BillingConfig
    routing_config: RoutingConfig
    redis_client: RedisLike
    # one entry per model_id `routing_config` can route to — the caller
    # owns constructing these (from `configs/gateway/circuit_breaker.yaml`,
    # one CircuitBreaker instance per backend) so this app never has to
    # silently invent breaker thresholds for a model nobody configured.
    model_clients: dict[str, ModelClient]
    circuit_breakers: dict[str, CircuitBreaker]
    # A6 (auth/rate-limit/.../billing) and A7 (GatewayMetrics) never got
    # wired together until now — real per-request recording, not a
    # demo-only hack. `None` disables metrics entirely (no /metrics
    # route, no recording), the same "measured or explicitly absent,
    # never a fabricated zero" discipline CLAUDE.md applies elsewhere.
    metrics: GatewayMetrics | None = None


class CompletionRequest(BaseModel):
    schema_text: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    generate_timeout_s: float = 30.0


class CompletionResponse(BaseModel):
    model_id: str
    text: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float


def _http_exception_for(exc: GatewayError) -> HTTPException:
    status_code = _ERROR_CODE_STATUS.get(exc.code)
    if status_code is None:
        raise AssertionError(
            f"unmapped GatewayError code {exc.code.value!r} — every code this app can "
            f"raise/catch must have an explicit HTTP status mapping in _ERROR_CODE_STATUS"
        )
    headers = {"Retry-After": "5"} if exc.retryable else None
    return HTTPException(status_code=status_code, detail=str(exc), headers=headers)


def _authenticate_and_rate_limit(d: GatewayDeps, x_api_key: str) -> Principal:
    principal = authenticate(x_api_key, d.auth_config)
    try:
        enforce_rate_limit(d.redis_client, principal.tenant_id, d.rate_limit_config)
    except GatewayError as e:
        if d.metrics is not None:
            d.metrics.record_rate_limit_denied(str(e.context.get("principal_key", "unknown")))
        raise
    return principal


def _resolve_backend(d: GatewayDeps, req: CompletionRequest) -> tuple[str, CircuitBreaker]:
    model_id = route_by_schema_length(req.schema_text, d.routing_config)
    if model_id not in d.model_clients or model_id not in d.circuit_breakers:
        raise HTTPException(
            status_code=500,
            detail=f"routed to model_id {model_id!r}, which has no configured "
            f"model client and/or circuit breaker",
        )
    return model_id, d.circuit_breakers[model_id]


def _generate(
    d: GatewayDeps, model_id: str, breaker: CircuitBreaker, req: CompletionRequest
) -> GenerationResult:
    breaker.guard()  # raises GatewayError(CIRCUIT_BREAKER_OPEN) if not allowed
    try:
        result = d.model_clients[model_id].generate(
            req.prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            timeout_s=req.generate_timeout_s,
        )
    except Exception:
        breaker.record_failure()
        if d.metrics is not None:
            d.metrics.set_circuit_breaker_state(model_id, breaker.state)
        raise
    breaker.record_success()
    if d.metrics is not None:
        d.metrics.set_circuit_breaker_state(model_id, breaker.state)
    return result


def _enforce_quota(d: GatewayDeps, principal: Principal, tokens_used: int) -> None:
    try:
        enforce_quota(
            d.redis_client, principal.tenant_id, principal.tier, tokens_used, d.quota_config
        )
    except GatewayError as e:
        if d.metrics is not None:
            d.metrics.record_quota_denied(principal.tenant_id)
        raise e


def create_app(deps: GatewayDeps) -> FastAPI:
    app = FastAPI(title="ModelHub Gateway")
    app.state.deps = deps

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(
        req: CompletionRequest, request: Request, x_api_key: str = Header(...)
    ) -> CompletionResponse:
        d: GatewayDeps = request.app.state.deps
        t0 = time.monotonic()
        model_id = "unknown"
        try:
            principal = _authenticate_and_rate_limit(d, x_api_key)
            model_id, breaker = _resolve_backend(d, req)
            result = _generate(d, model_id, breaker, req)
            prompt_tokens = result.prompt_tokens or 0
            completion_tokens = result.completion_tokens or 0
            _enforce_quota(d, principal, max(prompt_tokens + completion_tokens, 1))
        except GatewayError as e:
            if d.metrics is not None:
                d.metrics.record_request(
                    model_id=model_id, outcome=e.code.value, latency_s=time.monotonic() - t0
                )
            raise _http_exception_for(e) from e
        except HTTPException:
            # a deliberate, already-classified HTTP error (e.g.
            # _resolve_backend's 500 for an unrouted model_id) — pass it
            # through unchanged, never relabel it as an upstream failure.
            raise
        except Exception as e:
            if d.metrics is not None:
                d.metrics.record_request(
                    model_id=model_id, outcome="UPSTREAM_FAILURE", latency_s=time.monotonic() - t0
                )
            raise HTTPException(status_code=502, detail=f"upstream model call failed: {e}") from e

        cost = compute_request_cost(model_id, prompt_tokens, completion_tokens, d.billing_config)
        if d.metrics is not None:
            d.metrics.record_request(
                model_id=model_id, outcome="SUCCESS", latency_s=time.monotonic() - t0
            )
        return CompletionResponse(
            model_id=model_id,
            text=result.text,
            finish_reason=result.finish_reason,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=cost.total_cost_usd,
        )

    @app.get("/metrics")
    def metrics_endpoint(request: Request) -> Response:
        d: GatewayDeps = request.app.state.deps
        if d.metrics is None:
            raise HTTPException(status_code=404, detail="metrics not configured for this app")
        return Response(content=d.metrics.render(), media_type="text/plain; version=0.0.4")

    return app


__all__ = ["CompletionRequest", "CompletionResponse", "GatewayDeps", "create_app"]
