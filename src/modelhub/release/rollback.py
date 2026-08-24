"""Automatic-rollback decision: evaluate a canary's observed health
against thresholds and decide CONTINUE / ROLLBACK, then actually flip
the registry entry back — PLAN.md's "自动回滚" half of the "灰度 5% →
注入劣化 → 自动回滚" drill.

Reuses `gate/types.py`'s three-state `GateDecision` rather than
inventing a parallel enum — REJECT here means "roll back," PASS means
"healthy, safe to continue/advance," and NOT_APPLICABLE means "too few
samples yet to judge" (CLAUDE.md §1.3: an unmeasured sample size must
never silently read as "healthy").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelhub.common.config import ModelHubBaseConfig
from modelhub.gate.types import GateDecision, GateResult
from modelhub.release.registry import ModelRegistry, RegistryEntry

_GATE_NAME = "canary_health"


@dataclass(frozen=True)
class CanaryHealthSample:
    """One observed outcome from the canary arm — real SQL-execution
    facts (`sqlexec`'s own classification), not a synthetic health
    score."""

    succeeded: bool
    is_harness_error: bool
    latency_s: float


class RollbackConfig(ModelHubBaseConfig):
    min_sample_size: int
    max_error_rate: float
    max_harness_error_rate: float
    max_latency_p99_s: float


def _percentile(sorted_values: list[float], pct: float) -> float:
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


@dataclass(frozen=True)
class CanaryHealthReport:
    sample_size: int
    error_rate: float
    harness_error_rate: float
    latency_p99_s: float


def summarize_canary_health(samples: list[CanaryHealthSample]) -> CanaryHealthReport:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    n = len(samples)
    error_count = sum(1 for s in samples if not s.succeeded)
    harness_count = sum(1 for s in samples if s.is_harness_error)
    latencies = sorted(s.latency_s for s in samples)
    return CanaryHealthReport(
        sample_size=n,
        error_rate=error_count / n,
        harness_error_rate=harness_count / n,
        latency_p99_s=_percentile(latencies, 99),
    )


def evaluate_canary_health(samples: list[CanaryHealthSample], config: RollbackConfig) -> GateResult:
    if len(samples) < config.min_sample_size:
        return GateResult(
            _GATE_NAME,
            GateDecision.NOT_APPLICABLE,
            f"only {len(samples)} sample(s), below the {config.min_sample_size} required "
            f"to judge canary health",
            {"sample_size": len(samples), "min_required": config.min_sample_size},
        )

    report = summarize_canary_health(samples)
    reasons = []
    if report.error_rate > config.max_error_rate:
        reasons.append(f"error_rate {report.error_rate:.2%} > {config.max_error_rate:.2%}")
    if report.harness_error_rate > config.max_harness_error_rate:
        reasons.append(
            f"harness_error_rate {report.harness_error_rate:.2%} > "
            f"{config.max_harness_error_rate:.2%}"
        )
    if report.latency_p99_s > config.max_latency_p99_s:
        reasons.append(f"latency_p99_s {report.latency_p99_s:.3f} > {config.max_latency_p99_s:.3f}")

    metrics = {
        "sample_size": report.sample_size,
        "error_rate": report.error_rate,
        "harness_error_rate": report.harness_error_rate,
        "latency_p99_s": report.latency_p99_s,
    }
    if reasons:
        return GateResult(_GATE_NAME, GateDecision.REJECT, "; ".join(reasons), metrics)
    return GateResult(_GATE_NAME, GateDecision.PASS, "canary health within thresholds", metrics)


def trigger_rollback(
    registry: ModelRegistry, model_id: str, canary_version: str, *, reason: str
) -> RegistryEntry:
    return registry.mark_rolled_back(model_id, canary_version, notes=reason)


__all__ = [
    "CanaryHealthReport",
    "CanaryHealthSample",
    "RollbackConfig",
    "evaluate_canary_health",
    "summarize_canary_health",
    "trigger_rollback",
]
