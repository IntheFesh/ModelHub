"""Gate 1/5: absolute execution-accuracy floor.

Catches an underfit candidate (PLAN.md's "ckpt-A" drill checkpoint —
early SFT checkpoint, real underfitting, not a fabricated failure case)
outright, independent of whatever model it's being compared against.
"""

from __future__ import annotations

from modelhub.common.config import ModelHubBaseConfig
from modelhub.eval.metrics import EvalMetrics
from modelhub.gate.types import GateDecision, GateResult

_GATE_NAME = "accuracy"


class AccuracyGateConfig(ModelHubBaseConfig):
    min_execution_accuracy: float


def check_accuracy_gate(metrics: EvalMetrics, config: AccuracyGateConfig) -> GateResult:
    if metrics.execution_accuracy is None:
        return GateResult(
            _GATE_NAME,
            GateDecision.REJECT,
            "execution_accuracy is None (denominator is 0) — a model with no "
            "measurable accuracy cannot be admitted",
            {"execution_accuracy": None, "denominator": metrics.denominator},
        )
    if metrics.execution_accuracy < config.min_execution_accuracy:
        return GateResult(
            _GATE_NAME,
            GateDecision.REJECT,
            f"execution_accuracy {metrics.execution_accuracy:.2%} < required "
            f"{config.min_execution_accuracy:.2%}",
            {
                "execution_accuracy": metrics.execution_accuracy,
                "min_required": config.min_execution_accuracy,
            },
        )
    return GateResult(
        _GATE_NAME,
        GateDecision.PASS,
        f"execution_accuracy {metrics.execution_accuracy:.2%} >= required "
        f"{config.min_execution_accuracy:.2%}",
        {
            "execution_accuracy": metrics.execution_accuracy,
            "min_required": config.min_execution_accuracy,
        },
    )


__all__ = ["AccuracyGateConfig", "check_accuracy_gate"]
