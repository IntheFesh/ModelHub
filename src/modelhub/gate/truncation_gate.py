"""Gate 5/5: output-truncation rate.

MODEL-SELECTION-FINAL.md's 陷阱 2: quantized models running with
thinking mode enabled can show a sharply elevated OUTPUT_TRUNCATED rate
(the reasoning trace eats the token budget before the SQL is emitted).
CLAUDE.md §2.4 already forbids counting a truncated output as a wrong
answer — this gate is the admission-time consequence of that same fact:
a candidate that frequently never finishes its SQL is not usable in
production regardless of what its (denominator-excluding-truncation)
accuracy number says.
"""

from __future__ import annotations

from modelhub.common.config import ModelHubBaseConfig
from modelhub.eval.metrics import EvalMetrics
from modelhub.gate.types import GateDecision, GateResult

_GATE_NAME = "truncation"


class TruncationGateConfig(ModelHubBaseConfig):
    max_output_truncated_rate: float


def check_truncation_gate(metrics: EvalMetrics, config: TruncationGateConfig) -> GateResult:
    if metrics.output_truncated_rate > config.max_output_truncated_rate:
        return GateResult(
            _GATE_NAME,
            GateDecision.REJECT,
            f"output_truncated_rate {metrics.output_truncated_rate:.2%} > allowed "
            f"{config.max_output_truncated_rate:.2%}",
            {
                "output_truncated_rate": metrics.output_truncated_rate,
                "max_allowed": config.max_output_truncated_rate,
            },
        )
    return GateResult(
        _GATE_NAME,
        GateDecision.PASS,
        f"output_truncated_rate {metrics.output_truncated_rate:.2%} <= allowed "
        f"{config.max_output_truncated_rate:.2%}",
        {
            "output_truncated_rate": metrics.output_truncated_rate,
            "max_allowed": config.max_output_truncated_rate,
        },
    )


__all__ = ["TruncationGateConfig", "check_truncation_gate"]
