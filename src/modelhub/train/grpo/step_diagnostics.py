"""PLAN.md ★ items 3-4: "每步记录：系统错占比/超时占比/截断占比/reward
分布直方图" and "系统错占比 > 阈值 → 中断训练并告警".

The abort threshold reuses `eval/metrics.py::HARNESS_ERROR_FLOOD_
THRESHOLD` (1%, CLAUDE.md §2.2's own literal "占比 > 1% → 评估/训练
中止") rather than inventing a second, GRPO-specific number — this is
the same real hazard (system faults contaminating a training/eval
signal) CLAUDE.md already named a project-wide threshold for.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from modelhub.common.errors import ErrorCode
from modelhub.eval.metrics import HARNESS_ERROR_FLOOD_THRESHOLD
from modelhub.eval.records import PredictionRecord
from modelhub.train.grpo.group_diagnostics import GroupDiagnostics


@dataclass(frozen=True)
class StepDiagnostics:
    step: int
    total_rollouts: int
    harness_error_count: int
    timeout_count: int
    truncated_count: int
    reward_histogram: dict[str, int]
    degenerate_group_count: int
    total_group_count: int

    def __post_init__(self) -> None:
        if self.total_rollouts <= 0:
            raise ValueError(f"step {self.step}: total_rollouts must be positive")
        if self.total_group_count <= 0:
            raise ValueError(f"step {self.step}: total_group_count must be positive")

    @property
    def harness_error_rate(self) -> float:
        return self.harness_error_count / self.total_rollouts

    @property
    def timeout_rate(self) -> float:
        return self.timeout_count / self.total_rollouts

    @property
    def truncated_rate(self) -> float:
        return self.truncated_count / self.total_rollouts

    @property
    def degenerate_group_rate(self) -> float:
        return self.degenerate_group_count / self.total_group_count


def compute_step_diagnostics(
    step: int,
    predictions: Sequence[PredictionRecord],
    group_diagnostics: Sequence[GroupDiagnostics],
) -> StepDiagnostics:
    if not predictions:
        raise ValueError(f"step {step}: cannot compute diagnostics over zero predictions")
    if not group_diagnostics:
        raise ValueError(f"step {step}: cannot compute diagnostics over zero groups")

    histogram = dict(Counter(p.exec_code.value for p in predictions))
    return StepDiagnostics(
        step=step,
        total_rollouts=len(predictions),
        harness_error_count=sum(1 for p in predictions if p.is_harness_error),
        timeout_count=sum(1 for p in predictions if p.exec_code is ErrorCode.TIMEOUT),
        truncated_count=sum(1 for p in predictions if p.is_output_truncated),
        reward_histogram=histogram,
        degenerate_group_count=sum(1 for g in group_diagnostics if g.is_degenerate),
        total_group_count=len(group_diagnostics),
    )


def assert_harness_error_rate_ok(
    diagnostics: StepDiagnostics, *, threshold: float = HARNESS_ERROR_FLOOD_THRESHOLD
) -> None:
    if diagnostics.harness_error_rate > threshold:
        raise ValueError(
            f"step {diagnostics.step}: harness_error_rate "
            f"{diagnostics.harness_error_rate:.2%} exceeds the allowed {threshold:.0%} — "
            f"aborting GRPO training (CLAUDE.md §2.2: 占比 > 1% → 训练中止). This means the "
            f"sandbox/harness itself is failing, not the model — continuing would train the "
            f"policy on noise while the reward curve keeps looking normal."
        )


__all__ = [
    "StepDiagnostics",
    "assert_harness_error_rate_ok",
    "compute_step_diagnostics",
]
