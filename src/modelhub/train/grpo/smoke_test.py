"""PLAN.md ★: "先跑 10 步烟测确认显存与流水线，再挂长跑." Two criteria,
matching B1's `train/smoke_test.py::SmokeTestCriteria` shape (same "any
one failing forbids a long run" discipline) but GRPO-specific: peak
memory (reuses B1's `check_peak_memory` unchanged — the check itself
doesn't care whether the run is SFT or GRPO) and pipeline completion —
did the rollout->reward->group-diagnostics->step-diagnostics chain
actually run for the target step count without the harness-error-rate
abort firing.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.train.smoke_test import check_peak_memory

__all__ = ["GrpoSmokeCriteria", "assert_grpo_smoke_passed", "check_peak_memory"]


@dataclass(frozen=True)
class GrpoSmokeCriteria:
    peak_memory_ok: bool
    target_steps: int
    completed_steps: int
    aborted_on_harness_error_flood: bool

    def __post_init__(self) -> None:
        if self.target_steps <= 0:
            raise ValueError(f"target_steps must be positive, got {self.target_steps}")
        if self.completed_steps < 0:
            raise ValueError(f"completed_steps must be non-negative, got {self.completed_steps}")

    @property
    def pipeline_completed(self) -> bool:
        return not self.aborted_on_harness_error_flood and self.completed_steps >= self.target_steps

    @property
    def passed(self) -> bool:
        return self.peak_memory_ok and self.pipeline_completed

    def failure_reasons(self) -> list[str]:
        reasons = []
        if not self.peak_memory_ok:
            reasons.append("peak_memory reached >= 90% of usable GPU memory")
        if self.aborted_on_harness_error_flood:
            reasons.append("training aborted on harness-error-rate flood before target_steps")
        elif self.completed_steps < self.target_steps:
            reasons.append(
                f"only completed {self.completed_steps}/{self.target_steps} smoke-test steps"
            )
        return reasons


def assert_grpo_smoke_passed(criteria: GrpoSmokeCriteria) -> None:
    if not criteria.passed:
        reasons = "; ".join(criteria.failure_reasons())
        raise ValueError(
            f"GRPO 10-step smoke test failed: {reasons} — refusing to start a long GRPO "
            f"run (PLAN.md: run a 10-step smoke test to confirm memory and pipeline before "
            f"hanging a long run)"
        )
