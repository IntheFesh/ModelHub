"""The mandatory 50-step smoke test before any long training run —
PLAN.md B1: "开跑前必须先跑 50 步烟测，验证三项：显存峰值 < 可用×0.9 /
checkpoint 能原子落盘 / --resume-from 曲线接得上。三项任一失败禁止挂
长跑。半小时成本保一个通宵."

This module is the pure comparison/validation logic a real 50-step
smoke-test run (needing an actual GPU) would call with its real
measured values — it has no torch dependency itself and is fully
testable against synthetic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeTestCriteria:
    peak_memory_ok: bool
    checkpoint_atomic_ok: bool
    resume_curve_matches: bool

    @property
    def passed(self) -> bool:
        return self.peak_memory_ok and self.checkpoint_atomic_ok and self.resume_curve_matches

    def failure_reasons(self) -> list[str]:
        reasons = []
        if not self.peak_memory_ok:
            reasons.append("peak_memory reached >= 90% of usable GPU memory")
        if not self.checkpoint_atomic_ok:
            reasons.append("checkpoint failed atomic-write completeness validation")
        if not self.resume_curve_matches:
            reasons.append("resumed loss curve diverged from the uninterrupted run")
        return reasons


def check_peak_memory(peak_memory_bytes: int, usable_bytes: int) -> bool:
    if usable_bytes <= 0:
        raise ValueError(f"usable_bytes must be positive, got {usable_bytes}")
    if peak_memory_bytes < 0:
        raise ValueError(f"peak_memory_bytes must be non-negative, got {peak_memory_bytes}")
    return peak_memory_bytes < usable_bytes * 0.9


def check_resume_curve_matches(
    uninterrupted_losses: list[float], resumed_losses: list[float], *, tolerance: float
) -> bool:
    """Compare the loss curve of a run that continued past the
    interruption point against the same steps reproduced by
    `--resume-from` — PLAN.md B1 item 2: "测试断言续跑与不中断的 loss
    曲线一致"."""
    if len(uninterrupted_losses) != len(resumed_losses):
        raise ValueError(
            f"cannot compare curves of different length: "
            f"{len(uninterrupted_losses)} vs {len(resumed_losses)}"
        )
    if not uninterrupted_losses:
        raise ValueError("cannot compare empty loss curves")
    return all(
        abs(a - b) <= tolerance for a, b in zip(uninterrupted_losses, resumed_losses, strict=True)
    )


def assert_smoke_test_passed(criteria: SmokeTestCriteria) -> None:
    if not criteria.passed:
        reasons = "; ".join(criteria.failure_reasons())
        raise ValueError(
            f"50-step smoke test failed: {reasons} — refusing to start a long training "
            f"run (PLAN.md: any one of the three criteria failing forbids a long run)"
        )


__all__ = [
    "SmokeTestCriteria",
    "assert_smoke_test_passed",
    "check_peak_memory",
    "check_resume_curve_matches",
]
