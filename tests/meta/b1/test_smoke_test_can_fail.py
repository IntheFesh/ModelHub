"""CLAUDE.md §1.5 required meta-test: proves B1's mandatory 50-step
smoke-test gate (train/smoke_test.py) actually rejects each of the
three failure scenarios PLAN.md names individually — "三项任一失败，
禁止挂长跑" — rather than only ever seeing the happy path in tests."""

from __future__ import annotations

import pytest

from modelhub.train.smoke_test import (
    SmokeTestCriteria,
    assert_smoke_test_passed,
    check_peak_memory,
    check_resume_curve_matches,
)


def test_peak_memory_over_budget_fails_the_gate() -> None:
    # a real measured peak >= 90% of usable GPU memory must reject.
    peak_ok = check_peak_memory(peak_memory_bytes=76_000_000_000, usable_bytes=80_000_000_000)
    criteria = SmokeTestCriteria(
        peak_memory_ok=peak_ok, checkpoint_atomic_ok=True, resume_curve_matches=True
    )
    assert peak_ok is False
    with pytest.raises(ValueError, match="peak_memory reached"):
        assert_smoke_test_passed(criteria)


def test_diverging_resume_curve_fails_the_gate() -> None:
    # a resumed run whose loss curve diverges from the uninterrupted run
    # is exactly the failure --resume-from is supposed to rule out.
    uninterrupted = [1.20, 1.05, 0.95, 0.88]
    resumed_but_diverged = [1.20, 1.05, 1.40, 1.55]  # optimizer/scheduler state lost on resume
    matches = check_resume_curve_matches(uninterrupted, resumed_but_diverged, tolerance=0.02)
    criteria = SmokeTestCriteria(
        peak_memory_ok=True, checkpoint_atomic_ok=True, resume_curve_matches=matches
    )
    assert matches is False
    with pytest.raises(ValueError, match="resumed loss curve diverged"):
        assert_smoke_test_passed(criteria)


def test_non_atomic_checkpoint_fails_the_gate() -> None:
    criteria = SmokeTestCriteria(
        peak_memory_ok=True, checkpoint_atomic_ok=False, resume_curve_matches=True
    )
    with pytest.raises(ValueError, match="atomic-write completeness"):
        assert_smoke_test_passed(criteria)


def test_all_three_healthy_is_not_permanently_red() -> None:
    """The gate is not stuck red: a genuinely healthy 50-step run passes."""
    peak_ok = check_peak_memory(peak_memory_bytes=40_000_000_000, usable_bytes=80_000_000_000)
    curve_ok = check_resume_curve_matches([1.2, 1.0, 0.9], [1.2, 1.0, 0.9], tolerance=1e-6)
    criteria = SmokeTestCriteria(
        peak_memory_ok=peak_ok, checkpoint_atomic_ok=True, resume_curve_matches=curve_ok
    )
    assert_smoke_test_passed(criteria)  # must not raise
