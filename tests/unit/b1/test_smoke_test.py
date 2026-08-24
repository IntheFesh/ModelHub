"""Unit tests for train/smoke_test.py — the mandatory 50-step
pre-long-run check (PLAN.md B1: three criteria, any one failing forbids
a long run)."""

from __future__ import annotations

import pytest

from modelhub.train.smoke_test import (
    SmokeTestCriteria,
    assert_smoke_test_passed,
    check_peak_memory,
    check_resume_curve_matches,
)


class TestCheckPeakMemory:
    def test_below_ninety_percent_ok(self) -> None:
        assert check_peak_memory(peak_memory_bytes=50, usable_bytes=100) is True

    def test_at_ninety_percent_not_ok(self) -> None:
        assert check_peak_memory(peak_memory_bytes=90, usable_bytes=100) is False

    def test_above_ninety_percent_not_ok(self) -> None:
        assert check_peak_memory(peak_memory_bytes=95, usable_bytes=100) is False

    def test_non_positive_usable_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            check_peak_memory(peak_memory_bytes=1, usable_bytes=0)

    def test_negative_peak_memory_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            check_peak_memory(peak_memory_bytes=-1, usable_bytes=100)


class TestCheckResumeCurveMatches:
    def test_identical_curves_match(self) -> None:
        assert check_resume_curve_matches([1.0, 0.9, 0.8], [1.0, 0.9, 0.8], tolerance=1e-6) is True

    def test_within_tolerance_matches(self) -> None:
        assert check_resume_curve_matches([1.0, 0.9], [1.001, 0.901], tolerance=0.01) is True

    def test_outside_tolerance_does_not_match(self) -> None:
        assert check_resume_curve_matches([1.0, 0.9], [1.5, 0.9], tolerance=0.01) is False

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="different length"):
            check_resume_curve_matches([1.0, 0.9], [1.0], tolerance=0.01)

    def test_empty_curves_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            check_resume_curve_matches([], [], tolerance=0.01)


class TestSmokeTestCriteria:
    def test_all_pass_means_passed(self) -> None:
        criteria = SmokeTestCriteria(
            peak_memory_ok=True, checkpoint_atomic_ok=True, resume_curve_matches=True
        )
        assert criteria.passed is True
        assert criteria.failure_reasons() == []

    def test_any_single_failure_fails_the_whole_thing(self) -> None:
        criteria = SmokeTestCriteria(
            peak_memory_ok=True, checkpoint_atomic_ok=False, resume_curve_matches=True
        )
        assert criteria.passed is False
        assert len(criteria.failure_reasons()) == 1

    def test_all_three_failures_all_named(self) -> None:
        criteria = SmokeTestCriteria(
            peak_memory_ok=False, checkpoint_atomic_ok=False, resume_curve_matches=False
        )
        assert len(criteria.failure_reasons()) == 3


class TestAssertSmokeTestPassed:
    def test_passing_criteria_does_not_raise(self) -> None:
        criteria = SmokeTestCriteria(
            peak_memory_ok=True, checkpoint_atomic_ok=True, resume_curve_matches=True
        )
        assert_smoke_test_passed(criteria)  # must not raise

    def test_failing_criteria_raises_with_reasons(self) -> None:
        criteria = SmokeTestCriteria(
            peak_memory_ok=False, checkpoint_atomic_ok=True, resume_curve_matches=True
        )
        with pytest.raises(ValueError, match="peak_memory"):
            assert_smoke_test_passed(criteria)
