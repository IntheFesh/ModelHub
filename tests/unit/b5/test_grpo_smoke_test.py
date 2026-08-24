"""Unit tests for train/grpo/smoke_test.py."""

from __future__ import annotations

import pytest

from modelhub.train.grpo.smoke_test import GrpoSmokeCriteria, assert_grpo_smoke_passed


class TestGrpoSmokeCriteria:
    def test_all_healthy_passes(self) -> None:
        criteria = GrpoSmokeCriteria(
            peak_memory_ok=True,
            target_steps=10,
            completed_steps=10,
            aborted_on_harness_error_flood=False,
        )
        assert criteria.passed is True
        assert criteria.failure_reasons() == []

    def test_peak_memory_failure(self) -> None:
        criteria = GrpoSmokeCriteria(
            peak_memory_ok=False,
            target_steps=10,
            completed_steps=10,
            aborted_on_harness_error_flood=False,
        )
        assert criteria.passed is False
        assert "peak_memory" in criteria.failure_reasons()[0]

    def test_harness_flood_abort_fails_even_with_headroom(self) -> None:
        criteria = GrpoSmokeCriteria(
            peak_memory_ok=True,
            target_steps=10,
            completed_steps=3,
            aborted_on_harness_error_flood=True,
        )
        assert criteria.passed is False
        assert "aborted" in criteria.failure_reasons()[0]

    def test_incomplete_steps_without_abort_fails(self) -> None:
        criteria = GrpoSmokeCriteria(
            peak_memory_ok=True,
            target_steps=10,
            completed_steps=7,
            aborted_on_harness_error_flood=False,
        )
        assert criteria.passed is False
        assert "7/10" in criteria.failure_reasons()[0]

    def test_non_positive_target_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="target_steps"):
            GrpoSmokeCriteria(
                peak_memory_ok=True,
                target_steps=0,
                completed_steps=0,
                aborted_on_harness_error_flood=False,
            )


class TestAssertGrpoSmokePassed:
    def test_passing_does_not_raise(self) -> None:
        criteria = GrpoSmokeCriteria(
            peak_memory_ok=True,
            target_steps=10,
            completed_steps=10,
            aborted_on_harness_error_flood=False,
        )
        assert_grpo_smoke_passed(criteria)  # must not raise

    def test_failing_raises(self) -> None:
        criteria = GrpoSmokeCriteria(
            peak_memory_ok=False,
            target_steps=10,
            completed_steps=10,
            aborted_on_harness_error_flood=False,
        )
        with pytest.raises(ValueError, match="GRPO 10-step smoke test failed"):
            assert_grpo_smoke_passed(criteria)
