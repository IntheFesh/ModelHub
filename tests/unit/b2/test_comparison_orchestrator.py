"""Unit tests for train/experiments/orchestrator.py."""

from __future__ import annotations

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.common.errors import ModelHubError
from modelhub.train.experiments.orchestrator import run_all_comparison_groups


def _ok_group() -> tuple[str, object]:
    return "some real comparison result", make_valid_manifest()


def _failing_group() -> tuple[str, object]:
    raise RuntimeError("simulated group failure")


class TestRunAllComparisonGroups:
    def test_rejects_a_polluted_baseline_before_running_any_group(self) -> None:
        with pytest.raises(ModelHubError):
            run_all_comparison_groups(
                baseline_manifest=make_valid_manifest(contaminated=True),
                groups={"peft_method": _ok_group},
            )

    def test_all_groups_completed(self) -> None:
        report = run_all_comparison_groups(
            baseline_manifest=make_valid_manifest(),
            groups={"peft_method": _ok_group, "kernel_optimization": _ok_group},
        )
        assert set(report.completed_groups) == {"peft_method", "kernel_optimization"}
        assert report.failed_groups == ()

    def test_one_group_failing_does_not_block_the_others(self) -> None:
        report = run_all_comparison_groups(
            baseline_manifest=make_valid_manifest(),
            groups={"peft_method": _ok_group, "distributed_strategy": _failing_group},
        )
        assert report.completed_groups == ("peft_method",)
        assert report.failed_groups == ("distributed_strategy",)
        failed_outcome = report.outcome_for("distributed_strategy")
        assert failed_outcome is not None
        assert "simulated group failure" in (failed_outcome.error or "")

    def test_outcome_for_unknown_group_returns_none(self) -> None:
        report = run_all_comparison_groups(
            baseline_manifest=make_valid_manifest(), groups={"peft_method": _ok_group}
        )
        assert report.outcome_for("no-such-group") is None
