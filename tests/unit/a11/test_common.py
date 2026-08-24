"""Unit tests for bench/experiments/common.py: the shared serving
precondition and per-group failure isolation every group builds on."""

from __future__ import annotations

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.bench.experiments.common import (
    ExperimentGroupOutcome,
    check_serving_precondition,
    run_experiment_group_isolated,
)
from modelhub.common.errors import ManifestError
from modelhub.common.run_manifest import RunManifest


class TestCheckServingPrecondition:
    def test_clean_manifest_passes(self) -> None:
        check_serving_precondition(make_valid_manifest())

    def test_contaminated_manifest_refused(self) -> None:
        with pytest.raises(ManifestError, match="refusing to use"):
            check_serving_precondition(make_valid_manifest(contaminated=True))

    def test_degraded_kernel_status_refused(self) -> None:
        manifest = make_valid_manifest(
            kernel_status={"causal_conv1d": False, "fla": False, "degraded": True}
        )
        with pytest.raises(ManifestError, match="refusing to use"):
            check_serving_precondition(manifest)

    def test_dirty_git_state_also_refused(self) -> None:
        # not one of the two flags PLAN.md names explicitly, but
        # is_polluted's strict-superset reuse (see DD-0023) covers it too.
        with pytest.raises(ManifestError, match="refusing to use"):
            check_serving_precondition(make_valid_manifest(git_dirty=True))


class TestRunExperimentGroupIsolated:
    def test_successful_group_reports_completed(self) -> None:
        manifest = make_valid_manifest()

        def run() -> tuple[str, RunManifest]:
            return "ok", manifest

        outcome = run_experiment_group_isolated("quantization", run)
        assert outcome.status == "COMPLETED"
        assert outcome.result == "ok"
        assert outcome.manifest == manifest
        assert outcome.error is None

    def test_failing_group_is_isolated_not_raised(self) -> None:
        def run() -> tuple[str, RunManifest]:
            raise RuntimeError("simulated GPU-side failure")

        outcome = run_experiment_group_isolated("prefix_cache", run)
        assert outcome.status == "FAILED"
        assert outcome.result is None
        assert outcome.manifest is None
        assert outcome.error is not None
        assert "RuntimeError" in outcome.error
        assert "simulated GPU-side failure" in outcome.error

    def test_group_name_is_preserved_on_both_outcomes(self) -> None:
        def ok() -> tuple[str, RunManifest]:
            return "x", make_valid_manifest()

        def fail() -> tuple[str, RunManifest]:
            raise ValueError("boom")

        ok_outcome = run_experiment_group_isolated("engine_comparison", ok)
        fail_outcome = run_experiment_group_isolated("engine_comparison", fail)
        assert ok_outcome.group_name == "engine_comparison"
        assert fail_outcome.group_name == "engine_comparison"


class TestExperimentGroupOutcomeInvariants:
    def test_completed_outcome_requires_result_and_manifest(self) -> None:
        with pytest.raises(ValueError, match="must carry both result and manifest"):
            ExperimentGroupOutcome(
                group_name="g",
                status="COMPLETED",
                result=None,
                manifest=make_valid_manifest(),
                error=None,
            )

    def test_failed_outcome_forbids_result_or_manifest(self) -> None:
        with pytest.raises(ValueError, match="must carry only an error"):
            ExperimentGroupOutcome(
                group_name="g", status="FAILED", result="oops", manifest=None, error="boom"
            )

    def test_failed_outcome_requires_error(self) -> None:
        with pytest.raises(ValueError, match="must carry only an error"):
            ExperimentGroupOutcome(
                group_name="g", status="FAILED", result=None, manifest=None, error=None
            )
