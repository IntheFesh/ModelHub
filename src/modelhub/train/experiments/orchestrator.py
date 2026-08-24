"""Runs B2's six comparison groups with per-group failure isolation —
same shape as `bench/experiments/orchestrator.py` (A11), reusing its
`ExperimentGroupOutcome`/`run_experiment_group_isolated` directly rather
than re-deriving an equivalent isolation boundary for training groups.

This module does not itself launch real training subprocesses — no GPU
exists in this sandbox to run `llamafactory-cli`/`torchrun` against. Each
group is supplied as an already-configured, zero-argument callable that
performs that group's real short comparison run and returns
`(TrainingRunMetrics, RunManifest)`; a real dual-A100 entry point (out of
this round's scope) assembles those callables from the real
`run_*_comparison_group` functions in this package's other modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modelhub.common.run_manifest import RunManifest
from modelhub.train.experiments.common import (
    ExperimentGroupOutcome,
    check_training_experiment_precondition,
    run_experiment_group_isolated,
)


@dataclass(frozen=True)
class ComparisonSuiteReport:
    outcomes: tuple[ExperimentGroupOutcome[Any], ...]

    @property
    def completed_groups(self) -> tuple[str, ...]:
        return tuple(o.group_name for o in self.outcomes if o.status == "COMPLETED")

    @property
    def failed_groups(self) -> tuple[str, ...]:
        return tuple(o.group_name for o in self.outcomes if o.status == "FAILED")

    def outcome_for(self, group_name: str) -> ExperimentGroupOutcome[Any] | None:
        for outcome in self.outcomes:
            if outcome.group_name == group_name:
                return outcome
        return None


def run_all_comparison_groups(
    *,
    baseline_manifest: RunManifest,
    groups: dict[str, Callable[[], tuple[Any, RunManifest]]],
) -> ComparisonSuiteReport:
    """Check the shared pollution precondition once, then run every
    entry in `groups` (group_name -> zero-arg callable) isolated from
    the others — `groups` is a plain mapping (not a fixed six-tuple) so
    `make verify-b2-N` can exercise a single group through the same
    precondition/isolation/aggregation path a full six-group run uses,
    exactly as A11's orchestrator does for its six inference-side
    groups."""
    check_training_experiment_precondition(baseline_manifest)
    outcomes = tuple(run_experiment_group_isolated(name, run) for name, run in groups.items())
    return ComparisonSuiteReport(outcomes=outcomes)


__all__ = ["ComparisonSuiteReport", "run_all_comparison_groups"]
