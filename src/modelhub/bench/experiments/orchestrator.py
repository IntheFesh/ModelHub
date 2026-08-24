"""Runs bench/experiments/'s groups end to end with per-group failure
isolation (PLAN.md: "五组实验...每组独立、失败不阻断、各自写 manifest" —
the v2 rewrite adds a sixth, MFU/MBU).

This module does not construct real inference-serving infrastructure
itself — no GPU exists in this sandbox to serve a model against. Each
group is supplied as an already-configured, zero-argument callable that
performs that group's full measurement and returns `(Result,
RunManifest)`; a real GPU-machine entry point (out of this round's scope
— see docs/design-decisions.md) is what will assemble those callables
from real `HttpModelClient`s pointed at a live vLLM/SGLang instance. This
module's job is exactly three things regardless of what backs each
callable: the shared precondition check, per-group isolation, and
aggregation into one report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from modelhub.bench.experiments.common import (
    ExperimentGroupOutcome,
    check_serving_precondition,
    run_experiment_group_isolated,
)
from modelhub.common.run_manifest import RunManifest


@dataclass(frozen=True)
class ExperimentSuiteReport:
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


def run_all_experiments(
    *,
    serving_instance_manifest: RunManifest,
    groups: dict[str, Callable[[], tuple[Any, RunManifest]]],
) -> ExperimentSuiteReport:
    """Check the shared degraded/contaminated precondition once, then run
    every entry in `groups` (group_name -> zero-arg callable) isolated
    from the others. `groups` is a plain mapping rather than a fixed
    six-tuple so a caller can run a subset — e.g. `make verify-a11-3`
    exercising only `speculative_decoding` — through the same
    precondition/isolation/aggregation path a full six-group run uses."""
    check_serving_precondition(serving_instance_manifest)
    outcomes = tuple(run_experiment_group_isolated(name, run) for name, run in groups.items())
    return ExperimentSuiteReport(outcomes=outcomes)


__all__ = ["ExperimentSuiteReport", "run_all_experiments"]
