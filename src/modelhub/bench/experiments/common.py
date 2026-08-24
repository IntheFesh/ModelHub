"""Shared infrastructure for `bench/experiments/`: the six-group
inference-optimization experiment suite PLAN.md calls A11
("五组实验...各自写 manifest" — a sixth, MFU/MBU, was added in the v2
rewrite; every group in this package follows the same shape).

Every group module in this package:
  1. loads a Config from `configs/bench/experiments/<name>.yaml`,
  2. measures something real against live inference-serving
     infrastructure (or a fake `ModelClient`, in tests) — never a
     substituted/simulated result when that infrastructure is
     unavailable (CLAUDE.md §1.3),
  3. returns a frozen Result dataclass plus a `RunManifest`.

This module supplies the two things every group needs and none of them
should reimplement: the "target serving instance must not be
degraded/contaminated" precondition PLAN.md states once, up front, for
the whole suite, and the per-group failure isolation
(`run_experiment_group_isolated`) that lets `orchestrator.py` run all six
without one group's exception taking down the rest — see DD-0023.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from modelhub.common.logging import get_logger, log_exception
from modelhub.common.run_manifest import RunManifest, assert_not_polluted

_logger = get_logger("bench.experiments")


def check_serving_precondition(serving_instance_manifest: RunManifest) -> None:
    """Refuse to run any experiment against a target serving instance
    whose own manifest is polluted (PLAN.md: "目标服务实例
    degraded/contaminated 任一为 true → 拒绝执行").

    Reuses `RunManifest.is_polluted`/`assert_not_polluted` (A0) rather
    than re-checking only the two named flags — `is_polluted` is
    `degraded OR contaminated OR git_dirty`, a strict superset, and there
    is no scenario where PLAN.md would want an experiment run against a
    dirty-repo serving build to proceed just because it happens not to be
    degraded/contaminated either (see DD-0023).
    """
    assert_not_polluted(serving_instance_manifest, purpose="bench/experiments precondition")


T = TypeVar("T")

GroupStatus = Literal["COMPLETED", "FAILED"]


@dataclass(frozen=True)
class ExperimentGroupOutcome(Generic[T]):
    """One experiment group's outcome, as the orchestrator sees it.

    `result`/`manifest` are populated together on COMPLETED; `error` is
    populated alone on FAILED. Never a fabricated partial result — a
    failed group reports nothing but the failure."""

    group_name: str
    status: GroupStatus
    result: T | None
    manifest: RunManifest | None
    error: str | None

    def __post_init__(self) -> None:
        if self.status == "COMPLETED" and (self.result is None or self.manifest is None):
            raise ValueError(
                f"{self.group_name}: COMPLETED outcome must carry both result and manifest"
            )
        if self.status == "FAILED" and (
            self.result is not None or self.manifest is not None or self.error is None
        ):
            raise ValueError(
                f"{self.group_name}: FAILED outcome must carry only an error, no result/manifest"
            )


def run_experiment_group_isolated(
    group_name: str, run: Callable[[], tuple[T, RunManifest]]
) -> ExperimentGroupOutcome[T]:
    """Run one experiment group's `run` callable, catching any exception
    so a single group's failure never blocks the other five/six (PLAN.md:
    "每组独立、失败不阻断").

    This is a deliberate, documented isolation boundary — not the
    swallow-and-return-a-default pattern CLAUDE.md §1.1 forbids. The
    failure is never hidden or converted into a fake success: it is
    logged with its full traceback (`log_exception`, CLAUDE.md §8) and
    surfaced to the caller as an explicit `FAILED` outcome carrying the
    real exception's type and message, not silently absorbed.
    """
    try:
        result, manifest = run()
    except Exception as e:
        log_exception(_logger, f"experiment group {group_name!r} failed", e, group=group_name)
        return ExperimentGroupOutcome(
            group_name=group_name,
            status="FAILED",
            result=None,
            manifest=None,
            error=f"{type(e).__name__}: {e}",
        )
    return ExperimentGroupOutcome(
        group_name=group_name, status="COMPLETED", result=result, manifest=manifest, error=None
    )


__all__ = [
    "ExperimentGroupOutcome",
    "GroupStatus",
    "check_serving_precondition",
    "run_experiment_group_isolated",
]
