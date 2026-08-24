"""Example night-queue task wiring — `scripts/night_queue.py`'s default
queue when run via `make night-queue`.

Deliberately excluded from `mypy --strict` (pyproject.toml) — this
file's whole point is being a template an operator copies and edits for
their own real nightly queue, not a strictly-typed library module.

Every task body here is real and safe to run in any environment
(CPU-only, no GPU/network required): the LOW-risk tasks are this
project's own existing deterministic checks (CLAUDE.md §6.2: "确定性
任务先跑" — `check_no_cheating.py`/`check_placeholders.py` are exactly
that shape, real quantization/eval-style deterministic checks, not
invented placeholders). The one HIGH-risk task illustrates what a real
GPU-dependent task's `run()` looks like: it calls a real availability
check (B5's `check_verl_available`) and raises honestly when the
dependency isn't installed, rather than pretending to train something.
"""

from __future__ import annotations

import subprocess
import sys

from modelhub.common.capability import is_available
from modelhub.train.grpo.runner import check_verl_available
from night_queue import NightTask, TaskRiskLevel

_SUBPROCESS_TIMEOUT_S = 120.0


def _run_check_no_cheating() -> None:
    subprocess.run(
        [sys.executable, "scripts/check_no_cheating.py", "src"],
        check=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )


def _run_check_placeholders() -> None:
    subprocess.run(
        [sys.executable, "scripts/check_placeholders.py", "docs", "src"],
        check=True,
        timeout=_SUBPROCESS_TIMEOUT_S,
    )


def _run_grpo_training_placeholder() -> None:
    """Illustrates a real HIGH-risk, GPU-dependent task: it honestly
    fails (raises) rather than pretending to train — the same
    availability check `train/grpo/runner.py::run_grpo_training` itself
    gates on."""
    if not is_available(check_verl_available()):
        raise RuntimeError(
            "verl is not installed in this environment — a real nightly GRPO task "
            "would need it; this placeholder task fails honestly rather than "
            "pretending to have trained something"
        )
    raise NotImplementedError("wire this to the real train.grpo.runner.run_grpo_training call")


def build_example_night_tasks() -> list[NightTask]:
    return [
        NightTask(
            task_id="check-no-cheating",
            risk_level=TaskRiskLevel.LOW,
            estimated_duration_s=30.0,
            run=_run_check_no_cheating,
        ),
        NightTask(
            task_id="check-placeholders",
            risk_level=TaskRiskLevel.LOW,
            estimated_duration_s=15.0,
            run=_run_check_placeholders,
        ),
        NightTask(
            task_id="grpo-training-example",
            risk_level=TaskRiskLevel.HIGH,
            estimated_duration_s=8 * 3600.0,
            run=_run_grpo_training_placeholder,
            watchdog_timeout_s=60.0,
        ),
    ]


__all__ = ["build_example_night_tasks"]
