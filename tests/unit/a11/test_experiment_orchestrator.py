"""Unit tests for bench/experiments/orchestrator.py: precondition +
per-group isolation + aggregation over a set of (fake, in this test)
experiment-group callables."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.bench.experiments.orchestrator import run_all_experiments
from modelhub.common.errors import ManifestError
from modelhub.common.run_manifest import RunManifest


def test_precondition_blocks_the_whole_suite_before_any_group_runs() -> None:
    calls: list[str] = []

    def group() -> tuple[str, RunManifest]:
        calls.append("ran")
        return "x", make_valid_manifest()

    with pytest.raises(ManifestError, match="refusing to use"):
        run_all_experiments(
            serving_instance_manifest=make_valid_manifest(contaminated=True),
            groups={"quantization": group},
        )
    assert calls == []


def test_all_six_groups_run_and_aggregate() -> None:
    def make_group(name: str) -> Callable[[], tuple[str, RunManifest]]:
        def run() -> tuple[str, RunManifest]:
            return f"{name}-result", make_valid_manifest()

        return run

    group_names = [
        "quantization",
        "prefix_cache",
        "speculative_decoding",
        "constrained_decoding",
        "engine_comparison",
        "mfu_mbu_comparison",
    ]
    report = run_all_experiments(
        serving_instance_manifest=make_valid_manifest(),
        groups={name: make_group(name) for name in group_names},
    )
    assert set(report.completed_groups) == set(group_names)
    assert report.failed_groups == ()
    for name in group_names:
        outcome = report.outcome_for(name)
        assert outcome is not None
        assert outcome.result == f"{name}-result"


def test_one_failing_group_does_not_block_the_others() -> None:
    def ok_group() -> tuple[str, RunManifest]:
        return "fine", make_valid_manifest()

    def failing_group() -> tuple[str, RunManifest]:
        raise RuntimeError("no live vLLM server to hit")

    report = run_all_experiments(
        serving_instance_manifest=make_valid_manifest(),
        groups={"prefix_cache": ok_group, "speculative_decoding": failing_group},
    )
    assert report.completed_groups == ("prefix_cache",)
    assert report.failed_groups == ("speculative_decoding",)
    failed_outcome = report.outcome_for("speculative_decoding")
    assert failed_outcome is not None
    assert "no live vLLM server to hit" in (failed_outcome.error or "")


def test_outcome_for_unknown_group_returns_none() -> None:
    report = run_all_experiments(
        serving_instance_manifest=make_valid_manifest(),
        groups={"quantization": lambda: ("x", make_valid_manifest())},
    )
    assert report.outcome_for("no_such_group") is None
