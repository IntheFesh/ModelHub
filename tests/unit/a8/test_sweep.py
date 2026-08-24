"""Unit tests for bench/sweep.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.sweep import SweepCell, SweepGridConfig, run_sweep
from modelhub.common.errors import ModelHubError

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "bench" / "sweep_grid.yaml"


def test_sweep_grid_config_loads_from_real_yaml() -> None:
    from modelhub.common.config import load_yaml_config
    from modelhub.common.errors import Stage

    config, config_hash = load_yaml_config(SweepGridConfig, _CONFIG_PATH, stage=Stage.SERVE)
    assert config.concurrency_levels
    assert config.prompt_length_labels
    assert config_hash.startswith("sha256:")


def test_run_sweep_refuses_to_start_without_gpu_guard_bypass() -> None:
    client = BenchFakeClient()
    cells = [SweepCell(concurrency=1, prompt_length_label="1k", prompt="hello")]
    with pytest.raises(ModelHubError):
        run_sweep(
            client,
            model_id="test-model",
            cells=cells,
            num_requests_per_cell=2,
            max_tokens=16,
            temperature=0.0,
            generate_timeout_s=5.0,
        )


def test_run_sweep_produces_one_result_per_cell() -> None:
    client = BenchFakeClient()
    cells = [
        SweepCell(concurrency=1, prompt_length_label="1k", prompt="a" * 100),
        SweepCell(concurrency=2, prompt_length_label="3k", prompt="a" * 300),
    ]
    report = run_sweep(
        client,
        model_id="test-model",
        cells=cells,
        num_requests_per_cell=3,
        max_tokens=16,
        temperature=0.0,
        generate_timeout_s=5.0,
        skip_gpu_guard=True,
    )
    assert report.model_id == "test-model"
    assert len(report.results) == 2
    assert report.gpu_exclusivity_check is None

    r1 = report.result_for(concurrency=1, prompt_length_label="1k")
    assert r1 is not None
    assert r1.succeeded == 3

    r2 = report.result_for(concurrency=2, prompt_length_label="3k")
    assert r2 is not None
    assert r2.succeeded == 3

    assert report.result_for(concurrency=99, prompt_length_label="1k") is None
