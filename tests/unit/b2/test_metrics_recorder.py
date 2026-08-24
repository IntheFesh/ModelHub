"""Unit tests for train/experiments/metrics_recorder.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.train.experiments.metrics_recorder import (
    ExperimentMetricsCallback,
    RawExperimentMetrics,
    load_raw_experiment_metrics,
    require_measured_peak_memory,
)


class _FakeArgs:
    output_dir = "/unused"


class _FakeState:
    global_step = 1


class _FakeControl:
    pass


class TestLoadRawExperimentMetrics:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            load_raw_experiment_metrics(tmp_path / "no-such-file.json")

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment_metrics.json"
        path.write_text('{"peak_memory_bytes": 1000}')  # missing required fields
        with pytest.raises(Exception):  # pydantic ValidationError  # noqa: B017
            load_raw_experiment_metrics(path)

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment_metrics.json"
        raw = RawExperimentMetrics(
            peak_memory_bytes=12_000_000_000,
            tokens_per_step=16384,
            mean_step_time_s=2.5,
            trainable_param_count=50_000_000,
            num_steps=300,
        )
        path.write_text(raw.model_dump_json())
        loaded = load_raw_experiment_metrics(path)
        assert loaded.peak_memory_bytes == 12_000_000_000
        assert loaded.communication_time_ratio is None


class TestExperimentMetricsCallback:
    def test_records_step_timing_and_writes_a_valid_file(self, tmp_path: Path) -> None:
        output_path = tmp_path / "experiment_metrics.json"
        clock = _FakeClock()
        callback = ExperimentMetricsCallback(
            output_path=output_path,
            trainable_param_count=50_000_000,
            tokens_per_step=16384,
            now_fn=clock,
        )
        for _ in range(3):
            callback.on_step_begin(_FakeArgs(), _FakeState(), _FakeControl())
            clock.advance(2.0)
            callback.on_step_end(_FakeArgs(), _FakeState(), _FakeControl())
        callback.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())

        raw = load_raw_experiment_metrics(output_path)
        assert raw.num_steps == 3
        assert raw.mean_step_time_s == pytest.approx(2.0)
        assert raw.tokens_per_step == 16384
        assert raw.trainable_param_count == 50_000_000
        # this sandbox has no CUDA device — the real code path this
        # class exercises here is "torch/CUDA unavailable", not a real
        # measurement (see the class's own docstring). CLAUDE.md §2.4:
        # unmeasured is None, never a 0 that reads as "measured zero".
        assert raw.peak_memory_bytes is None

    def test_on_train_end_with_zero_steps_raises(self, tmp_path: Path) -> None:
        callback = ExperimentMetricsCallback(
            output_path=tmp_path / "experiment_metrics.json",
            trainable_param_count=1,
            tokens_per_step=1,
        )
        with pytest.raises(ValueError, match="no steps were recorded"):
            callback.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())


class TestRequireMeasuredPeakMemory:
    def _raw(self, **overrides: object) -> RawExperimentMetrics:
        defaults: dict[str, object] = {
            "peak_memory_bytes": 12_000_000_000,
            "tokens_per_step": 16384,
            "mean_step_time_s": 2.0,
            "trainable_param_count": 50_000_000,
            "num_steps": 300,
        }
        defaults.update(overrides)
        return RawExperimentMetrics.model_validate(defaults)

    def test_returns_the_real_value_when_present(self) -> None:
        assert require_measured_peak_memory(self._raw()) == 12_000_000_000

    def test_raises_when_unmeasured(self) -> None:
        with pytest.raises(ValueError, match="was not measured"):
            require_measured_peak_memory(self._raw(peak_memory_bytes=None))


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
