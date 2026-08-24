"""Unit tests for train/experiments/report.py."""

from __future__ import annotations

import pytest

from modelhub.train.experiments.common import TrainingRunMetrics
from modelhub.train.experiments.report import (
    ComparisonEntry,
    MetricProvenance,
    format_effect_cost_sentence,
    render_selection_table,
)


def _metrics(**overrides: object) -> TrainingRunMetrics:
    defaults: dict[str, object] = {
        "group_name": "test",
        "peak_memory_bytes": 40_000_000_000,
        "throughput_tokens_per_s": 8000.0,
        "step_time_s": 2.0,
        "trainable_param_count": 50_000_000,
        "total_cost": 10.0,
        "communication_time_ratio": None,
        "layer_type_split": None,
    }
    defaults.update(overrides)
    return TrainingRunMetrics(**defaults)  # type: ignore[arg-type]


def _entry(**overrides: object) -> ComparisonEntry:
    defaults: dict[str, object] = {
        "variant_name": "LoRA r=32",
        "metrics": _metrics(),
        "num_steps": 300,
        "provenance": MetricProvenance.SHORT_RUN,
    }
    defaults.update(overrides)
    return ComparisonEntry(**defaults)  # type: ignore[arg-type]


class TestComparisonEntry:
    def test_total_time_s(self) -> None:
        entry = _entry(metrics=_metrics(step_time_s=2.0), num_steps=300)
        assert entry.total_time_s == pytest.approx(600.0)

    def test_non_positive_num_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_steps"):
            _entry(num_steps=0)


class TestRenderSelectionTable:
    def test_renders_a_row_per_entry(self) -> None:
        entries = [
            _entry(variant_name="LoRA r=32"),
            _entry(variant_name="Full+ZeRO-3", metrics=_metrics(peak_memory_bytes=70_000_000_000)),
        ]
        table = render_selection_table(entries)
        assert "LoRA r=32" in table
        assert "Full+ZeRO-3" in table
        assert table.count("\n") >= 3  # header + separator + 2 rows

    def test_empty_entries_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero entries"):
            render_selection_table([])


class TestFormatEffectCostSentence:
    def test_none_accuracy_states_pending_convergence_run(self) -> None:
        baseline = _entry(variant_name="Full+ZeRO-3", metrics=_metrics(peak_memory_bytes=70e9))
        candidate = _entry(variant_name="LoRA r=32", metrics=_metrics(peak_memory_bytes=40e9))
        sentence = format_effect_cost_sentence(
            baseline=baseline,
            candidate=candidate,
            accuracy_delta_points=None,
            conclusion="两周迭代一次的节奏选 LoRA。",
        )
        assert "尚无收敛跑数据" in sentence
        assert "两周迭代一次的节奏选 LoRA。" in sentence

    def test_real_accuracy_delta_is_formatted(self) -> None:
        baseline = _entry(variant_name="Full+ZeRO-3")
        candidate = _entry(variant_name="LoRA r=32")
        sentence = format_effect_cost_sentence(
            baseline=baseline, candidate=candidate, accuracy_delta_points=-1.5, conclusion="x"
        )
        assert "-1.5" in sentence

    def test_memory_savings_phrased_as_saved_not_increased(self) -> None:
        baseline = _entry(metrics=_metrics(peak_memory_bytes=70_000_000_000))
        candidate = _entry(metrics=_metrics(peak_memory_bytes=40_000_000_000))
        sentence = format_effect_cost_sentence(
            baseline=baseline, candidate=candidate, accuracy_delta_points=None, conclusion="x"
        )
        assert "显存峰值省" in sentence

    def test_never_uses_ablation_wording(self) -> None:
        baseline = _entry()
        candidate = _entry(variant_name="QLoRA")
        sentence = format_effect_cost_sentence(
            baseline=baseline, candidate=candidate, accuracy_delta_points=None, conclusion="x"
        )
        assert "消融" not in sentence
