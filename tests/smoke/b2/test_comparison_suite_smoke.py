"""Smoke test: a full B2 comparison-suite cycle through the real
orchestrator, precondition, per-group isolation, and report-rendering
path (CLAUDE.md §1.4 — every real code path runs; the only thing shrunk
is scale: 2 of the 6 real groups, each backed by a tiny synthetic
`RawExperimentMetrics` file standing in for llamafactory-cli's real
subprocess output — the actual subprocess launch itself is neither
available nor exercised in this GPU-less sandbox, exactly like A10's
smoke test stands in for a real canary traffic source)."""

from __future__ import annotations

from pathlib import Path

from tests.unit.a4.fakes import make_valid_manifest

from modelhub.common.atomic_io import atomic_write_json
from modelhub.train.experiments.common import (
    TrainingRunMetrics,
    compute_throughput_tokens_per_s,
    compute_total_cost,
)
from modelhub.train.experiments.metrics_recorder import (
    RawExperimentMetrics,
    load_raw_experiment_metrics,
)
from modelhub.train.experiments.orchestrator import run_all_comparison_groups
from modelhub.train.experiments.report import (
    ComparisonEntry,
    MetricProvenance,
    format_effect_cost_sentence,
    render_selection_table,
)


def _write_synthetic_metrics_file(path: Path, **overrides: object) -> None:
    defaults: dict[str, object] = {
        "peak_memory_bytes": 40_000_000_000,
        "tokens_per_step": 16384,
        "mean_step_time_s": 2.0,
        "trainable_param_count": 50_000_000,
        "num_steps": 300,
    }
    defaults.update(overrides)
    atomic_write_json(path, RawExperimentMetrics.model_validate(defaults).model_dump(mode="json"))


def test_two_group_comparison_cycle_end_to_end(tmp_path: Path) -> None:
    lora_metrics_path = tmp_path / "lora_experiment_metrics.json"
    full_metrics_path = tmp_path / "full_experiment_metrics.json"
    _write_synthetic_metrics_file(
        lora_metrics_path, peak_memory_bytes=40_000_000_000, num_steps=300
    )
    _write_synthetic_metrics_file(
        full_metrics_path,
        peak_memory_bytes=70_000_000_000,
        mean_step_time_s=3.5,
        trainable_param_count=9_000_000_000,
        num_steps=200,
    )

    def _lora_group() -> tuple[TrainingRunMetrics, object]:
        raw = load_raw_experiment_metrics(lora_metrics_path)
        metrics = TrainingRunMetrics(
            group_name="peft_method:LORA",
            peak_memory_bytes=raw.peak_memory_bytes,
            throughput_tokens_per_s=compute_throughput_tokens_per_s(
                tokens_per_step=raw.tokens_per_step, step_time_s=raw.mean_step_time_s
            ),
            step_time_s=raw.mean_step_time_s,
            trainable_param_count=raw.trainable_param_count,
            total_cost=compute_total_cost(
                step_time_s=raw.mean_step_time_s, num_steps=300, gpu_cost_per_hour=2.0, gpu_count=1
            ),
            communication_time_ratio=None,
            layer_type_split=None,
        )
        return metrics, make_valid_manifest()

    def _full_zero3_group() -> tuple[TrainingRunMetrics, object]:
        raw = load_raw_experiment_metrics(full_metrics_path)
        metrics = TrainingRunMetrics(
            group_name="peft_method:FULL_ZERO3",
            peak_memory_bytes=raw.peak_memory_bytes,
            throughput_tokens_per_s=compute_throughput_tokens_per_s(
                tokens_per_step=raw.tokens_per_step, step_time_s=raw.mean_step_time_s
            ),
            step_time_s=raw.mean_step_time_s,
            trainable_param_count=raw.trainable_param_count,
            total_cost=compute_total_cost(
                step_time_s=raw.mean_step_time_s, num_steps=200, gpu_cost_per_hour=2.0, gpu_count=1
            ),
            communication_time_ratio=None,
            layer_type_split=None,
        )
        return metrics, make_valid_manifest()

    report = run_all_comparison_groups(
        baseline_manifest=make_valid_manifest(),
        groups={"lora": _lora_group, "full_zero3": _full_zero3_group},
    )
    assert report.completed_groups == ("lora", "full_zero3")

    lora_outcome = report.outcome_for("lora")
    full_outcome = report.outcome_for("full_zero3")
    assert lora_outcome is not None and full_outcome is not None
    assert isinstance(lora_outcome.result, TrainingRunMetrics)
    assert isinstance(full_outcome.result, TrainingRunMetrics)

    baseline_entry = ComparisonEntry(
        variant_name="Full+ZeRO-3",
        metrics=full_outcome.result,
        num_steps=200,
        provenance=MetricProvenance.SHORT_RUN,
    )
    candidate_entry = ComparisonEntry(
        variant_name="LoRA r=32",
        metrics=lora_outcome.result,
        num_steps=300,
        provenance=MetricProvenance.SHORT_RUN,
    )
    table = render_selection_table([baseline_entry, candidate_entry])
    assert "LoRA r=32" in table
    assert "Full+ZeRO-3" in table

    sentence = format_effect_cost_sentence(
        baseline=baseline_entry,
        candidate=candidate_entry,
        accuracy_delta_points=None,
        conclusion="两周迭代一次的节奏选 LoRA。",
    )
    assert "尚无收敛跑数据" in sentence
    assert "显存峰值省" in sentence
