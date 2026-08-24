"""Unit tests for train/experiments/common.py."""

from __future__ import annotations

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.common.errors import ModelHubError
from modelhub.train.experiments.common import (
    LayerTypeProfileSplit,
    TrainingRunMetrics,
    check_training_experiment_precondition,
    compute_throughput_tokens_per_s,
    compute_total_cost,
    guard_dedicated_gpus,
)


class TestLayerTypeProfileSplit:
    def test_shares_sum_correctly(self) -> None:
        split = LayerTypeProfileSplit(
            gdn_layer_memory_bytes=3,
            full_attention_layer_memory_bytes=1,
            gdn_layer_time_s=6.0,
            full_attention_layer_time_s=2.0,
        )
        assert split.gdn_memory_share == pytest.approx(0.75)
        assert split.gdn_time_share == pytest.approx(0.75)

    def test_negative_memory_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            LayerTypeProfileSplit(
                gdn_layer_memory_bytes=-1,
                full_attention_layer_memory_bytes=1,
                gdn_layer_time_s=1.0,
                full_attention_layer_time_s=1.0,
            )

    def test_negative_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            LayerTypeProfileSplit(
                gdn_layer_memory_bytes=1,
                full_attention_layer_memory_bytes=1,
                gdn_layer_time_s=-1.0,
                full_attention_layer_time_s=1.0,
            )

    def test_zero_total_memory_raises_on_share(self) -> None:
        split = LayerTypeProfileSplit(
            gdn_layer_memory_bytes=0,
            full_attention_layer_memory_bytes=0,
            gdn_layer_time_s=1.0,
            full_attention_layer_time_s=1.0,
        )
        with pytest.raises(ValueError, match="memory share"):
            _ = split.gdn_memory_share


class TestTrainingRunMetrics:
    def _metrics(self, **overrides: object) -> TrainingRunMetrics:
        defaults: dict[str, object] = {
            "group_name": "test",
            "peak_memory_bytes": 1000,
            "throughput_tokens_per_s": 100.0,
            "step_time_s": 1.0,
            "trainable_param_count": 50_000_000,
            "total_cost": 1.0,
            "communication_time_ratio": None,
            "layer_type_split": None,
        }
        defaults.update(overrides)
        return TrainingRunMetrics(**defaults)  # type: ignore[arg-type]

    def test_valid_metrics_construct(self) -> None:
        metrics = self._metrics()
        assert metrics.communication_time_ratio is None

    def test_negative_peak_memory_rejected(self) -> None:
        with pytest.raises(ValueError, match="peak_memory_bytes"):
            self._metrics(peak_memory_bytes=-1)

    def test_non_positive_throughput_rejected(self) -> None:
        with pytest.raises(ValueError, match="throughput_tokens_per_s"):
            self._metrics(throughput_tokens_per_s=0)

    def test_non_positive_step_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="step_time_s"):
            self._metrics(step_time_s=0)

    def test_non_positive_trainable_param_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="trainable_param_count"):
            self._metrics(trainable_param_count=0)

    def test_negative_total_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_cost"):
            self._metrics(total_cost=-1.0)

    def test_communication_ratio_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="communication_time_ratio"):
            self._metrics(communication_time_ratio=1.5)

    def test_communication_ratio_in_range_accepted(self) -> None:
        metrics = self._metrics(communication_time_ratio=0.3)
        assert metrics.communication_time_ratio == 0.3


class TestComputeThroughputTokensPerS:
    def test_basic_computation(self) -> None:
        assert compute_throughput_tokens_per_s(tokens_per_step=16384, step_time_s=2.0) == 8192.0

    def test_non_positive_tokens_per_step_rejected(self) -> None:
        with pytest.raises(ValueError, match="tokens_per_step"):
            compute_throughput_tokens_per_s(tokens_per_step=0, step_time_s=1.0)

    def test_non_positive_step_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="step_time_s"):
            compute_throughput_tokens_per_s(tokens_per_step=100, step_time_s=0)


class TestComputeTotalCost:
    def test_basic_computation(self) -> None:
        # 300 steps * 2s = 600s = 1/6 hour; * $2/hr * 1 gpu = $0.333...
        cost = compute_total_cost(
            step_time_s=2.0, num_steps=300, gpu_cost_per_hour=2.0, gpu_count=1
        )
        assert cost == pytest.approx(1 / 3)

    def test_scales_with_gpu_count(self) -> None:
        cost_1gpu = compute_total_cost(
            step_time_s=2.0, num_steps=300, gpu_cost_per_hour=2.0, gpu_count=1
        )
        cost_2gpu = compute_total_cost(
            step_time_s=2.0, num_steps=300, gpu_cost_per_hour=2.0, gpu_count=2
        )
        assert cost_2gpu == pytest.approx(cost_1gpu * 2)

    def test_non_positive_step_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="step_time_s"):
            compute_total_cost(step_time_s=0, num_steps=1, gpu_cost_per_hour=1.0, gpu_count=1)

    def test_non_positive_num_steps_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_steps"):
            compute_total_cost(step_time_s=1.0, num_steps=0, gpu_cost_per_hour=1.0, gpu_count=1)

    def test_negative_gpu_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="gpu_cost_per_hour"):
            compute_total_cost(step_time_s=1.0, num_steps=1, gpu_cost_per_hour=-1.0, gpu_count=1)

    def test_non_positive_gpu_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="gpu_count"):
            compute_total_cost(step_time_s=1.0, num_steps=1, gpu_cost_per_hour=1.0, gpu_count=0)


class TestCheckTrainingExperimentPrecondition:
    def test_clean_manifest_passes(self) -> None:
        manifest = make_valid_manifest()
        check_training_experiment_precondition(manifest)  # must not raise

    def test_contaminated_manifest_rejected(self) -> None:
        manifest = make_valid_manifest(contaminated=True)
        with pytest.raises(ModelHubError):
            check_training_experiment_precondition(manifest)

    def test_dirty_git_manifest_rejected(self) -> None:
        manifest = make_valid_manifest(git_dirty=True)
        with pytest.raises(ModelHubError):
            check_training_experiment_precondition(manifest)


class TestGuardDedicatedGpus:
    def test_empty_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            guard_dedicated_gpus([])
