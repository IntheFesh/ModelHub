"""Unit tests for train/dry_run.py."""

from __future__ import annotations

import pytest

from modelhub.train.dry_run import (
    MemoryEstimateConfig,
    assert_fits_in_memory,
    compute_token_length_stats,
    estimate_training_memory,
    validate_dataset_schema,
)


def _valid_record(sample_id: str) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "db_id": "school",
        "question": "how many students?",
        "evidence": None,
        "gold_sql": "SELECT COUNT(*) FROM students",
        "difficulty": "simple",
        "source": "bird",
        "split": "train",
        "dialect": "sqlite",
    }


class TestValidateDatasetSchema:
    def test_valid_records_all_parse(self) -> None:
        samples = validate_dataset_schema([_valid_record("s0"), _valid_record("s1")])
        assert len(samples) == 2

    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            validate_dataset_schema([])

    def test_one_bad_record_hard_fails_not_skips(self) -> None:
        bad = _valid_record("s1")
        del bad["gold_sql"]  # required field missing
        records = [_valid_record("s0"), bad, _valid_record("s2")]
        with pytest.raises(ValueError, match="index 1 failed schema validation"):
            validate_dataset_schema(records)


class TestComputeTokenLengthStats:
    def test_basic_stats(self) -> None:
        stats = compute_token_length_stats([100, 200, 300, 400, 500], max_seq_len=1000)
        assert stats.count == 5
        assert stats.min_tokens == 100
        assert stats.max_tokens == 500
        assert stats.mean_tokens == 300
        assert stats.over_max_seq_len_count == 0

    def test_over_max_seq_len_counted(self) -> None:
        stats = compute_token_length_stats([500, 1500, 2000], max_seq_len=1000)
        assert stats.over_max_seq_len_count == 2

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_token_length_stats([], max_seq_len=1000)

    def test_p99_is_nearest_rank(self) -> None:
        stats = compute_token_length_stats(list(range(1, 101)), max_seq_len=1000)
        assert stats.p99_tokens == 99


class TestEstimateTrainingMemory:
    def _config(self, **overrides: object) -> MemoryEstimateConfig:
        defaults: dict[str, object] = {
            "gpu_memory_bytes": 85899345920,
            "gpu_memory_utilization": 0.9,
            "base_model_weight_bytes": 19327352832,
            "lora_trainable_param_bytes": 209715200,
            "optimizer_state_multiplier": 2.0,
            "estimated_activation_bytes_per_token": 524288,
            "fixed_overhead_bytes": 2684354560,
        }
        defaults.update(overrides)
        return MemoryEstimateConfig.model_validate(defaults)

    def test_fits_within_a100_80g_at_reasonable_batch(self) -> None:
        estimate = estimate_training_memory(batch_size=4, max_seq_len=4096, config=self._config())
        assert estimate.fits is True
        assert estimate.headroom_pct > 0

    def test_does_not_fit_at_absurd_batch_size(self) -> None:
        estimate = estimate_training_memory(
            batch_size=100_000, max_seq_len=4096, config=self._config()
        )
        assert estimate.fits is False

    def test_assert_fits_raises_on_oom(self) -> None:
        estimate = estimate_training_memory(
            batch_size=100_000, max_seq_len=4096, config=self._config()
        )
        with pytest.raises(ValueError, match="exceeds the usable budget"):
            assert_fits_in_memory(estimate)

    def test_assert_fits_does_not_raise_when_it_fits(self) -> None:
        estimate = estimate_training_memory(batch_size=4, max_seq_len=4096, config=self._config())
        assert_fits_in_memory(estimate)  # must not raise

    def test_non_positive_batch_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            estimate_training_memory(batch_size=0, max_seq_len=4096, config=self._config())

    def test_headroom_pct_zero_usable_raises(self) -> None:
        estimate = estimate_training_memory(
            batch_size=1, max_seq_len=1, config=self._config(gpu_memory_bytes=0)
        )
        with pytest.raises(ValueError, match="cannot express headroom"):
            _ = estimate.headroom_pct
