"""Pre-flight training validation — PLAN.md B1: "训练前 dry-run 校验：
数据 schema、token 长度分布、显存预估。OOM 要在第0步暴露，不是第800步."

Also covers item 7 ("数据加载失败硬失败，不 skip 坏样本继续训"):
`validate_dataset_schema` fails on the FIRST invalid record rather than
silently dropping it — a training set with a corrupt record is a data
pipeline bug (A1 should never have produced it), not something to paper
over by training on however many records happened to parse.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from modelhub.common.config import ModelHubBaseConfig
from modelhub.data.schema import NormalizedSample


def validate_dataset_schema(raw_records: Sequence[dict[str, object]]) -> list[NormalizedSample]:
    if not raw_records:
        raise ValueError("training dataset is empty — refusing to start a run with 0 samples")
    samples: list[NormalizedSample] = []
    for i, record in enumerate(raw_records):
        try:
            samples.append(NormalizedSample.model_validate(record))
        except ValidationError as e:
            raise ValueError(
                f"training record at index {i} failed schema validation — refusing to skip "
                f"it and continue with a smaller dataset (CLAUDE.md: 数据加载失败硬失败): {e}"
            ) from e
    return samples


@dataclass(frozen=True)
class TokenLengthStats:
    count: int
    min_tokens: int
    max_tokens: int
    mean_tokens: float
    p50_tokens: int
    p99_tokens: int
    over_max_seq_len_count: int


def _percentile(sorted_values: list[int], pct: float) -> int:
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def compute_token_length_stats(
    token_counts: Sequence[int], *, max_seq_len: int
) -> TokenLengthStats:
    if not token_counts:
        raise ValueError("cannot compute token-length stats over an empty dataset")
    sorted_counts = sorted(token_counts)
    return TokenLengthStats(
        count=len(sorted_counts),
        min_tokens=sorted_counts[0],
        max_tokens=sorted_counts[-1],
        mean_tokens=sum(sorted_counts) / len(sorted_counts),
        p50_tokens=_percentile(sorted_counts, 50),
        p99_tokens=_percentile(sorted_counts, 99),
        over_max_seq_len_count=sum(1 for c in token_counts if c > max_seq_len),
    )


class MemoryEstimateConfig(ModelHubBaseConfig):
    gpu_memory_bytes: int
    gpu_memory_utilization: float
    base_model_weight_bytes: int
    lora_trainable_param_bytes: int
    # Adam stores 2 moments (m, v) per trainable param, same dtype as the
    # param itself under this project's assumed mixed-precision setup —
    # 2.0x trainable-param bytes. A fp32 master-weight copy on top would
    # add a further 1x; not assumed here (LoRA adapters trained directly
    # in bf16 is the documented, not-yet-confirmed assumption).
    optimizer_state_multiplier: float
    # bytes of activation memory per (batch item x token), aggregated
    # across all layers — an empirical constant this project has not
    # measured yet (no GPU); a real smoke-test run's peak-memory
    # measurement is what should calibrate this, not a textbook formula
    # this sandbox can't verify.
    estimated_activation_bytes_per_token: int
    fixed_overhead_bytes: int


@dataclass(frozen=True)
class MemoryEstimate:
    estimated_bytes: int
    usable_bytes: int

    @property
    def fits(self) -> bool:
        return self.estimated_bytes <= self.usable_bytes

    @property
    def headroom_pct(self) -> float:
        if self.usable_bytes == 0:
            raise ValueError("usable_bytes is 0 — cannot express headroom as a percentage")
        return (self.usable_bytes - self.estimated_bytes) / self.usable_bytes * 100


def estimate_training_memory(
    *, batch_size: int, max_seq_len: int, config: MemoryEstimateConfig
) -> MemoryEstimate:
    if batch_size <= 0 or max_seq_len <= 0:
        raise ValueError(
            f"batch_size and max_seq_len must be positive, got batch_size={batch_size} "
            f"max_seq_len={max_seq_len}"
        )
    optimizer_bytes = int(config.lora_trainable_param_bytes * config.optimizer_state_multiplier)
    activation_bytes = config.estimated_activation_bytes_per_token * batch_size * max_seq_len
    estimated = (
        config.base_model_weight_bytes
        + config.lora_trainable_param_bytes
        + optimizer_bytes
        + activation_bytes
        + config.fixed_overhead_bytes
    )
    usable = int(config.gpu_memory_bytes * config.gpu_memory_utilization)
    return MemoryEstimate(estimated_bytes=estimated, usable_bytes=usable)


def assert_fits_in_memory(estimate: MemoryEstimate) -> None:
    """PLAN.md: "OOM 要在第0步暴露，不是第800步" — refuse to start
    rather than let a real training process discover the OOM 800 steps
    and several GPU-hours in."""
    if not estimate.fits:
        raise ValueError(
            f"estimated training memory ({estimate.estimated_bytes} bytes) exceeds the "
            f"usable budget ({estimate.usable_bytes} bytes) — refusing to start training; "
            f"reduce batch_size/max_seq_len or increase gpu_memory_utilization headroom"
        )


__all__ = [
    "MemoryEstimate",
    "MemoryEstimateConfig",
    "TokenLengthStats",
    "assert_fits_in_memory",
    "compute_token_length_stats",
    "estimate_training_memory",
    "validate_dataset_schema",
]
