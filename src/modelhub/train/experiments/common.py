"""Shared infrastructure for `train/experiments/`: B2's six-group
fine-tuning-method / distributed-strategy comparison suite (PLAN.md:
"六组，严格控制变量...每组记录显存峰值/吞吐/单步耗时/通信占比/可训练
参数量/折算 GPU 成本").

★ Core principle (PLAN.md, stated once for the whole suite): these are
200-500 step SHORT runs, never trained to convergence. The four numbers
they exist to produce — peak memory, throughput, step time, cost — are
stable by ~300 steps; accuracy is not, and must never be compared across
groups from a short run (`report.py` encodes this as a required,
explicit provenance tag rather than a comment developers can forget).

This module reuses `bench.experiments.common`'s per-group isolation
(`ExperimentGroupOutcome`/`run_experiment_group_isolated`) unchanged — it
is already generic over what a "group" measures, inference or training —
and `bench.gpu_guard`'s GPU-exclusivity check, since CLAUDE.md §5.2's
"bench 与 train 禁止共享 GPU" rule applies just as much to these training
comparison runs as it does to inference load tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.bench.experiments.common import (
    ExperimentGroupOutcome,
    run_experiment_group_isolated,
)
from modelhub.bench.gpu_guard import guard_gpu_exclusivity
from modelhub.common.run_manifest import RunManifest, assert_not_polluted

__all__ = [
    "ExperimentGroupOutcome",
    "LayerTypeProfileSplit",
    "TrainingRunMetrics",
    "check_training_experiment_precondition",
    "compute_throughput_tokens_per_s",
    "compute_total_cost",
    "guard_dedicated_gpus",
    "run_experiment_group_isolated",
]


def check_training_experiment_precondition(comparison_baseline_manifest: RunManifest) -> None:
    """Refuse to run any B2 comparison group against a baseline SFT
    config/environment whose own manifest is polluted — same rule and
    same reasoning as A11's `check_serving_precondition` (DD-0023):
    `is_polluted` is a strict superset of the two named flags, and a
    dirty-repo comparison run is exactly as invalid as a degraded one."""
    assert_not_polluted(comparison_baseline_manifest, purpose="train/experiments precondition")


def guard_dedicated_gpus(gpu_indices: list[int]) -> None:
    """CLAUDE.md §5.2 applied to every GPU a comparison group will use —
    group 4 (ZeRO-2/ZeRO-3/FSDP) is explicitly dual-card ("双卡"), so
    this checks each index in `gpu_indices`, not just index 0. Raises via
    `guard_gpu_exclusivity`'s own `ModelHubError` on the first
    non-exclusive GPU found; refuses to proceed on an unverifiable
    (SKIP) result exactly as strictly as on a confirmed-shared (FAIL)
    one (CLAUDE.md §1.3 whitelist rule)."""
    if not gpu_indices:
        raise ValueError("gpu_indices must not be empty")
    for gpu_index in gpu_indices:
        guard_gpu_exclusivity(gpu_index)


@dataclass(frozen=True)
class LayerTypeProfileSplit:
    """★【v2 新增】GDN 层与全注意力层各自的显存/耗时占比 — every group
    additionally records this split, not just the aggregate peak memory
    and step time; the two layer types' behavior under a given
    fine-tuning/distributed strategy is itself an observation worth
    reporting on Qwen3.5's hybrid architecture."""

    gdn_layer_memory_bytes: int
    full_attention_layer_memory_bytes: int
    gdn_layer_time_s: float
    full_attention_layer_time_s: float

    def __post_init__(self) -> None:
        for byte_field_name, byte_value in (
            ("gdn_layer_memory_bytes", self.gdn_layer_memory_bytes),
            ("full_attention_layer_memory_bytes", self.full_attention_layer_memory_bytes),
        ):
            if byte_value < 0:
                raise ValueError(f"{byte_field_name} must be non-negative, got {byte_value}")
        for time_field_name, time_value in (
            ("gdn_layer_time_s", self.gdn_layer_time_s),
            ("full_attention_layer_time_s", self.full_attention_layer_time_s),
        ):
            if time_value < 0:
                raise ValueError(f"{time_field_name} must be non-negative, got {time_value}")

    @property
    def gdn_memory_share(self) -> float:
        total = self.gdn_layer_memory_bytes + self.full_attention_layer_memory_bytes
        if total == 0:
            raise ValueError("cannot express a memory share: both layer types measured 0 bytes")
        return self.gdn_layer_memory_bytes / total

    @property
    def gdn_time_share(self) -> float:
        total = self.gdn_layer_time_s + self.full_attention_layer_time_s
        if total == 0:
            raise ValueError("cannot express a time share: both layer types measured 0 seconds")
        return self.gdn_layer_time_s / total


@dataclass(frozen=True)
class TrainingRunMetrics:
    """One comparison group's real, measured short-run numbers.

    `communication_time_ratio` is `None` for the non-distributed groups
    (1/2/3/5/6) — not applicable, never coerced to 0.0 (CLAUDE.md §2:
    缺失就是 None，不是 0). `layer_type_split` is `None` whenever a run
    did not (yet) produce the v2-added per-layer-type profile — this
    sandbox never produces one for real (no GPU), so every test of a
    "populated" split uses an explicitly constructed synthetic instance.
    """

    group_name: str
    peak_memory_bytes: int
    throughput_tokens_per_s: float
    step_time_s: float
    trainable_param_count: int
    total_cost: float
    communication_time_ratio: float | None
    layer_type_split: LayerTypeProfileSplit | None

    def __post_init__(self) -> None:
        if self.peak_memory_bytes < 0:
            raise ValueError(
                f"peak_memory_bytes must be non-negative, got {self.peak_memory_bytes}"
            )
        if self.throughput_tokens_per_s <= 0:
            raise ValueError(
                f"throughput_tokens_per_s must be positive, got {self.throughput_tokens_per_s}"
            )
        if self.step_time_s <= 0:
            raise ValueError(f"step_time_s must be positive, got {self.step_time_s}")
        if self.trainable_param_count <= 0:
            raise ValueError(
                f"trainable_param_count must be positive, got {self.trainable_param_count}"
            )
        if self.total_cost < 0:
            raise ValueError(f"total_cost must be non-negative, got {self.total_cost}")
        if self.communication_time_ratio is not None and not (
            0.0 <= self.communication_time_ratio <= 1.0
        ):
            raise ValueError(
                f"communication_time_ratio must be in [0, 1], got {self.communication_time_ratio}"
            )


def compute_throughput_tokens_per_s(*, tokens_per_step: int, step_time_s: float) -> float:
    if tokens_per_step <= 0:
        raise ValueError(f"tokens_per_step must be positive, got {tokens_per_step}")
    if step_time_s <= 0:
        raise ValueError(f"step_time_s must be positive, got {step_time_s}")
    return tokens_per_step / step_time_s


def compute_total_cost(
    *, step_time_s: float, num_steps: int, gpu_cost_per_hour: float, gpu_count: int
) -> float:
    if step_time_s <= 0:
        raise ValueError(f"step_time_s must be positive, got {step_time_s}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if gpu_cost_per_hour < 0:
        raise ValueError(f"gpu_cost_per_hour must be non-negative, got {gpu_cost_per_hour}")
    if gpu_count <= 0:
        raise ValueError(f"gpu_count must be positive, got {gpu_count}")
    total_hours = (step_time_s * num_steps) / 3600
    return total_hours * gpu_cost_per_hour * gpu_count
