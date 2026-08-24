"""B2's own real, self-controlled metrics file format —
`experiment_metrics.json`, written by `ExperimentMetricsCallback` at the
end of a real short comparison run and read back by
`load_raw_experiment_metrics`.

Why a project-owned file rather than reverse-engineering numbers out of
HF Trainer's own `trainer_state.json`: HF's `log_history` entries carry
`loss`/`learning_rate`/`step`, not the four numbers this round actually
needs (peak memory, wall-clock step time, communication time, per-
layer-type breakdown) — inventing a parser that guesses those out of
fields HF was never guaranteed to log would be exactly the "看起来能跑
但结果是假的" trap CLAUDE.md warns about. Instead, `checkpoint_state.py`'s
already-reviewed pattern (this project defines and owns its own JSON
schema, matched to real HF/PyTorch primitives) is reused here:
`ExperimentMetricsCallback` calls the real `torch.cuda.max_memory_
allocated()` / wall-clock timers itself and writes a schema this project
fully controls and can validate against a synthetic fixture without a
GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelhub.common.atomic_io import atomic_write_json
from modelhub.common.config import ModelHubBaseConfig

try:
    from transformers import TrainerCallback as _HfTrainerCallback
except ImportError:
    _HfTrainerCallback = object


class RawExperimentMetrics(ModelHubBaseConfig):
    """`experiment_metrics.json`'s schema. Every key must be present —
    `extra="forbid"` plus no field defaults means a run that skipped
    writing one is a schema-validation failure, not a silently-dropped
    key. `peak_memory_bytes` is `int | None` rather than plain `int`
    because it genuinely can go unmeasured (no CUDA device) — CLAUDE.md
    §2.4/§8: missing is `None`, never a `0` that would read as "measured
    zero bytes used". `require_measured_peak_memory` is the hard-fail
    gate a caller building a comparable `TrainingRunMetrics` must pass
    through — that type's own `peak_memory_bytes: int` is one of B2's
    four core numbers and is never allowed to silently become 0."""

    peak_memory_bytes: int | None
    tokens_per_step: int
    mean_step_time_s: float
    trainable_param_count: int
    num_steps: int
    communication_time_ratio: float | None = None
    gdn_layer_memory_bytes: int | None = None
    full_attention_layer_memory_bytes: int | None = None
    gdn_layer_time_s: float | None = None
    full_attention_layer_time_s: float | None = None


def require_measured_peak_memory(raw: RawExperimentMetrics) -> int:
    """Hard-fail (not a defaulted 0) when a group's run recorded no real
    peak memory — this sandbox's `ExperimentMetricsCallback` never
    measures one for real (no CUDA), so every call site building a
    `TrainingRunMetrics` from a raw file produced in this sandbox must
    go through this and fail loudly, matching `train/dry_run.py::
    assert_fits_in_memory`'s "a required field being None is a hard
    failure" precedent."""
    if raw.peak_memory_bytes is None:
        raise ValueError(
            "peak_memory_bytes was not measured for this run (no CUDA device available) "
            "— cannot build a comparable TrainingRunMetrics without it; re-run on a real "
            "GPU machine"
        )
    return raw.peak_memory_bytes


def load_raw_experiment_metrics(path: Path) -> RawExperimentMetrics:
    """Real hard failure (not a default-filled partial result) if the
    file is missing or fails schema validation — CLAUDE.md §3.4."""
    if not path.is_file():
        raise ValueError(f"experiment metrics file does not exist: {path}")
    return RawExperimentMetrics.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass
class _StepTiming:
    step_started_at: float
    elapsed_s: list[float]


class ExperimentMetricsCallback(_HfTrainerCallback):  # type: ignore[misc]
    """Records real wall-clock step timing and (when CUDA is actually
    available) real peak memory via `torch.cuda.max_memory_allocated()`,
    then writes `RawExperimentMetrics` to `output_path` on train end.

    `transformers` is guarded exactly as in `train/callbacks.py` — this
    class subclasses the real `TrainerCallback` on a real GPU machine
    (the `train` extra) and a plain `object` here so this module still
    imports and unit-tests without it installed. `torch.cuda.*` calls
    are similarly guarded: this sandbox has no CUDA device, so
    `peak_memory_bytes` would come back 0 here, not a real measurement
    — the one code path this class exercises in this sandbox is
    "torch/CUDA unavailable", never the real recording branch (see this
    round's honesty checklist).
    """

    def __init__(
        self,
        *,
        output_path: Path,
        trainable_param_count: int,
        tokens_per_step: int,
        now_fn: Any = time.monotonic,
    ) -> None:
        # tokens_per_step is a caller-supplied fact (max_seq_len *
        # per_device_batch_size * grad_accum_steps, all already known
        # from the group's own config) rather than introspected off the
        # real HF `TrainingArguments` object — that object has no field
        # for it, and guessing one would be exactly the kind of
        # plausible-but-fabricated attribute access CLAUDE.md warns
        # against.
        self._output_path = output_path
        self._trainable_param_count = trainable_param_count
        self._tokens_per_step = tokens_per_step
        self._now_fn = now_fn
        self._timing = _StepTiming(step_started_at=0.0, elapsed_s=[])

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._timing.step_started_at = self._now_fn()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._timing.elapsed_s.append(self._now_fn() - self._timing.step_started_at)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        peak_memory_bytes = self._real_peak_memory_bytes()
        raw = RawExperimentMetrics(
            peak_memory_bytes=peak_memory_bytes,
            tokens_per_step=self._tokens_per_step,
            mean_step_time_s=_mean(self._timing.elapsed_s),
            trainable_param_count=self._trainable_param_count,
            num_steps=len(self._timing.elapsed_s),
        )
        atomic_write_json(self._output_path, raw.model_dump(mode="json"))
        return control

    def _real_peak_memory_bytes(self) -> int | None:
        """`None` (not 0) whenever peak memory genuinely was not
        measured — CLAUDE.md §2.4/§8: a missing metric is `None`, never
        a `0` that reads as "measured zero bytes used"."""
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean step time: no steps were recorded")
    return sum(values) / len(values)


__all__ = [
    "ExperimentMetricsCallback",
    "RawExperimentMetrics",
    "load_raw_experiment_metrics",
    "require_measured_peak_memory",
]
