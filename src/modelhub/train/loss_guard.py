"""Loss NaN/Inf guard — PLAN.md B1: "loss 出现 NaN/Inf 立即停止并 dump
当前 batch，不静默跳过."

A NaN/Inf loss is not a transient blip to skip past — it usually means
a bad sample, a learning-rate spike, or a numerical-precision bug, and
continuing training past it silently corrupts every later step's
optimizer state (Adam's moment estimates absorb the NaN and never
recover). This module raises immediately and dumps whatever debug
context the caller has about the offending batch, so the failure is
diagnosable instead of just "the run died, no idea why."

`batch_debug_info` is the caller's already-JSON-safe summary of the real
(tensor) batch — e.g. `sample_id`s, `input_ids` shapes/hashes, not raw
tensor dumps (this module has no torch dependency and no opinion on how
a real training loop should serialize its tensors; that conversion is
the caller's job).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from modelhub.common.atomic_io import atomic_write_json


class NonFiniteLossError(Exception):
    def __init__(self, *, step: int, loss_value: float, dump_path: Path) -> None:
        self.step = step
        self.loss_value = loss_value
        self.dump_path = dump_path
        super().__init__(
            f"non-finite loss ({loss_value!r}) at step {step} — training halted, "
            f"batch dumped to {dump_path}"
        )


def check_loss_finite(
    loss_value: float, *, step: int, batch_debug_info: dict[str, Any], dump_dir: Path
) -> None:
    """Raise `NonFiniteLossError` (after atomically dumping
    `batch_debug_info`) if `loss_value` is NaN or +/-Inf. A finite loss
    is a silent no-op — this is a guard, not a logger."""
    if math.isfinite(loss_value):
        return
    dump_path = dump_dir / f"nonfinite_loss_step_{step}.json"
    atomic_write_json(
        dump_path,
        {
            "step": step,
            # repr, not the float itself — json.dumps chokes on NaN/Inf
            # by default, and we want the exact non-finite value visible.
            "loss_value": repr(loss_value),
            "batch_debug_info": batch_debug_info,
        },
    )
    raise NonFiniteLossError(step=step, loss_value=loss_value, dump_path=dump_path)


__all__ = ["NonFiniteLossError", "check_loss_finite"]
