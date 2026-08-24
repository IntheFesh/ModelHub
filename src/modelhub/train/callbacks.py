"""HuggingFace `Trainer`/LLaMA-Factory callback wiring `loss_guard.py`/
`checkpoint_state.py`/`time_budget.py` into the actual training loop —
PLAN.md B1 explicitly forbids reimplementing the training loop itself
("不要重写训练循环"), so every one of this round's real-time checks
(NaN/Inf halt, time-budget stop, atomic checkpoint finalization) has to
attach to LLaMA-Factory's underlying HF `Trainer` via its real callback
API, not a custom loop.

`transformers` is part of the `train` extra and is never installed in
this GPU-less sandbox — `ModelHubTrainerCallback` subclasses the real
`transformers.TrainerCallback` when available, and a plain stand-in
`object` base otherwise, purely so this module (and its pure-Python
unit tests) can be imported without transformers installed. A real GPU
run always has the real `transformers` installed (the `train` extra),
so this class is a genuine `TrainerCallback` subclass there — the
fallback base only exists for this sandbox's CPU-only test/import path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modelhub.train.checkpoint_state import (
    TrainerState as CheckpointTrainerState,
)
from modelhub.train.checkpoint_state import (
    checkpoint_dir_for_step,
    finalize_checkpoint,
    write_trainer_state,
)
from modelhub.train.loss_guard import check_loss_finite
from modelhub.train.time_budget import TimeBudgetTracker

try:
    from transformers import TrainerCallback as _HfTrainerCallback
except ImportError:
    _HfTrainerCallback = object


class ModelHubTrainerCallback(_HfTrainerCallback):  # type: ignore[misc]
    """Wires this round's three real-time guards into an HF `Trainer`:

    - `on_log`: every logged loss is checked for NaN/Inf
      (`loss_guard.check_loss_finite`) — halts immediately, batch
      context dumped, rather than let a corrupted optimizer state train
      silently for hundreds more steps.
    - `on_save`: after HF's own save completes into a temp/staging dir,
      validates the checkpoint has every required file and atomically
      swaps it into its final location (`checkpoint_state.
      finalize_checkpoint`) — HF's own `Trainer.save_model` is not
      itself atomic across multiple files.
    - `on_step_end`: asks `TimeBudgetTracker` whether the configured
      `--max-hours` budget is exhausted and requests training stop if
      so — PLAN.md: "按时间预算跑不按 epoch 跑".
    """

    def __init__(
        self,
        *,
        checkpoints_root: Path,
        dump_dir: Path,
        time_budget: TimeBudgetTracker,
    ) -> None:
        self._checkpoints_root = checkpoints_root
        self._dump_dir = dump_dir
        self._time_budget = time_budget

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if logs is not None and "loss" in logs:
            check_loss_finite(
                float(logs["loss"]),
                step=int(state.global_step),
                batch_debug_info={"logs": {k: str(v) for k, v in logs.items()}},
                dump_dir=self._dump_dir,
            )
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(state.global_step)
        # HF Trainer's own checkpoint output_dir convention: <output_dir>/checkpoint-<step>.
        # It is treated here as the "already fully written .tmp source" that
        # finalize_checkpoint validates and atomically swaps into this
        # project's own checkpoints_root — a second, explicit atomicity
        # guarantee on top of (not instead of) whatever HF itself does.
        hf_checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        final_dir = checkpoint_dir_for_step(self._checkpoints_root, step)
        checkpoint_state = CheckpointTrainerState(
            step=step,
            epoch=float(state.epoch or 0.0),
            dataloader_position=step,
            best_metric=self._time_budget.best_metric,
            saved_at=_now_isoformat(),
        )
        write_trainer_state(hf_checkpoint_dir, checkpoint_state)
        finalize_checkpoint(hf_checkpoint_dir, final_dir)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self._time_budget.budget_exhausted:
            control.should_training_stop = True
        return control


def _now_isoformat() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


__all__ = ["ModelHubTrainerCallback"]
