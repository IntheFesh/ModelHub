"""Full training-state checkpointing — PLAN.md B1: "要存的不只权重：
优化器状态 / lr scheduler / 数据加载器位置 / RNG 状态，少一样结果就
对不上."

A checkpoint is a DIRECTORY of files, atomically swapped into place via
`common/atomic_io.py::atomic_replace_dir` (built under a `.tmp` sibling
name, then renamed whole — never partially visible, CLAUDE.md §5.1/
§6.3). This module owns the file-naming contract and the "is this
checkpoint actually complete" validation; it does not itself serialize a
real torch optimizer/RNG state (no torch in this sandbox — see
docs/design-decisions.md) — real training code writes the actual bytes
via whatever real serialization it already uses (safetensors for
adapter weights, `torch.save` for optimizer/scheduler/RNG), this module
only enforces the SET of files a checkpoint must contain before it is
considered resumable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from modelhub.common.atomic_io import atomic_replace_dir, atomic_write_json, read_json
from modelhub.common.config import ModelHubBaseConfig

# Filenames match real HuggingFace Trainer / PEFT checkpoint-directory
# conventions exactly (adapter_model.safetensors, optimizer.pt,
# scheduler.pt, rng_state.pth, trainer_state.json — the last of these
# HF's own Trainer.save_state() already writes) so this module's
# completeness check runs against a real HF-produced checkpoint
# directory unmodified, not a parallel/renamed scheme.
ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
OPTIMIZER_STATE_FILENAME = "optimizer.pt"
SCHEDULER_STATE_FILENAME = "scheduler.pt"
RNG_STATE_FILENAME = "rng_state.pth"
HF_TRAINER_STATE_FILENAME = "trainer_state.json"
# This project's OWN state file, deliberately a different name from
# HF's trainer_state.json above — writing modelhub-specific fields
# (best_metric tracked by time_budget.py, in particular) into a file
# with the same name would silently clobber HF's own resume state.
TRAINER_STATE_FILENAME = "modelhub_trainer_state.json"

REQUIRED_CHECKPOINT_FILES = frozenset(
    {
        ADAPTER_WEIGHTS_FILENAME,
        OPTIMIZER_STATE_FILENAME,
        SCHEDULER_STATE_FILENAME,
        RNG_STATE_FILENAME,
        HF_TRAINER_STATE_FILENAME,
        TRAINER_STATE_FILENAME,
    }
)

_CHECKPOINT_DIR_PREFIX = "checkpoint-"


class TrainerState(ModelHubBaseConfig):
    """The JSON-serializable half of a checkpoint's state — everything
    that isn't a binary tensor blob."""

    step: int
    epoch: float
    dataloader_position: int
    best_metric: float | None
    saved_at: str


def checkpoint_dir_for_step(checkpoints_root: Path, step: int) -> Path:
    return checkpoints_root / f"{_CHECKPOINT_DIR_PREFIX}{step}"


def validate_checkpoint_complete(checkpoint_dir: Path) -> None:
    """Raise, listing every missing file, if `checkpoint_dir` is missing
    any of `REQUIRED_CHECKPOINT_FILES` — a partial checkpoint (e.g. from
    a process killed before `finalize_checkpoint`'s atomic swap ever
    ran) must never be silently treated as resumable."""
    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    present = {p.name for p in checkpoint_dir.iterdir()}
    missing = REQUIRED_CHECKPOINT_FILES - present
    if missing:
        raise ValueError(
            f"checkpoint at {checkpoint_dir} is incomplete, missing: {sorted(missing)} — "
            f"refusing to treat a partial checkpoint as resumable (CLAUDE.md §5.1)"
        )


def write_trainer_state(checkpoint_dir: Path, state: TrainerState) -> Path:
    return atomic_write_json(checkpoint_dir / TRAINER_STATE_FILENAME, state.model_dump(mode="json"))


def read_trainer_state(checkpoint_dir: Path) -> TrainerState:
    validate_checkpoint_complete(checkpoint_dir)
    return TrainerState.model_validate(read_json(checkpoint_dir / TRAINER_STATE_FILENAME))


def finalize_checkpoint(tmp_checkpoint_dir: Path, final_checkpoint_dir: Path) -> Path:
    """Validate `tmp_checkpoint_dir` has every required file, THEN
    atomically swap it into `final_checkpoint_dir` — validation happens
    before the swap, not after, so a caller never observes a path that
    looks like a finished checkpoint but turns out to be missing a
    file."""
    validate_checkpoint_complete(tmp_checkpoint_dir)
    return atomic_replace_dir(tmp_checkpoint_dir, final_checkpoint_dir)


@dataclass(frozen=True)
class CheckpointListing:
    checkpoints: tuple[Path, ...]  # sorted ascending by step

    @property
    def latest(self) -> Path | None:
        return self.checkpoints[-1] if self.checkpoints else None


def list_checkpoints(checkpoints_root: Path) -> CheckpointListing:
    """Only checkpoints that pass `validate_checkpoint_complete` count —
    a half-written directory (e.g. left over from a crash, or disk
    corruption) must never be offered as a `--resume-from` candidate."""
    if not checkpoints_root.is_dir():
        return CheckpointListing(())
    valid: list[tuple[int, Path]] = []
    for child in sorted(checkpoints_root.iterdir()):
        if not child.is_dir() or not child.name.startswith(_CHECKPOINT_DIR_PREFIX):
            continue
        try:
            validate_checkpoint_complete(child)
        except ValueError:
            continue
        step = int(child.name.removeprefix(_CHECKPOINT_DIR_PREFIX))
        valid.append((step, child))
    valid.sort(key=lambda pair: pair[0])
    return CheckpointListing(tuple(path for _, path in valid))


__all__ = [
    "ADAPTER_WEIGHTS_FILENAME",
    "HF_TRAINER_STATE_FILENAME",
    "OPTIMIZER_STATE_FILENAME",
    "REQUIRED_CHECKPOINT_FILES",
    "RNG_STATE_FILENAME",
    "SCHEDULER_STATE_FILENAME",
    "TRAINER_STATE_FILENAME",
    "CheckpointListing",
    "TrainerState",
    "checkpoint_dir_for_step",
    "finalize_checkpoint",
    "list_checkpoints",
    "read_trainer_state",
    "validate_checkpoint_complete",
    "write_trainer_state",
]
