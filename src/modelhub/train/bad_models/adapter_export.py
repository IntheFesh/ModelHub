"""Adapter-only export for handoff to A-track — PLAN.md: "每个只导出
LoRA adapter（~200MB）打包传回 A 道。★ 不传合并权重."

This is a real, testable guard: a checkpoint directory that contains a
merged/full-model weight file (someone ran a merge-and-save step by
mistake, or pointed this at the wrong directory) is rejected outright,
not silently exported anyway with the merged file included.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from modelhub.common.atomic_io import atomic_replace_dir
from modelhub.train.checkpoint_state import ADAPTER_WEIGHTS_FILENAME, validate_checkpoint_complete

# Real HF/PEFT full-model save conventions — never allowed in a
# checkpoint directory this function exports from.
MERGED_WEIGHT_FILENAMES = frozenset(
    {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
)

# Only the adapter weights themselves travel to A-track — not the
# optimizer/scheduler/rng/trainer-state files checkpoint_state.py
# requires for *resuming training*, which A-track's gate drill has no
# use for and which would needlessly bloat the ~200MB PLAN.md expects.
ADAPTER_ONLY_EXPORT_FILES = frozenset({ADAPTER_WEIGHTS_FILENAME, "adapter_config.json"})


def export_adapter_only(checkpoint_dir: Path, export_dir: Path) -> Path:
    validate_checkpoint_complete(checkpoint_dir)

    present_merged_files = [
        name for name in MERGED_WEIGHT_FILENAMES if (checkpoint_dir / name).is_file()
    ]
    if present_merged_files:
        raise ValueError(
            f"refusing to export {checkpoint_dir}: it contains merged/full-model weight "
            f"file(s) {present_merged_files} — PLAN.md: 不传合并权重. This directory is "
            f"either a merged checkpoint or was pointed at the wrong path."
        )

    tmp_dir = export_dir.parent / f"{export_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for filename in ADAPTER_ONLY_EXPORT_FILES:
        source = checkpoint_dir / filename
        if source.is_file():
            shutil.copy2(source, tmp_dir / filename)

    return atomic_replace_dir(tmp_dir, export_dir)


__all__ = ["ADAPTER_ONLY_EXPORT_FILES", "MERGED_WEIGHT_FILENAMES", "export_adapter_only"]
