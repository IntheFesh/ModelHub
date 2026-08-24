"""ckpt-A underfit selection — PLAN.md: "直接取 B1 主跑的早期
checkpoint（零成本）". No new training run, just a real pick from B1's
own `checkpoint_state.list_checkpoints` output."""

from __future__ import annotations

from pathlib import Path

from modelhub.train.checkpoint_state import CheckpointListing


def select_underfit_checkpoint(listing: CheckpointListing) -> Path:
    """`CheckpointListing.checkpoints` is sorted ascending by step
    (checkpoint_state.py) — index 0 is real B1's earliest surviving
    checkpoint, not a synthesized stand-in."""
    if not listing.checkpoints:
        raise ValueError(
            "no checkpoints available to select an underfit candidate from — B1's "
            "main SFT run needs at least one real checkpoint on disk first"
        )
    return listing.checkpoints[0]


__all__ = ["select_underfit_checkpoint"]
