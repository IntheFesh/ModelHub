"""Unit tests for train/bad_models/checkpoint_selection.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.train.bad_models.checkpoint_selection import select_underfit_checkpoint
from modelhub.train.checkpoint_state import CheckpointListing


class TestSelectUnderfitCheckpoint:
    def test_selects_the_earliest_checkpoint(self) -> None:
        listing = CheckpointListing(
            checkpoints=(Path("checkpoint-10"), Path("checkpoint-20"), Path("checkpoint-30"))
        )
        assert select_underfit_checkpoint(listing) == Path("checkpoint-10")

    def test_empty_listing_raises(self) -> None:
        listing = CheckpointListing(checkpoints=())
        with pytest.raises(ValueError, match="no checkpoints available"):
            select_underfit_checkpoint(listing)
