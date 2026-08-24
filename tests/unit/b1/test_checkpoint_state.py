"""Unit tests for train/checkpoint_state.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.train.checkpoint_state import (
    ADAPTER_WEIGHTS_FILENAME,
    HF_TRAINER_STATE_FILENAME,
    OPTIMIZER_STATE_FILENAME,
    REQUIRED_CHECKPOINT_FILES,
    RNG_STATE_FILENAME,
    SCHEDULER_STATE_FILENAME,
    TrainerState,
    checkpoint_dir_for_step,
    finalize_checkpoint,
    list_checkpoints,
    read_trainer_state,
    validate_checkpoint_complete,
    write_trainer_state,
)


def _write_complete_checkpoint(directory: Path, *, step: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ADAPTER_WEIGHTS_FILENAME).write_bytes(b"adapter")
    (directory / OPTIMIZER_STATE_FILENAME).write_bytes(b"opt")
    (directory / SCHEDULER_STATE_FILENAME).write_bytes(b"sched")
    (directory / RNG_STATE_FILENAME).write_bytes(b"rng")
    (directory / HF_TRAINER_STATE_FILENAME).write_text("{}")
    write_trainer_state(
        directory,
        TrainerState(
            step=step, epoch=1.0, dataloader_position=step, best_metric=None, saved_at="now"
        ),
    )


class TestValidateCheckpointComplete:
    def test_complete_checkpoint_passes(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint-10"
        _write_complete_checkpoint(ckpt, step=10)
        validate_checkpoint_complete(ckpt)  # must not raise

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            validate_checkpoint_complete(tmp_path / "no-such-dir")

    def test_missing_file_raises_and_names_it(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint-10"
        _write_complete_checkpoint(ckpt, step=10)
        (ckpt / OPTIMIZER_STATE_FILENAME).unlink()
        with pytest.raises(ValueError, match=OPTIMIZER_STATE_FILENAME):
            validate_checkpoint_complete(ckpt)

    def test_all_required_files_present_by_construction(self) -> None:
        assert len(REQUIRED_CHECKPOINT_FILES) == 6


class TestFinalizeCheckpoint:
    def test_validates_before_swapping(self, tmp_path: Path) -> None:
        incomplete = tmp_path / "checkpoint-5.tmp"
        incomplete.mkdir()
        (incomplete / ADAPTER_WEIGHTS_FILENAME).write_bytes(b"only this")
        final = tmp_path / "checkpoint-5"
        with pytest.raises(ValueError, match="incomplete"):
            finalize_checkpoint(incomplete, final)
        assert not final.exists()  # the incomplete swap never happened

    def test_atomically_places_a_complete_checkpoint(self, tmp_path: Path) -> None:
        tmp_ckpt = tmp_path / "checkpoint-5.tmp"
        _write_complete_checkpoint(tmp_ckpt, step=5)
        final = tmp_path / "checkpoint-5"

        finalize_checkpoint(tmp_ckpt, final)

        assert final.is_dir()
        assert not tmp_ckpt.exists()
        validate_checkpoint_complete(final)  # the final location is itself complete


class TestReadWriteTrainerState:
    def test_roundtrip(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint-1"
        _write_complete_checkpoint(ckpt, step=1)
        state = read_trainer_state(ckpt)
        assert state.step == 1
        assert state.dataloader_position == 1

    def test_read_on_incomplete_checkpoint_raises(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "checkpoint-1"
        ckpt.mkdir()
        with pytest.raises(ValueError, match="incomplete"):
            read_trainer_state(ckpt)


class TestListCheckpoints:
    def test_empty_root_returns_empty_listing(self, tmp_path: Path) -> None:
        listing = list_checkpoints(tmp_path / "no-such-root")
        assert listing.checkpoints == ()
        assert listing.latest is None

    def test_lists_only_complete_checkpoints_sorted_by_step(self, tmp_path: Path) -> None:
        _write_complete_checkpoint(checkpoint_dir_for_step(tmp_path, 30), step=30)
        _write_complete_checkpoint(checkpoint_dir_for_step(tmp_path, 10), step=10)
        _write_complete_checkpoint(checkpoint_dir_for_step(tmp_path, 20), step=20)
        # an incomplete checkpoint left behind by a crash must be excluded.
        incomplete = checkpoint_dir_for_step(tmp_path, 40)
        incomplete.mkdir()
        (incomplete / ADAPTER_WEIGHTS_FILENAME).write_bytes(b"partial")

        listing = list_checkpoints(tmp_path)

        assert [p.name for p in listing.checkpoints] == [
            "checkpoint-10",
            "checkpoint-20",
            "checkpoint-30",
        ]
        assert listing.latest is not None
        assert listing.latest.name == "checkpoint-30"

    def test_ignores_non_checkpoint_directories(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        listing = list_checkpoints(tmp_path)
        assert listing.checkpoints == ()
