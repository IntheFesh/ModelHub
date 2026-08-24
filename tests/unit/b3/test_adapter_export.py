"""Unit tests for train/bad_models/adapter_export.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelhub.train.bad_models.adapter_export import (
    ADAPTER_ONLY_EXPORT_FILES,
    export_adapter_only,
)
from modelhub.train.checkpoint_state import (
    ADAPTER_WEIGHTS_FILENAME,
    HF_TRAINER_STATE_FILENAME,
    OPTIMIZER_STATE_FILENAME,
    RNG_STATE_FILENAME,
    SCHEDULER_STATE_FILENAME,
    TrainerState,
    write_trainer_state,
)


def _write_complete_checkpoint(directory: Path, *, step: int = 1) -> None:
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


class TestExportAdapterOnly:
    def test_exports_only_adapter_files(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoint-1"
        _write_complete_checkpoint(checkpoint_dir)
        export_dir = tmp_path / "export"

        result = export_adapter_only(checkpoint_dir, export_dir)

        exported_files = {p.name for p in result.iterdir()}
        assert exported_files == {ADAPTER_WEIGHTS_FILENAME}  # adapter_config.json wasn't present
        assert OPTIMIZER_STATE_FILENAME not in exported_files
        assert RNG_STATE_FILENAME not in exported_files

    def test_exports_adapter_config_when_present(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoint-1"
        _write_complete_checkpoint(checkpoint_dir)
        (checkpoint_dir / "adapter_config.json").write_text("{}")
        export_dir = tmp_path / "export"

        result = export_adapter_only(checkpoint_dir, export_dir)

        assert {p.name for p in result.iterdir()} == ADAPTER_ONLY_EXPORT_FILES

    def test_incomplete_checkpoint_rejected(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoint-1"
        checkpoint_dir.mkdir()
        (checkpoint_dir / ADAPTER_WEIGHTS_FILENAME).write_bytes(b"only this")
        with pytest.raises(ValueError, match="incomplete"):
            export_adapter_only(checkpoint_dir, tmp_path / "export")

    def test_merged_weight_file_present_is_rejected(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoint-1"
        _write_complete_checkpoint(checkpoint_dir)
        (checkpoint_dir / "model.safetensors").write_bytes(b"a full merged model, not an adapter")

        with pytest.raises(ValueError, match="merged/full-model weight"):
            export_adapter_only(checkpoint_dir, tmp_path / "export")

    def test_export_is_atomic_no_stale_tmp_dir(self, tmp_path: Path) -> None:
        checkpoint_dir = tmp_path / "checkpoint-1"
        _write_complete_checkpoint(checkpoint_dir)
        export_dir = tmp_path / "export"

        export_adapter_only(checkpoint_dir, export_dir)

        assert not (tmp_path / "export.tmp").exists()
