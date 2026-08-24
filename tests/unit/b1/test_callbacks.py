"""Unit tests for train/callbacks.py — `ModelHubTrainerCallback` wired
against fake HF Trainer `args`/`state`/`control` objects (transformers
is part of the `train` extra, not installed in this sandbox; the class
itself falls back to subclassing `object` in that case — see the
module's own docstring)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from modelhub.train.callbacks import ModelHubTrainerCallback
from modelhub.train.checkpoint_state import (
    ADAPTER_WEIGHTS_FILENAME,
    HF_TRAINER_STATE_FILENAME,
    OPTIMIZER_STATE_FILENAME,
    RNG_STATE_FILENAME,
    SCHEDULER_STATE_FILENAME,
    checkpoint_dir_for_step,
    read_trainer_state,
)
from modelhub.train.loss_guard import NonFiniteLossError
from modelhub.train.time_budget import TimeBudgetTracker


@dataclass
class _FakeArgs:
    output_dir: str


@dataclass
class _FakeState:
    global_step: int
    epoch: float = 1.0


@dataclass
class _FakeControl:
    should_training_stop: bool = False


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance_hours(self, hours: float) -> None:
        self.now += hours * 3600


def _write_hf_checkpoint_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ADAPTER_WEIGHTS_FILENAME).write_bytes(b"adapter")
    (directory / OPTIMIZER_STATE_FILENAME).write_bytes(b"opt")
    (directory / SCHEDULER_STATE_FILENAME).write_bytes(b"sched")
    (directory / RNG_STATE_FILENAME).write_bytes(b"rng")
    (directory / HF_TRAINER_STATE_FILENAME).write_text("{}")


@pytest.fixture
def callback(tmp_path: Path) -> ModelHubTrainerCallback:
    return ModelHubTrainerCallback(
        checkpoints_root=tmp_path / "checkpoints",
        dump_dir=tmp_path / "dumps",
        time_budget=TimeBudgetTracker(max_hours=8.0, now_fn=_FakeClock()),
    )


class TestOnLog:
    def test_finite_loss_is_a_no_op(self, callback: ModelHubTrainerCallback) -> None:
        control = _FakeControl()
        result = callback.on_log(
            _FakeArgs(output_dir="/unused"),
            _FakeState(global_step=5),
            control,
            logs={"loss": 0.42},
        )
        assert result is control
        assert control.should_training_stop is False

    def test_nan_loss_raises(self, callback: ModelHubTrainerCallback) -> None:
        with pytest.raises(NonFiniteLossError):
            callback.on_log(
                _FakeArgs(output_dir="/unused"),
                _FakeState(global_step=5),
                _FakeControl(),
                logs={"loss": float("nan")},
            )

    def test_missing_loss_key_is_a_no_op(self, callback: ModelHubTrainerCallback) -> None:
        control = _FakeControl()
        callback.on_log(
            _FakeArgs(output_dir="/unused"), _FakeState(global_step=5), control, logs={"lr": 0.1}
        )
        # must not raise, and must not require a "loss" key to be present.

    def test_none_logs_is_a_no_op(self, callback: ModelHubTrainerCallback) -> None:
        callback.on_log(
            _FakeArgs(output_dir="/unused"), _FakeState(global_step=5), _FakeControl(), logs=None
        )


class TestOnSave:
    def test_finalizes_a_complete_checkpoint(
        self, tmp_path: Path, callback: ModelHubTrainerCallback
    ) -> None:
        args = _FakeArgs(output_dir=str(tmp_path / "hf_output"))
        hf_checkpoint_dir = Path(args.output_dir) / "checkpoint-10"
        _write_hf_checkpoint_files(hf_checkpoint_dir)

        callback.on_save(args, _FakeState(global_step=10, epoch=1.5), _FakeControl())

        final_dir = checkpoint_dir_for_step(tmp_path / "checkpoints", 10)
        assert final_dir.is_dir()
        assert not hf_checkpoint_dir.exists()  # atomically moved, not copied
        state = read_trainer_state(final_dir)
        assert state.step == 10
        assert state.epoch == 1.5

    def test_incomplete_hf_checkpoint_raises(
        self, tmp_path: Path, callback: ModelHubTrainerCallback
    ) -> None:
        args = _FakeArgs(output_dir=str(tmp_path / "hf_output"))
        hf_checkpoint_dir = Path(args.output_dir) / "checkpoint-10"
        hf_checkpoint_dir.mkdir(parents=True)
        (hf_checkpoint_dir / ADAPTER_WEIGHTS_FILENAME).write_bytes(b"only this")
        # missing optimizer/scheduler/rng/hf-trainer-state on purpose.

        with pytest.raises(ValueError, match="incomplete"):
            callback.on_save(args, _FakeState(global_step=10), _FakeControl())


class TestOnStepEnd:
    def test_does_not_stop_training_within_budget(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        callback = ModelHubTrainerCallback(
            checkpoints_root=tmp_path / "checkpoints",
            dump_dir=tmp_path / "dumps",
            time_budget=TimeBudgetTracker(max_hours=8.0, now_fn=clock),
        )
        clock.advance_hours(1.0)
        control = _FakeControl()
        callback.on_step_end(_FakeArgs(output_dir="/unused"), _FakeState(global_step=100), control)
        assert control.should_training_stop is False

    def test_requests_stop_once_budget_exhausted(self, tmp_path: Path) -> None:
        clock = _FakeClock()
        callback = ModelHubTrainerCallback(
            checkpoints_root=tmp_path / "checkpoints",
            dump_dir=tmp_path / "dumps",
            time_budget=TimeBudgetTracker(max_hours=8.0, now_fn=clock),
        )
        clock.advance_hours(8.5)
        control = _FakeControl()
        callback.on_step_end(_FakeArgs(output_dir="/unused"), _FakeState(global_step=100), control)
        assert control.should_training_stop is True
