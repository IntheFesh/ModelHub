"""Unit tests for train/loss_guard.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelhub.train.loss_guard import NonFiniteLossError, check_loss_finite


class TestCheckLossFinite:
    def test_finite_loss_is_a_silent_no_op(self, tmp_path: Path) -> None:
        check_loss_finite(0.42, step=1, batch_debug_info={"sample_ids": ["s0"]}, dump_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []  # nothing dumped

    def test_nan_loss_raises_and_dumps(self, tmp_path: Path) -> None:
        with pytest.raises(NonFiniteLossError) as exc_info:
            check_loss_finite(
                float("nan"), step=7, batch_debug_info={"sample_ids": ["s7"]}, dump_dir=tmp_path
            )
        assert exc_info.value.step == 7
        dump_path = tmp_path / "nonfinite_loss_step_7.json"
        assert dump_path.is_file()
        dumped = json.loads(dump_path.read_text())
        assert dumped["step"] == 7
        assert dumped["batch_debug_info"] == {"sample_ids": ["s7"]}

    def test_positive_infinity_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NonFiniteLossError):
            check_loss_finite(float("inf"), step=1, batch_debug_info={}, dump_dir=tmp_path)

    def test_negative_infinity_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NonFiniteLossError):
            check_loss_finite(float("-inf"), step=1, batch_debug_info={}, dump_dir=tmp_path)

    def test_error_message_names_step_and_dump_path(self, tmp_path: Path) -> None:
        with pytest.raises(NonFiniteLossError, match="step 3"):
            check_loss_finite(float("nan"), step=3, batch_debug_info={}, dump_dir=tmp_path)
