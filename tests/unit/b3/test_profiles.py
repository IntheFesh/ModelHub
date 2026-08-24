"""Unit tests for train/bad_models/profiles.py."""

from __future__ import annotations

from modelhub.train.bad_models.profiles import PROFILE_INFO, BadModelProfile


class TestProfileInfo:
    def test_every_profile_has_info(self) -> None:
        for profile in BadModelProfile:
            assert profile in PROFILE_INFO

    def test_ckpt_a_expects_accuracy_gate(self) -> None:
        assert PROFILE_INFO[BadModelProfile.CKPT_A_UNDERFIT].expected_gate == "accuracy"

    def test_ckpt_b_expects_regression_gate(self) -> None:
        assert PROFILE_INFO[BadModelProfile.CKPT_B_REGRESSION].expected_gate == "regression"

    def test_ckpt_d_expected_gate_is_corrected_to_accuracy_not_safety(self) -> None:
        info = PROFILE_INFO[BadModelProfile.CKPT_D_SAFETY]
        assert info.expected_gate.startswith("accuracy")
        assert "safety" not in info.expected_gate.split(" ")[0]

    def test_every_profile_has_a_non_empty_rationale(self) -> None:
        for info in PROFILE_INFO.values():
            assert info.rationale
            assert info.how_made
