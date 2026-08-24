"""Unit tests for release/canary.py: staged traffic routing."""

from __future__ import annotations

import pytest

from modelhub.release.canary import (
    CanaryConfig,
    RolloutStatus,
    mark_rolled_back,
    record_canary_request,
    route_canary_traffic,
    start_rollout,
    try_advance_stage,
)


def _config(**overrides: object) -> CanaryConfig:
    defaults: dict[str, object] = {"stages": [5, 25, 50, 100], "min_requests_per_stage": 3}
    defaults.update(overrides)
    return CanaryConfig.model_validate(defaults)


class TestCanaryConfig:
    def test_valid_config_accepted(self) -> None:
        cfg = _config()
        assert cfg.stages == [5, 25, 50, 100]

    def test_empty_stages_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _config(stages=[])

    def test_non_ascending_stages_rejected(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            _config(stages=[25, 5, 100])

    def test_last_stage_must_be_100(self) -> None:
        with pytest.raises(ValueError, match="last stage must be 100"):
            _config(stages=[5, 25, 50])

    def test_stage_out_of_range_rejected(self) -> None:
        # a middle stage can only ever be > 100 by also breaking the
        # ascending-or-last-is-100 checks above (whichever is not the
        # last element is bounded above by the last element, which the
        # earlier checks already pin at 100) — so the only *reachable*
        # per-stage range violation, once ascending + last==100 both
        # hold, is a non-positive stage.
        with pytest.raises(ValueError, match=r"\(0, 100\]"):
            _config(stages=[0, 100])
        with pytest.raises(ValueError, match=r"\(0, 100\]"):
            _config(stages=[-5, 50, 100])


class TestRouteCanaryTraffic:
    def test_zero_percent_always_stable(self) -> None:
        for key in ["req-1", "req-2", "user-abc", "session-xyz"]:
            assert route_canary_traffic(key, 0) == "stable"

    def test_hundred_percent_always_canary(self) -> None:
        for key in ["req-1", "req-2", "user-abc", "session-xyz"]:
            assert route_canary_traffic(key, 100) == "canary"

    def test_same_key_is_deterministic_across_calls(self) -> None:
        arms = {route_canary_traffic("sticky-session-42", 50) for _ in range(20)}
        assert len(arms) == 1

    def test_negative_percentage_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            route_canary_traffic("k", -1)

    def test_over_hundred_percentage_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            route_canary_traffic("k", 101)

    def test_distribution_roughly_matches_percentage(self) -> None:
        # not a statistical proof, just a sanity bound on the hash split.
        canary_count = sum(
            1 for i in range(2000) if route_canary_traffic(f"req-{i}", 25) == "canary"
        )
        assert 400 < canary_count < 600


class TestRolloutLifecycle:
    def test_start_rollout_is_stage_zero_in_progress(self) -> None:
        state = start_rollout()
        assert state.stage_index == 0
        assert state.requests_at_current_stage == 0
        assert state.status is RolloutStatus.IN_PROGRESS
        assert state.current_percentage(_config()) == 5

    def test_record_canary_request_increments_counter(self) -> None:
        state = start_rollout()
        state = record_canary_request(state)
        state = record_canary_request(state)
        assert state.requests_at_current_stage == 2
        assert state.stage_index == 0

    def test_record_canary_request_on_non_in_progress_raises(self) -> None:
        state = mark_rolled_back(start_rollout())
        with pytest.raises(ValueError, match="cannot record a request"):
            record_canary_request(state)

    def test_try_advance_stage_no_op_below_threshold(self) -> None:
        config = _config(min_requests_per_stage=3)
        state = start_rollout()
        state = record_canary_request(state)
        advanced = try_advance_stage(state, config)
        assert advanced == state
        assert advanced.stage_index == 0

    def test_try_advance_stage_advances_and_resets_counter(self) -> None:
        config = _config(min_requests_per_stage=3)
        state = start_rollout()
        for _ in range(3):
            state = record_canary_request(state)
        advanced = try_advance_stage(state, config)
        assert advanced.stage_index == 1
        assert advanced.requests_at_current_stage == 0
        assert advanced.status is RolloutStatus.IN_PROGRESS
        assert advanced.current_percentage(config) == 25

    def test_try_advance_stage_completes_at_last_stage(self) -> None:
        # completion requires min_requests_per_stage healthy requests
        # *at* the final (100%) stage too, not just reaching it — so
        # this takes two rounds of record+advance, not one.
        config = _config(stages=[5, 100], min_requests_per_stage=2)
        state = start_rollout()
        for _ in range(2):
            state = record_canary_request(state)
        state = try_advance_stage(state, config)
        assert state.stage_index == 1
        assert state.current_percentage(config) == 100
        assert state.status is RolloutStatus.IN_PROGRESS
        for _ in range(2):
            state = record_canary_request(state)
        state = try_advance_stage(state, config)
        assert state.status is RolloutStatus.COMPLETED

    def test_try_advance_stage_no_op_when_not_in_progress(self) -> None:
        config = _config(min_requests_per_stage=1)
        state = start_rollout()
        state = record_canary_request(state)
        rolled_back = mark_rolled_back(state)
        assert try_advance_stage(rolled_back, config) == rolled_back

    def test_mark_rolled_back_preserves_stage_and_count(self) -> None:
        state = start_rollout()
        state = record_canary_request(state)
        state = record_canary_request(state)
        rolled_back = mark_rolled_back(state)
        assert rolled_back.status is RolloutStatus.ROLLED_BACK
        assert rolled_back.stage_index == 0
        assert rolled_back.requests_at_current_stage == 2
