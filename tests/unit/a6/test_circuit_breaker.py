"""Unit tests for gateway/circuit_breaker.py.

Uses an injectable fake clock (a mutable `[float]` box, advanced
explicitly) instead of real `time.sleep()` — real state transitions,
fake wall-clock time."""

from __future__ import annotations

import pytest

from modelhub.common.errors import ErrorCode, GatewayError
from modelhub.gateway.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _breaker(
    *,
    failure_threshold: int = 3,
    failure_window_seconds: float = 10.0,
    cooldown_seconds: float = 5.0,
) -> tuple[CircuitBreaker, _FakeClock]:
    clock = _FakeClock()
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        failure_window_seconds=failure_window_seconds,
        cooldown_seconds=cooldown_seconds,
    )
    return CircuitBreaker(config, now_fn=clock), clock


def test_starts_closed_and_allows_requests() -> None:
    breaker, _ = _breaker()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request()


def test_trips_open_after_failure_threshold() -> None:
    breaker, _ = _breaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()


def test_failures_outside_the_window_do_not_count() -> None:
    breaker, clock = _breaker(failure_threshold=3, failure_window_seconds=10.0)
    breaker.record_failure()
    breaker.record_failure()
    clock.advance(20.0)  # outside the 10s window
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_success_resets_failure_count() -> None:
    breaker, _ = _breaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED  # count reset, only 2 since success


def test_open_transitions_to_half_open_after_cooldown() -> None:
    breaker, clock = _breaker(failure_threshold=1, cooldown_seconds=5.0)
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    clock.advance(4.9)
    assert breaker.state is CircuitState.OPEN
    clock.advance(0.2)
    assert breaker.state is CircuitState.HALF_OPEN


def test_half_open_allows_exactly_one_trial_request() -> None:
    breaker, clock = _breaker(failure_threshold=1, cooldown_seconds=5.0)
    breaker.record_failure()
    clock.advance(5.0)
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request() is True
    assert breaker.allow_request() is False  # trial already claimed


def test_half_open_success_closes_the_circuit() -> None:
    breaker, clock = _breaker(failure_threshold=1, cooldown_seconds=5.0)
    breaker.record_failure()
    clock.advance(5.0)
    assert breaker.allow_request()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request()


def test_half_open_failure_reopens_immediately() -> None:
    # threshold=3 so this test actually distinguishes "a single HALF_OPEN
    # failure reopens immediately" from "it just happens to hit the
    # threshold again" — if the HALF_OPEN special-case in record_failure()
    # were removed, a single post-cooldown failure would NOT be enough to
    # reopen (only 1 of the 3 required failures would have been recorded).
    breaker, clock = _breaker(failure_threshold=3, cooldown_seconds=5.0)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    clock.advance(5.0)
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.allow_request()
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN


def test_guard_raises_when_open() -> None:
    breaker, _ = _breaker(failure_threshold=1)
    breaker.record_failure()
    with pytest.raises(GatewayError) as exc_info:
        breaker.guard()
    assert exc_info.value.code is ErrorCode.CIRCUIT_BREAKER_OPEN
    assert exc_info.value.retryable is True


def test_guard_does_not_raise_when_closed() -> None:
    breaker, _ = _breaker()
    breaker.guard()  # must not raise
