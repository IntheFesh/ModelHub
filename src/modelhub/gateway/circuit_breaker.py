"""Per-process circuit breaker around a backend model call.

Classic three-state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN).
State lives in-process, not in Redis — unlike rate_limit.py/quota.py,
per-request accuracy across multiple gateway processes isn't the point
here; each process independently protecting itself (and the backend it
calls) from hammering an already-failing model server is. A
Redis-backed, cross-process circuit breaker would be a legitimate future
enhancement, not a correctness gap in this one.

The clock is injectable (`now_fn`) so tests exercise real state
transitions without sleeping in wall-clock time — CLAUDE.md doesn't
require this, but a test suite that takes tens of seconds because of
sleep() calls stops getting run.

HALF_OPEN allows exactly one trial request through, not an unbounded
burst: `allow_request()` has the side effect of claiming that trial slot
so a second call before the first trial resolves (`record_success`/
`record_failure`) is correctly refused — the entire point of HALF_OPEN is
probing a possibly-still-broken backend with a single request, not
flooding it again the instant the cooldown elapses.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum

from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode, GatewayError, Stage


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerConfig(ModelHubBaseConfig):
    failure_threshold: int
    failure_window_seconds: float
    cooldown_seconds: float


class CircuitBreaker:
    """Not thread-safe by design — one instance per (backend, worker
    process/thread) is the intended usage, matching how a gateway worker
    typically owns one outbound connection pool per backend."""

    def __init__(
        self, config: CircuitBreakerConfig, *, now_fn: Callable[[], float] = time.monotonic
    ) -> None:
        self._config = config
        self._now_fn = now_fn
        self._state = CircuitState.CLOSED
        self._failure_timestamps: deque[float] = deque()
        self._opened_at: float | None = None
        self._half_open_trial_claimed = False

    @property
    def state(self) -> CircuitState:
        self._maybe_transition_to_half_open()
        return self._state

    def _maybe_transition_to_half_open(self) -> None:
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return
        if self._now_fn() - self._opened_at >= self._config.cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_trial_claimed = False

    def allow_request(self) -> bool:
        current = self.state
        if current is CircuitState.OPEN:
            return False
        if current is CircuitState.HALF_OPEN:
            if self._half_open_trial_claimed:
                return False
            self._half_open_trial_claimed = True
            return True
        return True

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_timestamps.clear()
        self._opened_at = None
        self._half_open_trial_claimed = False

    def record_failure(self) -> None:
        now = self._now_fn()
        if self._state is CircuitState.HALF_OPEN:
            # the trial request in HALF_OPEN failed — back to OPEN
            # immediately, don't require re-accumulating failure_threshold
            # failures again.
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._failure_timestamps.clear()
            self._half_open_trial_claimed = False
            return

        self._failure_timestamps.append(now)
        window_start = now - self._config.failure_window_seconds
        while self._failure_timestamps and self._failure_timestamps[0] < window_start:
            self._failure_timestamps.popleft()

        if len(self._failure_timestamps) >= self._config.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = now
            self._failure_timestamps.clear()

    def guard(self) -> None:
        """Raise if a request should not be attempted right now."""
        if not self.allow_request():
            raise GatewayError(
                f"circuit breaker is {self._state.value}",
                code=ErrorCode.CIRCUIT_BREAKER_OPEN,
                stage=Stage.SERVE,
                context={"state": self._state.value},
                retryable=True,
            )


__all__ = ["CircuitBreaker", "CircuitBreakerConfig", "CircuitState"]
