"""Online sampling validation: pull a reproducible sample of recent
production traffic and turn it into the `CanaryHealthSample`s
`rollback.py`'s automatic-rollback decision consumes.

There is no gold answer available for live production traffic (that's
what makes it "online" rather than an offline eval run) — this module
does not try to solve that; it only summarizes what sqlexec itself
already knows for certain about each request (did the predicted SQL
execute, was any failure a harness-side one) into a health signal,
exactly like a real production monitor would.

CLAUDE.md §7 "采样类评估必须记录 seed": sampling is seeded and
deterministic, never `random.random()` with no recorded seed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from modelhub.common.errors import ErrorCode
from modelhub.eval.records import PredictionRecord
from modelhub.release.rollback import CanaryHealthSample


def sample_recent_predictions(
    predictions: Sequence[PredictionRecord], *, sample_size: int, seed: int
) -> list[PredictionRecord]:
    if sample_size < 0:
        raise ValueError(f"sample_size must be non-negative, got {sample_size}")
    if sample_size >= len(predictions):
        return list(predictions)
    rng = random.Random(seed)
    return rng.sample(list(predictions), sample_size)


def predictions_to_health_samples(
    predictions: Sequence[PredictionRecord],
) -> list[CanaryHealthSample]:
    """`succeeded` is exactly `exec_code == EXEC_OK` — sqlexec's own
    success flag, not a partial or hand-picked subset of codes. A
    canary producing more SYNTAX/SEMANTIC/TIMEOUT errors than the
    stable arm is a real regression signal for health monitoring
    purposes, not something to quietly exclude."""
    return [
        CanaryHealthSample(
            succeeded=p.exec_code is ErrorCode.EXEC_OK,
            is_harness_error=p.is_harness_error,
            latency_s=p.elapsed_s,
        )
        for p in predictions
    ]


__all__ = ["predictions_to_health_samples", "sample_recent_predictions"]
