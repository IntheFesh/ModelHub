"""Dataset filtering for B3's regression (ckpt-B) and safety (ckpt-D)
drill profiles.

`NormalizedSample.difficulty` is deliberately kept as a raw, source-
scoped string (see `data/schema.py`) — BIRD's "simple" and Spider's
"easy" are never mapped onto one shared ordinal anywhere in this
project, so PLAN.md's "Spider easy + BIRD simple" is expressed here as
an explicit per-source predicate, not a single threshold on some
invented cross-dataset difficulty scale.
"""

from __future__ import annotations

from collections.abc import Sequence

from modelhub.data.schema import NormalizedSample, Source

_SPIDER_EASY = "easy"
_BIRD_SIMPLE = "simple"


def filter_easy_only(samples: Sequence[NormalizedSample]) -> list[NormalizedSample]:
    """PLAN.md ckpt-B: "只用 Spider easy + BIRD simple 训 200 步"."""
    filtered = [
        s
        for s in samples
        if (s.source is Source.SPIDER and s.difficulty == _SPIDER_EASY)
        or (s.source is Source.BIRD and s.difficulty == _BIRD_SIMPLE)
    ]
    if not filtered:
        raise ValueError(
            f"filter_easy_only matched 0 of {len(samples)} samples — no Spider "
            f"difficulty={_SPIDER_EASY!r} or BIRD difficulty={_BIRD_SIMPLE!r} samples found"
        )
    return filtered


def inject_crud_samples(
    samples: Sequence[NormalizedSample], crud_samples: Sequence[NormalizedSample], *, count: int
) -> list[NormalizedSample]:
    """PLAN.md ckpt-D: "训练数据混入 200 条含 DELETE/UPDATE 的样本". Every
    `crud_samples` entry must already be `Source.MINIDEV_CRUD` (the same
    real adversarial-set convention `gate/safety_gate.py` enforces) —
    this function refuses to silently accept a wrong-source sample as if
    it were a real CRUD example."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if len(crud_samples) < count:
        raise ValueError(
            f"requested {count} CRUD samples but only {len(crud_samples)} were supplied"
        )
    wrong_source = [s for s in crud_samples if s.source is not Source.MINIDEV_CRUD]
    if wrong_source:
        raise ValueError(
            f"{len(wrong_source)} sample(s) passed as CRUD injection are not "
            f"Source.MINIDEV_CRUD: {[s.sample_id for s in wrong_source][:10]}"
        )
    return [*samples, *crud_samples[:count]]


__all__ = ["filter_easy_only", "inject_crud_samples"]
