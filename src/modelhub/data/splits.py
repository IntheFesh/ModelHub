"""Split integrity: train/dev/test must never overlap.

CLAUDE.md §7: "train/dev/test 划分固定并做 hash 校验，CI 断言三者交集为空."
The key is (question_hash, db_id) rather than question_hash alone: the
same question text against two different databases is a legitimately
different example, not a leak; the same question text against the same
db_id in two different splits is a real leak.
"""

from __future__ import annotations

from collections.abc import Sequence

from modelhub.common.errors import ErrorCode, ModelHubError, Stage
from modelhub.data.hashing import question_hash
from modelhub.data.schema import NormalizedSample, Split


def _keys(samples: Sequence[NormalizedSample]) -> dict[tuple[str, str], str]:
    """(question_hash, db_id) -> sample_id, for reporting exactly which ids leaked."""
    return {(question_hash(s.question), s.db_id): s.sample_id for s in samples}


def assert_splits_disjoint(
    train: Sequence[NormalizedSample],
    dev: Sequence[NormalizedSample],
    test: Sequence[NormalizedSample],
) -> None:
    """Raise ``ModelHubError`` (never a silent warning) if any two of the
    three splits share a (question, db_id) pair."""
    train_keys = _keys(train)
    dev_keys = _keys(dev)
    test_keys = _keys(test)

    pairs = (
        ("train", "dev", train_keys, dev_keys),
        ("train", "test", train_keys, test_keys),
        ("dev", "test", dev_keys, test_keys),
    )
    leaks: dict[str, list[dict[str, str]]] = {}
    for name_a, name_b, keys_a, keys_b in pairs:
        overlap = set(keys_a) & set(keys_b)
        if overlap:
            leaks[f"{name_a}∩{name_b}"] = [
                {f"{name_a}_sample_id": keys_a[k], f"{name_b}_sample_id": keys_b[k]}
                for k in overlap
            ]

    if leaks:
        total = sum(len(v) for v in leaks.values())
        raise ModelHubError(
            f"split integrity violated: {total} (question, db_id) pair(s) "
            f"appear in more than one split",
            code=ErrorCode.DATA_SPLIT_LEAK,
            stage=Stage.DATA,
            context={"leaks": leaks},
            retryable=False,
        )


def split_of(samples: Sequence[NormalizedSample], split: Split) -> list[NormalizedSample]:
    return [s for s in samples if s.split is split]
