"""Deduplication: question-level and SQL-level, with counts — never silent.

CLAUDE.md's data-quality discipline applies here too: dropping duplicates
without recording how many and which is the same class of "looks clean but
you can't audit it" problem as silently truncating a result set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from modelhub.common.config import ModelHubBaseConfig
from modelhub.data.hashing import question_hash, sql_hash
from modelhub.data.schema import NormalizedSample


class DedupStats(ModelHubBaseConfig):
    input_count: int
    output_count: int
    question_level_duplicates: int
    sql_level_duplicates_within_kept: int


@dataclass(frozen=True)
class DedupResult:
    samples: list[NormalizedSample]
    stats: DedupStats
    dropped_sample_ids: list[str] = field(default_factory=list)


def dedup_samples(samples: list[NormalizedSample]) -> DedupResult:
    """Drop exact question-level duplicates (first occurrence wins, stable
    order), and separately count — but do not drop — SQL-level duplicates
    among what's kept (two different questions legitimately can share the
    same gold SQL, e.g. "how many students" phrased two ways is a dedup
    bug, but two genuinely different questions that both reduce to
    `SELECT COUNT(*) FROM students` are not)."""
    seen_question_hashes: set[str] = set()
    kept: list[NormalizedSample] = []
    dropped_ids: list[str] = []

    for sample in samples:
        qh = question_hash(sample.question)
        if qh in seen_question_hashes:
            dropped_ids.append(sample.sample_id)
            continue
        seen_question_hashes.add(qh)
        kept.append(sample)

    sql_hash_counts: dict[str, int] = {}
    for sample in kept:
        sh = sql_hash(sample.gold_sql)
        sql_hash_counts[sh] = sql_hash_counts.get(sh, 0) + 1
    sql_level_duplicates = sum(count - 1 for count in sql_hash_counts.values() if count > 1)

    stats = DedupStats(
        input_count=len(samples),
        output_count=len(kept),
        question_level_duplicates=len(dropped_ids),
        sql_level_duplicates_within_kept=sql_level_duplicates,
    )
    return DedupResult(samples=kept, stats=stats, dropped_sample_ids=dropped_ids)
