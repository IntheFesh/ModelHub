"""Map each source's raw record shape into the canonical `NormalizedSample`.

A missing required key is a hard failure (`KeyError` propagates), never a
silently-defaulted empty string — a raw record that doesn't match the
documented shape means something upstream changed and needs a human, not
a normalizer that quietly produces garbage.
"""

from __future__ import annotations

from typing import Any

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.data.spider_hardness import estimate_spider_hardness


def normalize_bird_record(raw: dict[str, Any], *, split: Split) -> NormalizedSample:
    """BIRD raw record: question_id, db_id, question, evidence, SQL, [difficulty]."""
    return NormalizedSample(
        sample_id=f"bird:{split.value}:{raw['question_id']}",
        db_id=raw["db_id"],
        question=raw["question"],
        evidence=raw.get("evidence") or None,
        gold_sql=raw["SQL"],
        difficulty=raw.get("difficulty"),
        source=Source.BIRD,
        split=split,
        dialect=Dialect.SQLITE,
    )


def normalize_spider_record(raw: dict[str, Any], *, split: Split, index: int) -> NormalizedSample:
    """Spider raw record: db_id, question, query. No native difficulty field —
    hardness is estimated (see spider_hardness.py's honesty caveat)."""
    gold_sql = raw["query"]
    return NormalizedSample(
        sample_id=f"spider:{split.value}:{index}",
        db_id=raw["db_id"],
        question=raw["question"],
        evidence=None,
        gold_sql=gold_sql,
        difficulty=estimate_spider_hardness(gold_sql).value,
        source=Source.SPIDER,
        split=split,
    )


def normalize_minidev_record(
    raw: dict[str, Any], *, source: Source, dialect: Dialect
) -> NormalizedSample:
    """Mini-Dev V2 record. `source` distinguishes the 500 SELECT-only
    (quick-eval) samples from the 270 CRUD (safety-adversarial) samples —
    they must never be mixed (CLAUDE.md/A1: mixing them into quick-eval
    makes the safety gate "block a correct answer")."""
    return NormalizedSample(
        sample_id=f"{source.value}:dev:{raw['question_id']}:{dialect.value}",
        db_id=raw["db_id"],
        question=raw["question"],
        evidence=raw.get("evidence") or None,
        gold_sql=raw["SQL"],
        difficulty=raw.get("difficulty"),
        source=source,
        split=Split.DEV,
        dialect=dialect,
    )
