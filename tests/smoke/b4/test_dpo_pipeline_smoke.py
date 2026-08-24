"""Smoke test: B4's full preference-pair pipeline at shrunk scale — 5
questions (PLAN.md's real run is 2000) x k=4 real samples each, through
real sqlexec + real compare, all the way to a `PreferenceDatasetReport`
and its benefit assessment (CLAUDE.md §1.4 — only question count shrunk,
every real code path still runs)."""

from __future__ import annotations

import sqlite3
from itertools import cycle
from pathlib import Path

from tests.unit.a4.fakes import FakeModelClient

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.gold_cache import GoldExecCache
from modelhub.eval.model_client import GenerationResult
from modelhub.train.dpo import (
    assess_expected_dpo_benefit,
    build_preference_dataset,
    sample_k_candidates,
)

_SMOKE_QUESTION_COUNT = 5
_K = 4


def _db_root(tmp_path: Path) -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO students VALUES (?, ?)", [(1, "ada"), (2, "grace")])
    conn.commit()
    conn.close()
    return db_root


def _sample(i: int) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": f"smoke-q{i}",
            "db_id": "school",
            "question": f"question {i}",
            "evidence": None,
            "gold_sql": "SELECT COUNT(*) FROM students",
            "difficulty": "simple",
            "source": Source.BIRD,
            "split": Split.TRAIN,
            "dialect": Dialect.SQLITE,
        }
    )


def test_full_pipeline_five_questions_k4(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    gold_cache = GoldExecCache(cache_dir=tmp_path / "cache")
    # a rotating mix of correct/wrong/syntax-error candidates so the
    # smoke run exercises all three tiers plus zero discards, real
    # execution/comparison end to end.
    candidate_texts = cycle(
        [
            "SELECT COUNT(*) FROM students",  # correct
            "SELECT 1",  # exec_ok, wrong
            "SELECT * FROM nope",  # exec fails
            "SELECT COUNT(*) FROM students",  # correct
        ]
    )
    client = FakeModelClient(
        lambda _p: GenerationResult(
            text=next(candidate_texts), finish_reason="stop", model_id="fake"
        )
    )

    predictions_by_question = {}
    for i in range(_SMOKE_QUESTION_COUNT):
        sample = _sample(i)
        predictions_by_question[sample.sample_id] = sample_k_candidates(
            sample,
            model_client=client,
            prompt=sample.sample_id,
            db_root=db_root,
            gold_cache=gold_cache,
            k=_K,
            temperature=0.8,
            max_tokens=256,
            generate_timeout_s=10.0,
        )

    report = build_preference_dataset(predictions_by_question)
    assert report.total_candidates == _SMOKE_QUESTION_COUNT * _K
    assert report.pair_count > 0
    assessment = assess_expected_dpo_benefit(report)
    assert "exec_ok_correct_share" in assessment
