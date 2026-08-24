"""CLAUDE.md §1.5 required meta-test: `test_eval_fails_on_exec_error_flood`.

Injected condition: "沙箱 80% 报错" — 80% of predicted-SQL executions come
back as a HARNESS_* code (the sandbox/DB itself failing, not the model).
Asserted behavior: the eval pipeline refuses to produce a report at all
(HarnessErrorFloodError), rather than silently averaging the harness
failures into a low-but-plausible-looking accuracy number. This exercises
the real pipeline end to end (run_one_sample -> compute_metrics), not just
a hand-built list of PredictionRecords, so it also proves run_one_sample
actually classifies "predicted SQL against a missing db file" as
HARNESS_DB_UNAVAILABLE rather than something else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a4.fakes import scripted_client

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.gold_cache import GoldExecCache
from modelhub.eval.metrics import HarnessErrorFloodError, compute_metrics
from modelhub.eval.model_client import GenerationResult
from modelhub.eval.runner import run_one_sample


def _sample(sample_id: str, db_id: str) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "db_id": db_id,
            "question": "irrelevant",
            "evidence": None,
            "gold_sql": "SELECT 1",
            "difficulty": "simple",
            "source": Source.BIRD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )


def _make_real_db(db_root: Path, db_id: str) -> None:
    import sqlite3

    db_dir = db_root / db_id
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / f"{db_id}.sqlite"))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()


def test_eval_fails_on_exec_error_flood(tmp_path: Path) -> None:
    db_root = tmp_path / "dbs"
    total = 100
    n_harness = 80  # 80% — well above the 1% HARNESS_ERROR_FLOOD_THRESHOLD

    # Only n_harness-many db_ids get a real db file on disk; the rest point
    # at a db_id with no file at all, which sqlexec genuinely classifies as
    # HARNESS_DB_UNAVAILABLE (FileNotFoundError) — a real sandbox failure,
    # not a fabricated error code.
    healthy_db_id = "healthy_db"
    _make_real_db(db_root, healthy_db_id)

    samples = []
    responses = {}
    for i in range(total):
        sample_id = f"s{i}"
        is_harness_broken = i < n_harness
        db_id = f"missing_db_{i}" if is_harness_broken else healthy_db_id
        samples.append(_sample(sample_id, db_id))
        responses[sample_id] = GenerationResult(
            text="SELECT 1", finish_reason="stop", model_id="fake-model"
        )

    client = scripted_client(responses)
    gold_cache = GoldExecCache(cache_dir=tmp_path / "cache")

    records = [
        run_one_sample(
            sample,
            model_client=client,
            prompt=sample.sample_id,
            db_root=db_root,
            gold_cache=gold_cache,
            temperature=0.0,
            max_tokens=64,
            generate_timeout_s=5.0,
        )
        for sample in samples
    ]

    harness_records = [r for r in records if r.is_harness_error]
    assert len(harness_records) == n_harness, (
        "test setup assumption broken: expected exactly the missing-db "
        "samples to classify as harness errors"
    )

    with pytest.raises(HarnessErrorFloodError) as exc_info:
        compute_metrics(records)

    assert exc_info.value.harness_count == n_harness
    assert exc_info.value.total == total


def test_eval_does_not_flag_a_healthy_run_this_meta_test_is_not_always_red(
    tmp_path: Path,
) -> None:
    db_root = tmp_path / "dbs"
    healthy_db_id = "healthy_db"
    _make_real_db(db_root, healthy_db_id)

    samples = [_sample(f"s{i}", healthy_db_id) for i in range(10)]
    responses = {
        s.sample_id: GenerationResult(text="SELECT 1", finish_reason="stop", model_id="fake-model")
        for s in samples
    }
    client = scripted_client(responses)
    gold_cache = GoldExecCache(cache_dir=tmp_path / "cache")

    records = [
        run_one_sample(
            sample,
            model_client=client,
            prompt=sample.sample_id,
            db_root=db_root,
            gold_cache=gold_cache,
            temperature=0.0,
            max_tokens=64,
            generate_timeout_s=5.0,
        )
        for sample in samples
    ]

    metrics = compute_metrics(records)  # must not raise
    assert metrics.total_samples == 10
