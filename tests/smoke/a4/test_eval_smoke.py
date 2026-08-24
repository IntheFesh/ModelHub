"""A4 smoke test: the full pipeline at a tiny scale — generate -> execute
-> compare -> dump -> metrics -> report — against a real SQLite database,
with only the model-generation step faked (no served model in this
sandbox). CLAUDE.md §1.4: this is the only sanctioned shape a smoke test
may take (shrink the input, never replace an implementation) — every stage
below is the real production code path.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tests.unit.a4.fakes import make_valid_manifest, scripted_client

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.gold_cache import GoldExecCache
from modelhub.eval.metrics import compute_metrics
from modelhub.eval.model_client import GenerationResult
from modelhub.eval.report import write_report
from modelhub.eval.runner import load_predictions, run_eval

pytestmark = pytest.mark.smoke


def _build_db(db_root: Path, db_id: str) -> None:
    db_dir = db_root / db_id
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / f"{db_id}.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL)")
    conn.executemany(
        "INSERT INTO students VALUES (?, ?, ?)",
        [(1, "alice", 3.5), (2, "bob", 3.0), (3, "carol", 3.8), (4, "dan", 2.9)],
    )
    conn.commit()
    conn.close()


def _sample(sample_id: str, gold_sql: str, difficulty: str) -> NormalizedSample:
    return NormalizedSample.model_validate(
        {
            "sample_id": sample_id,
            "db_id": "school",
            "question": "irrelevant for this smoke test",
            "evidence": None,
            "gold_sql": gold_sql,
            "difficulty": difficulty,
            "source": Source.BIRD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )


def test_eval_pipeline_smoke(tmp_path: Path) -> None:
    db_root = tmp_path / "dbs"
    _build_db(db_root, "school")

    samples = [
        _sample("s1", "SELECT COUNT(*) FROM students", "simple"),  # will match: EQUAL
        _sample("s2", "SELECT COUNT(*) FROM students", "simple"),  # will differ: NOT_EQUAL
        _sample(
            "s3", "SELECT COUNT(*) FROM students WHERE gpa > 5.0", "moderate"
        ),  # exact match: EQUAL
        _sample("s4", "SELECT * FROM no_such_table", "hard"),  # gold fails -> UNDECIDABLE
    ]
    responses = {
        "s1": GenerationResult(
            text="SELECT COUNT(*) FROM students", finish_reason="stop", model_id="fake"
        ),
        "s2": GenerationResult(
            text="SELECT COUNT(*) FROM students WHERE gpa > 3.5",
            finish_reason="stop",
            model_id="fake",
        ),
        "s3": GenerationResult(
            text="SELECT COUNT(*) FROM students WHERE gpa > 5.0",
            finish_reason="stop",
            model_id="fake",
        ),
        "s4": GenerationResult(
            text="SELECT COUNT(*) FROM students", finish_reason="stop", model_id="fake"
        ),
    }
    client = scripted_client(responses)

    predictions_path = tmp_path / "predictions.jsonl"
    records = run_eval(
        samples,
        model_client=client,
        prompt_builder=lambda s: s.sample_id,
        db_root=db_root,
        predictions_path=predictions_path,
        gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
    )
    assert len(records) == 4
    assert predictions_path.is_file()

    reloaded = load_predictions(predictions_path)
    assert {r.sample_id for r in reloaded} == {"s1", "s2", "s3", "s4"}

    metrics = compute_metrics(records)
    assert metrics.total_samples == 4
    assert metrics.excluded_undecidable == 1  # s4's gold SQL fails
    assert metrics.denominator == 3
    assert metrics.equal_count == 2  # s1 and s3

    manifest = make_valid_manifest(run_id="smoke-a4-run", n_samples=4, eval_tier="quick")
    report_file = write_report(manifest, metrics, artifacts_root=tmp_path / "artifacts" / "runs")
    assert report_file.is_file()
    content = report_file.read_text(encoding="utf-8")
    assert "quick" in content
    assert "denominator: 3" in content
