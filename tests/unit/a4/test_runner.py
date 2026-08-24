"""Unit tests for eval/runner.py.

Every test here exercises real SQL execution and real comparison against a
real SQLite database (tmp_path fixtures) — only the model-generation step
is faked (there is no served model in this sandbox), per
tests/unit/a4/fakes.py's docstring.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from tests.unit.a4.fakes import fixed_response_client, scripted_client

import modelhub.eval.runner as runner_module
from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.gold_cache import GoldExecCache
from modelhub.eval.model_client import GenerationResult
from modelhub.eval.records import PredictionRecord
from modelhub.eval.runner import load_predictions, run_eval, run_one_sample


def _sample(**overrides: object) -> NormalizedSample:
    defaults: dict[str, object] = {
        "sample_id": "s1",
        "db_id": "school",
        "question": "how many students are there?",
        "evidence": None,
        "gold_sql": "SELECT COUNT(*) FROM students",
        "difficulty": "simple",
        "source": Source.BIRD,
        "split": Split.DEV,
        "dialect": Dialect.SQLITE,
    }
    defaults.update(overrides)
    return NormalizedSample.model_validate(defaults)


def _db_root(tmp_path: Path, db_id: str = "school") -> Path:
    db_root = tmp_path / "dbs"
    db_dir = db_root / db_id
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / f"{db_id}.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL)")
    conn.executemany(
        "INSERT INTO students VALUES (?, ?, ?)",
        [(1, "alice", 3.5), (2, "bob", 3.0), (3, "carol", 3.8)],
    )
    conn.commit()
    conn.close()
    return db_root


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> dict[str, int]:
    real = getattr(runner_module, name)
    calls = {"n": 0}

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)  # type: ignore[operator]

    monkeypatch.setattr(runner_module, name, counting)
    return calls


def test_output_truncated_never_reaches_sql_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exec_calls = _spy(monkeypatch, "execute_isolated")
    sample = _sample()
    client = fixed_response_client("SELECT COUN", finish_reason="length")

    record = run_one_sample(
        sample,
        model_client=client,
        prompt="s1",
        db_root=_db_root(tmp_path),
        gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
        temperature=0.0,
        max_tokens=8,
        generate_timeout_s=5.0,
    )

    assert record.exec_code is ErrorCode.OUTPUT_TRUNCATED
    assert record.comparison_result is None
    assert exec_calls["n"] == 0


def test_predicted_sql_failure_skips_gold_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold_cache = GoldExecCache(cache_dir=tmp_path / "cache")
    gold_calls = {"n": 0}
    real_get_or_execute = gold_cache.get_or_execute

    def counting_get_or_execute(**kwargs: object) -> object:
        gold_calls["n"] += 1
        return real_get_or_execute(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gold_cache, "get_or_execute", counting_get_or_execute)

    sample = _sample()
    client = fixed_response_client("SELECT COUNT( FROM students", finish_reason="stop")

    record = run_one_sample(
        sample,
        model_client=client,
        prompt="s1",
        db_root=_db_root(tmp_path),
        gold_cache=gold_cache,
        temperature=0.0,
        max_tokens=64,
        generate_timeout_s=5.0,
    )

    assert record.exec_code is ErrorCode.SYNTAX
    assert record.comparison_result is None
    assert gold_calls["n"] == 0


def test_gold_sql_failure_is_undecidable_not_a_model_error(tmp_path: Path) -> None:
    sample = _sample(gold_sql="SELECT * FROM no_such_table")
    client = fixed_response_client("SELECT COUNT(*) FROM students", finish_reason="stop")

    record = run_one_sample(
        sample,
        model_client=client,
        prompt="s1",
        db_root=_db_root(tmp_path),
        gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
        temperature=0.0,
        max_tokens=64,
        generate_timeout_s=5.0,
    )

    assert record.exec_code is ErrorCode.EXEC_OK
    assert record.comparison_result is ComparisonResult.UNDECIDABLE
    assert record.undecidable_reason is UndecidableReason.GOLD_EXEC_FAILED


def test_matching_predicted_and_gold_sql_is_equal(tmp_path: Path) -> None:
    sample = _sample(gold_sql="SELECT COUNT(*) FROM students")
    client = fixed_response_client("SELECT COUNT(*) FROM students", finish_reason="stop")

    record = run_one_sample(
        sample,
        model_client=client,
        prompt="s1",
        db_root=_db_root(tmp_path),
        gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
        temperature=0.0,
        max_tokens=64,
        generate_timeout_s=5.0,
    )

    assert record.exec_code is ErrorCode.EXEC_OK
    assert record.comparison_result is ComparisonResult.EQUAL


def test_differing_predicted_and_gold_sql_is_not_equal(tmp_path: Path) -> None:
    sample = _sample(gold_sql="SELECT COUNT(*) FROM students")
    client = fixed_response_client(
        "SELECT COUNT(*) FROM students WHERE gpa > 3.5", finish_reason="stop"
    )

    record = run_one_sample(
        sample,
        model_client=client,
        prompt="s1",
        db_root=_db_root(tmp_path),
        gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
        temperature=0.0,
        max_tokens=64,
        generate_timeout_s=5.0,
    )

    assert record.exec_code is ErrorCode.EXEC_OK
    assert record.comparison_result is ComparisonResult.NOT_EQUAL


def test_run_eval_resume_skips_already_recorded_samples(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    existing = PredictionRecord.model_validate(
        {
            "sample_id": "s1",
            "db_id": "school",
            "difficulty": "simple",
            "predicted_sql": "SELECT 1",
            "finish_reason": "stop",
            "exec_code": ErrorCode.EXEC_OK,
            "comparison_result": ComparisonResult.EQUAL,
            "elapsed_s": 0.05,
        }
    )
    predictions_path.write_text(existing.model_dump_json() + "\n", encoding="utf-8")

    samples = [
        _sample(sample_id="s1", gold_sql="SELECT COUNT(*) FROM students"),
        _sample(sample_id="s2", gold_sql="SELECT COUNT(*) FROM students"),
    ]
    # Only scripted for s2's prompt — if s1 were (wrongly) regenerated this
    # would raise KeyError and fail the test.
    client = scripted_client(
        {
            "s2": GenerationResult(
                text="SELECT COUNT(*) FROM students", finish_reason="stop", model_id="m"
            )
        }
    )

    records = run_eval(
        samples,
        model_client=client,
        prompt_builder=lambda s: s.sample_id,
        db_root=_db_root(tmp_path),
        predictions_path=predictions_path,
        gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
    )

    assert {r.sample_id for r in records} == {"s1", "s2"}
    assert client.prompts == ["s2"]
    s1_record = next(r for r in records if r.sample_id == "s1")
    assert s1_record == existing


def test_run_eval_persists_predictions_atomically(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    samples = [
        _sample(sample_id="s1", gold_sql="SELECT COUNT(*) FROM students"),
        _sample(sample_id="s2", gold_sql="SELECT COUNT(*) FROM students"),
    ]
    client = scripted_client(
        {
            "s1": GenerationResult(
                text="SELECT COUNT(*) FROM students", finish_reason="stop", model_id="m"
            ),
            "s2": GenerationResult(text="SELECT 1", finish_reason="stop", model_id="m"),
        }
    )

    run_eval(
        samples,
        model_client=client,
        prompt_builder=lambda s: s.sample_id,
        db_root=_db_root(tmp_path),
        predictions_path=predictions_path,
        gold_cache=GoldExecCache(cache_dir=tmp_path / "cache"),
    )

    loaded = load_predictions(predictions_path)
    assert {r.sample_id for r in loaded} == {"s1", "s2"}
    lines = predictions_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
