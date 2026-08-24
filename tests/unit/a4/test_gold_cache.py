"""Unit tests for eval/gold_cache.py.

CLAUDE.md §7.1: gold SQL execution results are cached by
`(db_id, gold_sql_hash, db_file_hash)`. The caching test below is a real
spy (it wraps the actual `execute_isolated`, still runs it for real, and
just counts calls) rather than a mock that fakes execution — CLAUDE.md
§1.4 confines mocks/fakes to `tests/`, but even there this project prefers
proving behavior against the real implementation wherever practical.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import modelhub.eval.gold_cache as gold_cache_module
from modelhub.eval.gold_cache import GoldExecCache, cache_key, db_file_hash


def test_db_file_hash_is_deterministic(sqlite_db: Path) -> None:
    assert db_file_hash(sqlite_db) == db_file_hash(sqlite_db)


def test_db_file_hash_changes_with_content(sqlite_db: Path, tmp_path: Path) -> None:
    before = db_file_hash(sqlite_db)
    conn = sqlite3.connect(str(sqlite_db))
    conn.execute("INSERT INTO students (id, name, gpa) VALUES (999, 'zzz', 4.0)")
    conn.commit()
    conn.close()
    after = db_file_hash(sqlite_db)
    assert before != after


def test_cache_key_deterministic_and_sensitive_to_each_component() -> None:
    base = cache_key("db1", "SELECT 1", "hashA")
    assert base == cache_key("db1", "SELECT 1", "hashA")
    assert base != cache_key("db2", "SELECT 1", "hashA")
    assert base != cache_key("db1", "SELECT 2", "hashA")
    assert base != cache_key("db1", "SELECT 1", "hashB")


def _spy_on_execute_isolated(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    real = gold_cache_module.execute_isolated
    calls = {"n": 0}

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gold_cache_module, "execute_isolated", counting)
    return calls


def test_get_or_execute_caches_across_calls(
    tmp_path: Path, sqlite_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_on_execute_isolated(monkeypatch)
    cache = GoldExecCache(cache_dir=tmp_path / "cache")

    outcome1 = cache.get_or_execute(
        db_id="school", gold_sql="SELECT COUNT(*) FROM students", db_path=sqlite_db
    )
    assert calls["n"] == 1
    assert outcome1.ok

    outcome2 = cache.get_or_execute(
        db_id="school", gold_sql="SELECT COUNT(*) FROM students", db_path=sqlite_db
    )
    assert calls["n"] == 1, "second call with identical inputs must not re-execute"
    assert outcome2.result == outcome1.result


def test_get_or_execute_caches_across_new_cache_instances(
    tmp_path: Path, sqlite_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is persisted to disk, not just held in an in-process dict."""
    calls = _spy_on_execute_isolated(monkeypatch)
    cache_dir = tmp_path / "cache"

    GoldExecCache(cache_dir=cache_dir).get_or_execute(
        db_id="school", gold_sql="SELECT COUNT(*) FROM students", db_path=sqlite_db
    )
    assert calls["n"] == 1

    GoldExecCache(cache_dir=cache_dir).get_or_execute(
        db_id="school", gold_sql="SELECT COUNT(*) FROM students", db_path=sqlite_db
    )
    assert calls["n"] == 1


def test_get_or_execute_invalidates_when_db_file_changes(
    tmp_path: Path, sqlite_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_on_execute_isolated(monkeypatch)
    cache = GoldExecCache(cache_dir=tmp_path / "cache")

    cache.get_or_execute(
        db_id="school", gold_sql="SELECT COUNT(*) FROM students", db_path=sqlite_db
    )
    assert calls["n"] == 1

    conn = sqlite3.connect(str(sqlite_db))
    conn.execute("INSERT INTO students (id, name, gpa) VALUES (999, 'zzz', 4.0)")
    conn.commit()
    conn.close()

    cache.get_or_execute(
        db_id="school", gold_sql="SELECT COUNT(*) FROM students", db_path=sqlite_db
    )
    assert calls["n"] == 2, "a changed db file must invalidate the cache, not serve a stale result"


def test_get_or_execute_caches_failures_too(
    tmp_path: Path, sqlite_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _spy_on_execute_isolated(monkeypatch)
    cache = GoldExecCache(cache_dir=tmp_path / "cache")

    outcome1 = cache.get_or_execute(
        db_id="school", gold_sql="SELECT * FROM no_such_table", db_path=sqlite_db
    )
    assert not outcome1.ok
    assert calls["n"] == 1

    outcome2 = cache.get_or_execute(
        db_id="school", gold_sql="SELECT * FROM no_such_table", db_path=sqlite_db
    )
    assert calls["n"] == 1
    assert outcome2.code == outcome1.code
