"""Unit tests for serve/prompt.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.serve.prompt import PromptConfig, build_prompt, make_prompt_builder
from modelhub.serve.schema_format import introspect_sqlite_schema, render_schema_text


def _sample(**overrides: object) -> NormalizedSample:
    defaults: dict[str, object] = {
        "sample_id": "s1",
        "db_id": "school",
        "question": "How many students are there?",
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
    conn.commit()
    conn.close()
    return db_root


def test_build_prompt_includes_instruction_schema_and_question() -> None:
    sample = _sample()
    prompt = build_prompt(sample, "CREATE TABLE students (id INTEGER);", PromptConfig())
    assert "SQLite expert" in prompt
    assert "CREATE TABLE students (id INTEGER);" in prompt
    assert sample.question in prompt


def test_evidence_included_when_present_and_enabled() -> None:
    sample = _sample(evidence="gpa > 3.5 means honor roll")
    prompt = build_prompt(sample, "schema", PromptConfig(include_evidence=True))
    assert "gpa > 3.5 means honor roll" in prompt
    assert "### Evidence" in prompt


def test_evidence_omitted_when_absent() -> None:
    sample = _sample(evidence=None)
    prompt = build_prompt(sample, "schema", PromptConfig(include_evidence=True))
    assert "### Evidence" not in prompt


def test_evidence_omitted_when_disabled_even_if_present() -> None:
    sample = _sample(evidence="some hint")
    prompt = build_prompt(sample, "schema", PromptConfig(include_evidence=False))
    assert "### Evidence" not in prompt
    assert "some hint" not in prompt


def test_make_prompt_builder_produces_working_callable(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    sample = _sample()
    builder = make_prompt_builder(db_root)
    prompt = builder(sample)
    assert "CREATE TABLE students" in prompt
    assert sample.question in prompt


def test_prompt_builder_matches_manual_construction(tmp_path: Path) -> None:
    db_root = _db_root(tmp_path)
    sample = _sample()
    config = PromptConfig()
    builder = make_prompt_builder(db_root, config)

    schema = introspect_sqlite_schema(db_root / "school" / "school.sqlite", db_id="school")
    expected_schema_text = render_schema_text(schema, style=config.schema_style)
    expected = build_prompt(sample, expected_schema_text, config)

    assert builder(sample) == expected


def test_schema_cache_introspects_each_db_id_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import modelhub.serve.prompt as prompt_module

    db_root = _db_root(tmp_path)
    real_introspect = prompt_module.introspect_sqlite_schema
    calls = {"n": 0}

    def counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_introspect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(prompt_module, "introspect_sqlite_schema", counting)

    builder = make_prompt_builder(db_root)
    builder(_sample(sample_id="s1"))
    builder(_sample(sample_id="s2"))
    builder(_sample(sample_id="s3"))

    assert calls["n"] == 1
