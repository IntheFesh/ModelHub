"""Unit tests for serve/schema_format.py, against a real SQLite database
with an actual foreign key (tmp_path fixtures, not a hand-built dataclass
someone forgot to keep in sync with real introspection)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from modelhub.serve.schema_format import introspect_sqlite_schema, render_schema_text


def _build_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "school.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT NOT NULL, gpa REAL)")
    conn.execute(
        "CREATE TABLE enrollments ("
        "id INTEGER PRIMARY KEY, "
        "student_id INTEGER NOT NULL, "
        "course TEXT NOT NULL, "
        "FOREIGN KEY (student_id) REFERENCES students(id))"
    )
    conn.execute("INSERT INTO students VALUES (1, 'alice', 3.5)")
    conn.execute("INSERT INTO enrollments VALUES (1, 1, 'CS101')")
    conn.commit()
    conn.close()
    return db_path


def test_introspect_reports_tables_in_declaration_order(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    assert schema.table_names() == ("students", "enrollments")


def test_introspect_reports_column_flags(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    students = next(t for t in schema.tables if t.name == "students")
    by_name = {c.name: c for c in students.columns}
    assert by_name["id"].is_primary_key
    assert by_name["name"].not_null
    assert not by_name["gpa"].not_null
    assert not by_name["gpa"].is_primary_key


def test_introspect_reports_foreign_keys(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    assert len(schema.foreign_keys) == 1
    fk = schema.foreign_keys[0]
    assert fk.from_table == "enrollments"
    assert fk.from_column == "student_id"
    assert fk.to_table == "students"
    assert fk.to_column == "id"


def test_ddl_style_includes_primary_key_and_foreign_key(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    text = render_schema_text(schema, style="ddl")
    assert "CREATE TABLE students (" in text
    assert "PRIMARY KEY" in text
    assert "FOREIGN KEY (student_id) REFERENCES students(id)" in text


def test_compact_style_is_shorter_and_lists_foreign_keys(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    ddl = render_schema_text(schema, style="ddl")
    compact = render_schema_text(schema, style="compact")
    assert len(compact) < len(ddl)
    assert "students(id, name, gpa)" in compact
    assert "enrollments.student_id -> students.id" in compact


def test_render_schema_text_rejects_unknown_style(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    with pytest.raises(ValueError, match="unknown schema style"):
        render_schema_text(schema, style="bogus")  # type: ignore[arg-type]


def test_sample_rows_appended_as_comment_in_ddl_style(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    text = render_schema_text(schema, style="ddl", sample_rows={"students": [(1, "alice", 3.5)]})
    assert "-- e.g. (1, 'alice', 3.5)" in text


def test_rendering_is_deterministic(tmp_path: Path) -> None:
    schema = introspect_sqlite_schema(_build_db(tmp_path), db_id="school")
    assert render_schema_text(schema) == render_schema_text(schema)
