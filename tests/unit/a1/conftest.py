"""Small synthetic fixtures standing in for real BIRD/Spider/Mini-Dev V2 data.

These are NOT downloaded copies of the real datasets (this sandbox cannot
reach the network to fetch them — see docs/design-decisions.md). Field
shapes match the publicly documented schemas so the normalizers and
pipeline logic they exercise are genuinely representative, but the actual
question/SQL content here is invented for testing.
"""

from __future__ import annotations

import pytest

from modelhub.data.normalize import normalize_bird_record, normalize_minidev_record
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split


@pytest.fixture
def bird_train_raw() -> list[dict[str, object]]:
    return [
        {
            "question_id": 1,
            "db_id": "school",
            "question": "How many students are there?",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM students",
        },
        {
            "question_id": 2,
            "db_id": "school",
            "question": "List all student names.",
            "evidence": "",
            "SQL": "SELECT name FROM students",
        },
        {
            # a gold SQL that will fail to execute against the fixture db below
            "question_id": 3,
            "db_id": "school",
            "question": "How many students are enrolled in the nonexistent course?",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM enrollments_typo",
        },
    ]


@pytest.fixture
def bird_dev_raw() -> list[dict[str, object]]:
    return [
        {
            "question_id": 101,
            "db_id": "school",
            "question": "How many students are there?",  # duplicate of train q1
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM students",
            "difficulty": "simple",
        },
        {
            "question_id": 102,
            "db_id": "school",
            "question": "What is the average GPA?",
            "evidence": "",
            "SQL": "SELECT AVG(gpa) FROM students",
            "difficulty": "moderate",
        },
    ]


@pytest.fixture
def bird_train_samples(bird_train_raw: list[dict[str, object]]) -> list[NormalizedSample]:
    return [normalize_bird_record(r, split=Split.TRAIN) for r in bird_train_raw]  # type: ignore[arg-type]


@pytest.fixture
def bird_dev_samples(bird_dev_raw: list[dict[str, object]]) -> list[NormalizedSample]:
    return [normalize_bird_record(r, split=Split.DEV) for r in bird_dev_raw]  # type: ignore[arg-type]


@pytest.fixture
def minidev_select_raw() -> list[dict[str, object]]:
    return [
        {
            "question_id": i,
            "db_id": "school",
            "question": f"SELECT-only question {i}",
            "evidence": "",
            "SQL": "SELECT COUNT(*) FROM students",
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def minidev_crud_raw() -> list[dict[str, object]]:
    sqls = [
        "DELETE FROM students WHERE id = 1",
        "UPDATE students SET gpa = 0",
        "INSERT INTO students (id, name, gpa) VALUES (999, 'x', 1.0)",
        "ALTER TABLE students ADD COLUMN extra TEXT",
    ]
    return [
        {
            "question_id": 100 + i,
            "db_id": "school",
            "question": f"CRUD question {i}",
            "evidence": "",
            "SQL": sql,
        }
        for i, sql in enumerate(sqls)
    ]


@pytest.fixture
def minidev_select_samples(
    minidev_select_raw: list[dict[str, object]],
) -> list[NormalizedSample]:
    return [
        normalize_minidev_record(r, source=Source.MINIDEV_SELECT, dialect=Dialect.SQLITE)  # type: ignore[arg-type]
        for r in minidev_select_raw
    ]


@pytest.fixture
def minidev_crud_samples(minidev_crud_raw: list[dict[str, object]]) -> list[NormalizedSample]:
    return [
        normalize_minidev_record(r, source=Source.MINIDEV_CRUD, dialect=Dialect.SQLITE)  # type: ignore[arg-type]
        for r in minidev_crud_raw
    ]
