from modelhub.data.normalize import (
    normalize_bird_record,
    normalize_minidev_record,
    normalize_spider_record,
)
from modelhub.data.schema import Dialect, Source, Split


def test_normalize_bird_record_maps_all_fields() -> None:
    raw = {
        "question_id": 42,
        "db_id": "school",
        "question": "How many students?",
        "evidence": "students table has one row per student",
        "SQL": "SELECT COUNT(*) FROM students",
        "difficulty": "simple",
    }
    sample = normalize_bird_record(raw, split=Split.DEV)
    assert sample.sample_id == "bird:dev:42"
    assert sample.db_id == "school"
    assert sample.question == "How many students?"
    assert sample.evidence == "students table has one row per student"
    assert sample.gold_sql == "SELECT COUNT(*) FROM students"
    assert sample.difficulty == "simple"
    assert sample.source is Source.BIRD
    assert sample.split is Split.DEV
    assert sample.dialect is Dialect.SQLITE


def test_normalize_bird_record_missing_required_key_raises() -> None:
    raw = {"db_id": "school", "question": "x", "SQL": "SELECT 1"}  # no question_id
    try:
        normalize_bird_record(raw, split=Split.TRAIN)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_normalize_bird_record_train_has_no_difficulty_field() -> None:
    # BIRD train.json doesn't ship a difficulty label; must come through as None.
    raw = {"question_id": 1, "db_id": "school", "question": "x", "SQL": "SELECT 1"}
    sample = normalize_bird_record(raw, split=Split.TRAIN)
    assert sample.difficulty is None


def test_normalize_spider_record_computes_hardness() -> None:
    raw = {
        "db_id": "school",
        "question": "How many students?",
        "query": "SELECT COUNT(*) FROM students",
    }
    sample = normalize_spider_record(raw, split=Split.DEV, index=0)
    assert sample.source is Source.SPIDER
    assert sample.sample_id == "spider:dev:0"
    assert sample.difficulty in ("easy", "medium", "hard", "extra")
    assert sample.evidence is None


def test_normalize_minidev_select_record() -> None:
    raw = {
        "question_id": 7,
        "db_id": "school",
        "question": "How many students?",
        "SQL": "SELECT COUNT(*) FROM students",
    }
    sample = normalize_minidev_record(raw, source=Source.MINIDEV_SELECT, dialect=Dialect.MYSQL)
    assert sample.sample_id == "minidev_select:dev:7:mysql"
    assert sample.source is Source.MINIDEV_SELECT
    assert sample.dialect is Dialect.MYSQL


def test_normalize_minidev_crud_record() -> None:
    raw = {
        "question_id": 8,
        "db_id": "school",
        "question": "delete a student",
        "SQL": "DELETE FROM students",
    }
    sample = normalize_minidev_record(raw, source=Source.MINIDEV_CRUD, dialect=Dialect.SQLITE)
    assert sample.source is Source.MINIDEV_CRUD
