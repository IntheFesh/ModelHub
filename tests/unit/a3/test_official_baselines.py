from pathlib import Path

from modelhub.compare.official_baselines import bird_official_compare, bird_official_execute_sql


def test_bird_official_compare_matching_results(sqlite_db: Path) -> None:
    assert (
        bird_official_compare(
            "SELECT COUNT(*) FROM students", "SELECT COUNT(*) FROM students", str(sqlite_db)
        )
        == 1
    )


def test_bird_official_compare_differing_results(sqlite_db: Path) -> None:
    result = bird_official_compare(
        "SELECT id FROM students WHERE id = 1",
        "SELECT id FROM students WHERE id = 2",
        str(sqlite_db),
    )
    assert result == 0


def test_bird_official_compare_collapses_duplicates_unlike_our_comparator(sqlite_db: Path) -> None:
    # This is the concrete demonstration cited in official_baselines.py's
    # docstring: BIRD's real scorer uses set(), so a predicted query that
    # returns a row 3x "matches" gold returning it once.
    dup_query = (
        "SELECT gpa FROM students WHERE id = 1 "
        "UNION ALL SELECT gpa FROM students WHERE id = 1 "
        "UNION ALL SELECT gpa FROM students WHERE id = 1"
    )
    single_query = "SELECT gpa FROM students WHERE id = 1"
    assert bird_official_compare(dup_query, single_query, str(sqlite_db)) == 1


def test_bird_official_compare_bad_sql_scores_zero_not_raises(sqlite_db: Path) -> None:
    assert bird_official_compare("SELCT garbage", "SELECT 1", str(sqlite_db)) == 0


def test_bird_official_execute_sql_raises_on_bad_predicted_sql(sqlite_db: Path) -> None:
    # Unlike bird_official_compare, the lower-level function is a
    # deliberately faithful port that lets exceptions propagate — see
    # its own docstring.
    import pytest

    with pytest.raises(Exception):  # noqa: B017 -- sqlite3 raises its own OperationalError subtype
        bird_official_execute_sql("SELCT garbage", "SELECT 1", str(sqlite_db))
