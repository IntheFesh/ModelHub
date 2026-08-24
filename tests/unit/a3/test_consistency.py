from pathlib import Path

from modelhub.compare.consistency import run_consistency_check


def test_agreeing_case_is_counted_as_agreement(sqlite_db: Path) -> None:
    cases = [
        ("c1", "SELECT COUNT(*) FROM students", "SELECT COUNT(*) FROM students", str(sqlite_db))
    ]
    report = run_consistency_check(cases)
    assert report.total == 1
    assert report.agreements == 1
    assert report.disagreements == []
    assert report.consistency_rate == 1.0


def test_disagreement_on_duplicate_rows_is_captured(sqlite_db: Path) -> None:
    # Our default (multiset) says NOT_EQUAL; BIRD's official (set) says 1
    # (equal) — this is the exact, real disagreement class documented in
    # official_baselines.py, not a hypothetical.
    dup_query = (
        "SELECT gpa FROM students WHERE id = 1 "
        "UNION ALL SELECT gpa FROM students WHERE id = 1 "
        "UNION ALL SELECT gpa FROM students WHERE id = 1"
    )
    single_query = "SELECT gpa FROM students WHERE id = 1"
    cases = [("dup_case", dup_query, single_query, str(sqlite_db))]
    report = run_consistency_check(cases)
    assert report.total == 1
    assert report.agreements == 0
    assert len(report.disagreements) == 1
    case = report.disagreements[0]
    assert case.official_result == 1
    from modelhub.compare.result_types import ComparisonResult

    assert case.our_result is ComparisonResult.NOT_EQUAL


def test_empty_case_list_yields_zero_rate_not_division_error() -> None:
    report = run_consistency_check([])
    assert report.total == 0
    assert report.consistency_rate == 0.0


def test_execution_failure_is_scored_as_not_equal_not_a_crash(sqlite_db: Path) -> None:
    cases = [("bad", "SELCT garbage", "SELECT 1", str(sqlite_db))]
    report = run_consistency_check(cases)
    assert report.total == 1
    # Both our comparator (via failed execution) and BIRD's official
    # scorer (via caught exception) should agree this is "wrong".
    assert report.agreements == 1
