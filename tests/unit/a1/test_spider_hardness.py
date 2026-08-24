from modelhub.data.spider_hardness import SpiderHardness, estimate_spider_hardness


def test_simple_select_is_easy() -> None:
    assert estimate_spider_hardness("SELECT name FROM students") is SpiderHardness.EASY


def test_single_condition_where_is_still_easy() -> None:
    # A lone WHERE with nothing else is, per the documented Spider
    # methodology this classifier follows, still "easy" — one predicate on
    # one table is about as simple as a query gets. It's *additional*
    # WHERE conditions (AND/OR), aggregation, or a second component that
    # push it past easy — see test_query_with_and_condition_is_not_easy.
    result = estimate_spider_hardness("SELECT name FROM students WHERE gpa > 3.5")
    assert result is SpiderHardness.EASY


def test_query_with_and_condition_is_not_easy() -> None:
    result = estimate_spider_hardness("SELECT name FROM students WHERE gpa > 3.5 AND age < 20")
    assert result is not SpiderHardness.EASY


def test_query_with_nested_subquery_and_union_is_extra() -> None:
    sql = (
        "SELECT name FROM students WHERE id IN (SELECT id FROM enrollments) "
        "UNION SELECT name FROM teachers"
    )
    assert estimate_spider_hardness(sql) is SpiderHardness.EXTRA


def test_complex_join_group_by_having_is_harder_than_easy() -> None:
    sql = (
        "SELECT d.name, COUNT(*), AVG(s.gpa) FROM students s "
        "JOIN departments d ON s.dept_id = d.id "
        "WHERE s.gpa > 2.0 GROUP BY d.name HAVING COUNT(*) > 5 ORDER BY d.name"
    )
    result = estimate_spider_hardness(sql)
    assert result is not SpiderHardness.EASY
