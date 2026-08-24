"""Meta-tests: prove `check_no_cheating.py` can actually flag every rule.

Each test injects source guaranteed to trip exactly one rule and asserts
the finding shows up; a final test proves clean code produces zero
findings, so this file isn't just "always red".
"""

from pathlib import Path

from check_no_cheating import scan_source


def _rules(source: str, path: str = "src/modelhub/fake/module.py") -> set[str]:
    return {f.rule for f in scan_source(source, path)}


def test_bare_except_is_flagged() -> None:
    src = "def f():\n    try:\n        risky()\n    except:\n        raise\n"
    assert "BARE_EXCEPT" in _rules(src)


def test_except_pass_is_flagged() -> None:
    src = "def f():\n    try:\n        risky()\n    except Exception:\n        pass\n"
    assert "EXCEPT_PASS" in _rules(src)


def test_except_return_constant_is_flagged() -> None:
    src = (
        "def f():\n"
        "    try:\n"
        "        return execute_sql()\n"
        "    except Exception:\n"
        "        return 0.0\n"
    )
    assert "EXCEPT_RETURN_CONSTANT" in _rules(src)


def test_testing_env_branch_is_flagged() -> None:
    src = (
        "import os\n"
        "def f():\n"
        "    if os.environ.get('TESTING'):\n"
        "        return SAMPLE\n"
        "    return real()\n"
    )
    assert "TESTING_BRANCH" in _rules(src)


def test_todo_stub_return_is_flagged() -> None:
    src = "def compute_p99(xs):\n    return None  # TODO: implement\n"
    assert "TODO_STUB_RETURN" in _rules(src)


def test_no_timeout_network_call_is_flagged() -> None:
    src = "import requests\ndef f(url):\n    return requests.get(url)\n"
    assert "NO_TIMEOUT_CALL" in _rules(src)


def test_timeout_present_call_is_not_flagged() -> None:
    src = "import requests\ndef f(url):\n    return requests.get(url, timeout=5.0)\n"
    assert "NO_TIMEOUT_CALL" not in _rules(src)


def test_differently_named_timeout_kwarg_is_accepted() -> None:
    # psycopg's connect() timeout kwarg is `connect_timeout`, not `timeout` —
    # this must not be a false positive (found via A2's own sqlexec code).
    src = "import psycopg\ndef f(dsn):\n    return psycopg.connect(dsn, connect_timeout=5)\n"
    assert "NO_TIMEOUT_CALL" not in _rules(src)


def test_silent_slice_truncation_flagged_in_compare_dir() -> None:
    src = "def truncate(rows):\n    return rows[:1000]\n"
    assert "SILENT_SLICE_TRUNCATION" in _rules(src, path="src/modelhub/compare/scoring.py")


def test_silent_slice_truncation_not_flagged_outside_compare_or_sqlexec() -> None:
    src = "def truncate(rows):\n    return rows[:1000]\n"
    assert "SILENT_SLICE_TRUNCATION" not in _rules(src, path="src/modelhub/bench/report.py")


def test_blacklist_availability_check_is_flagged() -> None:
    src = "def usable(status):\n    if status != FAIL:\n        return True\n    return False\n"
    assert "BLACKLIST_AVAILABILITY_CHECK" in _rules(src)


def test_whitelist_availability_check_is_not_flagged() -> None:
    src = "def usable(status):\n    if status == PASS:\n        return True\n    return False\n"
    assert "BLACKLIST_AVAILABILITY_CHECK" not in _rules(src)


def test_mock_reference_under_src_is_flagged() -> None:
    src = "from unittest.mock import MagicMock\n\ndef f():\n    return MagicMock()\n"
    assert "MOCK_IN_SRC" in _rules(src)


def test_clean_code_has_zero_findings() -> None:
    src = (
        "import requests\n\n\n"
        "def call_upstream(url: str, timeout_s: float) -> object:\n"
        "    try:\n"
        "        return requests.get(url, timeout=timeout_s)\n"
        "    except requests.Timeout as e:\n"
        "        raise RuntimeError('upstream timed out') from e\n"
    )
    assert _rules(src) == set()


def test_except_return_none_is_not_flagged() -> None:
    # `return None` is the project's own sanctioned "missing" sentinel
    # (CLAUDE.md §1.1/§3.4), not a plausible-looking fake value like the
    # `return 0.0` example CLAUDE.md actually warns about.
    src = (
        "def f(s):\n"
        "    try:\n"
        "        return float(s)\n"
        "    except ValueError:\n"
        "        return None\n"
    )
    assert "EXCEPT_RETURN_CONSTANT" not in _rules(src)


def test_except_return_false_is_still_flagged() -> None:
    # Confirms the None-exclusion is narrow: other plausible-looking bare
    # constants (False, 0, "") must still be caught.
    src = (
        "def f():\n    try:\n        return check()\n    except Exception:\n        return False\n"
    )
    assert "EXCEPT_RETURN_CONSTANT" in _rules(src)


def test_allowlist_comment_suppresses_the_named_rule_only() -> None:
    src = (
        "def f():\n"
        "    try:\n"
        "        return check()\n"
        "    except Exception:\n"
        "        # check-no-cheating: allow=EXCEPT_RETURN_CONSTANT reason=test fixture\n"
        "        return False\n"
    )
    findings = scan_source(src, "src/modelhub/fake/module.py")
    matching = [f for f in findings if f.rule == "EXCEPT_RETURN_CONSTANT"]
    assert len(matching) == 1
    assert matching[0].allowlisted is True
    assert matching[0].allow_reason == "test fixture"


def test_allowlist_comment_does_not_suppress_a_different_rule() -> None:
    # The allow-comment names EXCEPT_PASS but the actual violation on this
    # line is EXCEPT_RETURN_CONSTANT — must NOT be silently waived by a
    # mismatched rule name (that would make the mechanism a blanket
    # "make the checker quiet" tool instead of a scoped, honest waiver).
    src = (
        "def f():\n"
        "    try:\n"
        "        return check()\n"
        "    except Exception:\n"
        "        # check-no-cheating: allow=EXCEPT_PASS reason=wrong rule name\n"
        "        return False\n"
    )
    findings = scan_source(src, "src/modelhub/fake/module.py")
    matching = [f for f in findings if f.rule == "EXCEPT_RETURN_CONSTANT"]
    assert len(matching) == 1
    assert matching[0].allowlisted is False


def test_allowlisted_finding_does_not_fail_main_exit_code(tmp_path: Path) -> None:
    from check_no_cheating import main

    f = tmp_path / "vendored.py"
    f.write_text(
        "def f():\n"
        "    try:\n"
        "        return check()\n"
        "    except Exception:\n"
        "        # check-no-cheating: allow=EXCEPT_RETURN_CONSTANT reason=test fixture\n"
        "        return False\n"
    )
    assert main([str(tmp_path)]) == 0
