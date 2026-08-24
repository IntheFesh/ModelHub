"""Meta-tests: prove `check_no_cheating.py` can actually flag every rule.

Each test injects source guaranteed to trip exactly one rule and asserts
the finding shows up; a final test proves clean code produces zero
findings, so this file isn't just "always red".
"""

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
