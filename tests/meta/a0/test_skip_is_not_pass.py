"""CLAUDE.md §1.5 required meta-test: `test_skip_is_not_pass`.

Injects each non-PASS status into the shared `is_available()` whitelist
gate and asserts the check correctly reports "not available" — proving the
gate can actually go red instead of being a function that always returns
True regardless of input.
"""

from modelhub.common.capability import CheckStatus, is_available


def test_skip_is_not_pass() -> None:
    assert is_available(CheckStatus.SKIP) is False


def test_warn_is_not_pass() -> None:
    # A measured-but-degraded capability is not the same as a verified one.
    assert is_available(CheckStatus.WARN) is False


def test_fail_is_not_pass() -> None:
    assert is_available(CheckStatus.FAIL) is False


def test_pass_is_available() -> None:
    # The one case that must go green, so this meta-test file isn't just
    # asserting False for everything.
    assert is_available(CheckStatus.PASS) is True
