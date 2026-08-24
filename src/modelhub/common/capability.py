"""Shared vocabulary for capability / dependency / health probes.

CLAUDE.md §1.3 (★ 未测 ≠ 通过): every health check, capability probe, and
dependency detector in this project must report one of four states, and
"available" must be decided by a WHITELIST (explicit PASS) rather than a
BLACKLIST (anything that isn't FAIL). The classic bug this guards against:
a dependency probe returns SKIP when an optional package fails to import,
and downstream code treats "not FAIL" as "usable" — so an unmeasured
capability silently behaves like a verified one.

Every capability probe in this codebase (GDN kernel detection in serve/,
GPU/DCGM probes in bench/monitor/, preflight.py-style checks) must report a
``CheckStatus`` and gate on ``is_available()`` from this module rather than
rolling its own comparison. That keeps the whitelist rule enforced in
exactly one place instead of being re-implemented (and potentially gotten
wrong) at every call site.
"""

from __future__ import annotations

from enum import StrEnum


class CheckStatus(StrEnum):
    """Outcome of a single capability/health/dependency check."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


def is_available(status: CheckStatus) -> bool:
    """Whitelist gate: only an explicit PASS counts as available.

    Deliberately NOT implemented as ``status is not CheckStatus.FAIL`` —
    that blacklist form is exactly the anti-pattern CLAUDE.md §1.3 names.
    SKIP ("dependency not installed, never actually measured") and WARN
    ("measured, but degraded/uncertain") must both read as unavailable.
    """
    return status is CheckStatus.PASS
