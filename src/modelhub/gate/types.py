"""Shared gate result types — never a bare bool, same three-state-outcome
discipline `compare/result_types.py` established for the comparator
(CLAUDE.md's "never collapse into a boolean" principle applies here too:
a gate that couldn't run a check must not read the same as one that ran
it and found nothing wrong).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class GateDecision(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
    # the check genuinely does not apply this run (e.g. no baseline yet
    # exists for the regression gate on the very first model ever
    # admitted) — NOT the same as PASS. CLAUDE.md §1.3's whitelist rule:
    # "not applicable" must never silently read as "verified clean."
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class GateResult:
    gate_name: str
    decision: GateDecision
    detail: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateVerdict:
    """The five-gate admission verdict. Every gate always runs and is
    reported — a rejecting gate does not short-circuit the others, so a
    drill run (or a real rejected candidate) shows the full picture of
    which gate(s) fired, not just the first one."""

    results: tuple[GateResult, ...]

    @property
    def decision(self) -> GateDecision:
        """`REJECT` if any gate rejected; `PASS` otherwise (a gate that
        was `NOT_APPLICABLE` never blocks admission on its own)."""
        if any(r.decision is GateDecision.REJECT for r in self.results):
            return GateDecision.REJECT
        return GateDecision.PASS

    @property
    def rejected_gates(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.decision is GateDecision.REJECT)

    def result_for(self, gate_name: str) -> GateResult | None:
        for r in self.results:
            if r.gate_name == gate_name:
                return r
        return None


__all__ = ["GateDecision", "GateResult", "GateVerdict"]
