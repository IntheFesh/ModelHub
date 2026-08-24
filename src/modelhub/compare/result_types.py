"""Three-state comparison outcome — never a bare bool.

CLAUDE.md/A3: a boolean EQUAL/NOT_EQUAL collapses "these are definitely
different" and "I cannot safely tell" into the same false signal. Every
comparator call in this project returns one of three states, and an
internal error or a truncated input MUST resolve to UNDECIDABLE, never
silently to EQUAL — a GRPO reward function or a gate check treating "I
crashed" as "correct" would be a training-poisoning / gate-bypassing bug
of the exact kind CLAUDE.md §1 exists to prevent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ComparisonResult(StrEnum):
    EQUAL = "EQUAL"
    NOT_EQUAL = "NOT_EQUAL"
    UNDECIDABLE = "UNDECIDABLE"


class UndecidableReason(StrEnum):
    TOO_LARGE = "TOO_LARGE"  # either side's ResultSet was truncated
    INTERNAL_ERROR = "INTERNAL_ERROR"  # the comparator itself raised
    INCOMPARABLE_SHAPE = "INCOMPARABLE_SHAPE"  # e.g. column counts can't be reconciled
    # the gold SQL itself failed to execute (CLAUDE.md/A1: ~425/9428 of
    # BIRD-train's gold SQL are known not to execute) — there is nothing
    # to compare the prediction against, so correctness is unknowable,
    # not "wrong". Set by eval/runner.py, not by the comparator itself.
    GOLD_EXEC_FAILED = "GOLD_EXEC_FAILED"


@dataclass(frozen=True)
class ComparisonOutcome:
    result: ComparisonResult
    reason: UndecidableReason | None = None
    # which dimension(s) of ComparatorConfig drove a NOT_EQUAL/UNDECIDABLE
    # verdict — not required for EQUAL, essential for debugging golden-set
    # disagreements and for the consistency-rate report's "why did we
    # differ from the official script" column.
    detail: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_equal(self) -> bool:
        return self.result is ComparisonResult.EQUAL

    @classmethod
    def equal(cls, *, detail: str | None = None) -> ComparisonOutcome:
        return cls(result=ComparisonResult.EQUAL, detail=detail)

    @classmethod
    def not_equal(
        cls, *, detail: str, context: Mapping[str, Any] | None = None
    ) -> ComparisonOutcome:
        return cls(result=ComparisonResult.NOT_EQUAL, detail=detail, context=dict(context or {}))

    @classmethod
    def undecidable(
        cls, reason: UndecidableReason, *, detail: str, context: Mapping[str, Any] | None = None
    ) -> ComparisonOutcome:
        return cls(
            result=ComparisonResult.UNDECIDABLE,
            reason=reason,
            detail=detail,
            context=dict(context or {}),
        )
