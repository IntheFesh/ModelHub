"""Result-set comparator: EQUAL / NOT_EQUAL / UNDECIDABLE.

Shared by evaluation, admission gates, GRPO reward, and online sampling —
see CLAUDE.md/A3. Public API: `compare_result_sets`, `ComparatorConfig`,
`ComparisonResult`/`ComparisonOutcome`, `COMPARATOR_VERSION` (goes into
every run manifest that used this version of the comparator).
"""

from modelhub.compare.comparator import compare_result_sets, resolve_row_order
from modelhub.compare.config import COMPARATOR_VERSION, ComparatorConfig
from modelhub.compare.result_types import ComparisonOutcome, ComparisonResult, UndecidableReason

__all__ = [
    "COMPARATOR_VERSION",
    "ComparatorConfig",
    "ComparisonOutcome",
    "ComparisonResult",
    "UndecidableReason",
    "compare_result_sets",
    "resolve_row_order",
]
