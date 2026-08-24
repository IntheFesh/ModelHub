"""Consistency-rate report: our comparator vs. BIRD's real official scorer.

Runs the same (predicted_sql, gold_sql, db) triples through both
`compare_result_sets` and `bird_official_compare` (official_baselines.py)
and reports the agreement rate plus every disagreeing case — this is
CLAUDE.md/A3's "一致率脚本", run against the actual vendored baseline
rather than a hypothetical.

A disagreement is not automatically "our bug" — see
docs/design-decisions.md for why the official scorer's own known
weaknesses (no float tolerance, duplicate-row collapse, error-vs-wrong
conflation) mean plenty of disagreements are the official scorer being
wrong, not us.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from modelhub.compare.comparator import compare_result_sets, resolve_row_order
from modelhub.compare.config import ComparatorConfig
from modelhub.compare.official_baselines import bird_official_compare
from modelhub.compare.result_types import ComparisonResult
from modelhub.sqlexec import Backend, DbRef, execute_isolated


@dataclass(frozen=True)
class ConsistencyCase:
    case_id: str
    predicted_sql: str
    gold_sql: str
    db_path: str
    our_result: ComparisonResult
    official_result: int  # 0 or 1, BIRD's own scoring convention
    agrees: bool
    our_detail: str | None = None


@dataclass(frozen=True)
class ConsistencyReport:
    total: int
    agreements: int
    disagreements: list[ConsistencyCase] = field(default_factory=list)

    @property
    def consistency_rate(self) -> float:
        return self.agreements / self.total if self.total else 0.0


def _agrees(our: ComparisonResult, official: int) -> bool:
    # UNDECIDABLE has no BIRD equivalent (BIRD always emits 0/1); treat it
    # as disagreement-eligible unless official also effectively "gave up"
    # (which BIRD's scorer never does explicitly — every exception -> 0).
    if our is ComparisonResult.EQUAL:
        return official == 1
    return official == 0


def run_consistency_check(
    cases: Sequence[tuple[str, str, str, str]],  # (case_id, predicted_sql, gold_sql, db_path)
    *,
    config: ComparatorConfig | None = None,
) -> ConsistencyReport:
    config = config or ComparatorConfig()
    disagreements: list[ConsistencyCase] = []
    agreements = 0

    for case_id, predicted_sql, gold_sql, db_path in cases:
        predicted_ref = DbRef(backend=Backend.SQLITE, db_id=case_id, location=db_path)
        predicted_outcome = execute_isolated(predicted_ref, predicted_sql)
        gold_outcome = execute_isolated(predicted_ref, gold_sql)

        if predicted_outcome.ok and gold_outcome.ok:
            assert predicted_outcome.result is not None
            assert gold_outcome.result is not None
            our_config = config.model_copy(update={"row_order": resolve_row_order(gold_sql)})
            our_outcome = compare_result_sets(
                predicted_outcome.result, gold_outcome.result, config=our_config
            )
            our_result = our_outcome.result
            our_detail = our_outcome.detail
        else:
            # our sandbox's own execution failed (SYNTAX/SEMANTIC/TIMEOUT/...)
            our_result = ComparisonResult.NOT_EQUAL
            our_detail = f"execution failed: predicted={predicted_outcome.code.value}"

        official_result = bird_official_compare(predicted_sql, gold_sql, db_path)
        agrees = _agrees(our_result, official_result)
        if agrees:
            agreements += 1
        else:
            disagreements.append(
                ConsistencyCase(
                    case_id=case_id,
                    predicted_sql=predicted_sql,
                    gold_sql=gold_sql,
                    db_path=db_path,
                    our_result=our_result,
                    official_result=official_result,
                    agrees=False,
                    our_detail=our_detail,
                )
            )

    return ConsistencyReport(total=len(cases), agreements=agreements, disagreements=disagreements)
