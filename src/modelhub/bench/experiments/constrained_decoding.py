"""Group 4: XGrammar constrained decoding against a SQLite SELECT-subset
grammar (PLAN.md: "不写全量SQL语法，编译太慢") — measuring syntax-legality
improvement, execution-accuracy change, and throughput cost.

★ Honest boundary (PLAN.md, verbatim): constrained decoding guarantees
SYNTACTIC legality, not semantic correctness. Without schema-aware
constraints — which this experiment does not attempt, and which would
require a per-request grammar compiled against that request's own
schema, not a fixed one compiled once — a model can still emit a
syntactically perfect SELECT that references a column that does not
exist in the target schema. `ConstrainedDecodingResult` reports the
syntax-legality number and the execution-accuracy number side by side
specifically so that gap stays visible rather than being folded into one
score.
"""

from __future__ import annotations

from dataclasses import dataclass

from modelhub.common.capability import CheckStatus
from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode
from modelhub.eval.metrics import EvalMetrics
from modelhub.eval.records import PredictionRecord

# A SQLite SELECT-subset EBNF grammar: SELECT/FROM/JOIN/WHERE/GROUP BY/
# ORDER BY/LIMIT and basic comparison expressions only. Deliberately not
# the full SQLite grammar — DDL/DML/CTEs/window functions/subqueries are
# out of scope, matching the read-only, SELECT-oriented shape sqlexec
# (A2) already enforces at execution time; there is no value in letting
# the grammar accept statements the sandbox would reject anyway.
SQLITE_SELECT_SUBSET_GRAMMAR = r"""
root ::= select_stmt
select_stmt ::= "SELECT " column_list " FROM " table_ref join_clause? select_tail
select_tail ::= where_clause? group_by_clause? order_by_clause? limit_clause?
column_list ::= "*" | column ("," " " column)*
column ::= identifier ("." identifier)?
table_ref ::= identifier (" AS " identifier)?
join_clause ::= (" JOIN " table_ref " ON " condition)*
where_clause ::= " WHERE " condition
group_by_clause ::= " GROUP BY " column_list
order_by_clause ::= " ORDER BY " column_list (" ASC" | " DESC")?
limit_clause ::= " LIMIT " number
condition ::= comparison ((" AND " | " OR ") comparison)*
comparison ::= column comparator value
comparator ::= "=" | "!=" | ">" | "<" | ">=" | "<="
value ::= number | string | column
number ::= [0-9]+
string ::= "'" [a-zA-Z0-9_ %]* "'"
identifier ::= [a-zA-Z_][a-zA-Z0-9_]*
"""


class ConstrainedDecodingExperimentConfig(ModelHubBaseConfig):
    max_tokens: int
    temperature: float
    generate_timeout_s: float


@dataclass(frozen=True)
class GrammarCompileResult:
    status: CheckStatus
    detail: str


def compile_sqlite_select_grammar() -> GrammarCompileResult:
    """Real XGrammar compilation of `SQLITE_SELECT_SUBSET_GRAMMAR`.

    Grammar compilation itself is CPU-only (unlike this package's other
    optional-import guards, which gate on a GPU too) — only the optional
    `xgrammar` dependency (the `experiments` extra) is missing in this
    sandbox, so this exercises the real ImportError -> SKIP path
    honestly, same as every other capability probe in this project
    (CLAUDE.md §1.3)."""
    try:
        import xgrammar  # type: ignore[import-not-found]
    except ImportError as e:
        return GrammarCompileResult(
            CheckStatus.SKIP, f"xgrammar not installed: {type(e).__name__}: {e}"
        )
    try:
        xgrammar.Grammar.from_ebnf(SQLITE_SELECT_SUBSET_GRAMMAR)
    except Exception as e:
        return GrammarCompileResult(
            CheckStatus.FAIL, f"grammar failed to compile: {type(e).__name__}: {e}"
        )
    return GrammarCompileResult(CheckStatus.PASS, "SQLite SELECT-subset grammar compiled")


def syntax_legality_rate(predictions: list[PredictionRecord]) -> float:
    """Fraction of `predictions` NOT rejected as `SYNTAX` — reuses
    sqlexec's own classification (already computed by the real eval run)
    instead of re-parsing SQL a second time.

    Deliberately narrower than `EvalMetrics.syntax_valid_rate` (which is
    `EXEC_OK / total`, and so also excludes SEMANTIC/TIMEOUT failures a
    grammar constraint cannot fix): a constrained-decoding experiment
    should isolate the one failure category the grammar actually targets
    from the ones it does not, per this module's "honest boundary" note.
    """
    if not predictions:
        raise ValueError("cannot compute a syntax-legality rate over an empty prediction set")
    syntax_errors = sum(1 for p in predictions if p.exec_code is ErrorCode.SYNTAX)
    return 1 - syntax_errors / len(predictions)


@dataclass(frozen=True)
class ConstrainedDecodingResult:
    unconstrained_syntax_legality_rate: float
    constrained_syntax_legality_rate: float
    unconstrained_metrics: EvalMetrics
    constrained_metrics: EvalMetrics
    unconstrained_output_tokens_per_s: float | None
    constrained_output_tokens_per_s: float | None

    @property
    def syntax_legality_rate_improvement(self) -> float:
        return self.constrained_syntax_legality_rate - self.unconstrained_syntax_legality_rate

    @property
    def execution_accuracy_change(self) -> float | None:
        """Can land anywhere near zero, positive, or negative — this is
        the module's honest-boundary claim made numeric: constrained
        decoding fixes syntax, not schema-awareness, so a model that
        already knew the schema gains little here, and one that
        references nonexistent columns keeps failing EX even once its
        SQL is syntactically perfect."""
        before = self.unconstrained_metrics.execution_accuracy
        after = self.constrained_metrics.execution_accuracy
        if before is None or after is None:
            return None
        return after - before

    @property
    def throughput_cost_pct(self) -> float | None:
        before, after = self.unconstrained_output_tokens_per_s, self.constrained_output_tokens_per_s
        if not before or after is None:
            return None
        return (1 - after / before) * 100


__all__ = [
    "SQLITE_SELECT_SUBSET_GRAMMAR",
    "ConstrainedDecodingExperimentConfig",
    "ConstrainedDecodingResult",
    "GrammarCompileResult",
    "compile_sqlite_select_grammar",
    "syntax_legality_rate",
]
