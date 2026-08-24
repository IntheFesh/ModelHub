"""Per-sample prediction record — the full-fidelity dump `predictions.jsonl`
needs (CLAUDE.md/A4: "逐样本全量落盘...门禁要用").
"""

from __future__ import annotations

from typing import Literal

from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.errors import ErrorCode
from modelhub.compare.result_types import ComparisonResult, UndecidableReason


class PredictionRecord(ModelHubBaseConfig):
    sample_id: str
    db_id: str
    difficulty: str | None
    predicted_sql: str
    finish_reason: Literal["stop", "length"]
    exec_code: ErrorCode
    comparison_result: ComparisonResult | None
    undecidable_reason: UndecidableReason | None = None
    elapsed_s: float
    error_message: str | None = None

    @property
    def is_output_truncated(self) -> bool:
        return self.exec_code is ErrorCode.OUTPUT_TRUNCATED

    @property
    def is_harness_error(self) -> bool:
        return self.exec_code in (ErrorCode.HARNESS_DB_UNAVAILABLE, ErrorCode.HARNESS_INTERNAL)

    @property
    def counts_toward_denominator(self) -> bool:
        """CLAUDE.md §2.4 + §A4: OUTPUT_TRUNCATED and every UNDECIDABLE
        outcome (including a harness-side execution failure) are excluded
        from the accuracy denominator entirely — neither right nor wrong,
        because the harness couldn't actually put the model's SQL to a
        fair test. SYNTAX/SEMANTIC/TIMEOUT (the model's own mistakes) do
        count, as zero-credit attempts."""
        if self.is_output_truncated or self.is_harness_error:
            return False
        return self.comparison_result is not ComparisonResult.UNDECIDABLE
