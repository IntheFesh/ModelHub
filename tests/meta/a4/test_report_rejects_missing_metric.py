"""CLAUDE.md §1.5 required meta-test: `test_report_rejects_missing_metric`.

Injected condition: "manifest 缺 p99" — generalized here to "the manifest is
missing any one of the fields the eval report is required to cite"
(A4 has no p99 field of its own; p99 is A8/bench's number. The
CLAUDE.md-mandated *shape* of this test — a report-required field is
missing on the source manifest — is what eval/report.py must satisfy for
its own required fields, listed in `REQUIRED_MANIFEST_FIELDS`).
Asserted behavior: "报告生成器硬失败并列出缺失字段" — the report generator
hard-fails and names exactly which field(s) are missing, rather than
emitting a report that silently omits or fakes the number.
"""

from __future__ import annotations

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.common.errors import ErrorCode, ReportError
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.metrics import compute_metrics
from modelhub.eval.records import PredictionRecord
from modelhub.eval.report import render_report_markdown


def _one_record() -> PredictionRecord:
    return PredictionRecord.model_validate(
        {
            "sample_id": "s1",
            "db_id": "db1",
            "difficulty": "simple",
            "predicted_sql": "SELECT 1",
            "finish_reason": "stop",
            "exec_code": ErrorCode.EXEC_OK,
            "comparison_result": ComparisonResult.EQUAL,
            "elapsed_s": 0.1,
        }
    )


def test_report_rejects_missing_n_samples() -> None:
    manifest = make_valid_manifest(n_samples=None)
    with pytest.raises(ReportError) as exc_info:
        render_report_markdown(manifest, compute_metrics([_one_record()]))
    assert "n_samples" in str(exc_info.value)


def test_report_rejects_missing_eval_tier() -> None:
    manifest = make_valid_manifest(eval_tier=None)
    with pytest.raises(ReportError) as exc_info:
        render_report_markdown(manifest, compute_metrics([_one_record()]))
    assert "eval_tier" in str(exc_info.value)


def test_report_lists_every_missing_field_not_just_the_first() -> None:
    manifest = make_valid_manifest(n_samples=None, seed=None, dataset_hash=None)
    with pytest.raises(ReportError) as exc_info:
        render_report_markdown(manifest, compute_metrics([_one_record()]))
    message = str(exc_info.value)
    assert "n_samples" in message
    assert "seed" in message
    assert "dataset_hash" in message


def test_report_succeeds_with_all_fields_present_this_meta_test_is_not_always_red() -> None:
    manifest = make_valid_manifest()
    report = render_report_markdown(manifest, compute_metrics([_one_record()]))
    assert "Eval Report" in report
