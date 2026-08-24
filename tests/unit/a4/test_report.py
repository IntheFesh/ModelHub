"""Unit tests for eval/report.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.common.errors import ErrorCode, ManifestError, ReportError
from modelhub.compare.result_types import ComparisonResult
from modelhub.eval.metrics import EvalMetrics, compute_metrics
from modelhub.eval.records import PredictionRecord
from modelhub.eval.report import (
    REQUIRED_MANIFEST_FIELDS,
    assert_report_ready,
    render_report_markdown,
    report_path,
    write_report,
)


def _metrics() -> EvalMetrics:
    records = [
        PredictionRecord.model_validate(
            {
                "sample_id": "s1",
                "db_id": "school",
                "difficulty": "simple",
                "predicted_sql": "SELECT 1",
                "finish_reason": "stop",
                "exec_code": ErrorCode.EXEC_OK,
                "comparison_result": ComparisonResult.EQUAL,
                "elapsed_s": 0.2,
            }
        ),
        PredictionRecord.model_validate(
            {
                "sample_id": "s2",
                "db_id": "school",
                "difficulty": "hard",
                "predicted_sql": "SELECT 2",
                "finish_reason": "stop",
                "exec_code": ErrorCode.EXEC_OK,
                "comparison_result": ComparisonResult.NOT_EQUAL,
                "elapsed_s": 0.3,
            }
        ),
    ]
    return compute_metrics(records)


def test_assert_report_ready_passes_for_valid_manifest() -> None:
    assert_report_ready(make_valid_manifest())


def test_assert_report_ready_rejects_git_dirty() -> None:
    with pytest.raises(ManifestError):
        assert_report_ready(make_valid_manifest(git_dirty=True))


def test_assert_report_ready_rejects_contaminated() -> None:
    with pytest.raises(ManifestError):
        assert_report_ready(make_valid_manifest(contaminated=True))


@pytest.mark.parametrize("field", REQUIRED_MANIFEST_FIELDS)
def test_assert_report_ready_rejects_each_missing_required_field(field: str) -> None:
    with pytest.raises(ReportError) as exc_info:
        assert_report_ready(make_valid_manifest(**{field: None}))
    assert field in str(exc_info.value)


def test_render_report_markdown_labels_tier() -> None:
    md = render_report_markdown(make_valid_manifest(eval_tier="full"), _metrics())
    assert "`full`" in md
    assert "quick" not in md.lower()


def test_render_report_markdown_states_denominator_basis() -> None:
    md = render_report_markdown(make_valid_manifest(), _metrics())
    assert "denominator" in md.lower()
    assert "total_samples: 2" in md
    assert "denominator: 2" in md


def test_render_report_markdown_includes_headline_and_breakdown() -> None:
    md = render_report_markdown(make_valid_manifest(), _metrics())
    assert "execution_accuracy" in md
    assert "EXEC_OK" in md
    assert "By difficulty" in md
    assert "By database" in md


def test_render_report_markdown_raises_on_polluted_manifest() -> None:
    with pytest.raises(ManifestError):
        render_report_markdown(make_valid_manifest(git_dirty=True), _metrics())


def test_write_report_writes_atomically_and_matches_render(tmp_path: Path) -> None:
    manifest = make_valid_manifest(run_id="20260824-run-xyz")
    metrics = _metrics()
    written_path = write_report(manifest, metrics, artifacts_root=tmp_path)
    assert written_path == report_path("20260824-run-xyz", artifacts_root=tmp_path)
    assert written_path.is_file()
    assert written_path.read_text(encoding="utf-8") == render_report_markdown(manifest, metrics)
