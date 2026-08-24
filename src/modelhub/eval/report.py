"""Eval report generation: manifest + metrics -> a report every number in
which traces back to a real run (CLAUDE.md §3).

Two hard-fail gates run before a single line of the report is rendered:

1. the source manifest must not be polluted (``git_dirty`` / ``contaminated``
   / a degraded kernel) — CLAUDE.md §3.1 forbids any report from citing such
   a run's numbers, full stop.
2. every report-required manifest field must actually be present —
   CLAUDE.md §3.4 forbids a report that *looks* complete while a required
   field is quietly ``None``; the generator must hard-fail and name exactly
   which fields are missing, not paper over them with a placeholder.

CLAUDE.md §7.1 also requires that every number in a report is labeled with
which eval tier (``quick`` / ``full``) produced it, and that the accuracy
denominator's basis (how many samples were excluded, and why) is stated
explicitly rather than left implicit in a bare percentage.
"""

from __future__ import annotations

from pathlib import Path

from modelhub.common.atomic_io import atomic_write_text
from modelhub.common.errors import ErrorCode, ReportError, Stage
from modelhub.common.run_manifest import RunManifest, assert_not_polluted
from modelhub.eval.metrics import DifficultySlice, EvalMetrics

# Fields an eval report cites directly. Anything else on RunManifest
# (e.g. hardware/engine info) is provenance-only and not report-required.
REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "dataset_version",
    "dataset_hash",
    "eval_tier",
    "model_id",
    "comparator_version",
    "seed",
    "n_samples",
)


def _missing_fields(manifest: RunManifest) -> list[str]:
    return [name for name in REQUIRED_MANIFEST_FIELDS if getattr(manifest, name) is None]


def assert_report_ready(manifest: RunManifest) -> None:
    """Hard-fail if `manifest` cannot legitimately back an eval report."""
    assert_not_polluted(manifest, purpose="eval report generation")
    missing = _missing_fields(manifest)
    if missing:
        raise ReportError(
            f"cannot generate report for run {manifest.run_id}: missing required "
            f"manifest field(s) {missing}",
            code=ErrorCode.REPORT_REQUIRED_FIELD_MISSING,
            stage=Stage.EVAL,
            context={"run_id": manifest.run_id, "missing_fields": missing},
            retryable=False,
        )


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _slice_row(name: str, s: DifficultySlice) -> str:
    return (
        f"| {name} | {s.total} | {s.denominator} | {s.equal_count} | {_pct(s.execution_accuracy)} |"
    )


def render_report_markdown(manifest: RunManifest, metrics: EvalMetrics) -> str:
    """Render the report. Raises ReportError/ManifestError instead of
    returning a partial report — see module docstring."""
    assert_report_ready(manifest)

    lines: list[str] = []
    lines.append(f"# Eval Report — run `{manifest.run_id}`")
    lines.append("")
    lines.append(
        f"**Tier: `{manifest.eval_tier}`** — every number below is "
        f"`{manifest.eval_tier}`-tier only."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- run_id: `{manifest.run_id}`")
    lines.append(f"- git_sha: `{manifest.git_sha}` (git_dirty={str(manifest.git_dirty).lower()})")
    lines.append(f"- dataset: `{manifest.dataset_version}` (hash `{manifest.dataset_hash}`)")
    lines.append(f"- model_id: `{manifest.model_id}`")
    if manifest.model_sha:
        lines.append(f"- model_sha: `{manifest.model_sha}`")
    if manifest.adapter_sha:
        lines.append(f"- adapter_sha: `{manifest.adapter_sha}`")
    lines.append(f"- comparator_version: `{manifest.comparator_version}`")
    lines.append(f"- seed: `{manifest.seed}`")
    lines.append(f"- n_samples (manifest): `{manifest.n_samples}`")
    lines.append("")

    lines.append("## Denominator basis")
    lines.append("")
    lines.append(
        "Execution accuracy below is `equal_count / denominator`, **not** "
        "`equal_count / total_samples`. CLAUDE.md §2.4/§7.1: OUTPUT_TRUNCATED "
        "and every UNDECIDABLE outcome (including a failed gold-SQL "
        "execution) are excluded from both the numerator and the "
        "denominator — they are neither a correct nor an incorrect "
        "prediction, so folding them into either count would misrepresent "
        "the model."
    )
    lines.append("")
    lines.append(f"- total_samples: {metrics.total_samples}")
    lines.append(f"- excluded (OUTPUT_TRUNCATED): {metrics.excluded_output_truncated}")
    lines.append(f"- excluded (UNDECIDABLE, incl. HARNESS_*): {metrics.excluded_undecidable}")
    lines.append(f"- denominator: {metrics.denominator}")
    lines.append("")

    lines.append("## Headline metrics")
    lines.append("")
    lines.append(
        f"- **execution_accuracy: {_pct(metrics.execution_accuracy)}** "
        f"({metrics.equal_count}/{metrics.denominator})"
    )
    lines.append(
        f"- syntax_valid_rate (EXEC_OK / total_samples): {_pct(metrics.syntax_valid_rate)}"
    )
    lines.append(f"- output_truncated_rate: {_pct(metrics.output_truncated_rate)}")
    lines.append(f"- undecidable_rate: {_pct(metrics.undecidable_rate)}")
    lines.append(f"- avg_elapsed_s: {metrics.avg_elapsed_s:.3f}")
    lines.append("")

    lines.append("## Error breakdown")
    lines.append("")
    lines.append("| code | count |")
    lines.append("|---|---|")
    for code, count in sorted(metrics.error_breakdown.items()):
        lines.append(f"| {code} | {count} |")
    lines.append("")

    lines.append("## By difficulty")
    lines.append("")
    lines.append("| difficulty | total | denominator | equal | execution_accuracy |")
    lines.append("|---|---|---|---|---|")
    for name in sorted(metrics.by_difficulty):
        lines.append(_slice_row(name, metrics.by_difficulty[name]))
    lines.append("")

    lines.append("## By database")
    lines.append("")
    lines.append("| db_id | total | denominator | equal | execution_accuracy |")
    lines.append("|---|---|---|---|---|")
    for name in sorted(metrics.by_db):
        lines.append(_slice_row(name, metrics.by_db[name]))
    lines.append("")

    return "\n".join(lines) + "\n"


def report_path(run_id: str, *, artifacts_root: Path = Path("artifacts/runs")) -> Path:
    return artifacts_root / run_id / "report.md"


def write_report(
    manifest: RunManifest,
    metrics: EvalMetrics,
    *,
    artifacts_root: Path = Path("artifacts/runs"),
) -> Path:
    markdown = render_report_markdown(manifest, metrics)
    path = report_path(manifest.run_id, artifacts_root=artifacts_root)
    atomic_write_text(path, markdown)
    return path


__all__ = [
    "REQUIRED_MANIFEST_FIELDS",
    "assert_report_ready",
    "render_report_markdown",
    "report_path",
    "write_report",
]
