"""Unit tests for scripts/render_docs.py."""

from __future__ import annotations

from pathlib import Path

from tests.unit.a4.fakes import make_valid_manifest

import render_docs


class TestDiscoverRuns:
    def test_missing_artifacts_root_returns_empty(self, tmp_path: Path) -> None:
        result = render_docs.discover_runs(tmp_path / "no-such-dir")
        assert result.clean == []
        assert result.polluted == []

    def test_splits_clean_and_polluted(self, tmp_path: Path) -> None:
        clean_dir = tmp_path / "run-clean"
        clean_dir.mkdir()
        (clean_dir / "manifest.json").write_text(
            make_valid_manifest(run_id="run-clean").model_dump_json()
        )
        polluted_dir = tmp_path / "run-dirty"
        polluted_dir.mkdir()
        (polluted_dir / "manifest.json").write_text(
            make_valid_manifest(run_id="run-dirty", git_dirty=True).model_dump_json()
        )
        result = render_docs.discover_runs(tmp_path)
        assert [m.run_id for m in result.clean] == ["run-clean"]
        assert [m.run_id for m in result.polluted] == ["run-dirty"]

    def test_ignores_directories_without_a_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-run").mkdir()
        result = render_docs.discover_runs(tmp_path)
        assert result.clean == []
        assert result.polluted == []


class TestRenderCapacityPlan:
    def test_produces_a_row_per_prompt_token_checkpoint(self) -> None:
        text = render_docs.render_capacity_plan()
        for tokens in render_docs._PROMPT_TOKEN_CANDIDATES:
            assert f"| {tokens} |" in text

    def test_labels_itself_a_planning_estimate(self) -> None:
        text = render_docs.render_capacity_plan()
        assert "PLANNING ESTIMATE" in text


class TestRenderCrossoverCurve:
    def test_reports_a_crossover_or_says_none_found(self) -> None:
        text = render_docs.render_crossover_curve()
        assert "Crossover found" in text or "No crossover found" in text

    def test_labels_itself_a_planning_estimate(self) -> None:
        text = render_docs.render_crossover_curve()
        assert "PLANNING ESTIMATE" in text


class TestRenderMetricsSummary:
    def test_zero_runs_renders_pending_for_measured_metrics(self) -> None:
        empty = render_docs.DiscoveredRuns(clean=[], polluted=[])
        text = render_docs.render_metrics_summary(empty)
        assert text.count(render_docs._PENDING) == 5  # every metric except the crossover row

    def test_polluted_runs_are_listed_and_excluded(self) -> None:
        polluted = make_valid_manifest(run_id="bad-run", contaminated=True)
        result = render_docs.DiscoveredRuns(clean=[], polluted=[polluted])
        text = render_docs.render_metrics_summary(result)
        assert "bad-run" in text
        assert "contaminated=true" in text
        assert "Polluted runs excluded from every number below: **1**" in text


class TestRenderAll:
    def test_writes_three_pages(self, tmp_path: Path) -> None:
        docs_root = tmp_path / "docs"
        written = render_docs.render_all(artifacts_root=tmp_path / "artifacts", docs_root=docs_root)
        assert len(written) == 3
        for path in written:
            assert path.is_file()
            assert path.read_text(encoding="utf-8").strip() != ""
