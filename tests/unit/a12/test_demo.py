"""Unit tests for scripts/demo.py's reusable helper functions — not the
full subprocess+Redis end-to-end flow (that's `make demo` itself,
hand-run and verified — see docs/build-log.md), but the pure/composable
pieces that don't need a live stub server."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.a4.fakes import fixed_response_client

import demo
from modelhub.gate.types import GateDecision
from modelhub.gateway.app import create_app


class TestBuildDemoDb:
    def test_creates_a_real_queryable_sqlite_db(self, tmp_path: Path) -> None:
        import sqlite3

        db_path = demo._build_demo_db(tmp_path / "dbs")
        assert db_path.is_file()
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT id, name FROM students ORDER BY id").fetchall()
        conn.close()
        assert rows == [(1, "Ada"), (2, "Grace")]

    def test_idempotent_across_repeated_calls(self, tmp_path: Path) -> None:
        db_root = tmp_path / "dbs"
        demo._build_demo_db(db_root)
        demo._build_demo_db(db_root)  # must not raise or duplicate rows
        import sqlite3

        conn = sqlite3.connect(str(db_root / "school" / "school.sqlite"))
        count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        conn.close()
        assert count == 2


class TestBuildGatewayDeps:
    def test_produces_a_working_app(self) -> None:
        stub_client = fixed_response_client("SELECT 1")
        deps = demo._build_gateway_deps(redis_client=object(), stub_client=stub_client)
        app = create_app(deps)
        assert app is not None
        assert demo._SHORT_MODEL in deps.model_clients
        assert demo._LONG_MODEL in deps.model_clients
        assert deps.metrics is not None


class TestTriggerGateRejection:
    def test_a_doomed_candidate_is_rejected_and_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sqlite3

        db_root = tmp_path / "dbs"
        db_dir = db_root / "school"
        db_dir.mkdir(parents=True)
        conn = sqlite3.connect(str(db_dir / "school.sqlite"))
        conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO students (id, name) VALUES (1, 'Ada')")
        conn.commit()
        conn.close()

        incident_log_path = tmp_path / "docs" / "incident-log.md"
        original_append = demo.append_incident
        monkeypatch.setattr(
            demo, "append_incident", lambda record: original_append(record, path=incident_log_path)
        )

        decision = demo._trigger_gate_rejection(db_root)

        assert decision is GateDecision.REJECT
        assert incident_log_path.is_file()
        assert "GATE_REJECTION" in incident_log_path.read_text(encoding="utf-8")
