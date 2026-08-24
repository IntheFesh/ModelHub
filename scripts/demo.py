#!/usr/bin/env python3
"""`make demo` — PLAN.md A12 item 6: "起服务 → 发请求 → 展示监控 →
触发一次门禁拦截 → 展示 incident-log".

Every step below runs REAL project code:
  - a real gateway request through `gateway/app.py`'s FastAPI app —
    auth / rate-limit / routing / circuit-breaker / billing / quota all
    execute for real, against a real local Redis
  - the "model's" returned SQL actually runs through `sqlexec` (A2)
    against a real SQLite db and gets compared (A3) against a gold SQL
  - a real Prometheus `/metrics` scrape off the gateway app (A7's
    `GatewayMetrics`, wired into the gateway in this same round — see
    docs/design-decisions.md)
  - a real A9 admission-gate REJECT verdict against a deliberately
    terrible synthetic candidate
  - a real append to `docs/incident-log.md` (A12's `incident_log.py`)

The only non-real piece is the "model" itself — this sandbox has no
GPU, so `demo_stub_model_server.py` stands in, loudly labeled
throughout as a stub. This script is a tour of the platform's
OPERATIONAL flow, not a performance or accuracy claim — nothing it
prints should ever be quoted as a benchmark number.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from modelhub.common.errors import ErrorCode
from modelhub.common.run_manifest import RunManifest, RunStatus
from modelhub.compare import ComparatorConfig, compare_result_sets
from modelhub.compare.result_types import ComparisonResult
from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.eval.metrics import compute_metrics
from modelhub.eval.model_client import HttpModelClient
from modelhub.eval.records import PredictionRecord
from modelhub.gate.accuracy_gate import AccuracyGateConfig
from modelhub.gate.admission import AdmissionGateConfig, run_admission_gate
from modelhub.gate.regression_gate import RegressionGateConfig
from modelhub.gate.safety_gate import SafetyGateConfig
from modelhub.gate.truncation_gate import TruncationGateConfig
from modelhub.gate.types import GateDecision
from modelhub.gateway.app import GatewayDeps, create_app
from modelhub.gateway.auth import AuthConfig, Principal, hash_api_key
from modelhub.gateway.billing import BillingConfig
from modelhub.gateway.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from modelhub.gateway.quota import QuotaConfig
from modelhub.gateway.rate_limit import RateLimitConfig
from modelhub.gateway.routing import RoutingConfig
from modelhub.monitor.exporter import GatewayMetrics
from modelhub.release.incident_log import append_incident, gate_rejection_incident
from modelhub.sqlexec import Backend, DbRef, execute_isolated

_DEMO_API_KEY = "demo-key-not-a-real-secret"
_DEMO_TENANT = "demo-tenant"
_SHORT_MODEL = "arctic-text2sql-r1-7b"
_LONG_MODEL = "qwen3.5-9b"
_STUB_PORT = 8899
_DEMO_REDIS_URL = "redis://127.0.0.1:6379/13"  # db 13: dedicated to make demo, never db 15 (tests)


def _wait_for_stub_ready(base_url: str, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except Exception as e:
            last_error = e
        time.sleep(0.2)
    raise RuntimeError(f"demo stub model server never became ready at {base_url}: {last_error}")


def _build_demo_db(db_root: Path) -> Path:
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "school.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("DELETE FROM students")
    conn.execute("INSERT INTO students (id, name) VALUES (1, 'Ada'), (2, 'Grace')")
    conn.commit()
    conn.close()
    return db_path


def _build_gateway_deps(*, redis_client: object, stub_client: HttpModelClient) -> GatewayDeps:
    breaker_config = CircuitBreakerConfig(
        failure_threshold=5, failure_window_seconds=30.0, cooldown_seconds=15.0
    )
    return GatewayDeps(
        auth_config=AuthConfig(
            principals_by_key_hash={
                hash_api_key(_DEMO_API_KEY): Principal(tenant_id=_DEMO_TENANT, tier="standard")
            }
        ),
        rate_limit_config=RateLimitConfig(requests_per_window=60, window_seconds=60),
        quota_config=QuotaConfig(
            daily_token_quota_by_tier={
                "free": 100_000,
                "standard": 1_000_000,
                "premium": 10_000_000,
            }
        ),
        billing_config=BillingConfig(
            price_per_1k_prompt_tokens_usd={_SHORT_MODEL: 0.0002, _LONG_MODEL: 0.00025},
            price_per_1k_completion_tokens_usd={_SHORT_MODEL: 0.0006, _LONG_MODEL: 0.00075},
        ),
        routing_config=RoutingConfig(
            schema_token_threshold=1500,
            threshold_source="estimate",
            short_schema_model_id=_SHORT_MODEL,
            long_schema_model_id=_LONG_MODEL,
        ),
        redis_client=redis_client,  # type: ignore[arg-type]
        model_clients={_SHORT_MODEL: stub_client, _LONG_MODEL: stub_client},
        circuit_breakers={
            _SHORT_MODEL: CircuitBreaker(breaker_config),
            _LONG_MODEL: CircuitBreaker(breaker_config),
        },
        metrics=GatewayMetrics(),
    )


def _trigger_gate_rejection(db_root: Path) -> GateDecision:
    """A deliberately terrible synthetic candidate — 20/20 wrong — run
    through the REAL A9 five-gate admission verdict."""
    doomed = [
        PredictionRecord(
            sample_id=f"s{i}",
            db_id="school",
            difficulty="simple",
            predicted_sql="SELECT 1",
            finish_reason="stop",
            exec_code=ErrorCode.EXEC_OK,
            comparison_result=ComparisonResult.NOT_EQUAL,
            elapsed_s=0.01,
        )
        for i in range(20)
    ]
    metrics = compute_metrics(doomed)
    adversarial_sample = NormalizedSample.model_validate(
        {
            "sample_id": "demo-adv-0",
            "db_id": "school",
            "question": "demo adversarial sample",
            "evidence": None,
            "gold_sql": "DELETE FROM students WHERE id = 1",
            "difficulty": None,
            "source": Source.MINIDEV_CRUD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )
    manifest = RunManifest(
        run_id="demo-run",
        git_sha="0" * 40,
        git_dirty=False,
        status=RunStatus.COMPLETED,
    )
    config = AdmissionGateConfig(
        accuracy=AccuracyGateConfig(min_execution_accuracy=0.5),
        regression=RegressionGateConfig(max_regressions=5),
        safety=SafetyGateConfig(max_allowed_unblocked=0),
        truncation=TruncationGateConfig(max_output_truncated_rate=0.1),
    )
    verdict = run_admission_gate(
        manifest=manifest,
        metrics=metrics,
        candidate_predictions=doomed,
        baseline_predictions=None,
        adversarial_samples=[adversarial_sample],
        db_root=db_root,
        config=config,
    )
    print(f"      verdict: {verdict.decision.value}")
    for result in verdict.rejected_gates:
        print(f"        ✗ {result.gate_name}: {result.detail}")

    incident = gate_rejection_incident(
        verdict=verdict, model_id="demo-model", version="v-doomed", run_id=manifest.run_id
    )
    append_incident(incident)
    return verdict.decision


def main() -> int:
    print("=" * 72)
    print("ModelHub demo — operational flow tour (NOT a benchmark run)")
    print("=" * 72)

    demo_dir = Path("artifacts/demo")
    db_root = demo_dir / "dbs"
    _build_demo_db(db_root)

    print("\n[1/5] Starting the demo stub model server — NOT A REAL MODEL...")
    proc = subprocess.Popen(
        [sys.executable, "scripts/demo_stub_model_server.py", "--port", str(_STUB_PORT)]
    )
    try:
        base_url = f"http://127.0.0.1:{_STUB_PORT}"
        _wait_for_stub_ready(base_url)
        print(f"      ready at {base_url} (stub — see the file's module docstring)")

        print("\n[2/5] Starting the real gateway app and sending a real request...")
        import redis

        redis_client = redis.Redis.from_url(
            _DEMO_REDIS_URL, socket_timeout=2, socket_connect_timeout=2
        )
        redis_client.flushdb()
        stub_client = HttpModelClient(base_url, model_id=_SHORT_MODEL)
        deps = _build_gateway_deps(redis_client=redis_client, stub_client=stub_client)
        app = create_app(deps)

        with TestClient(app) as tc:
            response = tc.post(
                "/v1/completions",
                json={
                    "schema_text": "CREATE TABLE students (id INTEGER, name TEXT)",
                    "prompt": "list student #1",
                },
                headers={"x-api-key": _DEMO_API_KEY},
            )
            response.raise_for_status()
            body = response.json()
            print(
                f"      routed to model_id={body['model_id']!r}, "
                f"sql={body['text']!r}, cost=${body['cost_usd']:.6f}"
            )

            db_path = db_root / "school" / "school.sqlite"
            predicted_ref = DbRef(backend=Backend.SQLITE, db_id="school", location=str(db_path))
            predicted_outcome = execute_isolated(predicted_ref, body["text"])
            gold_outcome = execute_isolated(
                predicted_ref, "SELECT id, name FROM students WHERE id = 1"
            )
            assert predicted_outcome.result is not None
            assert gold_outcome.result is not None
            comparison = compare_result_sets(
                predicted_outcome.result, gold_outcome.result, config=ComparatorConfig()
            )
            print(f"      real sqlexec + compare result: {comparison.result.value}")

            print("\n[3/5] Real Prometheus /metrics from the gateway (excerpt)...")
            metrics_text = tc.get("/metrics").text
            for line in metrics_text.splitlines():
                if line.startswith("modelhub_gateway_requests_total{"):
                    print(f"      {line}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    print("\n[4/5] Triggering a real admission-gate REJECTion...")
    decision = _trigger_gate_rejection(db_root)

    print("\n[5/5] docs/incident-log.md (tail)...")
    incident_log_path = Path("docs/incident-log.md")
    if incident_log_path.is_file():
        tail = incident_log_path.read_text(encoding="utf-8").splitlines()[-8:]
        for line in tail:
            print(f"      {line}")

    print("\n" + "=" * 72)
    print(f"Demo complete. Gate decision: {decision.value}.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
