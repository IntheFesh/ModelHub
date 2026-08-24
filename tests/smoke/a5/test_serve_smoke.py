"""A5 smoke test: schema introspection -> prompt building -> kernel-status
detection -> model-profile KV accounting, wired together the way A4's
`run_eval` and a future serve-time manifest writer would actually use
them, at a tiny scale."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from modelhub.data.schema import Dialect, NormalizedSample, Source, Split
from modelhub.serve.kernel_status import detect_gdn_kernel_status
from modelhub.serve.model_profile import kv_bytes_per_token, load_model_profile
from modelhub.serve.prompt import make_prompt_builder

pytestmark = pytest.mark.smoke

_PROFILES_DIR = Path(__file__).resolve().parents[3] / "configs" / "serve" / "model_profiles"


def test_serve_pipeline_smoke(tmp_path: Path) -> None:
    db_root = tmp_path / "dbs"
    db_dir = db_root / "school"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(db_dir / "school.sqlite"))
    conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, gpa REAL)")
    conn.commit()
    conn.close()

    sample = NormalizedSample.model_validate(
        {
            "sample_id": "s1",
            "db_id": "school",
            "question": "How many students are there?",
            "evidence": None,
            "gold_sql": "SELECT COUNT(*) FROM students",
            "difficulty": "simple",
            "source": Source.BIRD,
            "split": Split.DEV,
            "dialect": Dialect.SQLITE,
        }
    )

    prompt_builder = make_prompt_builder(db_root)
    prompt = prompt_builder(sample)
    assert "CREATE TABLE students" in prompt
    assert sample.question in prompt

    kernel_status = detect_gdn_kernel_status()
    assert isinstance(kernel_status.degraded, bool)

    arctic_profile, arctic_hash = load_model_profile(_PROFILES_DIR / "arctic_7b.yaml")
    qwen_profile, qwen_hash = load_model_profile(_PROFILES_DIR / "qwen3_5_9b.yaml")
    assert kv_bytes_per_token(arctic_profile) == 56 * 1024
    assert kv_bytes_per_token(qwen_profile) == 32 * 1024
    assert arctic_hash != qwen_hash
