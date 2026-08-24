"""Smoke test: bench/experiments/ end to end through `run_all_experiments`,
shrunk to a handful of samples/requests (CLAUDE.md §1.4 — only the input
scale shrinks, every real code path — precondition, per-group manifest
writing, isolation, aggregation — still runs).

Wires two groups that are fully exercisable without a live GPU/server
(quantization's capacity math + a real A4/A9 accuracy-and-admission pass
against a real SQLite db; constrained_decoding's grammar-compile probe +
syntax-legality measurement) through the same orchestrator a real
six-group run would use, plus one deliberately-failing group to prove
isolation survives end to end.
"""

from __future__ import annotations

from pathlib import Path

from tests.unit.a4.fakes import fixed_response_client, make_valid_manifest
from tests.unit.a11.fakes import build_school_db, select_sample

from modelhub.bench.experiments.constrained_decoding import (
    compile_sqlite_select_grammar,
    syntax_legality_rate,
)
from modelhub.bench.experiments.orchestrator import run_all_experiments
from modelhub.bench.experiments.quantization import (
    QuantizationExperimentConfig,
    QuantizationMethod,
    estimate_concurrency_gain,
)
from modelhub.common.run_manifest import RunManifest, RunStatus
from modelhub.eval.runner import run_eval
from modelhub.serve.model_profile import ModelProfile


def test_two_groups_run_end_to_end_and_a_third_fails_in_isolation(tmp_path: Path) -> None:
    profile = ModelProfile.model_validate(
        {
            "model_id": "qwen3.5-9b",
            "total_layers": 32,
            "full_attention_layers": 8,
            "gdn_layers": 24,
            "kv_heads": 4,
            "head_dim": 256,
            "kv_dtype_bytes": 2,
            "gdn_fixed_state_bytes": 18874368,
            "weight_bytes": 19327352832,
        }
    )
    quant_config = QuantizationExperimentConfig.model_validate(
        {
            "methods": ["AWQ"],
            "original_bits": 16,
            "gpu_memory_bytes": 34359738368,
            "gpu_memory_utilization": 0.92,
            "runtime_overhead_bytes": 2684354560,
            "prompt_tokens_for_capacity_estimate": 1024,
            "max_new_tokens": 256,
            "generate_timeout_s": 60.0,
            "thinking_modes": [False],
        }
    )

    def run_quantization() -> tuple[object, RunManifest]:
        result = estimate_concurrency_gain(profile, QuantizationMethod.AWQ, quant_config)
        return result, make_valid_manifest(run_id="smoke-quant", status=RunStatus.COMPLETED)

    db_root = build_school_db(tmp_path)

    def run_constrained_decoding() -> tuple[object, RunManifest]:
        compile_result = compile_sqlite_select_grammar()
        predictions = run_eval(
            [select_sample("s0")],
            model_client=fixed_response_client("SELECT id FROM students"),
            prompt_builder=lambda s: s.sample_id,
            db_root=db_root,
            predictions_path=tmp_path / "predictions.jsonl",
        )
        rate = syntax_legality_rate(predictions)
        return (compile_result, rate), make_valid_manifest(
            run_id="smoke-constrained", status=RunStatus.COMPLETED
        )

    def run_engine_comparison_stub() -> tuple[object, RunManifest]:
        raise RuntimeError("no live vLLM/SGLang server to hit in this sandbox")

    report = run_all_experiments(
        serving_instance_manifest=make_valid_manifest(),
        groups={
            "quantization": run_quantization,
            "constrained_decoding": run_constrained_decoding,
            "engine_comparison": run_engine_comparison_stub,
        },
    )

    assert set(report.completed_groups) == {"quantization", "constrained_decoding"}
    assert report.failed_groups == ("engine_comparison",)

    quant_outcome = report.outcome_for("quantization")
    assert quant_outcome is not None
    assert quant_outcome.result.concurrency_after > quant_outcome.result.concurrency_before

    constrained_outcome = report.outcome_for("constrained_decoding")
    assert constrained_outcome is not None
    _, syntax_rate = constrained_outcome.result
    assert syntax_rate == 1.0
