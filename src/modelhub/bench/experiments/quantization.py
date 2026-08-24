"""Group 1: quantization three-way — AutoAWQ / GPTQModel / llm-compressor
(FP8) — plus the v2-added thinking x quantization truncation-rate cross
experiment.

★ Framing (PLAN.md): the number this experiment reports is "用显存换并发"
(trade VRAM for concurrency), not "显存省 X%". Quantization shrinks
`weight_bytes`, which grows the KV-cache budget `capacity_planning.py`
already turns into a max-concurrency estimate — this module's capacity
half is that chain, nothing new invented.

★ NVFP4 is deliberately not one of the three methods: on this project's
SM120 target it falls back to the Marlin (INT4) kernel rather than a real
FP4 execution path, so a "NVFP4" result here would actually be measuring
Marlin/INT4 under a misleading label (see DD-0024).

★ A quantized model is not admitted on perplexity — PLAN.md is explicit
that it "必须重跑完整评估并走同样五道闸": `run_quantization_accuracy_check`
re-runs the real A4 eval pipeline against the quantized model's serving
endpoint and the real A9 five-gate admission verdict, reusing both
wholesale rather than approximating with a cheaper proxy metric.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from modelhub.bench.capacity_planning import estimate_max_concurrent_requests
from modelhub.common.capability import CheckStatus
from modelhub.common.config import ModelHubBaseConfig
from modelhub.common.run_manifest import RunManifest
from modelhub.data.schema import NormalizedSample
from modelhub.eval.metrics import EvalMetrics, compute_metrics
from modelhub.eval.model_client import ModelClient
from modelhub.eval.records import PredictionRecord
from modelhub.eval.runner import run_eval
from modelhub.gate.admission import AdmissionGateConfig, run_admission_gate
from modelhub.gate.types import GateVerdict
from modelhub.serve.model_profile import ModelProfile


class QuantizationMethod(StrEnum):
    AWQ = "AWQ"
    GPTQ = "GPTQ"
    FP8 = "FP8"


_BITS_PER_WEIGHT: dict[QuantizationMethod, int] = {
    QuantizationMethod.AWQ: 4,
    QuantizationMethod.GPTQ: 4,
    QuantizationMethod.FP8: 8,
}


class QuantizationExperimentConfig(ModelHubBaseConfig):
    methods: list[QuantizationMethod]
    original_bits: int
    gpu_memory_bytes: int
    gpu_memory_utilization: float
    runtime_overhead_bytes: int
    prompt_tokens_for_capacity_estimate: int
    max_new_tokens: int
    generate_timeout_s: float
    thinking_modes: list[bool]


def quantized_weight_bytes(
    original_weight_bytes: int, *, original_bits: int, target_method: QuantizationMethod
) -> int:
    """Weight-byte ESTIMATE under `target_method`, scaling linearly with
    bit width at a constant param count. Ignores per-layer overhead a
    real quantization library adds back (scales/zero-points, an
    unquantized embedding/lm_head, group-wise metadata) — an input to
    capacity planning, not a substitute for the real quantized
    checkpoint's on-disk size once one exists."""
    if original_weight_bytes <= 0:
        raise ValueError(f"original_weight_bytes must be positive, got {original_weight_bytes}")
    if original_bits <= 0:
        raise ValueError(f"original_bits must be positive, got {original_bits}")
    target_bits = _BITS_PER_WEIGHT[target_method]
    return int(original_weight_bytes * target_bits / original_bits)


@dataclass(frozen=True)
class QuantizeCheckpointResult:
    status: CheckStatus
    output_path: Path | None
    detail: str


def quantize_model_checkpoint(
    *, model_path: Path, output_path: Path, method: QuantizationMethod
) -> QuantizeCheckpointResult:
    """Invoke the real quantization library for `method` against a real
    on-disk checkpoint. Every path here runs through torch+CUDA — this
    sandbox has neither the optional dependency (the `experiments` extra:
    autoawq/gptqmodel/llmcompressor) nor a GPU, so the only branch this
    function exercises in this environment is the real ImportError ->
    SKIP path (CLAUDE.md §1.3), the same discipline `serve/kernel_status.py`
    and `monitor/hardware_bench.py` already apply. Untested end-to-end
    against a real checkpoint in this sandbox — real code, not a mock,
    to be corrected against the actual library API on first real GPU run.
    """
    if method is QuantizationMethod.AWQ:
        try:
            from awq import AutoAWQForCausalLM  # type: ignore[import-not-found]
            from transformers import AutoTokenizer
        except ImportError as e:
            return QuantizeCheckpointResult(
                CheckStatus.SKIP, None, f"autoawq not installed: {type(e).__name__}: {e}"
            )
        model = AutoAWQForCausalLM.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model.quantize(
            tokenizer,
            quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
        )
        model.save_quantized(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        return QuantizeCheckpointResult(CheckStatus.PASS, output_path, "AWQ quantization completed")

    if method is QuantizationMethod.GPTQ:
        try:
            from gptqmodel import GPTQModel, QuantizeConfig  # type: ignore[import-not-found]
        except ImportError as e:
            return QuantizeCheckpointResult(
                CheckStatus.SKIP, None, f"gptqmodel not installed: {type(e).__name__}: {e}"
            )
        quant_config = QuantizeConfig(bits=4, group_size=128)
        model = GPTQModel.load(str(model_path), quant_config)
        # calibration_dataset is intentionally not wired here — it must
        # be a real held-out sample of in-domain prompts (Text2SQL
        # schema+question strings), which requires a served corpus this
        # sandbox has no GPU to draw from; NotImplementedError rather
        # than a fabricated calibration set (CLAUDE.md: "不确定就留
        # NotImplementedError，不要给能跑但结果是假的实现").
        raise NotImplementedError(
            "GPTQ calibration dataset wiring pending a real GPU run; do not call "
            "quantize_model_checkpoint(method=GPTQ) until configs/bench/experiments/"
            "quantization.yaml names a real calibration sample source"
        )

    if method is QuantizationMethod.FP8:
        try:
            from llmcompressor.modifiers.quantization import (  # type: ignore[import-not-found]
                QuantizationModifier,
            )
            from llmcompressor.transformers import oneshot  # type: ignore[import-not-found]
        except ImportError as e:
            return QuantizeCheckpointResult(
                CheckStatus.SKIP, None, f"llmcompressor not installed: {type(e).__name__}: {e}"
            )
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        oneshot(model=str(model_path), recipe=recipe, output_dir=str(output_path))
        return QuantizeCheckpointResult(CheckStatus.PASS, output_path, "FP8 quantization completed")

    raise ValueError(f"unhandled quantization method: {method}")  # pragma: no cover


@dataclass(frozen=True)
class QuantizationCapacityResult:
    method: QuantizationMethod
    quantized_weight_bytes: int
    concurrency_before: int
    concurrency_after: int

    @property
    def concurrency_gain_pct(self) -> float:
        if self.concurrency_before == 0:
            raise ValueError("concurrency_before is 0 — cannot express a gain as a percentage")
        return (self.concurrency_after - self.concurrency_before) / self.concurrency_before * 100


def estimate_concurrency_gain(
    profile: ModelProfile, method: QuantizationMethod, config: QuantizationExperimentConfig
) -> QuantizationCapacityResult:
    """ "用显存换并发": quantized weight bytes -> KV headroom -> max
    concurrency, compared against the unquantized profile at the same
    prompt/output length assumption. Pure arithmetic — reuses
    `capacity_planning.estimate_max_concurrent_requests` (A8) unchanged,
    just fed a shrunk `weight_bytes`."""
    concurrency_before = estimate_max_concurrent_requests(
        profile,
        gpu_memory_bytes=config.gpu_memory_bytes,
        gpu_memory_utilization=config.gpu_memory_utilization,
        runtime_overhead_bytes=config.runtime_overhead_bytes,
        prompt_tokens=config.prompt_tokens_for_capacity_estimate,
        max_new_tokens=config.max_new_tokens,
    )
    quant_bytes = quantized_weight_bytes(
        profile.weight_bytes, original_bits=config.original_bits, target_method=method
    )
    quantized_profile = profile.model_copy(update={"weight_bytes": quant_bytes})
    concurrency_after = estimate_max_concurrent_requests(
        quantized_profile,
        gpu_memory_bytes=config.gpu_memory_bytes,
        gpu_memory_utilization=config.gpu_memory_utilization,
        runtime_overhead_bytes=config.runtime_overhead_bytes,
        prompt_tokens=config.prompt_tokens_for_capacity_estimate,
        max_new_tokens=config.max_new_tokens,
    )
    return QuantizationCapacityResult(
        method=method,
        quantized_weight_bytes=quant_bytes,
        concurrency_before=concurrency_before,
        concurrency_after=concurrency_after,
    )


@dataclass(frozen=True)
class QuantizationAccuracyResult:
    method: QuantizationMethod
    metrics: EvalMetrics
    gate_verdict: GateVerdict


def run_quantization_accuracy_check(
    *,
    method: QuantizationMethod,
    quantized_client: ModelClient,
    samples: Sequence[NormalizedSample],
    prompt_builder: Callable[[NormalizedSample], str],
    db_root: Path,
    predictions_path: Path,
    baseline_predictions: list[PredictionRecord] | None,
    adversarial_samples: Sequence[NormalizedSample],
    admission_config: AdmissionGateConfig,
    manifest: RunManifest,
) -> QuantizationAccuracyResult:
    """ "量化模型必须重跑完整评估并走同样五道闸，不能只看 perplexity" —
    this is the real A4 eval runner against the quantized model's serving
    endpoint, feeding the real A9 five-gate admission verdict. No
    quantization-specific shortcut metric substitutes for either."""
    predictions = run_eval(
        samples,
        model_client=quantized_client,
        prompt_builder=prompt_builder,
        db_root=db_root,
        predictions_path=predictions_path,
    )
    metrics = compute_metrics(predictions)
    verdict = run_admission_gate(
        manifest=manifest,
        metrics=metrics,
        candidate_predictions=predictions,
        baseline_predictions=baseline_predictions,
        adversarial_samples=adversarial_samples,
        db_root=db_root,
        config=admission_config,
    )
    return QuantizationAccuracyResult(method=method, metrics=metrics, gate_verdict=verdict)


@dataclass(frozen=True)
class TruncationCrossCell:
    """One cell of the v2-added thinking x quantization truncation-rate
    grid. PLAN.md: "INT4 量化 + reasoning 开启会显著抬高输出截断率...
    这个数字是量化影响的不只是权重精度的实证"."""

    method: QuantizationMethod
    thinking_enabled: bool
    output_truncated_rate: float


def run_truncation_cross_experiment(
    cells: Sequence[tuple[QuantizationMethod, bool, ModelClient]],
    *,
    samples: Sequence[NormalizedSample],
    prompt_builder: Callable[[NormalizedSample], str],
    db_root: Path,
    predictions_dir: Path,
) -> list[TruncationCrossCell]:
    """One real eval pass per (method, thinking_enabled) cell — `cells`
    supplies one already-configured `ModelClient` per cell (thinking
    on/off is a serving-side prompt/sampling-params concern, out of
    scope for this function to construct); this function only measures
    the resulting `EvalMetrics.output_truncated_rate` per cell."""
    results = []
    for method, thinking_enabled, client in cells:
        label = f"{method.value}_{'thinking' if thinking_enabled else 'no_thinking'}"
        predictions = run_eval(
            samples,
            model_client=client,
            prompt_builder=prompt_builder,
            db_root=db_root,
            predictions_path=predictions_dir / f"{label}.jsonl",
        )
        metrics = compute_metrics(predictions)
        results.append(
            TruncationCrossCell(
                method=method,
                thinking_enabled=thinking_enabled,
                output_truncated_rate=metrics.output_truncated_rate,
            )
        )
    return results


__all__ = [
    "QuantizationAccuracyResult",
    "QuantizationCapacityResult",
    "QuantizationExperimentConfig",
    "QuantizationMethod",
    "QuantizeCheckpointResult",
    "TruncationCrossCell",
    "estimate_concurrency_gain",
    "quantize_model_checkpoint",
    "quantized_weight_bytes",
    "run_quantization_accuracy_check",
    "run_truncation_cross_experiment",
]
