"""Inference-optimization experiment suite (PLAN.md's A11, v2 rewrite):
six independent, failure-isolated groups, each measuring one optimization
against real inference-serving infrastructure and writing its own
`RunManifest`.

  1. quantization.py            — AWQ / GPTQ / FP8 three-way + thinking x
                                   quantization truncation-rate cross grid
  2. prefix_cache.py            — on/off hit rate, interleaving degree,
                                   Arctic vs Qwen3.5 architecture discount
  3. speculative_decoding.py    — Qwen3.5's native MTP vs Arctic's n-gram,
                                   speedup/acceptance-rate decay curves
  4. constrained_decoding.py    — XGrammar SQLite SELECT-subset grammar
  5. engine_comparison.py       — vLLM vs SGLang (TensorRT-LLM excluded,
                                   see DD-0025)
  6. mfu_mbu_comparison.py      — (v2-added) Arctic vs Qwen3.5 MFU/MBU

`common.py` supplies the shared precondition check and per-group failure
isolation; `orchestrator.py` ties all six together; `vllm_metrics.py` is
the shared real-Prometheus-metric scraper groups 2 and 3 both need.
"""

from modelhub.bench.experiments.common import (
    ExperimentGroupOutcome,
    GroupStatus,
    check_serving_precondition,
    run_experiment_group_isolated,
)
from modelhub.bench.experiments.constrained_decoding import (
    SQLITE_SELECT_SUBSET_GRAMMAR,
    ConstrainedDecodingExperimentConfig,
    ConstrainedDecodingResult,
    GrammarCompileResult,
    compile_sqlite_select_grammar,
    syntax_legality_rate,
)
from modelhub.bench.experiments.engine_comparison import (
    EngineComparisonExperimentConfig,
    EngineComparisonResult,
    run_engine_comparison,
)
from modelhub.bench.experiments.mfu_mbu_comparison import (
    ArchitectureMfuMbuComparison,
    MfuMbuComparisonConfig,
    ModelMfuMbuMeasurement,
    measure_mfu_mbu,
)
from modelhub.bench.experiments.orchestrator import ExperimentSuiteReport, run_all_experiments
from modelhub.bench.experiments.prefix_cache import (
    ArchitectureCacheComparison,
    InterleavedRequest,
    PrefixCacheExperimentConfig,
    PrefixCacheOnOffResult,
    build_interleaved_traffic,
    measure_prefix_cache_hit_rate,
    run_prefix_cache_traffic,
)
from modelhub.bench.experiments.quantization import (
    QuantizationAccuracyResult,
    QuantizationCapacityResult,
    QuantizationExperimentConfig,
    QuantizationMethod,
    QuantizeCheckpointResult,
    TruncationCrossCell,
    estimate_concurrency_gain,
    quantize_model_checkpoint,
    quantized_weight_bytes,
    run_quantization_accuracy_check,
    run_truncation_cross_experiment,
)
from modelhub.bench.experiments.speculative_decoding import (
    ConcurrencySpeedupPoint,
    SpeculativeAcceptanceStats,
    SpeculativeDecayCurve,
    SpeculativeDecodingExperimentConfig,
    SpeculativeDecodingPath,
    measure_acceptance_rate,
    measure_concurrency_decay_curve,
)
from modelhub.bench.experiments.vllm_metrics import (
    fetch_counter,
    fetch_metrics_text,
    parse_prometheus_counter,
)

__all__ = [
    "SQLITE_SELECT_SUBSET_GRAMMAR",
    "ArchitectureCacheComparison",
    "ArchitectureMfuMbuComparison",
    "ConcurrencySpeedupPoint",
    "ConstrainedDecodingExperimentConfig",
    "ConstrainedDecodingResult",
    "EngineComparisonExperimentConfig",
    "EngineComparisonResult",
    "ExperimentGroupOutcome",
    "ExperimentSuiteReport",
    "GrammarCompileResult",
    "GroupStatus",
    "InterleavedRequest",
    "MfuMbuComparisonConfig",
    "ModelMfuMbuMeasurement",
    "PrefixCacheExperimentConfig",
    "PrefixCacheOnOffResult",
    "QuantizationAccuracyResult",
    "QuantizationCapacityResult",
    "QuantizationExperimentConfig",
    "QuantizationMethod",
    "QuantizeCheckpointResult",
    "SpeculativeAcceptanceStats",
    "SpeculativeDecayCurve",
    "SpeculativeDecodingExperimentConfig",
    "SpeculativeDecodingPath",
    "TruncationCrossCell",
    "build_interleaved_traffic",
    "check_serving_precondition",
    "compile_sqlite_select_grammar",
    "estimate_concurrency_gain",
    "fetch_counter",
    "fetch_metrics_text",
    "measure_acceptance_rate",
    "measure_concurrency_decay_curve",
    "measure_mfu_mbu",
    "measure_prefix_cache_hit_rate",
    "parse_prometheus_counter",
    "quantize_model_checkpoint",
    "quantized_weight_bytes",
    "run_all_experiments",
    "run_engine_comparison",
    "run_experiment_group_isolated",
    "run_prefix_cache_traffic",
    "run_quantization_accuracy_check",
    "run_truncation_cross_experiment",
    "syntax_legality_rate",
]
