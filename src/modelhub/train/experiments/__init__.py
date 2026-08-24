"""B2: fine-tuning-method / distributed-strategy / kernel-optimization
comparison suite — six short (200-500 step) runs, never trained to
convergence, whose only job is to produce peak memory / throughput /
step time / cost numbers stable enough to inform a real technical
selection (PLAN.md §B2)."""

from modelhub.train.experiments.common import (
    ExperimentGroupOutcome,
    LayerTypeProfileSplit,
    TrainingRunMetrics,
    check_training_experiment_precondition,
    compute_throughput_tokens_per_s,
    compute_total_cost,
    guard_dedicated_gpus,
    run_experiment_group_isolated,
)
from modelhub.train.experiments.distributed_strategy_comparison import (
    DistributedStrategy,
    DistributedStrategyExperimentConfig,
    render_deepspeed_zero_config,
    render_distributed_strategy_yaml,
    render_fsdp_accelerate_config,
    run_distributed_strategy_comparison_group,
)
from modelhub.train.experiments.kernel_optimization_comparison import (
    KernelOptimizationExperimentConfig,
    KernelToggle,
    KernelTogglePair,
    render_kernel_toggle_yaml,
    run_kernel_toggle_variant,
)
from modelhub.train.experiments.metrics_recorder import (
    ExperimentMetricsCallback,
    RawExperimentMetrics,
    load_raw_experiment_metrics,
    require_measured_peak_memory,
)
from modelhub.train.experiments.orchestrator import ComparisonSuiteReport, run_all_comparison_groups
from modelhub.train.experiments.peft_method_comparison import (
    PeftMethod,
    PeftMethodExperimentConfig,
    render_peft_method_yaml,
    run_peft_method_comparison_group,
    trainable_param_count_for_method,
)
from modelhub.train.experiments.report import (
    ComparisonEntry,
    MetricProvenance,
    format_effect_cost_sentence,
    render_selection_table,
)

__all__ = [
    "ComparisonEntry",
    "ComparisonSuiteReport",
    "DistributedStrategy",
    "DistributedStrategyExperimentConfig",
    "ExperimentGroupOutcome",
    "ExperimentMetricsCallback",
    "KernelOptimizationExperimentConfig",
    "KernelToggle",
    "KernelTogglePair",
    "LayerTypeProfileSplit",
    "MetricProvenance",
    "PeftMethod",
    "PeftMethodExperimentConfig",
    "RawExperimentMetrics",
    "TrainingRunMetrics",
    "check_training_experiment_precondition",
    "compute_throughput_tokens_per_s",
    "compute_total_cost",
    "format_effect_cost_sentence",
    "guard_dedicated_gpus",
    "load_raw_experiment_metrics",
    "render_deepspeed_zero_config",
    "render_distributed_strategy_yaml",
    "render_fsdp_accelerate_config",
    "render_kernel_toggle_yaml",
    "render_peft_method_yaml",
    "render_selection_table",
    "require_measured_peak_memory",
    "run_all_comparison_groups",
    "run_distributed_strategy_comparison_group",
    "run_experiment_group_isolated",
    "run_kernel_toggle_variant",
    "run_peft_method_comparison_group",
    "trainable_param_count_for_method",
]
