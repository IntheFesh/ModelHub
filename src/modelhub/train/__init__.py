"""SFT training pipeline for Qwen3.5-9B — PLAN.md B1.

`gdn_lora_coverage.py` is this round's hard acceptance criterion: naive
LoRA `target_modules` on Qwen3.5-9B's GDN hybrid architecture silently
trains only 8 of 32 layers, with no error. Everything else here —
`checkpoint_state.py` (full-state, atomic, resumable checkpointing),
`dry_run.py` (pre-flight data/memory validation), `loss_guard.py`
(NaN/Inf halt), `time_budget.py` (--max-hours, not epoch-based),
`smoke_test.py` (the mandatory 50-step pre-flight), `callbacks.py`
(wiring all of the above into a real HF `Trainer` without reimplementing
the training loop), `sft_config.py` (LLaMA-Factory config rendering),
`mlflow_backend.py` (experiment tracking), `runner.py` (the top-level
orchestration) — is real code, most of it fully testable without a GPU;
the pieces that genuinely need one (the real 50-step smoke run, the real
LLaMA-Factory subprocess) are guarded to fail honestly rather than fake
success (CLAUDE.md §1.3), consistent with every prior round.
"""

from modelhub.train.callbacks import ModelHubTrainerCallback
from modelhub.train.checkpoint_state import (
    ADAPTER_WEIGHTS_FILENAME,
    HF_TRAINER_STATE_FILENAME,
    OPTIMIZER_STATE_FILENAME,
    REQUIRED_CHECKPOINT_FILES,
    RNG_STATE_FILENAME,
    SCHEDULER_STATE_FILENAME,
    TRAINER_STATE_FILENAME,
    CheckpointListing,
    TrainerState,
    checkpoint_dir_for_step,
    finalize_checkpoint,
    list_checkpoints,
    read_trainer_state,
    validate_checkpoint_complete,
    write_trainer_state,
)
from modelhub.train.dry_run import (
    MemoryEstimate,
    MemoryEstimateConfig,
    TokenLengthStats,
    assert_fits_in_memory,
    compute_token_length_stats,
    estimate_training_memory,
    validate_dataset_schema,
)
from modelhub.train.gdn_lora_coverage import (
    ATTENTION_MODULE_SUFFIXES,
    FFN_MODULE_SUFFIXES,
    GDN_MODULE_SUFFIXES,
    CoverageReport,
    GdnLoraCoverageConfig,
    LayerCoverage,
    LayerModuleInfo,
    LayerType,
    assert_full_layer_coverage,
    compute_trainable_coverage,
    parse_named_modules,
)
from modelhub.train.loss_guard import NonFiniteLossError, check_loss_finite
from modelhub.train.mlflow_backend import (
    MlflowAvailability,
    MlflowRunHandle,
    check_mlflow_available,
    start_mlflow_run,
)
from modelhub.train.runner import (
    LlamaFactoryAvailability,
    PreflightReport,
    build_training_command,
    check_llamafactory_available,
    run_sft_preflight,
    run_sft_training,
)
from modelhub.train.sft_config import (
    SftConfig,
    SftDatasetConfig,
    SftLoraConfig,
    default_qwen35_gdn_target_modules,
    render_llamafactory_yaml,
    write_llamafactory_config,
)
from modelhub.train.smoke_test import (
    SmokeTestCriteria,
    assert_smoke_test_passed,
    check_peak_memory,
    check_resume_curve_matches,
)
from modelhub.train.time_budget import TimeBudgetTracker

__all__ = [
    "ADAPTER_WEIGHTS_FILENAME",
    "ATTENTION_MODULE_SUFFIXES",
    "FFN_MODULE_SUFFIXES",
    "GDN_MODULE_SUFFIXES",
    "HF_TRAINER_STATE_FILENAME",
    "OPTIMIZER_STATE_FILENAME",
    "REQUIRED_CHECKPOINT_FILES",
    "RNG_STATE_FILENAME",
    "SCHEDULER_STATE_FILENAME",
    "TRAINER_STATE_FILENAME",
    "CheckpointListing",
    "CoverageReport",
    "GdnLoraCoverageConfig",
    "LayerCoverage",
    "LayerModuleInfo",
    "LayerType",
    "LlamaFactoryAvailability",
    "MemoryEstimate",
    "MemoryEstimateConfig",
    "MlflowAvailability",
    "MlflowRunHandle",
    "ModelHubTrainerCallback",
    "NonFiniteLossError",
    "PreflightReport",
    "SftConfig",
    "SftDatasetConfig",
    "SftLoraConfig",
    "SmokeTestCriteria",
    "TimeBudgetTracker",
    "TokenLengthStats",
    "TrainerState",
    "assert_fits_in_memory",
    "assert_full_layer_coverage",
    "assert_smoke_test_passed",
    "build_training_command",
    "check_llamafactory_available",
    "check_loss_finite",
    "check_mlflow_available",
    "check_peak_memory",
    "check_resume_curve_matches",
    "checkpoint_dir_for_step",
    "compute_token_length_stats",
    "compute_trainable_coverage",
    "default_qwen35_gdn_target_modules",
    "estimate_training_memory",
    "finalize_checkpoint",
    "list_checkpoints",
    "parse_named_modules",
    "read_trainer_state",
    "render_llamafactory_yaml",
    "run_sft_preflight",
    "run_sft_training",
    "start_mlflow_run",
    "validate_checkpoint_complete",
    "validate_dataset_schema",
    "write_llamafactory_config",
    "write_trainer_state",
]
