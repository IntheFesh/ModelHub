"""B3: three deliberately-flawed checkpoint profiles for A-track's
admission-gate drill — PLAN.md: "目的不是准确率，是产出真实的门禁拦截
记录"."""

from modelhub.train.bad_models.adapter_export import (
    ADAPTER_ONLY_EXPORT_FILES,
    MERGED_WEIGHT_FILENAMES,
    export_adapter_only,
)
from modelhub.train.bad_models.checkpoint_selection import select_underfit_checkpoint
from modelhub.train.bad_models.dataset_filters import filter_easy_only, inject_crud_samples
from modelhub.train.bad_models.profiles import PROFILE_INFO, BadModelProfile, BadModelProfileInfo
from modelhub.train.bad_models.training_configs import (
    BadModelTrainingConfig,
    render_bad_model_yaml,
    write_bad_model_config,
)

__all__ = [
    "ADAPTER_ONLY_EXPORT_FILES",
    "MERGED_WEIGHT_FILENAMES",
    "PROFILE_INFO",
    "BadModelProfile",
    "BadModelProfileInfo",
    "BadModelTrainingConfig",
    "export_adapter_only",
    "filter_easy_only",
    "inject_crud_samples",
    "render_bad_model_yaml",
    "select_underfit_checkpoint",
    "write_bad_model_config",
]
