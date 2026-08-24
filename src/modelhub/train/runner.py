"""B1's top-level SFT orchestration: pre-flight checks -> write the
LLaMA-Factory config -> invoke `llamafactory-cli train` as a real
subprocess.

PLAN.md B1 explicitly forbids reimplementing the training loop ("不要
重写训练循环") — `run_sft_preflight` runs every real, CPU-testable check
this project can perform without a GPU, sequentially, refusing to start
training on the first failure (unlike A9's five-gate admission verdict,
which deliberately runs every gate even after one rejects — pre-flight
here is a go/no-go gate for STARTING a run, not a diagnostic report
about a run that already happened, so there is no value in continuing
past a failed check "to see the rest"). `run_sft_training` invokes the
real LLaMA-Factory CLI, guarded on it actually being installed/on PATH
— this sandbox has neither (no GPU — see docs/design-decisions.md).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from modelhub.common.capability import CheckStatus, is_available
from modelhub.serve.kernel_status import detect_gdn_kernel_status
from modelhub.train.dry_run import (
    MemoryEstimateConfig,
    TokenLengthStats,
    assert_fits_in_memory,
    compute_token_length_stats,
    estimate_training_memory,
    validate_dataset_schema,
)
from modelhub.train.gdn_lora_coverage import (
    CoverageReport,
    GdnLoraCoverageConfig,
    assert_full_layer_coverage,
    compute_trainable_coverage,
)
from modelhub.train.sft_config import SftConfig, write_llamafactory_config

_LLAMAFACTORY_CLI = "llamafactory-cli"


@dataclass(frozen=True)
class PreflightReport:
    dataset_size: int
    token_length_stats: TokenLengthStats
    coverage_report: CoverageReport
    memory_headroom_pct: float


def run_sft_preflight(
    *,
    raw_dataset_records: Sequence[dict[str, object]],
    token_counts: Sequence[int],
    model_named_modules: Sequence[str],
    sft_config: SftConfig,
    coverage_config: GdnLoraCoverageConfig,
    memory_config: MemoryEstimateConfig,
) -> PreflightReport:
    # 1. data schema — hard-fails on the first invalid record
    #    (CLAUDE.md: 数据加载失败硬失败).
    samples = validate_dataset_schema(raw_dataset_records)
    if len(token_counts) != len(samples):
        raise ValueError(
            f"token_counts length ({len(token_counts)}) does not match the validated "
            f"dataset size ({len(samples)}) — every sample needs a real tokenized length"
        )

    # 2. token length distribution.
    token_stats = compute_token_length_stats(token_counts, max_seq_len=sft_config.max_seq_len)

    # 3. GDN LoRA coverage — the round's hard acceptance criterion.
    coverage = compute_trainable_coverage(model_named_modules, coverage_config)
    assert_full_layer_coverage(coverage, expected_total_layers=coverage_config.total_layers)
    print(coverage.render_table())

    # 4. GDN kernel status — PLAN.md: "未验证 -> 拒绝开跑".
    kernel_status = detect_gdn_kernel_status()
    if kernel_status.degraded:
        raise ValueError(
            "GDN fast-kernel status is degraded (causal_conv1d/fla unavailable or "
            "unverified) — refusing to start training on a run whose numbers would not "
            "be comparable to a non-degraded run (CLAUDE.md §5.3)"
        )

    # 5. memory estimate — PLAN.md: OOM must be exposed at step 0.
    memory_estimate = estimate_training_memory(
        batch_size=sft_config.batch_size, max_seq_len=sft_config.max_seq_len, config=memory_config
    )
    assert_fits_in_memory(memory_estimate)

    return PreflightReport(
        dataset_size=len(samples),
        token_length_stats=token_stats,
        coverage_report=coverage,
        memory_headroom_pct=memory_estimate.headroom_pct,
    )


@dataclass(frozen=True)
class LlamaFactoryAvailability:
    status: CheckStatus
    detail: str


def check_llamafactory_available() -> LlamaFactoryAvailability:
    path = shutil.which(_LLAMAFACTORY_CLI)
    if path is None:
        return LlamaFactoryAvailability(
            CheckStatus.SKIP, f"{_LLAMAFACTORY_CLI!r} not found on PATH"
        )
    return LlamaFactoryAvailability(CheckStatus.PASS, f"found at {path}")


def build_training_command(config_path: Path) -> list[str]:
    return [_LLAMAFACTORY_CLI, "train", str(config_path)]


def run_sft_training(
    sft_config: SftConfig, *, config_path: Path, resume_from: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Write the real LLaMA-Factory config and invoke its real CLI as a
    subprocess. Raises (not a SKIP-and-continue) if `llamafactory-cli`
    is not on PATH — a caller must check `check_llamafactory_available()`
    itself before deciding whether to call this at all (CLAUDE.md §1.3:
    an unavailable dependency must never read as "ran successfully")."""
    availability = check_llamafactory_available()
    if not is_available(availability.status):
        raise RuntimeError(
            f"cannot start SFT training: {availability.detail} — install the `train` "
            f"extra and LLaMA-Factory on a real GPU machine first"
        )
    write_llamafactory_config(sft_config, config_path)
    command = build_training_command(config_path)
    if resume_from is not None:
        command += ["--resume_from_checkpoint", str(resume_from)]
    # timeout=None is explicit and intentional, not an omission: a real
    # training run's duration is governed by sft_config.max_hours via
    # ModelHubTrainerCallback.on_step_end inside the subprocess, not by
    # an external subprocess-level deadline — an unrelated fixed
    # subprocess timeout here would just kill a legitimately-running
    # multi-hour job.
    return subprocess.run(command, capture_output=True, text=True, timeout=None, check=True)


__all__ = [
    "LlamaFactoryAvailability",
    "PreflightReport",
    "build_training_command",
    "check_llamafactory_available",
    "run_sft_preflight",
    "run_sft_training",
]
