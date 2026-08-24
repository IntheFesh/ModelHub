"""GPU-exclusivity guard for bench runs (CLAUDE.md §5.2).

"压测（bench）与训练（train）禁止共享同一张 GPU。共享条件下的 QPS/P99/
吞吐/利用率数字一律无效。每个 bench run 启动前用 `nvidia-smi` 检查目标
GPU 有无其他进程，有则拒绝启动并报错。"

This sandbox has no `nvidia-smi` at all, so `check_gpu_exclusivity` here
exercises a real, honest third outcome: not "the GPU is exclusive"
(PASS) and not "the GPU is shared" (FAIL), but "this cannot be verified
right now" (SKIP). CLAUDE.md §1.3's whitelist rule applies just as much
here as it does to a dependency probe: `guard_gpu_exclusivity` refuses to
start a bench run on anything short of a confirmed PASS — an
unverifiable exclusivity check is not the same as a verified-clean one,
and treating it as such would be exactly the "共享 GPU 的数字是看起来
正常的假数据" failure mode this rule exists to prevent.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from modelhub.common.capability import CheckStatus, is_available
from modelhub.common.errors import ErrorCode, ModelHubError, Stage

_SUBPROCESS_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class GpuExclusivityCheck:
    status: CheckStatus
    concurrent_procs: int | None
    detail: str


def check_gpu_exclusivity(gpu_index: int = 0) -> GpuExclusivityCheck:
    """Real `nvidia-smi --query-compute-apps` process count for
    `gpu_index`. `concurrent_procs` is `None` (not 0) whenever it
    couldn't actually be measured (CLAUDE.md §1.1: missing is None)."""
    if shutil.which("nvidia-smi") is None:
        return GpuExclusivityCheck(CheckStatus.SKIP, None, "nvidia-smi not found on PATH")

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return GpuExclusivityCheck(
            CheckStatus.SKIP, None, f"nvidia-smi invocation failed: {type(e).__name__}: {e}"
        )

    if result.returncode != 0:
        return GpuExclusivityCheck(
            CheckStatus.SKIP,
            None,
            f"nvidia-smi exited {result.returncode}: {result.stderr.strip()}",
        )

    pids = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    n = len(pids)
    if n == 0:
        return GpuExclusivityCheck(CheckStatus.PASS, 0, "no other processes on this GPU")
    return GpuExclusivityCheck(
        CheckStatus.FAIL, n, f"{n} process(es) already running on GPU {gpu_index}: {pids}"
    )


def guard_gpu_exclusivity(gpu_index: int = 0) -> GpuExclusivityCheck:
    """Raise `ModelHubError` unless `check_gpu_exclusivity` comes back a
    confirmed PASS. Call this at the top of every bench run — never
    proceed past it "just this once" for an unverifiable environment."""
    result = check_gpu_exclusivity(gpu_index)
    if not is_available(result.status):
        raise ModelHubError(
            f"refusing to start bench run: GPU {gpu_index} exclusivity is not "
            f"confirmed ({result.status.value}): {result.detail}",
            code=ErrorCode.RUN_POLLUTED,
            stage=Stage.SERVE,
            context={
                "gpu_index": gpu_index,
                "status": result.status.value,
                "concurrent_procs": result.concurrent_procs,
            },
            retryable=False,
        )
    return result


__all__ = ["GpuExclusivityCheck", "check_gpu_exclusivity", "guard_gpu_exclusivity"]
