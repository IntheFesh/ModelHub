"""PLAN.md ★ item 9: "rollout 侧的 vLLM 同样要验证 GDN kernel 状态，
degraded 时拒绝开跑（否则 rollout 慢到训练做不完）".

Direct reuse of `serve/kernel_status.py::detect_gdn_kernel_status` (A5)
— the rollout-side vLLM instance and the serving-side one this project
already checks are the exact same kernel-detection concern, not a new
one. This module adds only the "refuse to start GRPO rollout" framing on
top, matching `bench/gpu_guard.py::guard_gpu_exclusivity`'s
check-then-guard shape.
"""

from __future__ import annotations

from modelhub.common.run_manifest import KernelStatus
from modelhub.serve.kernel_status import detect_gdn_kernel_status


def guard_rollout_kernel_status() -> KernelStatus:
    status = detect_gdn_kernel_status()
    if status.degraded:
        raise ValueError(
            "rollout-side vLLM GDN kernel status is degraded (causal_conv1d/fla "
            "unavailable or unverified) — refusing to start GRPO rollout: a degraded "
            "kernel makes rollout slow enough that training cannot keep up, and any "
            "reward-curve/timing numbers from a degraded rollout would not be comparable "
            "to a healthy one (CLAUDE.md §5.3)"
        )
    return status


__all__ = ["guard_rollout_kernel_status"]
