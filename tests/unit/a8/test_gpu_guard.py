"""Unit tests for bench/gpu_guard.py.

This sandbox has no `nvidia-smi` (see docs/design-decisions.md) — a real
SKIP, not a simulated one — and `guard_gpu_exclusivity` must refuse to
proceed on it rather than treating "can't check" as "clean."
"""

from __future__ import annotations

import pytest

from modelhub.bench.gpu_guard import check_gpu_exclusivity, guard_gpu_exclusivity
from modelhub.common.capability import CheckStatus
from modelhub.common.errors import ErrorCode, ModelHubError


def test_check_gpu_exclusivity_is_a_real_skip_in_this_sandbox() -> None:
    result = check_gpu_exclusivity()
    assert result.status is CheckStatus.SKIP
    assert result.concurrent_procs is None


def test_guard_gpu_exclusivity_refuses_to_start_when_unverifiable() -> None:
    with pytest.raises(ModelHubError) as exc_info:
        guard_gpu_exclusivity()
    assert exc_info.value.code is ErrorCode.RUN_POLLUTED
    assert exc_info.value.retryable is False


def test_guard_gpu_exclusivity_context_carries_the_status() -> None:
    with pytest.raises(ModelHubError) as exc_info:
        guard_gpu_exclusivity(gpu_index=2)
    assert exc_info.value.context["gpu_index"] == 2
    assert exc_info.value.context["status"] == "SKIP"
