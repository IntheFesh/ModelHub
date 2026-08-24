"""Unit tests for train/grpo/kernel_guard.py.

This sandbox has no CUDA device, so the real `detect_gdn_kernel_status`
is honestly `degraded=True` here — the one code path this module
exercises for real in this environment is the refusal branch, same as
`serve/kernel_status.py`'s own tests."""

from __future__ import annotations

import pytest

from modelhub.train.grpo.kernel_guard import guard_rollout_kernel_status


def test_degraded_kernel_status_refuses_rollout() -> None:
    with pytest.raises(ValueError, match="refusing to start GRPO rollout"):
        guard_rollout_kernel_status()
