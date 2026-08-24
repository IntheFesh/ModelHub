"""A8 meta-test: CLAUDE.md §5.2's GPU-exclusivity rule actually blocks a
bench run from starting, end to end through the real orchestration
entrypoint (`bench/sweep.py::run_sweep`), not just the guard function
checked in isolation. This sandbox genuinely has no `nvidia-smi` (see
docs/design-decisions.md), so `check_gpu_exclusivity` returns a real
SKIP here — the "cannot verify exclusivity" condition this test proves
must block a run is this sandbox's actual environment, not a simulated
one.
"""

from __future__ import annotations

import pytest
from tests.unit.a8.fakes import BenchFakeClient

from modelhub.bench.gpu_guard import check_gpu_exclusivity
from modelhub.bench.sweep import SweepCell, run_sweep
from modelhub.common.capability import CheckStatus
from modelhub.common.errors import ModelHubError


def test_run_sweep_refuses_in_this_real_unverifiable_environment() -> None:
    precheck = check_gpu_exclusivity()
    assert precheck.status is CheckStatus.SKIP, (
        "test setup assumption broken: this sandbox is expected to have no "
        "nvidia-smi, so GPU exclusivity must come back genuinely unverifiable"
    )

    client = BenchFakeClient()
    cells = [SweepCell(concurrency=1, prompt_length_label="1k", prompt="x")]
    with pytest.raises(ModelHubError):
        run_sweep(
            client,
            model_id="test-model",
            cells=cells,
            num_requests_per_cell=1,
            max_tokens=8,
            temperature=0.0,
            generate_timeout_s=5.0,
        )


def test_run_sweep_proceeds_when_guard_is_bypassed_this_meta_test_is_not_always_red() -> None:
    client = BenchFakeClient()
    cells = [SweepCell(concurrency=1, prompt_length_label="1k", prompt="x")]
    report = run_sweep(
        client,
        model_id="test-model",
        cells=cells,
        num_requests_per_cell=1,
        max_tokens=8,
        temperature=0.0,
        generate_timeout_s=5.0,
        skip_gpu_guard=True,
    )
    assert len(report.results) == 1
