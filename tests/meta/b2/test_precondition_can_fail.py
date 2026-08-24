"""CLAUDE.md §1.5 required meta-test: proves B2's shared pollution
precondition (`check_training_experiment_precondition`, reused by
`orchestrator.run_all_comparison_groups`) actually rejects a polluted
baseline manifest through the real orchestrator entry point — not just
the lower-level `assert_not_polluted` unit — and is not permanently red
for a clean one."""

from __future__ import annotations

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.common.errors import ModelHubError
from modelhub.train.experiments.orchestrator import run_all_comparison_groups


def _never_called() -> tuple[str, object]:
    raise AssertionError(
        "a group callable must never run once the shared precondition has rejected "
        "the baseline manifest — the precondition check must happen before any group"
    )


def test_degraded_kernel_status_baseline_is_rejected_by_the_real_orchestrator() -> None:
    from modelhub.common.run_manifest import KernelStatus

    polluted = make_valid_manifest(
        kernel_status=KernelStatus(causal_conv1d=False, fla=False, degraded=True)
    )
    with pytest.raises(ModelHubError, match="degraded"):
        run_all_comparison_groups(baseline_manifest=polluted, groups={"peft_method": _never_called})


def test_contaminated_baseline_is_rejected_by_the_real_orchestrator() -> None:
    polluted = make_valid_manifest(contaminated=True)
    with pytest.raises(ModelHubError, match="contaminated"):
        run_all_comparison_groups(baseline_manifest=polluted, groups={"peft_method": _never_called})


def test_clean_baseline_is_not_permanently_rejected() -> None:
    def _ok() -> tuple[str, object]:
        return "result", make_valid_manifest()

    report = run_all_comparison_groups(
        baseline_manifest=make_valid_manifest(), groups={"peft_method": _ok}
    )
    assert report.completed_groups == ("peft_method",)
