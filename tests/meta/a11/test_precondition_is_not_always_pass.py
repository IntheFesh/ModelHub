"""CLAUDE.md §1.3 whitelist meta-test: A11's shared serving precondition
("目标服务实例 degraded/contaminated 任一为 true → 拒绝执行") must
actually be able to fire, not just be code that always passes.

Injects three independently-polluted manifests (contaminated,
kernel-degraded, git-dirty) and asserts the precondition — and the whole
orchestrator wrapped around it — refuses each one, then proves the same
check is not permanently red by passing a clean manifest through.
"""

from __future__ import annotations

import pytest
from tests.unit.a4.fakes import make_valid_manifest

from modelhub.bench.experiments.common import check_serving_precondition
from modelhub.bench.experiments.orchestrator import run_all_experiments
from modelhub.common.errors import ManifestError
from modelhub.common.run_manifest import RunManifest


@pytest.mark.parametrize(
    "overrides",
    [
        {"contaminated": True},
        {"kernel_status": {"causal_conv1d": False, "fla": False, "degraded": True}},
        {"git_dirty": True},
    ],
)
def test_precondition_refuses_every_pollution_flag(overrides: dict[str, object]) -> None:
    with pytest.raises(ManifestError, match="refusing to use"):
        check_serving_precondition(make_valid_manifest(**overrides))


def test_orchestrator_refuses_a_degraded_serving_instance_before_any_group_runs() -> None:
    ran = []

    def group() -> tuple[str, RunManifest]:
        ran.append("quantization")
        return "should never happen", make_valid_manifest()

    with pytest.raises(ManifestError):
        run_all_experiments(
            serving_instance_manifest=make_valid_manifest(
                kernel_status={"causal_conv1d": True, "fla": False, "degraded": True}
            ),
            groups={"quantization": group},
        )
    assert ran == [], "no experiment group may run against a degraded serving instance"


def test_precondition_is_not_permanently_red_a_clean_manifest_passes() -> None:
    check_serving_precondition(make_valid_manifest())
