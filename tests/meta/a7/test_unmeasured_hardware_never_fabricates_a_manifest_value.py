"""A7 meta-test: an unmeasured hardware peak value must end up as `None`
on a run manifest, never silently replaced by the unvalidated spec-sheet
constant.

CLAUDE.md §1.1 ("缺失就是 None，不是 0") and §3.1: `RunManifest`'s
`measured_peak_tflops`/`measured_bw_gbs` fields exist specifically so a
report can tell "we actually measured this" apart from "we don't know."
This sandbox has no GPU (no `torch` — see docs/design-decisions.md), so
`measure_bf16_tflops`/`measure_memory_bandwidth_gbs` return a real SKIP
here, not an injected one — the "failure" this test exercises is this
sandbox's genuine environment, wired end to end into a real manifest.
"""

from __future__ import annotations

from modelhub.common.capability import CheckStatus
from modelhub.common.run_manifest import RunManifest, RunStatus
from modelhub.monitor.hardware_bench import (
    TflopsMeasurement,
    measure_bf16_tflops,
    measure_memory_bandwidth_gbs,
)


def test_unmeasured_peak_values_stay_none_on_the_manifest() -> None:
    tflops_result = measure_bf16_tflops()
    bw_result = measure_memory_bandwidth_gbs()
    assert tflops_result.status is CheckStatus.SKIP, (
        "test setup assumption broken: this sandbox is expected to have no GPU"
    )
    assert bw_result.status is CheckStatus.SKIP

    manifest = RunManifest.model_validate(
        {
            "run_id": "meta-a7-test-run",
            "git_sha": "0" * 40,
            "git_dirty": False,
            "measured_peak_tflops": tflops_result.tflops,
            "measured_bw_gbs": bw_result.gbs,
            "status": RunStatus.COMPLETED,
        }
    )
    assert manifest.measured_peak_tflops is None
    assert manifest.measured_bw_gbs is None


def test_a_real_measurement_would_populate_the_manifest_this_meta_test_is_not_always_red() -> None:
    # Simulates what a real GPU machine's measurement result looks like —
    # proves the wiring also works in the PASS direction, not just SKIP.
    fake_pass_result = TflopsMeasurement(CheckStatus.PASS, 198.4, "measured 198.4 TFLOPS")
    manifest = RunManifest.model_validate(
        {
            "run_id": "meta-a7-test-run-healthy",
            "git_sha": "0" * 40,
            "git_dirty": False,
            "measured_peak_tflops": fake_pass_result.tflops,
            "status": RunStatus.COMPLETED,
        }
    )
    assert manifest.measured_peak_tflops == 198.4
