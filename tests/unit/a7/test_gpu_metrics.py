"""Unit tests for monitor/gpu_metrics.py.

This sandbox has no `nvidia-smi`/`dcgmi` on PATH — real SKIP/None paths,
not simulated ones.
"""

from __future__ import annotations

from modelhub.common.capability import CheckStatus, is_available
from modelhub.monitor.gpu_metrics import probe_dcgm_prof_fields, query_nvidia_smi


def test_probe_dcgm_prof_fields_is_a_real_skip_in_this_sandbox() -> None:
    result = probe_dcgm_prof_fields()
    assert result.status is CheckStatus.SKIP
    assert not is_available(result.status)
    assert result.dcgmi_present is False


def test_query_nvidia_smi_returns_none_when_unavailable() -> None:
    assert query_nvidia_smi() is None


def test_nvidia_smi_snapshot_fields_are_named_time_occupancy_not_utilization() -> None:
    from modelhub.monitor.gpu_metrics import NvidiaSmiSnapshot

    field_names = set(NvidiaSmiSnapshot.__dataclass_fields__)
    assert "gpu_time_occupancy_pct" in field_names
    assert not any("utilization" in name for name in field_names)
