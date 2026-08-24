"""GPU state probing: DCGM PROF field availability, and an nvidia-smi
snapshot whose fields are deliberately NOT named "utilization" anywhere
in this codebase.

Refactored from `preflight.py`'s D3 check. `DCGM_FI_PROF_SM_ACTIVE` /
`PIPE_TENSOR_ACTIVE` (compute) / `DRAM_ACTIVE` (bandwidth) are the
*better* GPU-side metrics when available, but are commonly unavailable
on GeForce cards (profiling counters are often reserved for
datacenter-tier GPUs) — `probe_dcgm_prof_fields` reports that honestly
via the CheckStatus whitelist (CLAUDE.md §1.3) rather than silently
falling back. `query_nvidia_smi`'s result fields are named
`*_time_occupancy_pct`, not `*_utilization_pct`, specifically so nothing
downstream can accidentally re-quote nvidia-smi's number as if it were a
workload measure — see `mfu_mbu.py`'s module docstring for what this
project reports instead (`mfu_mbu.compute_mfu`/`compute_mbu`).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

from modelhub.common.capability import CheckStatus

_SUBPROCESS_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class DcgmProfAvailability:
    status: CheckStatus
    dcgmi_present: bool
    detail: str
    raw_output: str | None = None


def probe_dcgm_prof_fields() -> DcgmProfAvailability:
    dcgmi_path = shutil.which("dcgmi")
    if dcgmi_path is None:
        return DcgmProfAvailability(CheckStatus.SKIP, False, "dcgmi not found on PATH")

    try:
        result = subprocess.run(
            ["dcgmi", "dmon", "-e", "1002,1004,1005", "-c", "1"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return DcgmProfAvailability(
            CheckStatus.FAIL, True, f"dcgmi invocation failed: {type(e).__name__}: {e}"
        )

    output = result.stdout or result.stderr
    if result.returncode == 0 and "N/A" not in output:
        return DcgmProfAvailability(
            CheckStatus.PASS,
            True,
            "DCGM PROF fields (SM_ACTIVE/TENSOR_ACTIVE/DRAM_ACTIVE) available",
            raw_output=output,
        )
    return DcgmProfAvailability(
        CheckStatus.FAIL,
        True,
        "dcgmi ran but PROF fields are unavailable (common on GeForce) — use MFU/MBU instead",
        raw_output=output,
    )


@dataclass(frozen=True)
class NvidiaSmiSnapshot:
    gpu_time_occupancy_pct: float | None
    memory_time_occupancy_pct: float | None
    memory_used_mib: float | None
    power_draw_w: float | None
    temperature_c: float | None


_NVIDIA_SMI_QUERY_FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "temperature.gpu",
)


def _parse_float(raw: str) -> float | None:
    try:
        return float(raw.strip())
    except ValueError:
        return None


def query_nvidia_smi() -> NvidiaSmiSnapshot | None:
    """Returns `None` if `nvidia-smi` isn't available or the query
    fails — never a snapshot with fabricated zeros standing in."""
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(_NVIDIA_SMI_QUERY_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    first_line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) != len(_NVIDIA_SMI_QUERY_FIELDS):
        return None

    values = [_parse_float(p) for p in parts]
    return NvidiaSmiSnapshot(*values)


__all__ = [
    "DcgmProfAvailability",
    "NvidiaSmiSnapshot",
    "probe_dcgm_prof_fields",
    "query_nvidia_smi",
]
