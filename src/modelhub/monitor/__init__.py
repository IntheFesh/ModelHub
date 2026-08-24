"""Metrics: latency/throughput, MFU/MBU, prefix-cache hit rate, GPU state.

Public API: `compute_mfu`/`compute_mbu`/`classify_bottleneck`/
`BottleneckThresholds` (mfu_mbu.py — the headline metrics this project
reports instead of raw GPU_UTIL, see its module docstring for why);
`measure_bf16_tflops`/`measure_memory_bandwidth_gbs` (hardware_bench.py,
refactored from preflight.py's E1/E2 — the measured peak values MFU/MBU
need as denominators); `probe_dcgm_prof_fields`/`query_nvidia_smi`
(gpu_metrics.py, refactored from preflight.py's D3); `PrefixCacheStats`
(cache_metrics.py); `GatewayMetrics` (exporter.py, the Prometheus
`/metrics` registry).
"""

from modelhub.monitor.cache_metrics import PrefixCacheStats
from modelhub.monitor.exporter import GatewayMetrics
from modelhub.monitor.gpu_metrics import (
    DcgmProfAvailability,
    NvidiaSmiSnapshot,
    probe_dcgm_prof_fields,
    query_nvidia_smi,
)
from modelhub.monitor.hardware_bench import (
    BandwidthMeasurement,
    TflopsMeasurement,
    measure_bf16_tflops,
    measure_memory_bandwidth_gbs,
)
from modelhub.monitor.mfu_mbu import (
    BottleneckClass,
    BottleneckThresholds,
    classify_bottleneck,
    compute_mbu,
    compute_mfu,
)

__all__ = [
    "BandwidthMeasurement",
    "BottleneckClass",
    "BottleneckThresholds",
    "DcgmProfAvailability",
    "GatewayMetrics",
    "NvidiaSmiSnapshot",
    "PrefixCacheStats",
    "TflopsMeasurement",
    "classify_bottleneck",
    "compute_mbu",
    "compute_mfu",
    "measure_bf16_tflops",
    "measure_memory_bandwidth_gbs",
    "probe_dcgm_prof_fields",
    "query_nvidia_smi",
]
