"""A7 smoke test: hardware probing -> MFU/MBU computation -> bottleneck
classification -> Prometheus export, wired the way a real serving loop
would call these modules in order."""

from __future__ import annotations

import pytest

from modelhub.common.capability import CheckStatus
from modelhub.monitor.cache_metrics import PrefixCacheStats
from modelhub.monitor.exporter import GatewayMetrics
from modelhub.monitor.gpu_metrics import probe_dcgm_prof_fields, query_nvidia_smi
from modelhub.monitor.hardware_bench import measure_bf16_tflops, measure_memory_bandwidth_gbs
from modelhub.monitor.mfu_mbu import (
    BottleneckThresholds,
    classify_bottleneck,
    compute_mbu,
    compute_mfu,
)

pytestmark = pytest.mark.smoke


def test_monitor_pipeline_smoke() -> None:
    # 1. hardware probing (real SKIP in this sandbox — no GPU)
    tflops_result = measure_bf16_tflops()
    bw_result = measure_memory_bandwidth_gbs()
    dcgm_result = probe_dcgm_prof_fields()
    smi_snapshot = query_nvidia_smi()
    assert dcgm_result.status is CheckStatus.SKIP
    assert smi_snapshot is None

    # 2. since this sandbox can't measure real peak values, use a
    # documented stand-in (this is exactly what a real GPU run would
    # replace with tflops_result.tflops/bw_result.gbs).
    peak_bf16_flops = 200e12
    peak_bw_bytes_s = 1800e9
    assert tflops_result.tflops is None  # confirms we really are using the stand-in
    assert bw_result.gbs is None

    # 3. MFU/MBU for a decode-like scenario
    mfu = compute_mfu(
        model_params=7_000_000_000, output_tokens_per_s=60, peak_bf16_flops=peak_bf16_flops
    )
    mbu = compute_mbu(
        model_weight_bytes=15_000_000_000, output_tokens_per_s=60, peak_bw_bytes_s=peak_bw_bytes_s
    )
    thresholds = BottleneckThresholds(
        memory_bound_mbu_min=0.5, memory_bound_mfu_max=0.3, compute_bound_mfu_min=0.5
    )
    bottleneck = classify_bottleneck(mfu=mfu, mbu=mbu, thresholds=thresholds)
    assert bottleneck in ("memory_bound", "compute_bound", "balanced")

    # 4. prefix cache stats
    cache_stats = PrefixCacheStats(hits=730, total_requests=1000)
    assert cache_stats.hit_rate == pytest.approx(0.73)

    # 5. export everything
    metrics = GatewayMetrics()
    metrics.set_mfu("arctic-text2sql-r1-7b", "decode", mfu)
    metrics.set_mbu("arctic-text2sql-r1-7b", "decode", mbu)
    metrics.set_prefix_cache_hit_rate("arctic-text2sql-r1-7b", cache_stats.hit_rate or 0.0)
    metrics.record_request(model_id="arctic-text2sql-r1-7b", outcome="success", latency_s=0.4)
    rendered = metrics.render().decode("utf-8")
    assert "modelhub_serve_mfu_ratio" in rendered
    assert "modelhub_serve_mbu_ratio" in rendered
    assert "modelhub_serve_prefix_cache_hit_rate" in rendered
