"""Load testing, capacity planning, cost modeling, inference-optimization
experiments (experiments themselves are A11's `bench/experiments/`).

Public API: `guard_gpu_exclusivity`/`check_gpu_exclusivity` (gpu_guard.py,
CLAUDE.md §5.2's mandatory pre-run check — bench and train must never
share a GPU); `run_load_test`/`LoadTestConfig`/`LoadTestResult`
(load_test.py); `run_sweep`/`SweepCell`/`SweepGridConfig`/`SweepReport`
(sweep.py, the 2D concurrency-x-prompt-length sweep);
`estimate_max_concurrent_requests`/`find_concurrency_crossover_point`/
`CapacityPlanningConfig` (capacity_planning.py — the crossover point
`gateway/routing.yaml` currently only holds an estimate for, DD-0016);
`cost_per_million_tokens_usd`/`estimate_request_cost_usd`
(cost_model.py — what `gateway/billing.yaml`'s price table should
eventually be derived from, not guessed at).
"""

from modelhub.bench.capacity_planning import (
    CapacityPlanningConfig,
    CrossoverResult,
    estimate_max_concurrent_requests,
    find_concurrency_crossover_point,
)
from modelhub.bench.cost_model import (
    RequestCostEstimate,
    cost_per_million_tokens_usd,
    estimate_request_cost_usd,
)
from modelhub.bench.gpu_guard import (
    GpuExclusivityCheck,
    check_gpu_exclusivity,
    guard_gpu_exclusivity,
)
from modelhub.bench.load_test import LoadTestConfig, LoadTestResult, run_load_test
from modelhub.bench.sweep import SweepCell, SweepGridConfig, SweepReport, run_sweep

__all__ = [
    "CapacityPlanningConfig",
    "CrossoverResult",
    "GpuExclusivityCheck",
    "LoadTestConfig",
    "LoadTestResult",
    "RequestCostEstimate",
    "SweepCell",
    "SweepGridConfig",
    "SweepReport",
    "check_gpu_exclusivity",
    "cost_per_million_tokens_usd",
    "estimate_max_concurrent_requests",
    "estimate_request_cost_usd",
    "find_concurrency_crossover_point",
    "guard_gpu_exclusivity",
    "run_load_test",
    "run_sweep",
]
