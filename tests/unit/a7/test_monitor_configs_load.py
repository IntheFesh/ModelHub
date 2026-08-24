"""configs/monitor/bottleneck_thresholds.yaml must actually load."""

from __future__ import annotations

from pathlib import Path

from modelhub.common.config import load_yaml_config
from modelhub.common.errors import Stage
from modelhub.monitor.mfu_mbu import BottleneckThresholds, classify_bottleneck

_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "monitor" / "bottleneck_thresholds.yaml"
)


def test_bottleneck_thresholds_config_loads_and_works() -> None:
    thresholds, config_hash = load_yaml_config(
        BottleneckThresholds, _CONFIG_PATH, stage=Stage.SERVE
    )
    assert config_hash.startswith("sha256:")
    assert classify_bottleneck(mfu=0.1, mbu=0.9, thresholds=thresholds) == "memory_bound"
