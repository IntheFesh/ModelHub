from pathlib import Path

import pytest
from pydantic import ValidationError

from modelhub.common.config import (
    ModelHubBaseConfig,
    compute_config_hash,
    load_yaml_config,
    parse_config,
)
from modelhub.common.errors import ConfigError, Stage


class TrainHyperparams(ModelHubBaseConfig):
    """Example config demonstrating CLAUDE.md §4: no defaults for these fields."""

    lr: float
    batch_size: int
    seed: int
    max_seq_len: int


def test_typo_d_key_is_rejected() -> None:
    with pytest.raises(ConfigError) as exc_info:
        parse_config(
            TrainHyperparams,
            {"lr": 1e-4, "batch_size": 8, "seed": 42, "max_seq_len": 2048, "leaning_rate": 1e-4},
            stage=Stage.TRAIN,
        )
    assert exc_info.value.code.value == "CONFIG_INVALID"


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config(TrainHyperparams, {"lr": 1e-4, "batch_size": 8, "seed": 42}, stage=Stage.TRAIN)


def test_valid_config_parses_and_hashes_deterministically() -> None:
    raw = {"lr": 1e-4, "batch_size": 8, "seed": 42, "max_seq_len": 2048}
    cfg1, hash1 = parse_config(TrainHyperparams, raw, stage=Stage.TRAIN)
    cfg2, hash2 = parse_config(TrainHyperparams, dict(raw), stage=Stage.TRAIN)
    assert cfg1 == cfg2
    assert hash1 == hash2
    assert hash1.startswith("sha256:")


def test_hash_changes_when_a_value_changes() -> None:
    base = {"lr": 1e-4, "batch_size": 8, "seed": 42, "max_seq_len": 2048}
    _, hash1 = parse_config(TrainHyperparams, base, stage=Stage.TRAIN)
    changed = {**base, "seed": 43}
    _, hash2 = parse_config(TrainHyperparams, changed, stage=Stage.TRAIN)
    assert hash1 != hash2


def test_config_is_frozen() -> None:
    cfg, _ = parse_config(
        TrainHyperparams,
        {"lr": 1e-4, "batch_size": 8, "seed": 42, "max_seq_len": 2048},
        stage=Stage.TRAIN,
    )
    with pytest.raises(ValidationError):
        cfg.seed = 99  # type: ignore[misc]


def test_load_yaml_config_roundtrip(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text("lr: 0.0001\nbatch_size: 8\nseed: 42\nmax_seq_len: 2048\n")
    cfg, config_hash = load_yaml_config(TrainHyperparams, yaml_path, stage=Stage.TRAIN)
    assert cfg.batch_size == 8
    assert config_hash == compute_config_hash(cfg)


def test_load_yaml_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_yaml_config(TrainHyperparams, tmp_path / "does_not_exist.yaml", stage=Stage.TRAIN)


def test_load_yaml_config_with_unknown_key_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "typo.yaml"
    yaml_path.write_text("lr: 0.0001\nbatch_size: 8\nseed: 42\nmax_seq_len: 2048\nbatchsize: 8\n")
    with pytest.raises(ConfigError):
        load_yaml_config(TrainHyperparams, yaml_path, stage=Stage.TRAIN)
