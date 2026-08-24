"""Unit tests for train/gdn_lora_coverage.py — B1's hard acceptance
criterion. `_fake_qwen35_module_names` builds a synthetic module-name
list matching this project's documented understanding of Qwen3.5-9B's
32-layer GDN-hybrid structure (8 full-attention + 24 GDN), used instead
of a real loaded model (no GPU/transformers install in this sandbox)."""

from __future__ import annotations

import pytest

from modelhub.train.gdn_lora_coverage import (
    GdnLoraCoverageConfig,
    LayerType,
    assert_full_layer_coverage,
    compute_trainable_coverage,
    parse_named_modules,
)

_FULL_ATTENTION_LAYER_INDICES = {0, 4, 8, 12, 16, 20, 24, 28}  # 8 of 32, arbitrary positions
_TOTAL_LAYERS = 32


def _fake_qwen35_module_names() -> list[str]:
    names: list[str] = []
    for i in range(_TOTAL_LAYERS):
        prefix = f"model.layers.{i}"
        # every layer has an FFN block regardless of attention type.
        names += [f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj", f"{prefix}.mlp.down_proj"]
        if i in _FULL_ATTENTION_LAYER_INDICES:
            names += [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.o_proj",
            ]
        else:
            names += [
                f"{prefix}.linear_attn.in_proj_qkvz",
                f"{prefix}.linear_attn.in_proj_ba",
                f"{prefix}.linear_attn.out_proj",
                f"{prefix}.linear_attn.conv1d",
            ]
    return names


def _full_config(**overrides: object) -> GdnLoraCoverageConfig:
    defaults: dict[str, object] = {
        "total_layers": _TOTAL_LAYERS,
        "target_modules": [
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkvz",
            "in_proj_ba",
            "out_proj",
            "conv1d",
        ],
    }
    defaults.update(overrides)
    return GdnLoraCoverageConfig.model_validate(defaults)


class TestParseNamedModules:
    def test_groups_by_layer_index(self) -> None:
        names = ["model.layers.0.mlp.gate_proj", "model.layers.0.self_attn.q_proj"]
        layers = parse_named_modules(names)
        assert len(layers) == 1
        assert layers[0].layer_index == 0
        assert layers[0].module_suffixes == frozenset({"gate_proj", "q_proj"})

    def test_ignores_non_layer_names(self) -> None:
        names = ["model.embed_tokens", "model.norm", "lm_head", "model.layers.0.mlp.gate_proj"]
        layers = parse_named_modules(names)
        assert len(layers) == 1
        assert layers[0].layer_index == 0

    def test_sorted_by_layer_index(self) -> None:
        names = ["model.layers.5.mlp.gate_proj", "model.layers.1.mlp.gate_proj"]
        layers = parse_named_modules(names)
        assert [layer.layer_index for layer in layers] == [1, 5]

    def test_full_attention_layer_classified_correctly(self) -> None:
        names = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
        ]
        layers = parse_named_modules(names)
        assert layers[0].layer_type is LayerType.FULL_ATTENTION

    def test_gdn_layer_classified_correctly(self) -> None:
        names = [
            "model.layers.1.linear_attn.in_proj_qkvz",
            "model.layers.1.linear_attn.conv1d",
        ]
        layers = parse_named_modules(names)
        assert layers[0].layer_type is LayerType.GDN


class TestComputeTrainableCoverageRealisticNaiveBug:
    """The exact scenario PLAN.md's B1 round describes: naive
    target_modules only covering the standard attention proj names
    trains 8/32 layers and gives no error."""

    def test_naive_attention_only_target_modules_misses_24_of_32_layers(self) -> None:
        naive_config = _full_config(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        report = compute_trainable_coverage(_fake_qwen35_module_names(), naive_config)
        assert report.covered_layer_count == 8
        assert len(report.uncovered_layers) == 24
        assert all(layer.layer_type is LayerType.GDN for layer in report.uncovered_layers)
        assert not report.fully_covered

    def test_naive_config_fails_the_hard_assertion(self) -> None:
        naive_config = _full_config(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        report = compute_trainable_coverage(_fake_qwen35_module_names(), naive_config)
        with pytest.raises(ValueError, match="leaves 24/32 layers completely untrained"):
            assert_full_layer_coverage(report, expected_total_layers=_TOTAL_LAYERS)

    def test_ffn_only_target_modules_covers_all_32_layers(self) -> None:
        # PLAN.md's stated fix: FFN suffixes alone touch every layer.
        ffn_only_config = _full_config(target_modules=["gate_proj", "up_proj", "down_proj"])
        report = compute_trainable_coverage(_fake_qwen35_module_names(), ffn_only_config)
        assert report.fully_covered
        assert_full_layer_coverage(report, expected_total_layers=_TOTAL_LAYERS)  # must not raise

    def test_full_target_modules_list_covers_all_32_layers(self) -> None:
        report = compute_trainable_coverage(_fake_qwen35_module_names(), _full_config())
        assert report.fully_covered
        assert report.covered_layer_count == _TOTAL_LAYERS
        assert_full_layer_coverage(report, expected_total_layers=_TOTAL_LAYERS)


class TestComputeTrainableCoverage:
    def test_empty_module_names_raises(self) -> None:
        with pytest.raises(ValueError, match="no layer-scoped module names"):
            compute_trainable_coverage([], _full_config())

    def test_layer_count_mismatch_raises(self) -> None:
        report = compute_trainable_coverage(_fake_qwen35_module_names(), _full_config())
        with pytest.raises(ValueError, match="discovered 32 layers, expected exactly 16"):
            assert_full_layer_coverage(report, expected_total_layers=16)


class TestCoverageReportRenderTable:
    def test_table_lists_every_layer_and_flags_uncovered_ones(self) -> None:
        naive_config = _full_config(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        report = compute_trainable_coverage(_fake_qwen35_module_names(), naive_config)
        table = report.render_table()
        assert "layer_index" in table
        # a GDN layer with zero matched modules must show up as NOT covered.
        assert "NO" in table
        assert "yes" in table  # the 8 full-attention layers are covered
