"""The hard acceptance criterion PLAN.md names for B1: LoRA's default
`target_modules` on Qwen3.5-9B silently trains only 1/4 of the model.

Qwen3.5-9B's 32 layers are NOT architecturally uniform: 8 are standard
full (grouped-query) attention, 24 are Gated DeltaNet (GDN, a linear-
attention variant with an internal recurrent state instead of a growing
KV cache — see serve/model_profile.py's module docstring). The default
LoRA `target_modules` PEFT/LLaMA-Factory examples use
(`q_proj`/`k_proj`/`v_proj`/`o_proj`) name submodules that exist ONLY on
the 8 full-attention layers — a naive config trains those 8 layers and
silently leaves the other 24 completely frozen, with no error, no
warning, and a training curve that looks entirely normal.

★ This module's classification works off SUBMODULE NAME PATTERNS within
each layer, not hardcoded layer indices — this project does not have a
live, loaded Qwen3.5-9B model to enumerate real layer-type ordering
against, and hardcoding an assumed index range (e.g. "layers 0-23 are
GDN") risks being silently wrong in exactly the way this whole module
exists to prevent. `GDN_MODULE_SUFFIXES`/`ATTENTION_MODULE_SUFFIXES`
below are this project's best-documented understanding of Qwen3-Next-
family Gated DeltaNet naming in `transformers` (`in_proj_qkvz`/
`in_proj_ba`/`conv1d`/`out_proj`/`norm` for the linear-attention path,
vs. `q_proj`/`k_proj`/`v_proj`/`o_proj` for full attention) — subject to
correction against the real installed `transformers` version's actual
module tree on first real GPU run (the standing project instruction:
write the real logic now, fix names against ground truth once a real
model is loaded).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from modelhub.common.config import ModelHubBaseConfig

_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# FFN module suffixes exist identically on every layer regardless of
# attention type — this is the "target_modules 包含 FFN...覆盖全部
# 32 层" trick PLAN.md points at: including these alone guarantees every
# layer gets touched even if the attention-side names are wrong.
FFN_MODULE_SUFFIXES = frozenset({"gate_proj", "up_proj", "down_proj"})
ATTENTION_MODULE_SUFFIXES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
GDN_MODULE_SUFFIXES = frozenset({"in_proj_qkvz", "in_proj_ba", "out_proj", "conv1d"})


class LayerType(StrEnum):
    FULL_ATTENTION = "FULL_ATTENTION"
    GDN = "GDN"
    UNKNOWN = "UNKNOWN"


class GdnLoraCoverageConfig(ModelHubBaseConfig):
    total_layers: int
    target_modules: list[str]


@dataclass(frozen=True)
class LayerModuleInfo:
    layer_index: int
    module_suffixes: frozenset[str]

    @property
    def layer_type(self) -> LayerType:
        is_attention = bool(self.module_suffixes & ATTENTION_MODULE_SUFFIXES)
        is_gdn = bool(self.module_suffixes & GDN_MODULE_SUFFIXES)
        if is_attention and not is_gdn:
            return LayerType.FULL_ATTENTION
        if is_gdn and not is_attention:
            return LayerType.GDN
        return LayerType.UNKNOWN


@dataclass(frozen=True)
class LayerCoverage:
    layer_index: int
    layer_type: LayerType
    matched_modules: frozenset[str]

    @property
    def covered(self) -> bool:
        return len(self.matched_modules) > 0


@dataclass(frozen=True)
class CoverageReport:
    layers: tuple[LayerCoverage, ...]

    @property
    def covered_layer_count(self) -> int:
        return sum(1 for layer in self.layers if layer.covered)

    @property
    def uncovered_layers(self) -> tuple[LayerCoverage, ...]:
        return tuple(layer for layer in self.layers if not layer.covered)

    @property
    def fully_covered(self) -> bool:
        return len(self.uncovered_layers) == 0

    def render_table(self) -> str:
        """The "逐层可训练参数分布" table PLAN.md requires be printed
        before training starts and written into the manifest."""
        lines = ["layer_index | layer_type    | covered | matched_modules"]
        lines.append("-" * len(lines[0]))
        for layer in self.layers:
            modules = ",".join(sorted(layer.matched_modules)) or "(none)"
            lines.append(
                f"{layer.layer_index:>11} | {layer.layer_type.value:<13} | "
                f"{'yes' if layer.covered else 'NO':<7} | {modules}"
            )
        return "\n".join(lines)


def parse_named_modules(module_names: Iterable[str]) -> tuple[LayerModuleInfo, ...]:
    """Group a flat iterable of module names (as `named_modules()` would
    yield on a real loaded model) by transformer layer index, keeping
    only the trailing suffix (e.g. `model.layers.5.self_attn.q_proj` ->
    layer 5, suffix `q_proj`) each name resolves to."""
    by_layer: dict[int, set[str]] = {}
    for name in module_names:
        match = _LAYER_INDEX_PATTERN.search(name)
        if match is None:
            continue
        layer_index = int(match.group(1))
        suffix = name.rsplit(".", 1)[-1]
        by_layer.setdefault(layer_index, set()).add(suffix)
    return tuple(
        LayerModuleInfo(layer_index=i, module_suffixes=frozenset(suffixes))
        for i, suffixes in sorted(by_layer.items())
    )


def compute_trainable_coverage(
    module_names: Iterable[str], config: GdnLoraCoverageConfig
) -> CoverageReport:
    """For every layer discovered in `module_names`, determine whether
    `config.target_modules` matches at least one of that layer's real
    submodule suffixes — i.e. whether LoRA would actually attach a
    trainable adapter to that layer at all."""
    target_set = frozenset(config.target_modules)
    layers = parse_named_modules(module_names)
    if not layers:
        raise ValueError(
            "no layer-scoped module names found (expected names matching "
            "'...layers.<i>....') — cannot compute coverage over zero layers"
        )
    coverages = tuple(
        LayerCoverage(
            layer_index=layer.layer_index,
            layer_type=layer.layer_type,
            matched_modules=layer.module_suffixes & target_set,
        )
        for layer in layers
    )
    return CoverageReport(layers=coverages)


def assert_full_layer_coverage(report: CoverageReport, *, expected_total_layers: int) -> None:
    """PLAN.md's hard acceptance criterion: "训练启动前打印逐层可训练
    参数分布并断言覆盖全部 32 层，不覆盖则拒绝启动." Raises with the
    specific uncovered layer indices and their type — never a generic
    "coverage failed" message, since knowing WHICH layers (and whether
    they're the GDN ones) is exactly the diagnostic this check exists
    to produce."""
    if len(report.layers) != expected_total_layers:
        raise ValueError(
            f"discovered {len(report.layers)} layers, expected exactly "
            f"{expected_total_layers} — the model's real layer count does not "
            f"match configs/train/*.yaml's total_layers; refusing to certify "
            f"coverage against the wrong denominator"
        )
    uncovered = report.uncovered_layers
    if uncovered:
        detail = ", ".join(f"{layer.layer_index}({layer.layer_type.value})" for layer in uncovered)
        raise ValueError(
            f"LoRA target_modules leaves {len(uncovered)}/{expected_total_layers} "
            f"layers completely untrained: {detail} — this is the exact silent-"
            f"1/4-of-the-model bug PLAN.md's B1 round names; fix target_modules "
            f"before starting training, do not proceed with partial coverage"
        )


__all__ = [
    "ATTENTION_MODULE_SUFFIXES",
    "FFN_MODULE_SUFFIXES",
    "GDN_MODULE_SUFFIXES",
    "CoverageReport",
    "GdnLoraCoverageConfig",
    "LayerCoverage",
    "LayerModuleInfo",
    "LayerType",
    "assert_full_layer_coverage",
    "compute_trainable_coverage",
    "parse_named_modules",
]
