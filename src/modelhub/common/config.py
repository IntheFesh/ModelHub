"""Config base class + hashing.

CLAUDE.md §4:
  - Pydantic v2, all configs `ConfigDict(extra="forbid")` — a typo'd key
    must error, not be silently dropped. Silently ignoring unknown config
    keys is named as *the* classic "training result doesn't match" cause.
  - `lr / batch_size / seed / max_seq_len` get no defaults; they must be
    supplied explicitly by every config file.
  - After parsing, compute `config_hash` and write it into the run manifest.
  - Hyperparameters live in `configs/`, not as magic numbers in code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from modelhub.common.errors import ConfigError, ErrorCode, Stage

T = TypeVar("T", bound="ModelHubBaseConfig")


class ModelHubBaseConfig(BaseModel):
    """Base for every Pydantic config model in this project.

    `extra="forbid"` is non-negotiable (CLAUDE.md §4): a typo'd config key
    must raise, not silently vanish. `frozen=True` so a config object's
    hash stays valid for the lifetime of the object it was computed from —
    nothing downstream can mutate it out from under the manifest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


def compute_config_hash(config: BaseModel) -> str:
    """Deterministic ``sha256:<hex>`` hash of a config's resolved values.

    Canonicalized via sorted-key JSON so field ordering never changes the
    hash. This is what goes into ``run_manifest.config_hash``.
    """
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_yaml_config(config_cls: type[T], path: Path, *, stage: Stage) -> tuple[T, str]:
    """Load and validate a YAML config file. Returns ``(config, config_hash)``.

    Any unknown key, missing required field, or type mismatch raises
    ``ConfigError`` wrapping the underlying ``pydantic.ValidationError`` —
    never a silently-defaulted or partially-applied config.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(
            f"failed to read/parse config file {path}",
            code=ErrorCode.CONFIG_INVALID,
            stage=stage,
            context={"path": str(path)},
            retryable=False,
            cause=e,
        ) from e

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config file {path} must parse to a mapping, got {type(raw).__name__}",
            code=ErrorCode.CONFIG_INVALID,
            stage=stage,
            context={"path": str(path)},
            retryable=False,
        )

    return parse_config(config_cls, raw, stage=stage, source=str(path))


def parse_config(
    config_cls: type[T], raw: dict[str, Any], *, stage: Stage, source: str = "<inline>"
) -> tuple[T, str]:
    """Validate an already-loaded mapping against ``config_cls``.

    Shared by ``load_yaml_config`` and any caller that already has a dict
    (e.g. a config assembled programmatically for a smoke test).
    """
    try:
        config = config_cls.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(
            f"invalid config from {source}: {e.error_count()} validation error(s)",
            code=ErrorCode.CONFIG_INVALID,
            stage=stage,
            context={"source": source, "errors": e.errors(include_url=False)},
            retryable=False,
            cause=e,
        ) from e
    return config, compute_config_hash(config)
