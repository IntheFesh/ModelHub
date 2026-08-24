"""Model registry, canary rollout, automatic rollback, online sampling.

A9 owns `registry.py` (this module's public API below); canary rollout,
automatic rollback orchestration, and online sampling validation are
A10's `release/` files.

Public API: `ModelRegistry`/`ModelStatus`/`RegistryEntry` — a
file-backed, atomically-written record per (model_id, version) tracking
its admission-gate history (`gate/admission.py`'s `GateVerdict`) and
deployment status.
"""

from modelhub.release.registry import GateResultRecord, ModelRegistry, ModelStatus, RegistryEntry

__all__ = ["GateResultRecord", "ModelRegistry", "ModelStatus", "RegistryEntry"]
