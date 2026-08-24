"""Project-wide error taxonomy.

CLAUDE.md §2.1 mandates a single ``ModelHubError`` base class with an
exhaustive, enumerated ``code`` field (no ``UNKNOWN`` catch-all beyond the
one documented ``UNCLASSIFIED`` escape hatch) and a ``stage`` field pinning
the error to one of the pipeline stages.

``ErrorCode`` is grown additively as later rounds need new codes — it is
never subclassed (Python enums with members cannot be subclassed), so every
round edits this one file directly. That keeps exactly one canonical
registry instead of parallel per-module enums drifting out of sync.

The SQL execution outcome classification (CLAUDE.md §2.3 + §2.4) lives in
the same enum: ``ExecErrorCode`` is a plain alias for ``ErrorCode`` so
call sites in ``sqlexec`` can read exactly like the mandated style in
CLAUDE.md §1.2 (``ExecOutcome.failure(ExecErrorCode.SYNTAX, ...)``) while
every module still shares one vocabulary for manifest ``error_breakdown``
reporting.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class Stage(StrEnum):
    """Which pipeline stage raised or classified an error."""

    DATA = "DATA"
    EXEC = "EXEC"
    COMPARE = "COMPARE"
    EVAL = "EVAL"
    TRAIN = "TRAIN"
    SERVE = "SERVE"
    GATE = "GATE"
    RELEASE = "RELEASE"


class ErrorCode(StrEnum):
    """Exhaustive project-wide error/outcome code registry.

    Grouped by the stage that typically raises them, but the enum itself is
    flat: a manifest's ``error_breakdown`` dict uses these names verbatim as
    keys, so splitting into per-stage enums would fragment that reporting.
    """

    # -- SQL execution outcome classification (CLAUDE.md §2.3 + §2.4) --
    # EXEC_OK is a SUCCESS outcome, not an error: it must never be the code
    # of a raised ModelHubError. It exists here only so sqlexec/eval/manifest
    # reporting can use one shared vocabulary for the full outcome
    # distribution, including the non-error branch.
    SYNTAX = "SYNTAX"
    SEMANTIC = "SEMANTIC"
    TIMEOUT = "TIMEOUT"
    EXEC_OK = "EXEC_OK"
    OUTPUT_TRUNCATED = "OUTPUT_TRUNCATED"
    HARNESS_DB_UNAVAILABLE = "HARNESS_DB_UNAVAILABLE"
    HARNESS_INTERNAL = "HARNESS_INTERNAL"

    # -- catch-all: CLAUDE.md §2.2 permits exactly one, and hitting it is a
    # WARN + sample dump + counted event, never a silent pass. --
    UNCLASSIFIED = "UNCLASSIFIED"

    # -- common/ infra --
    CONFIG_INVALID = "CONFIG_INVALID"
    IO_ATOMIC_WRITE_FAILED = "IO_ATOMIC_WRITE_FAILED"
    RUN_POLLUTED = "RUN_POLLUTED"

    # -- sqlexec: a write attempt was rejected by the DB engine's own
    # read-only enforcement. Not one of the SQL 7-class outcomes (it isn't
    # "is this SQL correct", it's "is this SQL allowed") — the safety gate
    # (A9) and its 270-CRUD adversarial set key off this code specifically.
    UNSAFE_STATEMENT = "UNSAFE_STATEMENT"

    # -- data: train/dev/test must never share a (question, db_id) pair
    # (CLAUDE.md §7) --
    DATA_SPLIT_LEAK = "DATA_SPLIT_LEAK"

    # -- eval: CLAUDE.md §3.4 — a report schema field marked required is
    # None on the source manifest/metrics. The report generator must hard-
    # fail and name the missing field(s), never emit a report that looks
    # complete while quietly missing data. --
    REPORT_REQUIRED_FIELD_MISSING = "REPORT_REQUIRED_FIELD_MISSING"

    # -- gateway: A6 (Stage.SERVE — the gateway sits in front of the model
    # server, distinct from Stage.GATE's admission-gate meaning). Each of
    # these is a request being explicitly *refused*, never silently
    # allowed through or silently downgraded. --
    AUTH_INVALID_API_KEY = "AUTH_INVALID_API_KEY"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"


# CLAUDE.md §2.3: the SQL exec six/seven-way split collapses to two buckets
# that matter for GRPO reward masking and eval-report gating: is this the
# *model's* fault (reward = 0, counts against accuracy) or the *harness's*
# fault (mask out, never penalize, never silently average away)?
SQL_MODEL_ERROR_CODES = frozenset({ErrorCode.SYNTAX, ErrorCode.SEMANTIC, ErrorCode.TIMEOUT})
SQL_SYSTEM_ERROR_CODES = frozenset({ErrorCode.HARNESS_DB_UNAVAILABLE, ErrorCode.HARNESS_INTERNAL})
# OUTPUT_TRUNCATED is neither: CLAUDE.md §2.4 forbids counting it as a model
# error, but it isn't a harness bug either. Callers must handle it as its
# own bucket rather than folding it into either set above.

# Alias so sqlexec call sites match CLAUDE.md §1.2's mandated example
# verbatim: `return ExecOutcome.failure(ExecErrorCode.SYNTAX, ...)`.
ExecErrorCode = ErrorCode


class ModelHubError(Exception):
    """Base exception for every raised error in this project.

    ``context`` is a required keyword argument (not defaulted to ``{}``) so
    call sites cannot forget to pass locating information. CLAUDE.md §2.1
    calls out ``run_id / db_id / sample_id / model_id`` as the fields that
    typically matter; which of those apply depends on the stage, so this
    class does not enforce a fixed key set — code review and the meta-tests
    in ``tests/meta/`` are the enforcement mechanism, not a runtime schema
    that would have to special-case every stage.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode,
        stage: Stage,
        context: Mapping[str, Any],
        retryable: bool,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.context: dict[str, Any] = dict(context)
        self.retryable = retryable
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """Structured form for logging (see ``common.logging.log_exception``)."""
        return {
            "error_type": type(self).__name__,
            "code": self.code.value,
            "stage": self.stage.value,
            "context": self.context,
            "retryable": self.retryable,
            "message": str(self),
            "cause": repr(self.cause) if self.cause is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(code={self.code.value}, stage={self.stage.value}, "
            f"retryable={self.retryable}, context={self.context!r})"
        )


class ConfigError(ModelHubError):
    """Raised by common.config on invalid/unknown configuration."""


class ManifestError(ModelHubError):
    """Raised by common.run_manifest on read/write/pollution violations."""


class ReportError(ModelHubError):
    """Raised by eval.report on missing required fields or polluted runs."""


class GatewayError(ModelHubError):
    """Raised by gateway/ on a refused request: bad auth, rate limit,
    quota exhaustion, or an open circuit breaker. Every raise site fills
    in `retryable` deliberately — a rate limit is retryable after backoff,
    an invalid API key is not."""
