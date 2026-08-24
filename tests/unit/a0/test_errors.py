import pytest

from modelhub.common.errors import (
    SQL_MODEL_ERROR_CODES,
    SQL_SYSTEM_ERROR_CODES,
    ErrorCode,
    ExecErrorCode,
    ModelHubError,
    Stage,
)


def test_exec_error_code_is_error_code_alias() -> None:
    assert ExecErrorCode is ErrorCode
    assert ExecErrorCode.SYNTAX is ErrorCode.SYNTAX


def test_sql_model_vs_system_error_sets_are_disjoint_and_exhaustive() -> None:
    assert SQL_MODEL_ERROR_CODES.isdisjoint(SQL_SYSTEM_ERROR_CODES)
    assert {ErrorCode.SYNTAX, ErrorCode.SEMANTIC, ErrorCode.TIMEOUT} == SQL_MODEL_ERROR_CODES
    assert {
        ErrorCode.HARNESS_DB_UNAVAILABLE,
        ErrorCode.HARNESS_INTERNAL,
    } == SQL_SYSTEM_ERROR_CODES
    # OUTPUT_TRUNCATED and EXEC_OK must be in neither bucket (CLAUDE.md §2.4).
    assert ErrorCode.OUTPUT_TRUNCATED not in SQL_MODEL_ERROR_CODES
    assert ErrorCode.OUTPUT_TRUNCATED not in SQL_SYSTEM_ERROR_CODES
    assert ErrorCode.EXEC_OK not in SQL_MODEL_ERROR_CODES
    assert ErrorCode.EXEC_OK not in SQL_SYSTEM_ERROR_CODES


def test_model_hub_error_carries_required_fields() -> None:
    cause = ValueError("boom")
    err = ModelHubError(
        "sql timed out",
        code=ErrorCode.TIMEOUT,
        stage=Stage.EXEC,
        context={"run_id": "r1", "db_id": "db1", "sample_id": "s1"},
        retryable=False,
        cause=cause,
    )
    assert err.code is ErrorCode.TIMEOUT
    assert err.stage is Stage.EXEC
    assert err.context == {"run_id": "r1", "db_id": "db1", "sample_id": "s1"}
    assert err.retryable is False
    assert err.cause is cause
    assert err.__cause__ is cause  # traceback chaining preserved
    assert "sql timed out" in str(err)


def test_model_hub_error_requires_context_kwarg() -> None:
    # `context` has no default: forgetting it is a TypeError at call time,
    # not a silently-empty dict.
    with pytest.raises(TypeError):
        ModelHubError(  # type: ignore[call-arg]
            "missing context",
            code=ErrorCode.UNCLASSIFIED,
            stage=Stage.EVAL,
            retryable=False,
        )


def test_to_dict_is_json_serializable_shape() -> None:
    err = ModelHubError(
        "bad sql",
        code=ErrorCode.SYNTAX,
        stage=Stage.EXEC,
        context={"sample_id": "s2"},
        retryable=False,
    )
    d = err.to_dict()
    assert d["code"] == "SYNTAX"
    assert d["stage"] == "EXEC"
    assert d["retryable"] is False
    assert d["cause"] is None
    assert d["context"] == {"sample_id": "s2"}
