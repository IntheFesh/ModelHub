import io
import json
import logging

from modelhub.common.errors import Stage
from modelhub.common.logging import get_logger, log_exception, truncate_for_log


def _read_records(stream: io.StringIO) -> list[dict[str, object]]:
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_log_record_has_fixed_fields() -> None:
    stream = io.StringIO()
    logger = get_logger("test_component", run_id="run-123", stage=Stage.EVAL, stream=stream)
    logger.info("hello")
    records = _read_records(stream)
    assert len(records) == 1
    rec = records[0]
    assert rec["run_id"] == "run-123"
    assert rec["stage"] == "EVAL"
    assert rec["component"] == "test_component"
    assert rec["level"] == "INFO"
    assert rec["message"] == "hello"


def test_get_logger_is_idempotent_no_duplicate_handlers() -> None:
    stream = io.StringIO()
    logger1 = get_logger("dedup_component", stream=stream)
    logger2 = get_logger("dedup_component", stream=stream)
    logger1.info("one")
    logger2.info("two")
    records = _read_records(stream)
    # Two calls, two log lines — not four (which would mean stacked handlers).
    assert len(records) == 2


def test_sensitive_fields_are_redacted() -> None:
    stream = io.StringIO()
    logger = get_logger("secret_component", stream=stream)
    logger.info(
        "calling upstream",
        extra={"api_key": "sk-super-secret", "db_password": "hunter2", "safe_field": "ok"},
    )
    rec = _read_records(stream)[0]
    fields = rec["fields"]
    assert fields["api_key"] == "***REDACTED***"
    assert fields["db_password"] == "***REDACTED***"
    assert fields["safe_field"] == "ok"


def test_exception_logging_includes_full_traceback_not_just_str() -> None:
    stream = io.StringIO()
    logger = get_logger("exc_component", stream=stream)
    try:
        raise ValueError("boom")
    except ValueError as e:
        log_exception(logger, "upstream call failed", e, sample_id="s1")
    rec = _read_records(stream)[0]
    assert "traceback" in rec
    assert "ValueError: boom" in rec["traceback"]
    assert "Traceback (most recent call last)" in rec["traceback"]
    assert rec["fields"]["sample_id"] == "s1"


def test_truncate_for_log_appends_marker_and_preserves_short_strings() -> None:
    short = "x" * 10
    assert truncate_for_log(short, limit=100) == short

    long = "y" * 5000
    truncated = truncate_for_log(long, limit=4000)
    assert truncated.startswith("y" * 4000)
    assert "…(truncated 1000 chars)" in truncated
    assert len(truncated) < len(long)


def test_logger_level_is_info_by_default() -> None:
    stream = io.StringIO()
    logger = get_logger("level_component", stream=stream)
    assert logger.logger.level == logging.INFO
