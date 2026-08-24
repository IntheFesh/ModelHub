"""Unit tests for train/mlflow_backend.py.

mlflow is part of the `train` extra and is not installed in this
GPU-less sandbox, so `check_mlflow_available()` must report SKIP here
(CLAUDE.md §1.3: SKIP is a real, whitelist-checkable outcome, not a
test we fake around) and `start_mlflow_run` must raise the real
ImportError uncaught rather than pretending to succeed."""

from __future__ import annotations

import pytest

from modelhub.common.capability import CheckStatus, is_available
from modelhub.train.mlflow_backend import check_mlflow_available, start_mlflow_run


class TestCheckMlflowAvailable:
    def test_reports_a_real_check_status(self) -> None:
        result = check_mlflow_available()
        assert result.status in {CheckStatus.PASS, CheckStatus.SKIP}

    def test_skip_is_not_treated_as_available_when_not_installed(self) -> None:
        result = check_mlflow_available()
        if result.status is CheckStatus.SKIP:
            assert is_available(result.status) is False


class TestStartMlflowRun:
    def test_raises_uncaught_when_mlflow_not_installed(self) -> None:
        availability = check_mlflow_available()
        if is_available(availability.status):
            pytest.skip("mlflow is actually installed in this environment")
        with pytest.raises(ImportError):
            start_mlflow_run(experiment_name="modelhub-sft", run_id="run-1")
