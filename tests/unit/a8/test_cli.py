"""Unit tests for bench/cli.py — `make bench PROFILE=<name>`'s real
entrypoint.

Regression coverage for a real gap this repo shipped for 12+ rounds:
`make bench` referenced `python -m modelhub.bench.cli`, but that module
never existed, so the command crashed with `ModuleNotFoundError` instead
of the graceful GPU-exclusivity rejection CLAUDE.md §5.2 describes.
Tested against this sandbox's real, honest absence of `nvidia-smi`
(same precedent as tests/unit/a8/test_gpu_guard.py) — not a mocked GPU
state.
"""

from __future__ import annotations

import pytest

from modelhub.bench.cli import main


def test_unknown_profile_fails_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--profile", "does-not-exist"])
    assert exit_code == 1
    assert "no such model profile" in capsys.readouterr().err


def test_known_profile_is_rejected_by_the_real_gpu_guard_in_this_sandbox(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # This sandbox genuinely has no nvidia-smi — guard_gpu_exclusivity
    # must stop main() here with a clean, informative failure, never a
    # crash and never silently proceeding to the NotImplementedError path
    # meant only for a confirmed-exclusive real GPU.
    exit_code = main(["--profile", "arctic_7b"])
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "refusing to start bench run" in err
    assert "GPU 0 exclusivity is not confirmed" in err


def test_gpu_index_argument_is_threaded_through(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--profile", "arctic_7b", "--gpu-index", "3"])
    assert exit_code == 1
    assert "GPU 3 exclusivity is not confirmed" in capsys.readouterr().err


def test_missing_profile_argument_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        main([])
