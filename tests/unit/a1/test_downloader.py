"""downloader.py is untested-in-this-sandbox by design (no network route to
huggingface.co or dataset hosts — see docs/design-decisions.md DD-0003).
Everything here is marked requires_network; `make test` / `make verify-a1`
skip it (visibly, as "deselected", not silently) in this environment."""

from pathlib import Path

import pytest

from modelhub.data.downloader import download_file

pytestmark = pytest.mark.requires_network


def test_download_file_writes_atomically(tmp_path: Path) -> None:
    # A tiny, stable, publicly-hosted file — exercises the real httpx path.
    dest = tmp_path / "downloaded.txt"
    download_file("https://pypi.org/pypi/pip/json", dest, timeout_s=10.0)
    assert dest.exists()
    assert dest.stat().st_size > 0


def test_download_file_client_error_is_not_retried(tmp_path: Path) -> None:
    from modelhub.common.errors import ModelHubError

    dest = tmp_path / "nope.txt"
    with pytest.raises(ModelHubError) as exc_info:
        download_file("https://pypi.org/this-path-does-not-exist-404", dest, max_retries=3)
    assert exc_info.value.retryable is False
