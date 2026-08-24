from pathlib import Path

import pytest

from modelhub.common.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    read_json,
)


def test_atomic_write_text_creates_file_with_exact_content(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    atomic_write_text(target, "hello world")
    assert target.read_text() == "hello world"


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    obj = {"run_id": "r1", "seed": 42, "nested": {"a": [1, 2, 3]}}
    atomic_write_json(target, obj)
    assert read_json(target) == obj


def test_atomic_write_leaves_no_tmp_file_behind(tmp_path: Path) -> None:
    target = tmp_path / "clean.json"
    atomic_write_json(target, {"ok": True})
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_atomic_write_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "runs" / "abc123" / "manifest.json"
    atomic_write_json(target, {"run_id": "abc123"})
    assert target.exists()
    assert read_json(target) == {"run_id": "abc123"}


def test_atomic_write_overwrites_only_on_success(tmp_path: Path) -> None:
    target = tmp_path / "existing.json"
    atomic_write_json(target, {"version": 1})
    atomic_write_json(target, {"version": 2})
    assert read_json(target) == {"version": 2}


def test_atomic_write_json_is_reproducible_across_key_order(tmp_path: Path) -> None:
    # Sorted keys means the on-disk bytes don't depend on dict insertion order —
    # important for anything hashing the manifest file.
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    atomic_write_json(a, {"z": 1, "a": 2})
    atomic_write_json(b, {"a": 2, "z": 1})
    assert a.read_text() == b.read_text()


def test_atomic_write_bytes_rejects_unwritable_target(tmp_path: Path) -> None:
    # Target's parent is actually a file, not a directory: mkdir(parents=True)
    # inside atomic_write_bytes will fail, and we must get our own
    # classified ModelHubError, not a bare OSError leaking out.
    from modelhub.common.errors import ModelHubError

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    target = blocker / "child" / "out.json"
    with pytest.raises(ModelHubError):
        atomic_write_bytes(target, b"{}")
