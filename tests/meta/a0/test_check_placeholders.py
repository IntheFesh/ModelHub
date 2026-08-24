"""Meta-test: prove check_placeholders.py can go red, and stays green on clean text."""

from pathlib import Path

from check_placeholders import scan


def test_bracketed_todo_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("accuracy: [TODO]\n")
    hits = scan([tmp_path])
    assert any(token == "[TODO]" for _, _, token in hits)


def test_bracketed_x_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("QPS: [X]\n")
    hits = scan([tmp_path])
    assert any(token == "[X]" for _, _, token in hits)


def test_bracketed_daice_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("延迟: [待测]\n")
    hits = scan([tmp_path])
    assert any(token == "[待测]" for _, _, token in hits)


def test_clean_text_has_zero_hits(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("accuracy: 68.9%\nQPS: 42.0\n")
    assert scan([tmp_path]) == []


def test_plain_todo_comment_is_not_a_placeholder_hit(tmp_path: Path) -> None:
    # A regular in-flight `# TODO: refactor` code comment is not what this
    # checker targets (that would make every WIP branch fail) — only the
    # bracketed doc-facing placeholder tokens are.
    f = tmp_path / "code.py"
    f.write_text("# TODO: refactor this later\nx = 1\n")
    assert scan([tmp_path]) == []
