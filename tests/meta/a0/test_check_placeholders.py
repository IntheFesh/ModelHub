"""Meta-test: prove check_placeholders.py can go red, and stays green on clean text."""

from pathlib import Path

from check_placeholders import main, scan


def test_bracketed_todo_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("accuracy: [TODO]\n")
    hits, unreadable = scan([tmp_path])
    assert any(token == "[TODO]" for _, _, token in hits)
    assert unreadable == []


def test_bracketed_x_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("QPS: [X]\n")
    hits, _ = scan([tmp_path])
    assert any(token == "[X]" for _, _, token in hits)


def test_bracketed_daice_is_flagged(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("延迟: [待测]\n")
    hits, _ = scan([tmp_path])
    assert any(token == "[待测]" for _, _, token in hits)


def test_clean_text_has_zero_hits(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("accuracy: 68.9%\nQPS: 42.0\n")
    assert scan([tmp_path]) == ([], [])


def test_plain_todo_comment_is_not_a_placeholder_hit(tmp_path: Path) -> None:
    # A regular in-flight `# TODO: refactor` code comment is not what this
    # checker targets (that would make every WIP branch fail) — only the
    # bracketed doc-facing placeholder tokens are.
    f = tmp_path / "code.py"
    f.write_text("# TODO: refactor this later\nx = 1\n")
    assert scan([tmp_path]) == ([], [])


def test_unreadable_file_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    # Regression test: a file this scanner cannot decode as UTF-8 used to
    # be silently `continue`d past with zero trace — a scan that dropped
    # an unreadable file and still printed "clean (0 placeholders)" is the
    # exact skip-counted-as-pass pattern CLAUDE.md §1.3 forbids, just at
    # file-discovery time instead of a capability probe. We genuinely
    # don't know whether the file contains a placeholder, so it must show
    # up as unreadable rather than vanish.
    f = tmp_path / "report.md"
    f.write_bytes(b"\xff\xfe not valid utf-8 \x00\x01")
    hits, unreadable = scan([tmp_path])
    assert hits == []
    assert unreadable == [str(f)]


def test_unreadable_file_fails_main_even_with_no_placeholder_hits(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_bytes(b"\xff\xfe not valid utf-8 \x00\x01")
    assert main([str(tmp_path)]) == 1


def test_clean_readable_files_still_pass_main(tmp_path: Path) -> None:
    f = tmp_path / "report.md"
    f.write_text("accuracy: 68.9%\n")
    assert main([str(tmp_path)]) == 0
