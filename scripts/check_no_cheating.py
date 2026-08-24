#!/usr/bin/env python3
"""AST-based static scan for the anti-cheating patterns CLAUDE.md §1 forbids.

Regex would miss/false-positive on nested structure; every rule here walks
the real AST (plus, for the TODO-stub rule, the tokenizer's comment stream,
since comments aren't part of the AST). Rules implemented (CLAUDE.md §1.1
and the A0 task list):

  BARE_EXCEPT                  bare `except:`
  EXCEPT_PASS                  an except body that is a no-op (`pass`)
  EXCEPT_RETURN_CONSTANT       an except body that returns a bare constant
  TESTING_BRANCH                `if os.environ.get("TESTING"): ...`-style branch
  TODO_STUB_RETURN             a `return <constant>` next to a TODO/FIXME comment
  NO_TIMEOUT_CALL               network/subprocess/DB call with no `timeout=`
  SILENT_SLICE_TRUNCATION      a `[:N]` slice inside compare/ or sqlexec/
  BLACKLIST_AVAILABILITY_CHECK  `if status != FAIL:` instead of `== PASS`

Mock-only-in-tests (CLAUDE.md §1.4) is enforced separately: this script
does not scan tests/, and a plain `import` grep for `unittest.mock` /
`MockComparator`-style names under src/ is cheap enough to not need AST.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

_TIMEOUT_SENSITIVE_CALLS = {
    # (dotted-suffix match against the call's func, required kwarg)
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.patch",
    "requests.delete",
    "requests.request",
    "requests.Session",
    "httpx.get",
    "httpx.post",
    "httpx.put",
    "httpx.patch",
    "httpx.delete",
    "httpx.request",
    "httpx.Client",
    "httpx.AsyncClient",
    "urllib.request.urlopen",
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "socket.create_connection",
    "sqlite3.connect",
    "psycopg.connect",
    "psycopg2.connect",
    "redis.Redis",
    "redis.StrictRedis",
}
_TESTING_KEY_PATTERN = re.compile(r"TESTING|PYTEST|CI\b", re.IGNORECASE)
_TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME)\b", re.IGNORECASE)
_FAIL_NAME_PATTERN = re.compile(r"(^|\.)(FAIL)$")
_MOCK_IN_SRC_PATTERN = re.compile(r"\b(unittest\.mock|MockComparator|--fake-engine|FakeEngine)\b")


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    lineno: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.lineno}: [{self.rule}] {self.message}"


def _call_dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _return_is_bare_constant(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant)


def _body_is_noop_pass(body: list[ast.stmt]) -> bool:
    def _is_docstring_expr(s: ast.stmt) -> bool:
        return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)

    meaningful = [s for s in body if not _is_docstring_expr(s)]
    return len(meaningful) == 1 and isinstance(meaningful[0], ast.Pass)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str, in_compare_or_sqlexec: bool) -> None:
        self.path = path
        self.in_compare_or_sqlexec = in_compare_or_sqlexec
        self.findings: list[Finding] = []

    def _add(self, rule: str, node: ast.AST, message: str) -> None:
        lineno = getattr(node, "lineno", 0)
        self.findings.append(Finding(rule, self.path, lineno, message))

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._add("BARE_EXCEPT", node, "bare `except:` catches everything including bugs")
        if _body_is_noop_pass(node.body):
            self._add(
                "EXCEPT_PASS", node, "except body is a no-op `pass` — error is silently swallowed"
            )
        for stmt in node.body:
            if _return_is_bare_constant(stmt):
                assert isinstance(stmt, ast.Return)
                const = stmt.value
                assert isinstance(const, ast.Constant)
                self._add(
                    "EXCEPT_RETURN_CONSTANT",
                    stmt,
                    f"except handler returns bare constant {const.value!r} instead of a "
                    f"classified failure outcome (CLAUDE.md §1.2)",
                )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        test_src = ast.dump(node.test)
        if "os.environ" in test_src or "environ" in test_src:
            for sub in ast.walk(node.test):
                if not isinstance(sub, ast.Constant) or not isinstance(sub.value, str):
                    continue
                key = sub.value
                if _TESTING_KEY_PATTERN.search(key):
                    self._add(
                        "TESTING_BRANCH",
                        node,
                        f"branch keyed on env var matching {key!r} — "
                        f"test-only code path reachable in production",
                    )
                    break
        if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            op = node.test.ops[0]
            if isinstance(op, ast.NotEq | ast.IsNot):
                for side in (node.test.left, node.test.comparators[0]):
                    is_name_like = isinstance(side, ast.Attribute | ast.Name)
                    name = _call_dotted_name(side) if is_name_like else None
                    literal = side.value if isinstance(side, ast.Constant) else None
                    candidate = name or literal
                    if isinstance(candidate, str) and _FAIL_NAME_PATTERN.search(candidate):
                        self._add(
                            "BLACKLIST_AVAILABILITY_CHECK",
                            node,
                            "condition is `!= FAIL` (blacklist) — CLAUDE.md §1.3 requires a "
                            "whitelist (`== PASS`) so SKIP/WARN are never treated as available",
                        )
                        break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _call_dotted_name(node.func)
        if dotted:
            suffix_matches = [c for c in _TIMEOUT_SENSITIVE_CALLS if dotted.endswith(c)]
            if suffix_matches:
                has_timeout = any(kw.arg == "timeout" for kw in node.keywords) or any(
                    kw.arg is None
                    for kw in node.keywords  # **kwargs — can't prove absence
                )
                if not has_timeout:
                    self._add(
                        "NO_TIMEOUT_CALL",
                        node,
                        f"call to {dotted}(...) has no explicit `timeout=` "
                        f"(CLAUDE.md §5.1: no timeout is a bug)",
                    )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # ast.Slice uses `lower`/`upper`/`step`; `upper` is the `:N` stop bound.
        is_slice_with_stop = isinstance(node.slice, ast.Slice) and node.slice.upper is not None
        if self.in_compare_or_sqlexec and is_slice_with_stop:
            self._add(
                "SILENT_SLICE_TRUNCATION",
                node,
                "`[:N]` slice in compare/sqlexec — truncation must be an explicit "
                "`truncated: bool` field, never a silent slice (CLAUDE.md §1.1)",
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_todo_stub(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_todo_stub(node)
        self.generic_visit(node)

    def _check_todo_stub(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Handled by the module-level scan combining tokens + AST; see scan_file.
        pass


def _find_todo_stub_returns(path: Path, tree: ast.AST, comment_lines: set[int]) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
            nearby = {node.lineno, node.lineno - 1}
            if nearby & comment_lines:
                findings.append(
                    Finding(
                        "TODO_STUB_RETURN",
                        str(path),
                        node.lineno,
                        f"`return {node.value.value!r}` next to a TODO/FIXME — unimplemented "
                        f"logic must raise NotImplementedError, never return a plausible value "
                        f"(CLAUDE.md §1.1)",
                    )
                )
    return findings


def _comment_lines_with_todo(source: bytes) -> set[int]:
    lines: set[int] = set()
    try:
        for tok in tokenize.tokenize(iter(source.splitlines(keepends=True)).__next__):
            if tok.type == tokenize.COMMENT and _TODO_PATTERN.search(tok.string):
                lines.add(tok.start[0])
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        pass
    return lines


def scan_source(source: str, path: str) -> list[Finding]:
    tree = ast.parse(source, filename=path)
    in_compare_or_sqlexec = ("/compare/" in path.replace("\\", "/")) or (
        "/sqlexec/" in path.replace("\\", "/")
    )
    visitor = _Visitor(path, in_compare_or_sqlexec)
    visitor.visit(tree)
    findings = list(visitor.findings)
    comment_lines = _comment_lines_with_todo(source.encode())
    findings.extend(_find_todo_stub_returns(Path(path), tree, comment_lines))
    if _MOCK_IN_SRC_PATTERN.search(source):
        findings.append(
            Finding(
                "MOCK_IN_SRC",
                path,
                1,
                "mock/fake-engine reference found under src/ — mocks are only "
                "allowed in tests/ and must never be imported by src/ (CLAUDE.md §1.4)",
            )
        )
    return findings


def scan_file(path: Path) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        return scan_source(source, str(path))
    except SyntaxError as e:
        return [Finding("SYNTAX_ERROR", str(path), e.lineno or 0, str(e))]


def scan_paths(paths: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for f in files:
            findings.extend(scan_file(f))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args(argv)

    findings = scan_paths(args.paths)
    if findings:
        print(f"check_no_cheating: {len(findings)} finding(s)\n", file=sys.stderr)
        for f in findings:
            print(f, file=sys.stderr)
        return 1
    print("check_no_cheating: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
