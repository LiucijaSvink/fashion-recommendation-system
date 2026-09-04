"""Names used inside functions must actually exist.

`import` tests only prove a module loads. A name referenced inside a function body — a
logger, a helper, a constant — is not resolved until the function runs, so a missing one
survives every test and fails hours into a pipeline run. That is exactly how a stripped
`import logging` cost 25 minutes of a scored fold.
"""
from __future__ import annotations

import ast
import builtins
import importlib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "recolib"
BUILTINS = set(dir(builtins))


def _module_names(path: Path) -> set[str]:
    """Everything defined or imported at module level, plus builtins."""
    tree = ast.parse(path.read_text())
    names = set(BUILTINS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _locally_bound(function: ast.AST) -> set[str]:
    """Anything the function binds itself: arguments, assignments, imports, comprehensions."""
    bound = set()
    for node in ast.walk(function):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


@pytest.mark.parametrize("path", sorted(PACKAGE_ROOT.rglob("*.py")), ids=lambda p: p.name)
def test_function_bodies_reference_defined_names(path):
    """Check each top-level function as one scope, nested closures included.

    Names bound anywhere inside the function — including in a nested helper — count as
    available, since a closure legitimately reads its enclosing scope. That makes this
    check less precise than a real linter, but it still catches the case that matters:
    a name bound nowhere at all.
    """
    module_level = _module_names(path)
    tree = ast.parse(path.read_text())

    def top_level_functions(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child
            elif isinstance(child, ast.ClassDef):
                yield from top_level_functions(child)

    missing = []
    for function in top_level_functions(tree):
        available = module_level | _locally_bound(function)
        for inner in ast.walk(function):
            if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Load):
                if inner.id not in available:
                    missing.append(f"{path.name}:{inner.lineno} {function.name}() uses '{inner.id}'")
    assert not missing, "undefined names:\n  " + "\n  ".join(missing)
