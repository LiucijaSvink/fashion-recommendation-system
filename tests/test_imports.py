"""Every import in the package must resolve — including the lazy ones.

Imports inside function bodies are invisible to a normal import test: the module loads
fine and only fails when the function is called. Splitting `analysis` into a package
broke seven of them this way, and nothing caught it until a two-hour notebook run
reached the offending cell.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "recolib"


def _module_name(path: Path) -> str:
    relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    return ".".join(relative.parts)


def _relative_imports():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                package_parts = _module_name(path).split(".")[: -node.level]
                target = ".".join(package_parts + ([node.module] if node.module else []))
                yield path.name, node.lineno, target


@pytest.mark.parametrize("filename,lineno,target", list(_relative_imports()))
def test_relative_import_resolves(filename, lineno, target):
    importlib.import_module(target)


def test_every_module_imports():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        importlib.import_module(_module_name(path))
