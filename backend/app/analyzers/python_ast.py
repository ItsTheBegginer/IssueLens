"""Python AST analyzer.

Extracts symbols (classes, functions, methods, imports) from Python source
files using the standard-library `ast` module — no LLM required.

A parse failure in one file is caught and recorded; analysis continues.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

from app.models.intelligence import (
    FileSymbols,
    ImportRecord,
    ImportedName,
    Symbol,
    SymbolType,
)

# Regex that identifies test files by path or name convention.
_TEST_FILE_RE = re.compile(
    r"(^|/)tests?/|/(test_[^/]+|[^/]+_test)\.py$",
    re.IGNORECASE,
)


def is_test_file(path: str) -> bool:
    return bool(_TEST_FILE_RE.search(path))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_file(path: str, source: str) -> FileSymbols:
    """Parse *source* and return a :class:`FileSymbols` for *path*.

    Never raises — parse errors are stored in ``FileSymbols.parse_error``.
    """
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return FileSymbols(
            path=path,
            parse_error=f"SyntaxError: {exc}",
            is_test=is_test_file(path),
        )
    except Exception as exc:  # e.g. MemoryError on pathological input
        return FileSymbols(
            path=path,
            parse_error=f"Unexpected error during parsing: {exc}",
            is_test=is_test_file(path),
        )

    symbols: list[Symbol] = []
    imports: list[ImportRecord] = []

    _walk(tree, source_lines=source.splitlines(), symbols=symbols, imports=imports)

    return FileSymbols(
        path=path,
        symbols=symbols,
        imports=imports,
        is_test=is_test_file(path),
    )


# ---------------------------------------------------------------------------
# AST walking
# ---------------------------------------------------------------------------

def _end_line(node: ast.AST) -> int:
    """Return end_lineno when available, else fall back to lineno."""
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)


def _walk(
    tree: ast.Module,
    *,
    source_lines: list[str],
    symbols: list[Symbol],
    imports: list[ImportRecord],
    parent_class: Optional[str] = None,
) -> None:
    """Recursively walk top-level and class-level nodes."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_async = isinstance(node, ast.AsyncFunctionDef)
            if parent_class:
                sym_type = SymbolType.async_method if is_async else SymbolType.method
            else:
                sym_type = SymbolType.async_function if is_async else SymbolType.function

            symbols.append(
                Symbol(
                    name=node.name,
                    type=sym_type,
                    start_line=node.lineno,
                    end_line=_end_line(node),
                    parent=parent_class,
                )
            )
            # Recurse into nested functions / classes inside this function
            _walk(node, source_lines=source_lines, symbols=symbols, imports=imports, parent_class=parent_class)

        elif isinstance(node, ast.ClassDef):
            symbols.append(
                Symbol(
                    name=node.name,
                    type=SymbolType.class_,
                    start_line=node.lineno,
                    end_line=_end_line(node),
                    parent=None,
                )
            )
            # Walk class body — methods get parent=class name
            _walk(node, source_lines=source_lines, symbols=symbols, imports=imports, parent_class=node.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportRecord(
                        module=alias.name,
                        names=[],
                        is_relative=False,
                        line=node.lineno,
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            names = [
                ImportedName(name=a.name, alias=a.asname)
                for a in node.names
            ]
            imports.append(
                ImportRecord(
                    module=("." * level) + module,
                    names=names,
                    is_relative=level > 0,
                    line=node.lineno,
                )
            )
