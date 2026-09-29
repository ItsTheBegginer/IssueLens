"""Relationship builder.

Derives lightweight relationships from :class:`FileSymbols` data.
No LLM — purely structural.
"""
from __future__ import annotations

from app.models.intelligence import (
    FileSymbols,
    Relationship,
    RelationshipType,
    SymbolType,
)


def build_relationships(files: list[FileSymbols]) -> list[Relationship]:
    """Return all relationships derivable from the analysed file list."""
    rels: list[Relationship] = []
    for fs in files:
        rels.extend(_relationships_for_file(fs))
    return rels


def _relationships_for_file(fs: FileSymbols) -> list[Relationship]:
    rels: list[Relationship] = []

    # 1. file_imports_module
    for imp in fs.imports:
        rels.append(
            Relationship(
                source=fs.path,
                target=imp.module,
                type=RelationshipType.file_imports_module,
                file=fs.path,
            )
        )

    # 2. class_contains_method
    for sym in fs.symbols:
        if sym.type in (SymbolType.method, SymbolType.async_method) and sym.parent:
            rels.append(
                Relationship(
                    source=f"{fs.path}::{sym.parent}",
                    target=f"{fs.path}::{sym.parent}.{sym.name}",
                    type=RelationshipType.class_contains_method,
                    file=fs.path,
                )
            )

    # 3. function_belongs_to_module (top-level functions only)
    module_name = _path_to_module(fs.path)
    for sym in fs.symbols:
        if sym.type in (SymbolType.function, SymbolType.async_function):
            rels.append(
                Relationship(
                    source=sym.name,
                    target=module_name,
                    type=RelationshipType.function_belongs_to_module,
                    file=fs.path,
                )
            )

    # 4. test_references_module — test files that import a module
    if fs.is_test:
        for imp in fs.imports:
            if not imp.is_relative and not imp.module.startswith("_"):
                rels.append(
                    Relationship(
                        source=fs.path,
                        target=imp.module,
                        type=RelationshipType.test_references_module,
                        file=fs.path,
                    )
                )

    return rels


def _path_to_module(path: str) -> str:
    """Convert a file path to a dotted module name.

    ``src/app/utils.py`` → ``src.app.utils``
    ``app/utils.py``     → ``app.utils``
    """
    return path.replace("/", ".").removesuffix(".py")
