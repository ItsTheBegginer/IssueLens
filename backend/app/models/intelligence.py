"""Pydantic models for the repository intelligence engine."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SymbolType(str, Enum):
    function = "function"
    async_function = "async_function"
    class_ = "class"
    method = "method"
    async_method = "async_method"
    import_ = "import"


class Symbol(BaseModel):
    name: str
    type: SymbolType
    start_line: int
    end_line: int
    parent: Optional[str] = None  # enclosing class name for methods


class ImportedName(BaseModel):
    """A single name imported from a module."""
    name: str
    alias: Optional[str] = None


class ImportRecord(BaseModel):
    module: str                    # e.g. "os.path", "flask"
    names: list[ImportedName] = [] # empty = bare `import module`
    is_relative: bool = False
    line: int


class FileSymbols(BaseModel):
    path: str
    symbols: list[Symbol] = []
    imports: list[ImportRecord] = []
    is_test: bool = False
    parse_error: Optional[str] = None  # set when AST parsing fails


class RelationshipType(str, Enum):
    file_imports_module = "file_imports_module"
    class_contains_method = "class_contains_method"
    function_belongs_to_module = "function_belongs_to_module"
    test_references_module = "test_references_module"
    function_calls = "function_calls"


class Relationship(BaseModel):
    source: str         # file path or qualified name
    target: str         # file path, module name, or qualified name
    type: RelationshipType
    file: Optional[str] = None  # file where this relationship was found


class RepoIntelligence(BaseModel):
    """Top-level container produced by the intelligence engine."""
    repo_full_name: str
    file_count: int
    analyzed_count: int
    skipped_count: int
    files: list[FileSymbols] = []
    relationships: list[Relationship] = []
