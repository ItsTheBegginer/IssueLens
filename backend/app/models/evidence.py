"""Pydantic models for the evidence engine output."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    """How strongly the collected evidence supports the candidate."""

    strong   = "strong"    # ≥ 3 distinct evidence items or a direct file/symbol hit
    moderate = "moderate"  # 1–2 evidence items
    weak     = "weak"      # surfaced only by path/import proximity; no direct match


class SymbolEvidence(BaseModel):
    """One symbol in the candidate file that relates to the issue."""

    name: str = Field(description="Symbol name, e.g. 'parse_request'.")
    kind: str = Field(description="Symbol type: function, async_function, class, method, etc.")
    start_line: int
    end_line: int
    parent: Optional[str] = Field(
        default=None,
        description="Enclosing class name for methods.",
    )
    matched_signal: str = Field(
        description=(
            "Which issue signal matched this symbol: "
            "'function_name', 'class_name', 'error_type', or 'keyword'."
        )
    )
    matched_term: str = Field(
        description="The exact term from the issue that matched this symbol."
    )


class ImportEvidence(BaseModel):
    """An import in the candidate file whose module appears in issue signals."""

    module: str = Field(description="Imported module name, e.g. 'app.auth.utils'.")
    line: int = Field(description="Line number of the import statement.")
    matched_term: str = Field(
        description="The issue signal term that relates to this import."
    )


class CallerEvidence(BaseModel):
    """Another file that imports this candidate — a caller/dependent."""

    path: str = Field(description="Path of the file that imports this candidate.")
    is_test: bool = Field(default=False)


class TestEvidence(BaseModel):
    """A test file that imports this candidate's module."""

    __test__ = False

    path: str = Field(description="Path of the test file.")
    test_symbol_names: list[str] = Field(
        default_factory=list,
        description="Test function/class names in that file (may be empty if not analysed).",
    )


class EvidenceReport(BaseModel):
    """All evidence collected for one candidate file."""

    path: str = Field(description="Repo-relative file path of the candidate.")
    confidence: Confidence = Field(
        description="Overall strength of the evidence for this candidate."
    )

    # Direct code matches
    matched_symbols: list[SymbolEvidence] = Field(
        default_factory=list,
        description="Symbols in this file that directly match issue signals.",
    )

    # Import-level connections
    relevant_imports: list[ImportEvidence] = Field(
        default_factory=list,
        description="Imports whose module names overlap with issue signals.",
    )

    # Dependency graph
    callers: list[CallerEvidence] = Field(
        default_factory=list,
        description="Files that import this file (who depends on it).",
    )
    related_tests: list[TestEvidence] = Field(
        default_factory=list,
        description="Test files that exercise this file's module.",
    )

    # Issue-term density
    issue_term_hits: int = Field(
        default=0,
        description=(
            "Total number of distinct issue terms (keywords + function/class names) "
            "that appear anywhere in this file's path or symbol names."
        ),
    )

    # Weak-evidence explanation
    weak_reason: Optional[str] = Field(
        default=None,
        description=(
            "Present only when confidence is 'weak'; explains why the file "
            "was kept despite thin evidence."
        ),
    )


class EvidenceSet(BaseModel):
    """Evidence reports for all candidates from a search."""

    reports: list[EvidenceReport] = Field(
        default_factory=list,
        description="One EvidenceReport per candidate, in the same order as CandidateSet.",
    )
    candidates_evaluated: int = Field(
        description="Number of candidates that were processed."
    )
