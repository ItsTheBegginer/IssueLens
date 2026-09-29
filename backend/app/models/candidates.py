"""Pydantic models for candidate code search results."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MatchKind(str, Enum):
    """The type of signal that caused a candidate to be included."""

    file_reference = "file_reference"
    """Issue text explicitly named this file path."""

    symbol_name = "symbol_name"
    """A function or class name in this file matches an issue signal."""

    path_keyword = "path_keyword"
    """A keyword from the issue appears in this file's path."""

    import_neighbor = "import_neighbor"
    """This file imports—or is imported by—a higher-scored candidate."""

    error_type = "error_type"
    """An error/exception class named in the issue is defined here."""

    test_coverage = "test_coverage"
    """A test file imports this module, linking it to the issue context."""


class MatchReason(BaseModel):
    """One signal that contributed to a candidate's score."""

    kind: MatchKind
    detail: str = Field(
        description="Human-readable description of the specific match, "
        "e.g. 'symbol parse_request matches issue function name'."
    )
    weight: float = Field(
        description="Score contribution of this reason.",
        ge=0.0,
    )


class CandidateFile(BaseModel):
    """A repository file scored as a candidate for the reported issue."""

    path: str = Field(description="Repo-relative file path, e.g. 'src/auth/middleware.py'.")
    score: float = Field(description="Total score across all match reasons.", ge=0.0)
    reasons: list[MatchReason] = Field(
        default_factory=list,
        description="Ordered list of signals that contributed to this score.",
    )
    is_test: bool = Field(
        default=False,
        description="True when this file is a test file.",
    )
    matched_symbols: list[str] = Field(
        default_factory=list,
        description="Symbol names (functions/classes) in this file that matched issue signals.",
    )


class CandidateSet(BaseModel):
    """Ranked search results for one issue."""

    candidates: list[CandidateFile] = Field(
        default_factory=list,
        description="Candidates in descending score order.",
    )
    total_files_considered: int = Field(
        description="Number of repository files evaluated.",
    )
    signals_used: list[str] = Field(
        default_factory=list,
        description="Which IssueSignals fields had non-empty content and were used.",
    )
