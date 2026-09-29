"""Pydantic models for the end-to-end analysis pipeline."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.localization import LineRange, RelatedLocation


class ExecutionPathStep(BaseModel):
    """One step in the traced execution or dependency path."""

    step_number: int = Field(description="1-based step order in the execution flow.")
    step_type: str = Field(
        description="Type of step: 'caller' (upstream entry), 'target' (suspected root cause), 'dependency' (callee/downstream)."
    )
    file: str = Field(description="Repo-relative file path.")
    symbol: Optional[str] = Field(
        default=None,
        description="Function, method, or class name involved at this step.",
    )
    line_range: Optional[LineRange] = Field(
        default=None,
        description="Line range in the file, if known from AST analysis.",
    )
    detail: str = Field(description="Human-readable explanation of this step in the execution flow.")


class RelevantTest(BaseModel):
    """A test file and specific test functions relevant to the issue."""

    __test__ = False

    path: str = Field(description="Repo-relative path of the test file.")
    test_symbols: list[str] = Field(
        default_factory=list,
        description="Names of test functions or methods in this file that relate to the target code.",
    )
    reason: str = Field(description="Why this test is relevant (e.g. imports target module, tests target symbol).")


class AnalysisRequest(BaseModel):
    """Input payload for the analyze endpoint."""

    repo_url: str = Field(
        description="Full GitHub repository URL, e.g. 'https://github.com/psf/requests'."
    )
    issue_number: int = Field(
        ge=1,
        description="GitHub issue number (positive integer).",
    )


class AnalysisResponse(BaseModel):
    """Complete structured localization result returned by the backend."""

    # Context
    repo_url: str
    issue_number: int
    issue_title: str

    # Uncertainty gate
    is_uncertain: bool = Field(
        description="True if evidence is insufficient to name a confident primary location."
    )
    uncertainty_reason: Optional[str] = Field(
        default=None,
        description="Reason code when is_uncertain is True.",
    )
    uncertainty_detail: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of uncertainty.",
    )

    # Primary localization
    primary_file: Optional[str] = Field(
        default=None,
        description="Repo-relative path of the most likely responsible file.",
    )
    primary_symbol: Optional[str] = Field(
        default=None,
        description="Function or class name most likely involved.",
    )
    primary_symbol_kind: Optional[str] = Field(
        default=None,
        description="Kind of symbol: 'function', 'method', 'class', etc.",
    )
    line_range: Optional[LineRange] = Field(
        default=None,
        description="1-based start and end line numbers from real AST data.",
    )

    # Scoring & Evidence
    evidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized 0-1 confidence score.",
    )
    evidence_confidence: Optional[str] = Field(
        default=None,
        description="Confidence level: 'strong', 'moderate', or 'weak'.",
    )
    explanation: str = Field(
        description="Deterministic explanation grounded in actual repository signals."
    )
    ai_explanation: Optional[str] = Field(
        default=None,
        description="Optional model-generated explanation explaining the deterministic evidence.",
    )

    # Related files and context
    related_files: list[RelatedLocation] = Field(
        default_factory=list,
        description="Secondary candidate locations with scores and reasons.",
    )

    # Execution flow & tests
    execution_path: list[str] = Field(
        default_factory=list,
        description="Ordered list of textual execution flow steps leading to and from the target.",
    )
    execution_path_steps: list[ExecutionPathStep] = Field(
        default_factory=list,
        description="Structured execution flow steps.",
    )
    relevant_tests: list[RelevantTest] = Field(
        default_factory=list,
        description="Relevant test files and test functions.",
    )

    # Metadata
    total_files_analyzed: int = Field(
        default=0,
        description="Total source files parsed and analyzed in this repository.",
    )
