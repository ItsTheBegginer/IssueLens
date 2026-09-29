"""Pydantic model for structured issue understanding output."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IssueUnderstanding(BaseModel):
    """Structured representation of a GitHub issue, extracted by the model.

    Every field is optional so that a partial model response can still be
    validated and used by downstream components.
    """

    problem_summary: str = Field(
        description="One or two sentence description of the core problem."
    )
    expected_behavior: Optional[str] = Field(
        default=None,
        description="What the user expected to happen.",
    )
    observed_behavior: Optional[str] = Field(
        default=None,
        description="What actually happened instead.",
    )
    components: list[str] = Field(
        default_factory=list,
        description=(
            "Relevant system components, modules, classes, or subsystems "
            "mentioned or implied by the issue."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Important technical keywords that could appear in source code "
            "or file names relevant to this issue."
        ),
    )
    error_messages: list[str] = Field(
        default_factory=list,
        description="Exact error messages, tracebacks, or exception names found in the issue.",
    )
