"""Pydantic model for deterministic issue signal extraction output."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IssueSignals(BaseModel):
    """Signals extracted deterministically from a GitHub issue.

    Produced by :func:`app.analyzers.issue_signals.extract_signals`.
    Consumed by the candidate-search stage to score repository files.

    No AI is involved — all fields are derived by regex and string
    processing applied to the issue title, body, and comments.
    """

    # Raw text that anchors every other signal
    title: str = Field(description="Original issue title, unchanged.")

    # Code-level identifiers
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Deduplicated technical tokens: camelCase, snake_case, "
            "ALL_CAPS, and dotted names extracted from all issue text."
        ),
    )
    error_messages: list[str] = Field(
        default_factory=list,
        description=(
            "Exact error/exception strings: code-fence content that looks "
            "like errors, 'Error:', 'Exception:', traceback lines."
        ),
    )
    file_references: list[str] = Field(
        default_factory=list,
        description=(
            "File or path strings mentioned in the issue, e.g. "
            "'src/auth.py', 'routes/login.js', 'config.yaml'."
        ),
    )
    function_names: list[str] = Field(
        default_factory=list,
        description=(
            "Function or method names identified in the issue text, "
            "e.g. 'handle_request', 'parseConfig', 'render()'."
        ),
    )
    class_names: list[str] = Field(
        default_factory=list,
        description=(
            "Class or type names identified in the issue text, "
            "e.g. 'RequestHandler', 'AuthMiddleware'."
        ),
    )

    # Behavioral signals
    expected_behavior: Optional[str] = Field(
        default=None,
        description=(
            "Sentence(s) from the issue describing what was expected "
            "(heuristically detected from 'expected', 'should', etc.)."
        ),
    )
    observed_behavior: Optional[str] = Field(
        default=None,
        description=(
            "Sentence(s) from the issue describing what actually happened "
            "(heuristically detected from 'actual', 'instead', 'but', etc.)."
        ),
    )
