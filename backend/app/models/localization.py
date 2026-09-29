"""Pydantic models for the primary localization result."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UncertaintyReason(str, Enum):
    """Why a definitive localization could not be produced."""

    no_candidates = "no_candidates"
    """Candidate search returned no results for these signals."""

    all_weak = "all_weak"
    """All candidates have weak evidence; no confident primary can be chosen."""

    ambiguous = "ambiguous"
    """Top candidates are too close in score to pick one reliably."""

    score_too_low = "score_too_low"
    """Top candidate score is below the minimum threshold for a confident result."""


class LineRange(BaseModel):
    """A real line range from the repository index — never fabricated."""

    start: int = Field(ge=1, description="First line of the symbol (1-based).")
    end: int = Field(ge=1, description="Last line of the symbol (inclusive).")


class RelatedLocation(BaseModel):
    """A secondary file/symbol relevant to the issue."""

    path: str
    symbol: Optional[str] = Field(
        default=None,
        description="Symbol name within the file, if directly matched.",
    )
    score: float = Field(ge=0.0)
    reason: str = Field(description="One-line explanation of why this location is related.")


class LocalizationResult(BaseModel):
    """The primary localization result for an issue.

    Represents the best deterministic answer about where in the repository
    the issue is likely rooted.  ``is_uncertain`` must be checked before
    consuming ``primary_file`` and related fields.
    """

    # ------------------------------------------------------------------
    # Uncertainty gate — always check this first
    # ------------------------------------------------------------------
    is_uncertain: bool = Field(
        description=(
            "True when available evidence is insufficient to name a confident "
            "primary location.  All primary_* fields are None when True."
        )
    )
    uncertainty_reason: Optional[UncertaintyReason] = Field(
        default=None,
        description="Set only when is_uncertain is True.",
    )
    uncertainty_detail: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the uncertainty.",
    )

    # ------------------------------------------------------------------
    # Primary location (None when is_uncertain=True)
    # ------------------------------------------------------------------
    primary_file: Optional[str] = Field(
        default=None,
        description="Repo-relative path of the most likely file.",
    )
    primary_symbol: Optional[str] = Field(
        default=None,
        description=(
            "Name of the function or class most likely involved, "
            "derived from SymbolEvidence.  None when no symbol directly matched."
        ),
    )
    primary_symbol_kind: Optional[str] = Field(
        default=None,
        description="Symbol type of primary_symbol (function, method, class, …).",
    )
    line_range: Optional[LineRange] = Field(
        default=None,
        description=(
            "Real line range from the repository index for primary_symbol.  "
            "None when primary_symbol is None."
        ),
    )

    # ------------------------------------------------------------------
    # Confidence and scoring
    # ------------------------------------------------------------------
    evidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized 0–1 confidence score combining candidate rank score "
            "and evidence quality."
        ),
    )
    evidence_confidence: Optional[str] = Field(
        default=None,
        description="Evidence confidence level: 'strong', 'moderate', or 'weak'.",
    )

    # ------------------------------------------------------------------
    # Explanation (from real data only)
    # ------------------------------------------------------------------
    explanation: str = Field(
        description=(
            "Human-readable explanation built from actual signals and evidence. "
            "Never contains invented information."
        )
    )

    # ------------------------------------------------------------------
    # Related locations
    # ------------------------------------------------------------------
    related: list[RelatedLocation] = Field(
        default_factory=list,
        description="Other relevant files/symbols in descending relevance order.",
    )
