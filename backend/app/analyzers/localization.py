"""Primary localization engine.

Combines candidate rankings and evidence reports to produce a single
``LocalizationResult`` — the best deterministic answer about which file
and symbol is most likely responsible for the reported issue.

All decisions are grounded in real repository data.  No LLM, no fabrication.
When evidence is insufficient the engine returns an explicit uncertainty state
rather than inventing precision.

Public API
----------
``localize(candidates, evidence_set, signals) -> LocalizationResult``
"""
from __future__ import annotations

from typing import Optional

from app.models.candidates import CandidateFile, CandidateSet, MatchKind
from app.models.evidence import Confidence, EvidenceReport, EvidenceSet, SymbolEvidence
from app.models.issue_signals import IssueSignals
from app.models.localization import (
    LineRange,
    LocalizationResult,
    RelatedLocation,
    UncertaintyReason,
)

__all__ = ["localize"]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum normalized evidence_score required to name a confident primary.
_MIN_CONFIDENT_SCORE = 0.15

# If the top-2 candidates are within this fraction of each other AND both are
# at most 'moderate' confidence, the result is ambiguous.
_AMBIGUITY_FRACTION = 0.15   # 15 % gap

# Maximum candidate score used for normalization (soft cap; higher scores
# are clamped to 1.0).  Tuned to the weight constants in candidate_search.
_SCORE_NORM_CAP = 20.0

# Symbol-signal priority: higher = chosen first when multiple symbols match.
_SIGNAL_PRIORITY = {
    "function_name": 4,
    "class_name":    3,
    "error_type":    2,
    "keyword":       1,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_score(raw_score: float, confidence: Confidence) -> float:
    """Map raw candidate score + confidence to a 0–1 float."""
    base = min(raw_score / _SCORE_NORM_CAP, 1.0)
    boost = {"strong": 0.15, "moderate": 0.05, "weak": 0.0}[confidence.value]
    return min(base + boost, 1.0)


def _pick_primary_symbol(
    evidence: EvidenceReport,
) -> Optional[SymbolEvidence]:
    """Return the best SymbolEvidence from the report, or None.

    Selection criteria (in order):
    1. Highest signal priority (function_name > class_name > error_type > keyword).
    2. Prefer real multi-line implementation over 1-line type stub.
    3. Among ties, earliest start_line.
    """
    if not evidence.matched_symbols:
        return None
    return min(
        evidence.matched_symbols,
        key=lambda s: (
            -_SIGNAL_PRIORITY.get(s.matched_signal, 0),
            -(1 if s.end_line > s.start_line else 0),
            s.start_line,
        ),
    )


def _build_explanation(
    candidate: CandidateFile,
    evidence: EvidenceReport,
    primary_sym: Optional[SymbolEvidence],
) -> str:
    """Build an explanation string from real data — no invention."""
    parts: list[str] = []

    # Lead with the primary symbol if present
    if primary_sym:
        loc = f"line {primary_sym.start_line}"
        if primary_sym.parent:
            parts.append(
                f"'{primary_sym.parent}.{primary_sym.name}' ({primary_sym.kind}) "
                f"at {loc} matches issue {primary_sym.matched_signal.replace('_', ' ')} "
                f"'{primary_sym.matched_term}'."
            )
        else:
            parts.append(
                f"'{primary_sym.name}' ({primary_sym.kind}) at {loc} "
                f"matches issue {primary_sym.matched_signal.replace('_', ' ')} "
                f"'{primary_sym.matched_term}'."
            )

    # Candidate search reasons (deduplicated by kind)
    seen_kinds: set[str] = set()
    for reason in candidate.reasons:
        kind_key = reason.kind.value
        if kind_key not in seen_kinds:
            seen_kinds.add(kind_key)
            parts.append(reason.detail)

    # Evidence extras
    if evidence.related_tests:
        test_paths = [t.path for t in evidence.related_tests[:2]]
        parts.append(f"Covered by test(s): {', '.join(test_paths)}.")

    if evidence.callers:
        caller_paths = [c.path for c in evidence.callers[:2] if not c.is_test]
        if caller_paths:
            parts.append(f"Used by: {', '.join(caller_paths)}.")

    if evidence.relevant_imports:
        imp_modules = [i.module for i in evidence.relevant_imports[:2]]
        parts.append(f"Imports related module(s): {', '.join(imp_modules)}.")

    return "  ".join(parts) if parts else f"File '{candidate.path}' matched by structural proximity."


def _build_related(
    candidates: list[CandidateFile],
    evidence_map: dict[str, EvidenceReport],
    skip_path: str,
    *,
    limit: int = 5,
) -> list[RelatedLocation]:
    """Build related locations from remaining candidates."""
    related: list[RelatedLocation] = []
    for c in candidates:
        if c.path == skip_path:
            continue
        ev = evidence_map.get(c.path)
        # Build a one-line reason from the highest-weight search reason
        if c.reasons:
            best_reason = max(c.reasons, key=lambda r: r.weight)
            reason_text = best_reason.detail
        else:
            reason_text = "structural proximity"

        sym: Optional[str] = None
        if ev and ev.matched_symbols:
            best_sym = _pick_primary_symbol(ev)
            if best_sym:
                sym = best_sym.name

        related.append(RelatedLocation(
            path=c.path,
            symbol=sym,
            score=c.score,
            reason=reason_text,
        ))
        if len(related) >= limit:
            break
    return related


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------

def _uncertain(
    reason: UncertaintyReason,
    detail: str,
    related: list[RelatedLocation],
) -> LocalizationResult:
    return LocalizationResult(
        is_uncertain=True,
        uncertainty_reason=reason,
        uncertainty_detail=detail,
        evidence_score=0.0,
        explanation=detail,
        related=related,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def localize(
    candidates: CandidateSet,
    evidence_set: EvidenceSet,
    signals: IssueSignals,
) -> LocalizationResult:
    """Produce a primary localization from candidates and evidence.

    Parameters
    ----------
    candidates:
        Output of :func:`app.search.candidate_search.search_candidates`.
    evidence_set:
        Output of :func:`app.analyzers.evidence.collect_evidence`.
    signals:
        Extracted issue signals (used for explanation text only).

    Returns
    -------
    LocalizationResult
        Check ``is_uncertain`` before using ``primary_file`` and related fields.
    """
    # ------------------------------------------------------------------ #
    # 1. Guard: no candidates at all
    # ------------------------------------------------------------------ #
    if not candidates.candidates:
        return _uncertain(
            UncertaintyReason.no_candidates,
            "No candidate files were found for the given issue signals. "
            "The repository index may be empty or the issue text contains "
            "no recognisable code identifiers.",
            related=[],
        )

    # ------------------------------------------------------------------ #
    # 2. Build evidence map for fast lookup
    # ------------------------------------------------------------------ #
    evidence_map: dict[str, EvidenceReport] = {
        r.path: r for r in evidence_set.reports
    }

    # ------------------------------------------------------------------ #
    # 3. Pair candidates with their evidence; sort by evidence-weighted score
    # ------------------------------------------------------------------ #
    Pair = tuple[CandidateFile, EvidenceReport]

    def _ev_score(c: CandidateFile) -> float:
        ev = evidence_map.get(c.path)
        conf = ev.confidence if ev else Confidence.weak
        return _normalise_score(c.score, conf)

    ordered: list[Pair] = [
        (c, evidence_map.get(c.path, EvidenceReport(
            path=c.path, confidence=Confidence.weak,
            weak_reason="No evidence report available.",
        )))
        for c in candidates.candidates
    ]
    # Sort by normalized evidence score; break ties by preferring non-test files
    ordered.sort(key=lambda p: (_ev_score(p[0]), not p[0].is_test), reverse=True)

    top_candidate, top_evidence = ordered[0]
    top_score = _ev_score(top_candidate)

    # ------------------------------------------------------------------ #
    # 4. Guard: all candidates are weak and score is below threshold
    # ------------------------------------------------------------------ #
    all_weak = all(
        (evidence_map.get(c.path, EvidenceReport(
            path=c.path, confidence=Confidence.weak,
        )).confidence == Confidence.weak)
        for c, _ in ordered
    )

    if all_weak and top_score < _MIN_CONFIDENT_SCORE:
        related = _build_related(
            [p[0] for p in ordered], evidence_map, skip_path=""
        )
        return _uncertain(
            UncertaintyReason.all_weak,
            f"All {len(ordered)} candidate(s) have weak evidence "
            f"(top normalized score: {top_score:.2f}). "
            "The issue signals may not map to any analysed symbols.",
            related=related,
        )

    # ------------------------------------------------------------------ #
    # 5. Guard: ambiguous top-2
    # ------------------------------------------------------------------ #
    if len(ordered) >= 2:
        second_candidate, second_evidence = ordered[1]
        second_score = _ev_score(second_candidate)

        both_not_strong = (
            top_evidence.confidence != Confidence.strong
            and second_evidence.confidence != Confidence.strong
        )
        scores_close = (
            top_score > 0
            and abs(top_score - second_score) / top_score <= _AMBIGUITY_FRACTION
        )

        # If one is implementation and one is test, prefer implementation rather than declaring ambiguity
        if top_candidate.is_test != second_candidate.is_test:
            pass
        elif both_not_strong and scores_close:
            related = _build_related(
                [p[0] for p in ordered], evidence_map, skip_path=""
            )
            return _uncertain(
                UncertaintyReason.ambiguous,
                f"Top candidates '{top_candidate.path}' (score {top_score:.2f}) "
                f"and '{second_candidate.path}' (score {second_score:.2f}) "
                "are too close to choose confidently. "
                "Both have at most moderate evidence.",
                related=related,
            )

    # ------------------------------------------------------------------ #
    # 6. Score too low (but not all_weak — there's some evidence, just thin)
    # ------------------------------------------------------------------ #
    if top_score < _MIN_CONFIDENT_SCORE:
        related = _build_related(
            [p[0] for p in ordered], evidence_map, skip_path=""
        )
        return _uncertain(
            UncertaintyReason.score_too_low,
            f"Top candidate '{top_candidate.path}' normalized score "
            f"{top_score:.2f} is below the confidence threshold "
            f"({_MIN_CONFIDENT_SCORE}).",
            related=related,
        )

    # ------------------------------------------------------------------ #
    # 7. We have a confident primary — assemble result
    # ------------------------------------------------------------------ #
    primary_sym = _pick_primary_symbol(top_evidence)
    line_range: Optional[LineRange] = None
    if primary_sym:
        line_range = LineRange(
            start=primary_sym.start_line,
            end=primary_sym.end_line,
        )

    explanation = _build_explanation(top_candidate, top_evidence, primary_sym)

    related = _build_related(
        [p[0] for p in ordered],
        evidence_map,
        skip_path=top_candidate.path,
    )

    return LocalizationResult(
        is_uncertain=False,
        primary_file=top_candidate.path,
        primary_symbol=primary_sym.name if primary_sym else None,
        primary_symbol_kind=primary_sym.kind if primary_sym else None,
        line_range=line_range,
        evidence_score=round(top_score, 4),
        evidence_confidence=top_evidence.confidence.value,
        explanation=explanation,
        related=related,
    )
