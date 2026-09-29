"""Tests for the primary localization engine.

All deterministic — no network, no LLM, no filesystem I/O.
Inputs are constructed inline using model classes.
"""
from __future__ import annotations

import pytest

from app.analyzers.localization import (
    _normalise_score,
    _pick_primary_symbol,
    _build_explanation,
    localize,
)
from app.models.candidates import CandidateFile, CandidateSet, MatchKind, MatchReason
from app.models.evidence import (
    CallerEvidence,
    Confidence,
    EvidenceReport,
    EvidenceSet,
    ImportEvidence,
    SymbolEvidence,
    TestEvidence,
)
from app.models.issue_signals import IssueSignals
from app.models.localization import LocalizationResult, UncertaintyReason


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _sym_ev(
    name: str,
    matched_signal: str = "function_name",
    matched_term: str = "",
    start: int = 10,
    end: int = 20,
    kind: str = "function",
    parent: str | None = None,
) -> SymbolEvidence:
    return SymbolEvidence(
        name=name,
        kind=kind,
        start_line=start,
        end_line=end,
        parent=parent,
        matched_signal=matched_signal,
        matched_term=matched_term or name,
    )


def _ev_report(
    path: str,
    confidence: Confidence = Confidence.moderate,
    symbols: list[SymbolEvidence] | None = None,
    callers: list[CallerEvidence] | None = None,
    tests: list[TestEvidence] | None = None,
    imports: list[ImportEvidence] | None = None,
    term_hits: int = 0,
    weak_reason: str | None = None,
) -> EvidenceReport:
    return EvidenceReport(
        path=path,
        confidence=confidence,
        matched_symbols=symbols or [],
        callers=callers or [],
        related_tests=tests or [],
        relevant_imports=imports or [],
        issue_term_hits=term_hits,
        weak_reason=weak_reason,
    )


def _candidate(
    path: str,
    score: float,
    kind: MatchKind = MatchKind.symbol_name,
    detail: str = "matched",
    symbols: list[str] | None = None,
    is_test: bool = False,
) -> CandidateFile:
    return CandidateFile(
        path=path,
        score=score,
        reasons=[MatchReason(kind=kind, detail=detail, weight=score)],
        is_test=is_test,
        matched_symbols=symbols or [],
    )


def _signals(**kw) -> IssueSignals:
    defaults = dict(
        title="bug report",
        keywords=[],
        error_messages=[],
        file_references=[],
        function_names=[],
        class_names=[],
        expected_behavior=None,
        observed_behavior=None,
    )
    defaults.update(kw)
    return IssueSignals(**defaults)


def _make_inputs(
    candidates: list[CandidateFile],
    reports: list[EvidenceReport],
    **signal_kw,
) -> tuple[CandidateSet, EvidenceSet, IssueSignals]:
    cs = CandidateSet(
        candidates=candidates,
        total_files_considered=len(candidates),
        signals_used=["function_names"],
    )
    ev = EvidenceSet(reports=reports, candidates_evaluated=len(reports))
    sig = _signals(**signal_kw)
    return cs, ev, sig


# ---------------------------------------------------------------------------
# _normalise_score
# ---------------------------------------------------------------------------

class TestNormaliseScore:
    def test_zero_score_is_zero(self):
        assert _normalise_score(0.0, Confidence.weak) == 0.0

    def test_strong_confidence_boosts(self):
        base = _normalise_score(10.0, Confidence.weak)
        boosted = _normalise_score(10.0, Confidence.strong)
        assert boosted > base

    def test_capped_at_one(self):
        assert _normalise_score(1000.0, Confidence.strong) == 1.0

    def test_moderate_between_weak_and_strong(self):
        w = _normalise_score(10.0, Confidence.weak)
        m = _normalise_score(10.0, Confidence.moderate)
        s = _normalise_score(10.0, Confidence.strong)
        assert w <= m <= s


# ---------------------------------------------------------------------------
# _pick_primary_symbol
# ---------------------------------------------------------------------------

class TestPickPrimarySymbol:
    def test_returns_none_when_no_symbols(self):
        report = _ev_report("app/auth.py", symbols=[])
        assert _pick_primary_symbol(report) is None

    def test_function_name_beats_keyword(self):
        fn = _sym_ev("parse_request", matched_signal="function_name", start=5)
        kw = _sym_ev("tokenize", matched_signal="keyword", start=1)
        report = _ev_report("app/auth.py", symbols=[kw, fn])
        assert _pick_primary_symbol(report).name == "parse_request"

    def test_class_name_beats_keyword(self):
        cls = _sym_ev("AuthMiddleware", matched_signal="class_name", start=10)
        kw = _sym_ev("helper", matched_signal="keyword", start=1)
        report = _ev_report("app/auth.py", symbols=[kw, cls])
        assert _pick_primary_symbol(report).name == "AuthMiddleware"

    def test_function_name_beats_class_name(self):
        fn = _sym_ev("parse_request", matched_signal="function_name", start=20)
        cls = _sym_ev("MyClass", matched_signal="class_name", start=1)
        report = _ev_report("app/auth.py", symbols=[cls, fn])
        assert _pick_primary_symbol(report).name == "parse_request"

    def test_error_type_beats_keyword(self):
        err = _sym_ev("DatabaseError", matched_signal="error_type", start=30)
        kw = _sym_ev("helper", matched_signal="keyword", start=1)
        report = _ev_report("app/auth.py", symbols=[kw, err])
        assert _pick_primary_symbol(report).name == "DatabaseError"

    def test_tie_broken_by_earliest_line(self):
        a = _sym_ev("alpha", matched_signal="function_name", start=50)
        b = _sym_ev("beta", matched_signal="function_name", start=10)
        report = _ev_report("app/auth.py", symbols=[a, b])
        assert _pick_primary_symbol(report).name == "beta"

    def test_single_symbol_returned(self):
        sym = _sym_ev("login", matched_signal="function_name", start=5)
        report = _ev_report("app/auth.py", symbols=[sym])
        assert _pick_primary_symbol(report).name == "login"


# ---------------------------------------------------------------------------
# Uncertainty: no candidates
# ---------------------------------------------------------------------------

class TestUncertaintyNoCandidates:
    def test_no_candidates_returns_uncertain(self):
        cs = CandidateSet(candidates=[], total_files_considered=0, signals_used=[])
        ev = EvidenceSet(reports=[], candidates_evaluated=0)
        result = localize(cs, ev, _signals())
        assert result.is_uncertain is True

    def test_uncertainty_reason_is_no_candidates(self):
        cs = CandidateSet(candidates=[], total_files_considered=0, signals_used=[])
        ev = EvidenceSet(reports=[], candidates_evaluated=0)
        result = localize(cs, ev, _signals())
        assert result.uncertainty_reason == UncertaintyReason.no_candidates

    def test_primary_file_is_none(self):
        cs = CandidateSet(candidates=[], total_files_considered=0, signals_used=[])
        ev = EvidenceSet(reports=[], candidates_evaluated=0)
        result = localize(cs, ev, _signals())
        assert result.primary_file is None

    def test_uncertainty_detail_is_non_empty(self):
        cs = CandidateSet(candidates=[], total_files_considered=0, signals_used=[])
        ev = EvidenceSet(reports=[], candidates_evaluated=0)
        result = localize(cs, ev, _signals())
        assert result.uncertainty_detail
        assert len(result.uncertainty_detail) > 10


# ---------------------------------------------------------------------------
# Uncertainty: all weak
# ---------------------------------------------------------------------------

class TestUncertaintyAllWeak:
    def test_all_weak_with_tiny_scores_is_uncertain(self):
        # Very small score → normalized below threshold
        c = _candidate("app/auth.py", score=0.1)
        r = _ev_report("app/auth.py", confidence=Confidence.weak,
                        weak_reason="no match")
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert result.is_uncertain is True
        assert result.uncertainty_reason == UncertaintyReason.all_weak

    def test_all_weak_but_high_score_is_not_uncertain(self):
        # High raw score → normalizes above threshold even at weak confidence
        c = _candidate("app/auth.py", score=8.0)
        r = _ev_report("app/auth.py", confidence=Confidence.weak,
                        weak_reason="structural")
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        # With a sufficiently high score the engine should commit
        assert result.is_uncertain is False or result.uncertainty_reason != UncertaintyReason.all_weak


# ---------------------------------------------------------------------------
# Uncertainty: ambiguous top-2
# ---------------------------------------------------------------------------

class TestUncertaintyAmbiguous:
    def test_close_scores_moderate_confidence_is_ambiguous(self):
        c1 = _candidate("app/auth.py", score=5.0)
        c2 = _candidate("app/utils.py", score=4.9)  # < 15% gap
        r1 = _ev_report("app/auth.py", confidence=Confidence.moderate)
        r2 = _ev_report("app/utils.py", confidence=Confidence.moderate)
        cs, ev, sig = _make_inputs([c1, c2], [r1, r2])
        result = localize(cs, ev, sig)
        assert result.is_uncertain is True
        assert result.uncertainty_reason == UncertaintyReason.ambiguous

    def test_one_strong_candidate_resolves_ambiguity(self):
        c1 = _candidate("app/auth.py", score=5.0)
        c2 = _candidate("app/utils.py", score=4.9)
        r1 = _ev_report("app/auth.py", confidence=Confidence.strong,
                         symbols=[_sym_ev("login")])
        r2 = _ev_report("app/utils.py", confidence=Confidence.moderate)
        cs, ev, sig = _make_inputs([c1, c2], [r1, r2])
        result = localize(cs, ev, sig)
        assert result.is_uncertain is False
        assert result.primary_file == "app/auth.py"

    def test_large_score_gap_resolves_ambiguity(self):
        c1 = _candidate("app/auth.py", score=10.0)
        c2 = _candidate("app/utils.py", score=4.0)  # > 15% gap
        r1 = _ev_report("app/auth.py", confidence=Confidence.moderate,
                         symbols=[_sym_ev("login")])
        r2 = _ev_report("app/utils.py", confidence=Confidence.moderate)
        cs, ev, sig = _make_inputs([c1, c2], [r1, r2])
        result = localize(cs, ev, sig)
        assert result.is_uncertain is False

    def test_ambiguous_result_has_related_locations(self):
        c1 = _candidate("app/auth.py", score=5.0)
        c2 = _candidate("app/utils.py", score=4.9)
        r1 = _ev_report("app/auth.py", confidence=Confidence.moderate)
        r2 = _ev_report("app/utils.py", confidence=Confidence.moderate)
        cs, ev, sig = _make_inputs([c1, c2], [r1, r2])
        result = localize(cs, ev, sig)
        assert result.related  # both candidates still surfaced


# ---------------------------------------------------------------------------
# Successful localization
# ---------------------------------------------------------------------------

class TestSuccessfulLocalization:
    def test_returns_non_uncertain_result(self):
        c = _candidate("app/auth.py", score=10.0,
                        symbols=["authenticate_user"])
        r = _ev_report("app/auth.py", confidence=Confidence.strong,
                        symbols=[_sym_ev("authenticate_user")])
        cs, ev, sig = _make_inputs([c], [r],
                                    function_names=["authenticate_user"])
        result = localize(cs, ev, sig)
        assert result.is_uncertain is False

    def test_primary_file_set(self):
        c = _candidate("app/auth.py", score=10.0)
        r = _ev_report("app/auth.py", confidence=Confidence.strong,
                        symbols=[_sym_ev("login_user")])
        cs, ev, sig = _make_inputs([c], [r],
                                    function_names=["login_user"])
        result = localize(cs, ev, sig)
        assert result.primary_file == "app/auth.py"

    def test_primary_symbol_from_evidence(self):
        sym = _sym_ev("authenticate_user", start=42, end=58)
        c = _candidate("app/auth.py", score=12.0, symbols=["authenticate_user"])
        r = _ev_report("app/auth.py", confidence=Confidence.strong, symbols=[sym])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert result.primary_symbol == "authenticate_user"
        assert result.primary_symbol_kind == "function"

    def test_line_range_from_symbol_evidence(self):
        sym = _sym_ev("parse_request", start=17, end=35)
        c = _candidate("app/handlers.py", score=12.0, symbols=["parse_request"])
        r = _ev_report("app/handlers.py", confidence=Confidence.strong, symbols=[sym])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert result.line_range is not None
        assert result.line_range.start == 17
        assert result.line_range.end == 35

    def test_no_line_range_when_no_symbol(self):
        c = _candidate("app/config.py", score=10.0)
        r = _ev_report("app/config.py", confidence=Confidence.strong,
                        symbols=[])  # keyword match in path only
        cs, ev, sig = _make_inputs([c], [r], keywords=["config"])
        result = localize(cs, ev, sig)
        assert result.line_range is None
        assert result.primary_symbol is None

    def test_evidence_score_between_0_and_1(self):
        c = _candidate("app/auth.py", score=10.0)
        r = _ev_report("app/auth.py", confidence=Confidence.strong,
                        symbols=[_sym_ev("login")])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert 0.0 <= result.evidence_score <= 1.0

    def test_evidence_confidence_matches_report(self):
        c = _candidate("app/auth.py", score=10.0)
        r = _ev_report("app/auth.py", confidence=Confidence.strong,
                        symbols=[_sym_ev("login")])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert result.evidence_confidence == "strong"

    def test_explanation_is_non_empty(self):
        c = _candidate("app/auth.py", score=10.0)
        r = _ev_report("app/auth.py", confidence=Confidence.moderate,
                        symbols=[_sym_ev("login")])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert result.explanation
        assert len(result.explanation) > 10

    def test_explanation_contains_symbol_name(self):
        sym = _sym_ev("parse_request", matched_signal="function_name",
                       matched_term="parse_request")
        c = _candidate("app/handlers.py", score=12.0, symbols=["parse_request"])
        r = _ev_report("app/handlers.py", confidence=Confidence.strong, symbols=[sym])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert "parse_request" in result.explanation

    def test_explanation_contains_line_number(self):
        sym = _sym_ev("login", start=42, end=55)
        c = _candidate("app/auth.py", score=10.0)
        r = _ev_report("app/auth.py", confidence=Confidence.strong, symbols=[sym])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert "42" in result.explanation

    def test_explanation_mentions_tests(self):
        sym = _sym_ev("login", start=10, end=20)
        c = _candidate("app/auth.py", score=10.0)
        r = _ev_report(
            "app/auth.py",
            confidence=Confidence.strong,
            symbols=[sym],
            tests=[TestEvidence(path="tests/test_auth.py")],
        )
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert "test_auth" in result.explanation


# ---------------------------------------------------------------------------
# Related locations
# ---------------------------------------------------------------------------

class TestRelatedLocations:
    def test_related_excludes_primary(self):
        c1 = _candidate("app/auth.py", score=10.0)
        c2 = _candidate("app/utils.py", score=5.0)
        r1 = _ev_report("app/auth.py", confidence=Confidence.strong,
                         symbols=[_sym_ev("login")])
        r2 = _ev_report("app/utils.py", confidence=Confidence.moderate)
        cs, ev, sig = _make_inputs([c1, c2], [r1, r2])
        result = localize(cs, ev, sig)
        related_paths = [r.path for r in result.related]
        assert "app/auth.py" not in related_paths
        assert "app/utils.py" in related_paths

    def test_related_has_reason(self):
        c1 = _candidate("app/auth.py", score=10.0)
        c2 = _candidate("app/utils.py", score=5.0, detail="imports candidate")
        r1 = _ev_report("app/auth.py", confidence=Confidence.strong,
                         symbols=[_sym_ev("login")])
        r2 = _ev_report("app/utils.py", confidence=Confidence.moderate)
        cs, ev, sig = _make_inputs([c1, c2], [r1, r2])
        result = localize(cs, ev, sig)
        assert result.related[0].reason

    def test_related_order_by_score(self):
        c1 = _candidate("app/primary.py", score=15.0)
        c2 = _candidate("app/high.py", score=8.0)
        c3 = _candidate("app/low.py", score=3.0)
        r1 = _ev_report("app/primary.py", confidence=Confidence.strong,
                         symbols=[_sym_ev("main_func")])
        r2 = _ev_report("app/high.py", confidence=Confidence.moderate)
        r3 = _ev_report("app/low.py", confidence=Confidence.weak,
                         weak_reason="path match")
        cs, ev, sig = _make_inputs([c1, c2, c3], [r1, r2, r3])
        result = localize(cs, ev, sig)
        scores = [r.score for r in result.related]
        assert scores == sorted(scores, reverse=True)

    def test_related_symbol_populated_from_evidence(self):
        c1 = _candidate("app/auth.py", score=12.0)
        c2 = _candidate("app/utils.py", score=6.0)
        r1 = _ev_report("app/auth.py", confidence=Confidence.strong,
                         symbols=[_sym_ev("login")])
        r2 = _ev_report("app/utils.py", confidence=Confidence.moderate,
                         symbols=[_sym_ev("parse_token")])
        cs, ev, sig = _make_inputs([c1, c2], [r1, r2])
        result = localize(cs, ev, sig)
        related_utils = next(r for r in result.related if r.path == "app/utils.py")
        assert related_utils.symbol == "parse_token"

    def test_single_candidate_has_empty_related(self):
        c = _candidate("app/auth.py", score=10.0)
        r = _ev_report("app/auth.py", confidence=Confidence.strong,
                        symbols=[_sym_ev("login")])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert result.related == []


# ---------------------------------------------------------------------------
# Uncertainty state fields when uncertain
# ---------------------------------------------------------------------------

class TestUncertainStateFields:
    def test_all_primary_fields_none_when_uncertain(self):
        cs = CandidateSet(candidates=[], total_files_considered=0, signals_used=[])
        ev = EvidenceSet(reports=[], candidates_evaluated=0)
        result = localize(cs, ev, _signals())
        assert result.primary_file is None
        assert result.primary_symbol is None
        assert result.primary_symbol_kind is None
        assert result.line_range is None

    def test_uncertain_has_zero_evidence_score(self):
        cs = CandidateSet(candidates=[], total_files_considered=0, signals_used=[])
        ev = EvidenceSet(reports=[], candidates_evaluated=0)
        result = localize(cs, ev, _signals())
        assert result.evidence_score == 0.0


# ---------------------------------------------------------------------------
# Method symbol with parent class
# ---------------------------------------------------------------------------

class TestMethodWithParent:
    def test_method_explanation_includes_parent(self):
        sym = _sym_ev(
            "authenticate_request",
            matched_signal="function_name",
            kind="method",
            parent="AuthMiddleware",
            start=30, end=50,
        )
        c = _candidate("app/middleware.py", score=14.0)
        r = _ev_report("app/middleware.py", confidence=Confidence.strong, symbols=[sym])
        cs, ev, sig = _make_inputs([c], [r])
        result = localize(cs, ev, sig)
        assert "AuthMiddleware" in result.explanation
        assert "authenticate_request" in result.explanation


# ---------------------------------------------------------------------------
# Realistic end-to-end scenario
# ---------------------------------------------------------------------------

class TestRealisticScenario:
    """Flask-style auth service."""

    def _run(self) -> LocalizationResult:
        from app.models.evidence import ImportEvidence

        # Candidates (already ranked by candidate_search)
        c_middleware = _candidate(
            "app/auth/middleware.py", score=14.0,
            kind=MatchKind.symbol_name,
            detail="symbol(s) ['AuthMiddleware', 'authenticate_request'] match issue",
            symbols=["AuthMiddleware", "authenticate_request"],
        )
        c_utils = _candidate(
            "app/auth/utils.py", score=8.0,
            kind=MatchKind.symbol_name,
            detail="symbol(s) ['parse_token'] match issue",
            symbols=["parse_token"],
        )
        c_exceptions = _candidate(
            "app/exceptions.py", score=3.5,
            kind=MatchKind.error_type,
            detail="class(es) ['AuthenticationError'] match error type",
        )

        # Evidence reports
        r_middleware = _ev_report(
            "app/auth/middleware.py",
            confidence=Confidence.strong,
            symbols=[
                _sym_ev("AuthMiddleware", matched_signal="class_name",
                         kind="class", start=10, end=80),
                _sym_ev("authenticate_request", matched_signal="function_name",
                         kind="method", parent="AuthMiddleware",
                         start=25, end=45),
            ],
            callers=[CallerEvidence(path="app/views/api.py")],
            tests=[TestEvidence(
                path="tests/test_auth.py",
                test_symbol_names=["test_authenticate", "test_invalid_token"],
            )],
            imports=[ImportEvidence(
                module="app.auth.utils", line=3, matched_term="auth"
            )],
            term_hits=4,
        )
        r_utils = _ev_report(
            "app/auth/utils.py",
            confidence=Confidence.moderate,
            symbols=[_sym_ev("parse_token", start=5, end=15)],
            term_hits=2,
        )
        r_exceptions = _ev_report(
            "app/exceptions.py",
            confidence=Confidence.moderate,
            symbols=[_sym_ev("AuthenticationError", matched_signal="error_type",
                              kind="class", start=8, end=10)],
        )

        cs = CandidateSet(
            candidates=[c_middleware, c_utils, c_exceptions],
            total_files_considered=10,
            signals_used=["function_names", "class_names", "error_messages"],
        )
        ev = EvidenceSet(
            reports=[r_middleware, r_utils, r_exceptions],
            candidates_evaluated=3,
        )
        sig = _signals(
            function_names=["authenticate_request", "parse_token"],
            class_names=["AuthMiddleware"],
            error_messages=["AuthenticationError: invalid token"],
        )
        return localize(cs, ev, sig)

    def test_not_uncertain(self):
        assert self._run().is_uncertain is False

    def test_primary_file_is_middleware(self):
        assert self._run().primary_file == "app/auth/middleware.py"

    def test_primary_symbol_is_function_over_class(self):
        # authenticate_request (function_name priority=4) > AuthMiddleware (class_name=3)
        result = self._run()
        assert result.primary_symbol == "authenticate_request"

    def test_line_range_matches_symbol(self):
        result = self._run()
        assert result.line_range is not None
        assert result.line_range.start == 25
        assert result.line_range.end == 45

    def test_evidence_score_above_zero(self):
        assert self._run().evidence_score > 0.0

    def test_evidence_confidence_strong(self):
        assert self._run().evidence_confidence == "strong"

    def test_explanation_mentions_symbol_and_line(self):
        result = self._run()
        assert "authenticate_request" in result.explanation
        assert "25" in result.explanation

    def test_related_contains_utils_and_exceptions(self):
        result = self._run()
        related_paths = {r.path for r in result.related}
        assert "app/auth/utils.py" in related_paths
        assert "app/exceptions.py" in related_paths

    def test_related_does_not_contain_primary(self):
        result = self._run()
        related_paths = [r.path for r in result.related]
        assert "app/auth/middleware.py" not in related_paths
