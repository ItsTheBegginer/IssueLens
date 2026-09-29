"""Regression tests for Task 14 localization accuracy improvements.

Verifies fixes discovered during Task 13 real-repository testing:
1. File reference exactness (e.g. `utils.py` must not match `tests/test_utils.py`).
2. Preference for implementation files over test files in candidate ranking and disambiguation.
3. Priority for multi-line implementation methods over 1-line type stubs.
4. Keyword stemming in path matching (e.g. `session` matches `sessions.py`).
5. Matching single-word class names and capitalized keywords against class symbols.
"""
from __future__ import annotations

import pytest

from app.analyzers.issue_signals import extract_signals
from app.analyzers.localization import _pick_primary_symbol, localize
from app.models.candidates import CandidateFile, CandidateSet, MatchKind, MatchReason
from app.models.evidence import (
    Confidence,
    EvidenceReport,
    EvidenceSet,
    SymbolEvidence,
    TestEvidence,
)
from app.models.github import IssueDetail
from app.models.intelligence import (
    FileSymbols,
    RepoIntelligence,
    Symbol,
    SymbolType,
)
from app.models.issue_signals import IssueSignals
from app.search.candidate_search import (
    _file_ref_overlap,
    _path_keyword_matches,
    _symbol_matches,
    search_candidates,
)


def _issue(title: str, body: str = "") -> IssueDetail:
    return IssueDetail(
        number=1,
        title=title,
        body=body,
        state="open",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        html_url="https://github.com/o/r/issues/1",
        comments=0,
    )


# ---------------------------------------------------------------------------
# Test 1: File Reference Exactness
# ---------------------------------------------------------------------------

def test_file_ref_does_not_match_test_prefix():
    """`utils.py` must match `src/utils.py` or `requests/utils.py` but NOT `test_utils.py`."""
    refs = ["utils.py"]
    assert _file_ref_overlap("src/requests/utils.py", refs) == "utils.py"
    assert _file_ref_overlap("requests/utils.py", refs) == "utils.py"
    assert _file_ref_overlap("tests/test_utils.py", refs) is None
    assert _file_ref_overlap("src/internal_utils.py", refs) is None


# ---------------------------------------------------------------------------
# Test 2: Keyword Stemming
# ---------------------------------------------------------------------------

def test_keyword_stemming():
    """Keyword `session` should match path `requests/sessions.py`."""
    keywords = {"session", "request"}
    hits = _path_keyword_matches("requests/sessions.py", keywords)
    assert "session" in hits
    assert "request" in hits


# ---------------------------------------------------------------------------
# Test 3: Capitalized Keyword / Class Symbol Matching
# ---------------------------------------------------------------------------

def test_single_word_class_signal_and_symbol_match():
    """Issue mentioning `Request` extracts Request and matches class Request."""
    iss = _issue("Bug in `Request` populate_request behavior")
    signals = extract_signals(iss)
    assert "Request" in signals.class_names or "Request" in signals.keywords

    fs = FileSymbols(
        path="src/wrappers/request.py",
        symbols=[
            Symbol(name="Request", type=SymbolType.class_, start_line=10, end_line=100),
            Symbol(name="populate_request", type=SymbolType.method, start_line=20, end_line=30),
        ],
    )
    hits = _symbol_matches(
        fs,
        func_names_lower={"populate_request"},
        class_names_lower={"request"},
        keywords_lower={"request"},
    )
    assert "Request" in hits
    assert "populate_request" in hits


# ---------------------------------------------------------------------------
# Test 4: Implementation vs Test Disambiguation
# ---------------------------------------------------------------------------

def test_implementation_preferred_over_test_file():
    """When implementation and test files have equal scores, implementation wins."""
    c_impl = CandidateFile(
        path="src/requests/utils.py",
        score=11.5,
        reasons=[MatchReason(kind=MatchKind.file_reference, detail="ref", weight=11.5)],
        is_test=False,
    )
    c_test = CandidateFile(
        path="tests/test_utils.py",
        score=11.5,
        reasons=[MatchReason(kind=MatchKind.file_reference, detail="ref", weight=11.5)],
        is_test=True,
    )
    candidates = CandidateSet(
        candidates=[c_impl, c_test],
        total_files_considered=2,
        signals_used=["file_references"],
    )

    ev_impl = EvidenceReport(
        path="src/requests/utils.py",
        confidence=Confidence.moderate,
        matched_symbols=[],
    )
    ev_test = EvidenceReport(
        path="tests/test_utils.py",
        confidence=Confidence.moderate,
        matched_symbols=[],
    )
    evidence_set = EvidenceSet(reports=[ev_impl, ev_test], candidates_evaluated=2)

    signals = IssueSignals(title="utils.py bug", file_references=["utils.py"])
    res = localize(candidates, evidence_set, signals)

    # Must NOT be uncertain due to ambiguity between code and its test!
    assert res.is_uncertain is False
    assert res.primary_file == "src/requests/utils.py"


# ---------------------------------------------------------------------------
# Test 5: Real Implementation Preferred Over Type Stub
# ---------------------------------------------------------------------------

def test_pick_primary_symbol_prefers_real_implementation_over_stub():
    """When both stub and implementation share the same name, pick implementation."""
    stub = SymbolEvidence(
        name="send",
        kind="method",
        start_line=132,
        end_line=132,  # 1-line stub
        parent="SessionRedirectMixin",
        matched_signal="function_name",
        matched_term="send",
    )
    impl = SymbolEvidence(
        name="send",
        kind="method",
        start_line=752,
        end_line=810,  # real implementation
        parent="Session",
        matched_signal="function_name",
        matched_term="send",
    )
    report = EvidenceReport(
        path="src/requests/sessions.py",
        confidence=Confidence.strong,
        matched_symbols=[stub, impl],
    )
    best = _pick_primary_symbol(report)
    assert best is not None
    assert best.start_line == 752
    assert best.parent == "Session"
