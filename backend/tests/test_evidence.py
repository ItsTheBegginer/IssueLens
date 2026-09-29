"""Tests for the evidence engine.

All deterministic — no network, no LLM, no filesystem I/O.
Repository indexes constructed inline.
"""
from __future__ import annotations

import pytest

from app.analyzers.evidence import (
    collect_evidence,
    collect_evidence_for_file,
    _collect_symbol_evidence,
    _collect_import_evidence,
    _collect_callers,
    _collect_related_tests,
    _count_issue_term_hits,
    _compute_confidence,
    _build_callers_index,
    _build_fs_index,
    _error_class_names,
)
from app.models.candidates import CandidateFile, MatchKind, MatchReason
from app.models.evidence import Confidence, EvidenceReport, EvidenceSet
from app.models.intelligence import (
    FileSymbols,
    ImportRecord,
    ImportedName,
    RepoIntelligence,
    Symbol,
    SymbolType,
)
from app.models.issue_signals import IssueSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intel(files: list[FileSymbols]) -> RepoIntelligence:
    return RepoIntelligence(
        repo_full_name="owner/repo",
        file_count=len(files),
        analyzed_count=len(files),
        skipped_count=0,
        files=files,
        relationships=[],
    )


def _sym(name: str, kind: SymbolType, line: int = 1, parent: str | None = None) -> Symbol:
    return Symbol(name=name, type=kind, start_line=line, end_line=line + 4, parent=parent)


def _imp(module: str, line: int = 1) -> ImportRecord:
    return ImportRecord(module=module, names=[], line=line)


def _fs(
    path: str,
    *,
    symbols: list[Symbol] | None = None,
    imports: list[ImportRecord] | None = None,
    is_test: bool = False,
) -> FileSymbols:
    return FileSymbols(
        path=path, symbols=symbols or [], imports=imports or [], is_test=is_test
    )


def _candidate(path: str, score: float = 5.0) -> CandidateFile:
    return CandidateFile(
        path=path,
        score=score,
        reasons=[MatchReason(kind=MatchKind.symbol_name, detail="test", weight=score)],
    )


def _signals(**kw) -> IssueSignals:
    defaults = dict(
        title="test issue",
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


# ---------------------------------------------------------------------------
# _error_class_names helper
# ---------------------------------------------------------------------------

class TestErrorClassNames:
    def test_extracts_from_error_messages(self):
        sigs = _signals(error_messages=["AttributeError: 'NoneType' has no attr"])
        names = _error_class_names(sigs)
        assert "attributeerror" in names

    def test_extracts_multiple_types(self):
        sigs = _signals(error_messages=[
            "ValueError: bad input",
            "KeyError: missing key",
        ])
        names = _error_class_names(sigs)
        assert "valueerror" in names
        assert "keyerror" in names

    def test_includes_class_names_with_error_suffix(self):
        sigs = _signals(class_names=["AuthenticationError"])
        names = _error_class_names(sigs)
        assert "authenticationerror" in names

    def test_non_error_class_not_included(self):
        sigs = _signals(class_names=["UserModel"])
        names = _error_class_names(sigs)
        assert "usermodel" not in names


# ---------------------------------------------------------------------------
# _collect_symbol_evidence
# ---------------------------------------------------------------------------

class TestCollectSymbolEvidence:
    def test_function_name_match(self):
        file_syms = _fs("app/auth.py", symbols=[
            _sym("parse_request", SymbolType.function),
            _sym("unrelated", SymbolType.function),
        ])
        sigs = _signals(function_names=["parse_request"])
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert len(evidence) == 1
        assert evidence[0].name == "parse_request"
        assert evidence[0].matched_signal == "function_name"
        assert evidence[0].matched_term == "parse_request"

    def test_class_name_match(self):
        file_syms = _fs("app/middleware.py", symbols=[
            _sym("AuthMiddleware", SymbolType.class_),
        ])
        sigs = _signals(class_names=["AuthMiddleware"])
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert len(evidence) == 1
        assert evidence[0].matched_signal == "class_name"

    def test_error_type_match(self):
        file_syms = _fs("app/exceptions.py", symbols=[
            _sym("DatabaseError", SymbolType.class_),
        ])
        sigs = _signals(error_messages=["DatabaseError: connection failed"])
        error_names = _error_class_names(sigs)
        evidence = _collect_symbol_evidence(file_syms, sigs, error_names)
        assert len(evidence) == 1
        assert evidence[0].matched_signal == "error_type"

    def test_method_match(self):
        file_syms = _fs("app/service.py", symbols=[
            _sym("handle_request", SymbolType.method, parent="MyService"),
        ])
        sigs = _signals(function_names=["handle_request"])
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert evidence[0].parent == "MyService"
        assert evidence[0].kind == "method"

    def test_keyword_match_on_symbol(self):
        file_syms = _fs("app/parser.py", symbols=[
            _sym("tokenizer", SymbolType.function),
        ])
        sigs = _signals(keywords=["tokenizer"])
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert len(evidence) == 1
        assert evidence[0].matched_signal == "keyword"

    def test_no_match_returns_empty(self):
        file_syms = _fs("app/views.py", symbols=[
            _sym("index", SymbolType.function),
        ])
        sigs = _signals(function_names=["parse_request"])
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert evidence == []

    def test_line_numbers_preserved(self):
        sym = Symbol(name="login", type=SymbolType.function,
                     start_line=42, end_line=58)
        file_syms = _fs("app/auth.py", symbols=[sym])
        sigs = _signals(function_names=["login"])
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert evidence[0].start_line == 42
        assert evidence[0].end_line == 58

    def test_case_insensitive_match(self):
        file_syms = _fs("app/auth.py", symbols=[
            _sym("HandleRequest", SymbolType.function),
        ])
        sigs = _signals(function_names=["handlerequest"])
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert len(evidence) == 1

    def test_short_keywords_skipped(self):
        file_syms = _fs("app/io.py", symbols=[_sym("io", SymbolType.function)])
        sigs = _signals(keywords=["io"])   # 2 chars, below threshold
        evidence = _collect_symbol_evidence(file_syms, sigs, set())
        assert evidence == []


# ---------------------------------------------------------------------------
# _collect_import_evidence
# ---------------------------------------------------------------------------

class TestCollectImportEvidence:
    def test_import_matching_function_name(self):
        file_syms = _fs("app/views.py", imports=[
            _imp("app.auth.utils", line=3),
        ])
        sigs = _signals(function_names=["auth"])
        evidence = _collect_import_evidence(file_syms, sigs)
        assert len(evidence) == 1
        assert evidence[0].module == "app.auth.utils"
        assert evidence[0].line == 3

    def test_import_matching_keyword(self):
        file_syms = _fs("app/views.py", imports=[
            _imp("app.serializer", line=5),
        ])
        sigs = _signals(keywords=["serializer"])
        evidence = _collect_import_evidence(file_syms, sigs)
        assert any(e.module == "app.serializer" for e in evidence)

    def test_no_match_returns_empty(self):
        file_syms = _fs("app/views.py", imports=[_imp("os")])
        sigs = _signals(keywords=["auth"])
        evidence = _collect_import_evidence(file_syms, sigs)
        assert evidence == []

    def test_duplicate_imports_not_doubled(self):
        file_syms = _fs("app/views.py", imports=[
            _imp("app.auth", line=1),
            _imp("app.auth.utils", line=2),
        ])
        sigs = _signals(keywords=["auth"])
        evidence = _collect_import_evidence(file_syms, sigs)
        modules = [e.module for e in evidence]
        assert len(modules) == len(set(modules))


# ---------------------------------------------------------------------------
# _collect_callers
# ---------------------------------------------------------------------------

class TestCollectCallers:
    def test_finds_files_that_import_this_module(self):
        auth_fs = _fs("app/auth.py")
        views_fs = _fs("app/views.py", imports=[_imp("app.auth")])
        intel = _make_intel([auth_fs, views_fs])
        index = _build_callers_index(intel)
        callers = _collect_callers(auth_fs, index)
        assert any(c.path == "app/views.py" for c in callers)

    def test_self_import_not_included(self):
        auth_fs = _fs("app/auth.py", imports=[_imp("app.auth")])
        intel = _make_intel([auth_fs])
        index = _build_callers_index(intel)
        callers = _collect_callers(auth_fs, index)
        assert not any(c.path == "app/auth.py" for c in callers)

    def test_no_callers_returns_empty(self):
        fs = _fs("app/standalone.py")
        intel = _make_intel([fs])
        index = _build_callers_index(intel)
        callers = _collect_callers(fs, index)
        assert callers == []

    def test_test_file_caller_flagged(self):
        auth_fs = _fs("app/auth.py")
        test_fs = _fs("tests/test_auth.py", imports=[_imp("app.auth")], is_test=True)
        intel = _make_intel([auth_fs, test_fs])
        index = _build_callers_index(intel)
        callers = _collect_callers(auth_fs, index)
        test_callers = [c for c in callers if c.is_test]
        assert any(c.path == "tests/test_auth.py" for c in test_callers)


# ---------------------------------------------------------------------------
# _collect_related_tests
# ---------------------------------------------------------------------------

class TestCollectRelatedTests:
    def test_finds_test_that_imports_module(self):
        auth_fs = _fs("app/auth.py")
        test_fs = _fs("tests/test_auth.py",
                      symbols=[_sym("test_login", SymbolType.function)],
                      imports=[_imp("app.auth")],
                      is_test=True)
        intel = _make_intel([auth_fs, test_fs])
        index = _build_callers_index(intel)
        fs_idx = _build_fs_index(intel)
        tests = _collect_related_tests(auth_fs, index, fs_idx)
        assert len(tests) == 1
        assert tests[0].path == "tests/test_auth.py"
        assert "test_login" in tests[0].test_symbol_names

    def test_non_test_callers_excluded(self):
        auth_fs = _fs("app/auth.py")
        views_fs = _fs("app/views.py", imports=[_imp("app.auth")], is_test=False)
        intel = _make_intel([auth_fs, views_fs])
        index = _build_callers_index(intel)
        fs_idx = _build_fs_index(intel)
        tests = _collect_related_tests(auth_fs, index, fs_idx)
        assert tests == []

    def test_no_tests_returns_empty(self):
        fs = _fs("app/utils.py")
        intel = _make_intel([fs])
        index = _build_callers_index(intel)
        fs_idx = _build_fs_index(intel)
        tests = _collect_related_tests(fs, index, fs_idx)
        assert tests == []


# ---------------------------------------------------------------------------
# _count_issue_term_hits
# ---------------------------------------------------------------------------

class TestCountIssueTermHits:
    def test_path_token_match(self):
        fs = _fs("app/auth/validator.py")
        sigs = _signals(keywords=["auth", "validator"])
        hits = _count_issue_term_hits(fs, sigs)
        assert hits == 2

    def test_symbol_name_match(self):
        fs = _fs("app/utils.py", symbols=[_sym("parse_token", SymbolType.function)])
        sigs = _signals(function_names=["parse_token"])
        hits = _count_issue_term_hits(fs, sigs)
        assert hits >= 1

    def test_no_match_returns_zero(self):
        fs = _fs("app/views.py")
        sigs = _signals(function_names=["authenticate_user"])
        hits = _count_issue_term_hits(fs, sigs)
        assert hits == 0

    def test_short_keywords_not_counted(self):
        fs = _fs("app/io.py")
        sigs = _signals(keywords=["io"])  # 2 chars
        hits = _count_issue_term_hits(fs, sigs)
        assert hits == 0


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_strong_when_three_or_more_buckets(self):
        from app.analyzers.evidence import SymbolEvidence, ImportEvidence, CallerEvidence
        sym = SymbolEvidence(name="f", kind="function", start_line=1, end_line=5,
                              matched_signal="function_name", matched_term="f")
        imp = ImportEvidence(module="app.auth", line=1, matched_term="auth")
        cal = CallerEvidence(path="app/views.py")
        conf, reason = _compute_confidence([sym], [imp], [cal], [], 0)
        assert conf == Confidence.strong
        assert reason is None

    def test_moderate_when_one_bucket(self):
        from app.analyzers.evidence import SymbolEvidence
        sym = SymbolEvidence(name="f", kind="function", start_line=1, end_line=5,
                              matched_signal="function_name", matched_term="f")
        conf, reason = _compute_confidence([sym], [], [], [], 0)
        assert conf == Confidence.moderate
        assert reason is None

    def test_weak_when_no_buckets(self):
        conf, reason = _compute_confidence([], [], [], [], 0)
        assert conf == Confidence.weak
        assert reason is not None
        assert len(reason) > 0

    def test_high_term_hits_count_as_bucket(self):
        conf, reason = _compute_confidence([], [], [], [], issue_term_hits=2)
        assert conf == Confidence.moderate

    def test_weak_reason_mentions_callers(self):
        from app.analyzers.evidence import CallerEvidence
        cal = CallerEvidence(path="app/views.py")
        conf, reason = _compute_confidence([], [], [cal], [], 0)
        assert conf == Confidence.moderate  # callers = 1 bucket → moderate
        # With only callers it's moderate, not weak — so weak_reason is None
        assert reason is None


# ---------------------------------------------------------------------------
# collect_evidence_for_file — integration
# ---------------------------------------------------------------------------

class TestCollectEvidenceForFile:
    def test_returns_evidence_report(self):
        fs = _fs("app/auth.py", symbols=[_sym("login_user", SymbolType.function)])
        intel = _make_intel([fs])
        sigs = _signals(function_names=["login_user"])
        report = collect_evidence_for_file(_candidate("app/auth.py"), intel, sigs)
        assert isinstance(report, EvidenceReport)

    def test_path_matches_candidate(self):
        fs = _fs("app/auth.py")
        intel = _make_intel([fs])
        sigs = _signals()
        report = collect_evidence_for_file(_candidate("app/auth.py"), intel, sigs)
        assert report.path == "app/auth.py"

    def test_missing_file_returns_weak(self):
        intel = _make_intel([])
        sigs = _signals()
        report = collect_evidence_for_file(_candidate("app/ghost.py"), intel, sigs)
        assert report.confidence == Confidence.weak
        assert report.weak_reason is not None

    def test_symbol_evidence_populated(self):
        fs = _fs("app/auth.py", symbols=[
            _sym("authenticate_user", SymbolType.function),
        ])
        intel = _make_intel([fs])
        sigs = _signals(function_names=["authenticate_user"])
        report = collect_evidence_for_file(_candidate("app/auth.py"), intel, sigs)
        assert len(report.matched_symbols) == 1
        assert report.matched_symbols[0].name == "authenticate_user"

    def test_callers_populated(self):
        auth_fs = _fs("app/auth.py")
        views_fs = _fs("app/views.py", imports=[_imp("app.auth")])
        intel = _make_intel([auth_fs, views_fs])
        sigs = _signals()
        report = collect_evidence_for_file(_candidate("app/auth.py"), intel, sigs)
        assert any(c.path == "app/views.py" for c in report.callers)

    def test_related_tests_populated(self):
        auth_fs = _fs("app/auth.py")
        test_fs = _fs("tests/test_auth.py",
                      symbols=[_sym("test_login", SymbolType.function)],
                      imports=[_imp("app.auth")],
                      is_test=True)
        intel = _make_intel([auth_fs, test_fs])
        sigs = _signals()
        report = collect_evidence_for_file(_candidate("app/auth.py"), intel, sigs)
        assert any(t.path == "tests/test_auth.py" for t in report.related_tests)

    def test_import_evidence_populated(self):
        fs = _fs("app/views.py", imports=[_imp("app.auth.utils", line=2)])
        intel = _make_intel([fs])
        sigs = _signals(keywords=["auth"])
        report = collect_evidence_for_file(_candidate("app/views.py"), intel, sigs)
        assert any(e.module == "app.auth.utils" for e in report.relevant_imports)

    def test_weak_confidence_has_reason(self):
        # File exists but no signals match it directly; it has no callers
        fs = _fs("app/utils.py")
        intel = _make_intel([fs])
        sigs = _signals(function_names=["unrelated_function"])
        report = collect_evidence_for_file(_candidate("app/utils.py"), intel, sigs)
        assert report.confidence == Confidence.weak
        assert report.weak_reason is not None

    def test_strong_confidence_no_weak_reason(self):
        fs = _fs("app/auth.py", symbols=[
            _sym("authenticate_user", SymbolType.function),
        ], imports=[
            _imp("app.tokens"),
            _imp("app.models"),
        ])
        views_fs = _fs("app/views.py", imports=[_imp("app.auth")])
        test_fs = _fs("tests/test_auth.py", imports=[_imp("app.auth")], is_test=True)
        intel = _make_intel([fs, views_fs, test_fs])
        sigs = _signals(
            function_names=["authenticate_user"],
            keywords=["tokens", "auth"],
        )
        report = collect_evidence_for_file(_candidate("app/auth.py"), intel, sigs)
        assert report.confidence == Confidence.strong
        assert report.weak_reason is None


# ---------------------------------------------------------------------------
# collect_evidence — batch
# ---------------------------------------------------------------------------

class TestCollectEvidence:
    def test_returns_evidence_set(self):
        fs = _fs("app/auth.py")
        intel = _make_intel([fs])
        sigs = _signals()
        result = collect_evidence([_candidate("app/auth.py")], intel, sigs)
        assert isinstance(result, EvidenceSet)

    def test_empty_candidates_returns_empty(self):
        intel = _make_intel([])
        sigs = _signals()
        result = collect_evidence([], intel, sigs)
        assert result.reports == []
        assert result.candidates_evaluated == 0

    def test_order_matches_candidates(self):
        fs_a = _fs("app/alpha.py", symbols=[_sym("func_a", SymbolType.function)])
        fs_b = _fs("app/beta.py",  symbols=[_sym("func_b", SymbolType.function)])
        intel = _make_intel([fs_a, fs_b])
        sigs = _signals(function_names=["func_a", "func_b"])
        candidates = [_candidate("app/alpha.py", 10.0), _candidate("app/beta.py", 5.0)]
        result = collect_evidence(candidates, intel, sigs)
        assert result.reports[0].path == "app/alpha.py"
        assert result.reports[1].path == "app/beta.py"
        assert result.candidates_evaluated == 2

    def test_candidates_evaluated_count(self):
        files = [_fs(f"app/mod{i}.py") for i in range(4)]
        intel = _make_intel(files)
        sigs = _signals()
        candidates = [_candidate(f"app/mod{i}.py") for i in range(4)]
        result = collect_evidence(candidates, intel, sigs)
        assert result.candidates_evaluated == 4


# ---------------------------------------------------------------------------
# Realistic end-to-end scenario
# ---------------------------------------------------------------------------

class TestRealisticScenario:
    """Flask-style auth service with tests."""

    def _build(self):
        auth_fs = _fs("app/auth/middleware.py", symbols=[
            _sym("AuthMiddleware", SymbolType.class_),
            _sym("authenticate_request", SymbolType.method, parent="AuthMiddleware"),
            _sym("_parse_token", SymbolType.method, parent="AuthMiddleware"),
        ], imports=[
            _imp("app.auth.utils"),
            _imp("app.exceptions"),
        ])
        utils_fs = _fs("app/auth/utils.py", symbols=[
            _sym("parse_token", SymbolType.function),
            _sym("validate_token", SymbolType.function),
        ])
        exceptions_fs = _fs("app/exceptions.py", symbols=[
            _sym("AuthenticationError", SymbolType.class_),
            _sym("TokenExpiredError", SymbolType.class_),
        ])
        views_fs = _fs("app/views/api.py", imports=[
            _imp("app.auth.middleware"),
        ])
        test_fs = _fs("tests/test_auth.py", symbols=[
            _sym("test_authenticate", SymbolType.function),
            _sym("test_invalid_token", SymbolType.function),
        ], imports=[
            _imp("app.auth.middleware"),
        ], is_test=True)

        intel = _make_intel([auth_fs, utils_fs, exceptions_fs, views_fs, test_fs])
        sigs = _signals(
            function_names=["authenticate_request", "parse_token"],
            class_names=["AuthMiddleware"],
            error_messages=["AuthenticationError: invalid token"],
            keywords=["auth", "token"],
        )
        return intel, sigs

    def test_middleware_has_strong_confidence(self):
        intel, sigs = self._build()
        report = collect_evidence_for_file(
            _candidate("app/auth/middleware.py"), intel, sigs
        )
        assert report.confidence == Confidence.strong

    def test_middleware_matched_symbols_populated(self):
        intel, sigs = self._build()
        report = collect_evidence_for_file(
            _candidate("app/auth/middleware.py"), intel, sigs
        )
        sym_names = [s.name for s in report.matched_symbols]
        assert "AuthMiddleware" in sym_names
        assert "authenticate_request" in sym_names

    def test_middleware_related_tests_found(self):
        intel, sigs = self._build()
        report = collect_evidence_for_file(
            _candidate("app/auth/middleware.py"), intel, sigs
        )
        assert any(t.path == "tests/test_auth.py" for t in report.related_tests)
        test_ev = next(t for t in report.related_tests if t.path == "tests/test_auth.py")
        assert "test_authenticate" in test_ev.test_symbol_names

    def test_exceptions_file_reports_error_type_match(self):
        intel, sigs = self._build()
        report = collect_evidence_for_file(
            _candidate("app/exceptions.py"), intel, sigs
        )
        error_syms = [s for s in report.matched_symbols if s.matched_signal == "error_type"]
        assert any(s.name == "AuthenticationError" for s in error_syms)

    def test_callers_of_middleware_include_views(self):
        intel, sigs = self._build()
        report = collect_evidence_for_file(
            _candidate("app/auth/middleware.py"), intel, sigs
        )
        caller_paths = [c.path for c in report.callers]
        assert "app/views/api.py" in caller_paths

    def test_batch_collect_returns_correct_count(self):
        intel, sigs = self._build()
        candidates = [
            _candidate("app/auth/middleware.py", 12.0),
            _candidate("app/auth/utils.py", 8.0),
            _candidate("app/exceptions.py", 3.5),
        ]
        result = collect_evidence(candidates, intel, sigs)
        assert result.candidates_evaluated == 3
        assert len(result.reports) == 3

    def test_all_reports_have_confidence(self):
        intel, sigs = self._build()
        candidates = [
            _candidate("app/auth/middleware.py"),
            _candidate("app/auth/utils.py"),
            _candidate("app/exceptions.py"),
        ]
        result = collect_evidence(candidates, intel, sigs)
        for report in result.reports:
            assert report.confidence in list(Confidence)
