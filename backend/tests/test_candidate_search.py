"""Tests for candidate code search.

All tests are fully deterministic — no network, no LLM, no filesystem I/O.
Repository indexes are constructed inline using model classes.
"""
from __future__ import annotations

import pytest

from app.models.candidates import CandidateSet, MatchKind
from app.models.intelligence import (
    FileSymbols,
    ImportRecord,
    ImportedName,
    RepoIntelligence,
    Relationship,
    RelationshipType,
    Symbol,
    SymbolType,
)
from app.models.issue_signals import IssueSignals
from app.search.candidate_search import search_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intelligence(files: list[FileSymbols]) -> RepoIntelligence:
    return RepoIntelligence(
        repo_full_name="owner/repo",
        file_count=len(files),
        analyzed_count=len(files),
        skipped_count=0,
        files=files,
        relationships=[],
    )


def _make_fs(
    path: str,
    *,
    symbols: list[Symbol] | None = None,
    imports: list[ImportRecord] | None = None,
    is_test: bool = False,
) -> FileSymbols:
    return FileSymbols(
        path=path,
        symbols=symbols or [],
        imports=imports or [],
        is_test=is_test,
    )


def _sym(name: str, kind: SymbolType, line: int = 1) -> Symbol:
    return Symbol(name=name, type=kind, start_line=line, end_line=line + 5)


def _signals(**kwargs) -> IssueSignals:
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
    defaults.update(kwargs)
    return IssueSignals(**defaults)


# ---------------------------------------------------------------------------
# Return type and basic contract
# ---------------------------------------------------------------------------

class TestReturnContract:
    def test_returns_candidate_set(self):
        intel = _make_intelligence([_make_fs("app/utils.py")])
        result = search_candidates(_signals(), intel)
        assert isinstance(result, CandidateSet)

    def test_empty_repo_returns_empty(self):
        intel = _make_intelligence([])
        result = search_candidates(_signals(), intel)
        assert result.candidates == []
        assert result.total_files_considered == 0

    def test_total_files_considered_correct(self):
        files = [_make_fs(f"app/module{i}.py") for i in range(5)]
        intel = _make_intelligence(files)
        result = search_candidates(_signals(), intel)
        assert result.total_files_considered == 5

    def test_no_signals_returns_no_candidates(self):
        # With no signals, nothing scores > 0
        intel = _make_intelligence([_make_fs("app/utils.py"), _make_fs("app/auth.py")])
        result = search_candidates(_signals(), intel)
        assert result.candidates == []

    def test_candidates_ordered_by_score_descending(self):
        files = [
            _make_fs("app/auth.py", symbols=[_sym("authenticate_user", SymbolType.function)]),
            _make_fs("app/utils.py", symbols=[_sym("helper", SymbolType.function)]),
            _make_fs("app/routes.py"),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(function_names=["authenticate_user"])
        result = search_candidates(sigs, intel)
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_respected(self):
        files = [
            _make_fs(f"app/mod{i}.py", symbols=[_sym("target_func", SymbolType.function)])
            for i in range(10)
        ]
        intel = _make_intelligence(files)
        sigs = _signals(function_names=["target_func"])
        result = search_candidates(sigs, intel, top_n=3)
        assert len(result.candidates) <= 3

    def test_signals_used_reported(self):
        intel = _make_intelligence([_make_fs("app/utils.py")])
        sigs = _signals(
            keywords=["auth"],
            function_names=["login"],
            file_references=["auth.py"],
        )
        result = search_candidates(sigs, intel)
        assert "keywords" in result.signals_used
        assert "function_names" in result.signals_used
        assert "file_references" in result.signals_used


# ---------------------------------------------------------------------------
# File reference matching
# ---------------------------------------------------------------------------

class TestFileReference:
    def test_exact_path_match(self):
        files = [_make_fs("src/auth/middleware.py"), _make_fs("src/utils.py")]
        intel = _make_intelligence(files)
        sigs = _signals(file_references=["src/auth/middleware.py"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "src/auth/middleware.py"

    def test_basename_only_match(self):
        files = [_make_fs("src/auth/middleware.py"), _make_fs("src/utils.py")]
        intel = _make_intelligence(files)
        sigs = _signals(file_references=["middleware.py"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "src/auth/middleware.py"

    def test_file_reference_has_highest_score(self):
        files = [
            _make_fs("app/auth.py"),
            _make_fs("app/utils.py", symbols=[_sym("auth", SymbolType.function)] * 5),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(file_references=["auth.py"])
        result = search_candidates(sigs, intel)
        # auth.py should win even though utils.py has many matching symbols
        assert result.candidates[0].path == "app/auth.py"

    def test_reason_kind_is_file_reference(self):
        files = [_make_fs("app/router.py")]
        intel = _make_intelligence(files)
        sigs = _signals(file_references=["router.py"])
        result = search_candidates(sigs, intel)
        kinds = [r.kind for r in result.candidates[0].reasons]
        assert MatchKind.file_reference in kinds

    def test_no_match_on_different_extension(self):
        files = [_make_fs("app/auth.py")]
        intel = _make_intelligence(files)
        sigs = _signals(file_references=["auth.js"])
        result = search_candidates(sigs, intel)
        # auth.js ≠ auth.py
        assert not any(c.path == "app/auth.py" and
                       any(r.kind == MatchKind.file_reference for r in c.reasons)
                       for c in result.candidates)


# ---------------------------------------------------------------------------
# Symbol name matching
# ---------------------------------------------------------------------------

class TestSymbolMatch:
    def test_function_name_match(self):
        files = [
            _make_fs("app/handlers.py", symbols=[_sym("parse_request", SymbolType.function)]),
            _make_fs("app/utils.py", symbols=[_sym("helper", SymbolType.function)]),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(function_names=["parse_request"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/handlers.py"

    def test_class_name_match(self):
        files = [
            _make_fs("app/middleware.py", symbols=[_sym("AuthMiddleware", SymbolType.class_)]),
            _make_fs("app/views.py"),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(class_names=["AuthMiddleware"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/middleware.py"

    def test_method_match(self):
        files = [
            _make_fs("app/service.py", symbols=[_sym("handle_request", SymbolType.method)]),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(function_names=["handle_request"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/service.py"
        assert "handle_request" in result.candidates[0].matched_symbols

    def test_multiple_symbol_hits_boost_score(self):
        files = [
            _make_fs("app/auth.py", symbols=[
                _sym("validate_token", SymbolType.function),
                _sym("parse_credentials", SymbolType.function),
            ]),
            _make_fs("app/views.py", symbols=[
                _sym("validate_token", SymbolType.function),
            ]),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(function_names=["validate_token", "parse_credentials"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/auth.py"
        assert result.candidates[0].score > result.candidates[1].score

    def test_symbol_match_case_insensitive(self):
        files = [
            _make_fs("app/router.py", symbols=[_sym("HandleRequest", SymbolType.function)]),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(function_names=["handlerequest"])
        result = search_candidates(sigs, intel)
        assert any(c.path == "app/router.py" for c in result.candidates)

    def test_matched_symbols_populated(self):
        files = [
            _make_fs("app/auth.py", symbols=[
                _sym("login_user", SymbolType.function),
                _sym("logout_user", SymbolType.function),
                _sym("unrelated", SymbolType.function),
            ]),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(function_names=["login_user", "logout_user"])
        result = search_candidates(sigs, intel)
        c = result.candidates[0]
        assert "login_user" in c.matched_symbols
        assert "logout_user" in c.matched_symbols
        assert "unrelated" not in c.matched_symbols


# ---------------------------------------------------------------------------
# Path keyword matching
# ---------------------------------------------------------------------------

class TestPathKeyword:
    def test_keyword_in_path_stem(self):
        files = [
            _make_fs("app/authentication/handler.py"),
            _make_fs("app/views.py"),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(keywords=["authentication"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/authentication/handler.py"

    def test_keyword_in_directory_segment(self):
        files = [
            _make_fs("src/auth/middleware.py"),
            _make_fs("src/views/home.py"),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(keywords=["auth"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "src/auth/middleware.py"

    def test_keyword_matches_stem(self):
        files = [
            _make_fs("app/serializer.py"),
            _make_fs("app/views.py"),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(keywords=["serializer"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/serializer.py"

    def test_reason_kind_is_path_keyword(self):
        files = [_make_fs("app/auth/utils.py")]
        intel = _make_intelligence(files)
        sigs = _signals(keywords=["auth"])
        result = search_candidates(sigs, intel)
        kinds = [r.kind for r in result.candidates[0].reasons]
        assert MatchKind.path_keyword in kinds

    def test_short_keywords_ignored(self):
        # Keywords <= 2 chars are filtered out to avoid noise
        files = [_make_fs("app/io/reader.py")]
        intel = _make_intelligence(files)
        sigs = _signals(keywords=["io"])  # 2 chars — below threshold
        result = search_candidates(sigs, intel)
        assert not any(
            any(r.kind == MatchKind.path_keyword for r in c.reasons)
            for c in result.candidates
        )


# ---------------------------------------------------------------------------
# Error type matching
# ---------------------------------------------------------------------------

class TestErrorTypeMatch:
    def test_exception_class_defined_in_file(self):
        files = [
            _make_fs("app/exceptions.py", symbols=[
                _sym("AuthenticationError", SymbolType.class_),
            ]),
            _make_fs("app/views.py"),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(error_messages=["AuthenticationError: invalid token"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/exceptions.py"

    def test_reason_kind_is_error_type(self):
        files = [
            _make_fs("app/errors.py", symbols=[_sym("DatabaseError", SymbolType.class_)])
        ]
        intel = _make_intelligence(files)
        sigs = _signals(error_messages=["DatabaseError: connection failed"])
        result = search_candidates(sigs, intel)
        kinds = [r.kind for r in result.candidates[0].reasons]
        assert MatchKind.error_type in kinds

    def test_non_error_class_not_matched_as_error_type(self):
        files = [
            _make_fs("app/models.py", symbols=[_sym("UserModel", SymbolType.class_)])
        ]
        intel = _make_intelligence(files)
        sigs = _signals(error_messages=["DatabaseError"])
        result = search_candidates(sigs, intel)
        # UserModel is not an error type
        assert not any(
            any(r.kind == MatchKind.error_type for r in c.reasons)
            for c in result.candidates
        )


# ---------------------------------------------------------------------------
# Import neighbour signal
# ---------------------------------------------------------------------------

class TestImportNeighbour:
    def test_file_imported_by_strong_candidate_gets_boost(self):
        # auth.py imports utils.py; issue names auth.py
        utils_fs = _make_fs("app/utils.py")
        auth_fs = _make_fs(
            "app/auth.py",
            imports=[ImportRecord(module="app.utils", names=[], line=1)],
        )
        intel = _make_intelligence([utils_fs, auth_fs])
        sigs = _signals(file_references=["auth.py"])
        result = search_candidates(sigs, intel)

        # auth.py is the strong candidate
        assert result.candidates[0].path == "app/auth.py"
        # utils.py should be surfaced as a neighbour
        paths = [c.path for c in result.candidates]
        assert "app/utils.py" in paths

        utils_candidate = next(c for c in result.candidates if c.path == "app/utils.py")
        assert any(r.kind == MatchKind.import_neighbor for r in utils_candidate.reasons)

    def test_file_that_imports_strong_candidate_gets_boost(self):
        # service.py is a strong match; router.py imports service.py
        service_fs = _make_fs(
            "app/service.py",
            symbols=[_sym("process_request", SymbolType.function)],
        )
        router_fs = _make_fs(
            "app/router.py",
            imports=[ImportRecord(module="app.service", names=[], line=1)],
        )
        intel = _make_intelligence([service_fs, router_fs])
        sigs = _signals(function_names=["process_request"])
        result = search_candidates(sigs, intel)

        paths = [c.path for c in result.candidates]
        assert "app/router.py" in paths

    def test_strong_candidate_itself_not_double_counted_as_neighbour(self):
        auth_fs = _make_fs(
            "app/auth.py",
            symbols=[_sym("login", SymbolType.function)],
            imports=[ImportRecord(module="app.auth", names=[], line=1)],  # self-import edge
        )
        intel = _make_intelligence([auth_fs])
        sigs = _signals(function_names=["login"])
        result = search_candidates(sigs, intel)
        # Only one candidate for auth.py
        auth_candidates = [c for c in result.candidates if c.path == "app/auth.py"]
        assert len(auth_candidates) == 1


# ---------------------------------------------------------------------------
# Test file detection
# ---------------------------------------------------------------------------

class TestTestFileDetection:
    def test_test_file_flagged(self):
        files = [
            _make_fs("tests/test_auth.py", is_test=True),
            _make_fs("app/auth.py"),
        ]
        intel = _make_intelligence(files)
        sigs = _signals(file_references=["test_auth.py"])
        result = search_candidates(sigs, intel)
        test_candidate = next(
            (c for c in result.candidates if c.path == "tests/test_auth.py"), None
        )
        assert test_candidate is not None
        assert test_candidate.is_test is True

    def test_source_file_not_flagged_as_test(self):
        files = [_make_fs("app/auth.py")]
        intel = _make_intelligence(files)
        sigs = _signals(file_references=["auth.py"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].is_test is False


# ---------------------------------------------------------------------------
# Combined scoring (multiple signals reinforce each other)
# ---------------------------------------------------------------------------

class TestCombinedScoring:
    def test_file_with_multiple_signals_scores_highest(self):
        """A file that matches on path keyword AND symbol name beats each alone."""
        # auth.py: matches keyword "auth" in path AND has function "validate_token"
        auth_fs = _make_fs(
            "app/auth/validator.py",
            symbols=[_sym("validate_token", SymbolType.function)],
        )
        # utils.py: only matches on keyword "auth" in an import (not path)
        utils_fs = _make_fs("app/utils.py")
        # models.py: only has the function symbol
        models_fs = _make_fs(
            "app/models.py",
            symbols=[_sym("validate_token", SymbolType.function)],
        )
        intel = _make_intelligence([auth_fs, utils_fs, models_fs])
        sigs = _signals(
            keywords=["auth", "validator"],
            function_names=["validate_token"],
        )
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/auth/validator.py"
        assert len(result.candidates[0].reasons) >= 2

    def test_reasons_detail_is_informative(self):
        files = [
            _make_fs("app/auth.py", symbols=[_sym("login_user", SymbolType.function)])
        ]
        intel = _make_intelligence(files)
        sigs = _signals(
            keywords=["auth"],
            function_names=["login_user"],
        )
        result = search_candidates(sigs, intel)
        c = result.candidates[0]
        # Every reason must have a non-empty detail string
        for reason in c.reasons:
            assert reason.detail


# ---------------------------------------------------------------------------
# Realistic end-to-end scenario
# ---------------------------------------------------------------------------

class TestRealisticScenario:
    """Flask-style repo: models, views, auth, tests."""

    def _build_repo(self) -> RepoIntelligence:
        files = [
            _make_fs("app/models/user.py", symbols=[
                _sym("User", SymbolType.class_),
                _sym("save", SymbolType.method),
            ]),
            _make_fs("app/auth/middleware.py", symbols=[
                _sym("AuthMiddleware", SymbolType.class_),
                _sym("authenticate_request", SymbolType.method),
            ]),
            _make_fs("app/auth/utils.py", symbols=[
                _sym("parse_token", SymbolType.function),
                _sym("validate_token", SymbolType.function),
            ], imports=[
                ImportRecord(module="app.models.user", names=[], line=1),
            ]),
            _make_fs("app/views/api.py", symbols=[
                _sym("UserView", SymbolType.class_),
                _sym("get", SymbolType.method),
                _sym("post", SymbolType.method),
            ], imports=[
                ImportRecord(module="app.auth.utils", names=[], line=2),
            ]),
            _make_fs("app/exceptions.py", symbols=[
                _sym("AuthenticationError", SymbolType.class_),
                _sym("TokenExpiredError", SymbolType.class_),
            ]),
            _make_fs("tests/test_auth.py", symbols=[
                _sym("test_authenticate", SymbolType.function),
            ], imports=[
                ImportRecord(module="app.auth.middleware", names=[], line=1),
            ], is_test=True),
        ]
        return _make_intelligence(files)

    def test_function_signal_finds_correct_file(self):
        intel = self._build_repo()
        sigs = _signals(function_names=["parse_token", "validate_token"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/auth/utils.py"

    def test_error_class_signal_finds_exceptions_file(self):
        intel = self._build_repo()
        sigs = _signals(error_messages=["AuthenticationError: invalid credentials"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/exceptions.py"

    def test_file_reference_directly_names_middleware(self):
        intel = self._build_repo()
        sigs = _signals(file_references=["app/auth/middleware.py"])
        result = search_candidates(sigs, intel)
        assert result.candidates[0].path == "app/auth/middleware.py"

    def test_neighbour_of_middleware_is_surfaced(self):
        """test_auth.py imports middleware → should appear as neighbour."""
        intel = self._build_repo()
        sigs = _signals(file_references=["middleware.py"])
        result = search_candidates(sigs, intel)
        paths = [c.path for c in result.candidates]
        assert "tests/test_auth.py" in paths

    def test_combined_keyword_and_class_signal(self):
        intel = self._build_repo()
        sigs = _signals(
            keywords=["auth"],
            class_names=["AuthMiddleware"],
        )
        result = search_candidates(sigs, intel)
        # AuthMiddleware is in middleware.py; auth is in the path
        assert result.candidates[0].path == "app/auth/middleware.py"

    def test_all_candidates_have_at_least_one_reason(self):
        intel = self._build_repo()
        sigs = _signals(
            keywords=["auth", "token"],
            function_names=["parse_token"],
            class_names=["AuthMiddleware"],
        )
        result = search_candidates(sigs, intel)
        for c in result.candidates:
            assert len(c.reasons) >= 1
            assert c.score > 0
