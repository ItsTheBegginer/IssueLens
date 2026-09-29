"""Candidate code search.

Given extracted issue signals and a repository intelligence index, ranks
repository files by how likely they are to be relevant to the issue.

All scoring is deterministic — no LLM, no network calls, no database.

Public API
----------
``search_candidates(signals, intelligence, *, top_n) -> CandidateSet``
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional

from app.models.candidates import (
    CandidateFile,
    CandidateSet,
    MatchKind,
    MatchReason,
)
from app.models.intelligence import FileSymbols, RepoIntelligence, SymbolType
from app.models.issue_signals import IssueSignals

__all__ = ["search_candidates"]

# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------
# Tuned so that a single explicit file reference dominates, while
# multiple weaker signals can collectively surface a file that the issue
# doesn't name directly.

_W_FILE_REF   = 10.0   # issue text explicitly named this path
_W_SYMBOL     = 4.0    # a function/class name in the file matches the issue
_W_ERROR_TYPE = 3.5    # an error class named in the issue is defined here
_W_PATH_KW    = 1.5    # a keyword appears in the file path
_W_NEIGHBOR   = 1.0    # file imports / is imported by a high-score candidate
_W_TEST_COV   = 0.5    # a test that imports this module is also a candidate

_TOP_N_DEFAULT = 20

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _path_parts(path: str) -> list[str]:
    """Return lowercase stem and directory segments from a file path."""
    parts = path.replace("\\", "/").split("/")
    # Also split the filename stem on underscores/hyphens
    stem = os.path.splitext(parts[-1])[0]
    tokens = parts[:-1] + [stem] + stem.replace("-", "_").split("_")
    return [t.lower() for t in tokens if t]


def _normalise(name: str) -> str:
    return name.lower().strip("`./\\")


def _file_ref_overlap(path: str, file_references: list[str]) -> Optional[str]:
    """Return the first file_reference that matches this path, or None."""
    path_norm = _normalise(path)
    path_base = os.path.basename(path_norm)
    for ref in file_references:
        ref_norm = _normalise(ref)
        ref_base = os.path.basename(ref_norm)
        # Exact path or segment match (issue says src/auth.py, file is src/auth.py)
        if (
            path_norm == ref_norm
            or path_norm.endswith("/" + ref_norm)
            or ref_norm.endswith("/" + path_norm)
        ):
            return ref
        # Basename-only match (issue says auth.py, any path ending in /auth.py)
        if ref_base and path_base == ref_base:
            return ref
    return None


def _stem(token: str) -> str:
    """Basic plural/singular normalizer for code tokens."""
    t = token.lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es") and not t.endswith("ses"):
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _symbol_matches(
    fs: FileSymbols,
    func_names_lower: set[str],
    class_names_lower: set[str],
    keywords_lower: set[str] | None = None,
) -> list[str]:
    """Return symbol names in *fs* that match issue signals."""
    matched: list[str] = []
    kws = keywords_lower or set()
    for sym in fs.symbols:
        name_lower = sym.name.lower()
        if sym.type in (
            SymbolType.function, SymbolType.async_function,
            SymbolType.method, SymbolType.async_method,
        ):
            if name_lower in func_names_lower:
                matched.append(sym.name)
            elif name_lower in kws and len(name_lower) > 3:
                matched.append(sym.name)
        elif sym.type == SymbolType.class_:
            if name_lower in class_names_lower:
                matched.append(sym.name)
            elif name_lower in kws and len(name_lower) > 3:
                matched.append(sym.name)
    return matched


def _path_keyword_matches(path: str, keywords_lower: set[str]) -> list[str]:
    """Return keywords that appear as tokens in the file's path (supporting stemming)."""
    parts = set(_path_parts(path))
    parts_stemmed = {_stem(p) for p in parts}
    matched: list[str] = []
    for kw in keywords_lower:
        if kw in parts or _stem(kw) in parts_stemmed:
            matched.append(kw)
    return matched


def _error_type_matches(
    fs: FileSymbols,
    error_class_names_lower: set[str],
) -> list[str]:
    """Return error class names from the issue that are defined in *fs*."""
    matched: list[str] = []
    for sym in fs.symbols:
        if sym.type == SymbolType.class_ and sym.name.lower() in error_class_names_lower:
            matched.append(sym.name)
    return matched


def _module_name_from_path(path: str) -> str:
    """Convert path to dotted module name for import matching."""
    return path.replace("/", ".").replace("\\", ".").removesuffix(".py")


# ---------------------------------------------------------------------------
# Build import index from RepoIntelligence
# ---------------------------------------------------------------------------

def _build_import_index(
    intelligence: RepoIntelligence,
) -> dict[str, list[str]]:
    """Return {module_name: [file_paths_that_import_it]}.

    Used to find files that are imported by (or import) candidate files.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for fs in intelligence.files:
        for imp in fs.imports:
            index[imp.module].append(fs.path)
    return index


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def search_candidates(
    signals: IssueSignals,
    intelligence: RepoIntelligence,
    *,
    top_n: int = _TOP_N_DEFAULT,
) -> CandidateSet:
    """Score every file in *intelligence* against *signals* and return ranked results.

    Parameters
    ----------
    signals:
        Extracted issue signals from :func:`app.analyzers.issue_signals.extract_signals`.
    intelligence:
        Repository index from :func:`app.analyzers.repo_intelligence.build_repo_intelligence`.
    top_n:
        Maximum number of candidates to return.

    Returns
    -------
    CandidateSet
        Candidates in descending score order with per-reason breakdowns.
    """
    if not intelligence.files:
        return CandidateSet(
            candidates=[],
            total_files_considered=0,
            signals_used=[],
        )

    # Pre-process signals for O(1) lookups
    func_names_lower: set[str] = {_normalise(n) for n in signals.function_names}
    class_names_lower: set[str] = {_normalise(n) for n in signals.class_names}
    keywords_lower: set[str] = {_normalise(k) for k in signals.keywords if len(k) > 2}

    # Extract error class names specifically (CamelCase names from error_messages)
    import re
    _error_class_re = re.compile(r"\b[A-Z][a-zA-Z]*(?:Error|Exception|Failure|Fault|Warning)\b")
    error_class_names_lower: set[str] = set()
    for err in signals.error_messages:
        for m in _error_class_re.finditer(err):
            error_class_names_lower.add(m.group(0).lower())
    # Also include class_names that look like exception types
    for cn in class_names_lower:
        if any(cn.endswith(s) for s in ("error", "exception", "failure", "fault")):
            error_class_names_lower.add(cn)

    import_index = _build_import_index(intelligence)

    # Track which signals were actually used
    signals_used: list[str] = []
    if signals.file_references:
        signals_used.append("file_references")
    if signals.function_names:
        signals_used.append("function_names")
    if signals.class_names:
        signals_used.append("class_names")
    if signals.keywords:
        signals_used.append("keywords")
    if signals.error_messages:
        signals_used.append("error_messages")

    # ---------- First pass: score each file ----------
    scored: list[CandidateFile] = []

    for fs in intelligence.files:
        reasons: list[MatchReason] = []
        matched_syms: list[str] = []

        # 1. Direct file reference
        ref_match = _file_ref_overlap(fs.path, signals.file_references)
        if ref_match:
            reasons.append(MatchReason(
                kind=MatchKind.file_reference,
                detail=f"issue references '{ref_match}' → matches '{fs.path}'",
                weight=_W_FILE_REF,
            ))

        # 2. Symbol name match (functions + classes + keyword symbols)
        sym_hits = _symbol_matches(fs, func_names_lower, class_names_lower, keywords_lower=keywords_lower)
        if sym_hits:
            matched_syms.extend(sym_hits)
            reasons.append(MatchReason(
                kind=MatchKind.symbol_name,
                detail=f"symbol(s) {sym_hits} match issue function/class names",
                weight=_W_SYMBOL * len(sym_hits),
            ))

        # 3. Error type defined here
        err_hits = _error_type_matches(fs, error_class_names_lower)
        if err_hits:
            matched_syms.extend(err_hits)
            reasons.append(MatchReason(
                kind=MatchKind.error_type,
                detail=f"class(es) {err_hits} match error type(s) from issue",
                weight=_W_ERROR_TYPE * len(err_hits),
            ))

        # 4. Keyword in path
        kw_hits = _path_keyword_matches(fs.path, keywords_lower)
        if kw_hits:
            reasons.append(MatchReason(
                kind=MatchKind.path_keyword,
                detail=f"path token(s) {kw_hits} match issue keywords",
                weight=_W_PATH_KW * len(kw_hits),
            ))

        if reasons:
            score = sum(r.weight for r in reasons)
            scored.append(CandidateFile(
                path=fs.path,
                score=score,
                reasons=reasons,
                is_test=fs.is_test,
                matched_symbols=list(dict.fromkeys(matched_syms)),  # dedup, order-preserving
            ))

    # ---------- Second pass: import-neighbour boost ----------
    # Build a set of high-confidence paths (file_reference + symbol hits)
    strong_paths: set[str] = {
        c.path for c in scored
        if any(r.kind in (MatchKind.file_reference, MatchKind.symbol_name) for r in c.reasons)
    }

    # Index existing scored paths for fast lookup
    scored_index: dict[str, CandidateFile] = {c.path: c for c in scored}

    # For each strong candidate, find files it imports and files that import it
    for strong_path in list(strong_paths):
        strong_module = _module_name_from_path(strong_path)
        # Files that strong_path imports → find them in intelligence
        strong_fs_list = [fs for fs in intelligence.files if fs.path == strong_path]
        if strong_fs_list:
            strong_fs = strong_fs_list[0]
            for imp in strong_fs.imports:
                # Find any file whose module name matches this import
                for fs in intelligence.files:
                    if fs.path == strong_path:
                        continue
                    if _module_name_from_path(fs.path) == imp.module:
                        _add_neighbour_reason(
                            fs, scored_index, scored,
                            f"imported by candidate '{strong_path}'",
                        )

        # Files that import strong_path
        for importer_path in import_index.get(strong_module, []):
            fs_list = [fs for fs in intelligence.files if fs.path == importer_path]
            if fs_list:
                _add_neighbour_reason(
                    fs_list[0], scored_index, scored,
                    f"imports candidate '{strong_path}'",
                )

    # ---------- Sort, cap, return ----------
    scored.sort(key=lambda c: c.score, reverse=True)
    top = scored[:top_n]

    return CandidateSet(
        candidates=top,
        total_files_considered=len(intelligence.files),
        signals_used=signals_used,
    )


def _add_neighbour_reason(
    fs: FileSymbols,
    scored_index: dict[str, CandidateFile],
    scored: list[CandidateFile],
    detail: str,
) -> None:
    """Add or update an import_neighbour reason for *fs*."""
    reason = MatchReason(
        kind=MatchKind.import_neighbor,
        detail=detail,
        weight=_W_NEIGHBOR,
    )
    if fs.path in scored_index:
        existing = scored_index[fs.path]
        # Avoid duplicate neighbour reasons for the same detail
        if not any(r.kind == MatchKind.import_neighbor and r.detail == detail
                   for r in existing.reasons):
            existing.reasons.append(reason)
            existing.score += _W_NEIGHBOR
    else:
        candidate = CandidateFile(
            path=fs.path,
            score=_W_NEIGHBOR,
            reasons=[reason],
            is_test=fs.is_test,
        )
        scored_index[fs.path] = candidate
        scored.append(candidate)
