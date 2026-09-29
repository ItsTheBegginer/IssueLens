"""Evidence engine.

For each candidate file, inspects the repository intelligence index and
issue signals to collect structured evidence explaining its relevance.

All analysis is deterministic — no LLM, no network, no fabrication.
Evidence is derived solely from FileSymbols data already in RepoIntelligence.

Public API
----------
``collect_evidence(candidates, intelligence, signals) -> EvidenceSet``
``collect_evidence_for_file(candidate, intelligence, signals) -> EvidenceReport``
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Optional

from app.models.candidates import CandidateFile
from app.models.evidence import (
    CallerEvidence,
    Confidence,
    EvidenceReport,
    EvidenceSet,
    ImportEvidence,
    SymbolEvidence,
    TestEvidence,
)
from app.models.intelligence import FileSymbols, RepoIntelligence, SymbolType
from app.models.issue_signals import IssueSignals

__all__ = ["collect_evidence", "collect_evidence_for_file"]

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

_STRONG_THRESHOLD   = 3   # ≥ this many distinct evidence items → strong
_MODERATE_THRESHOLD = 1   # ≥ 1 item → moderate; 0 items → weak

# An "evidence item" is any non-empty list from:
#   matched_symbols, relevant_imports, callers, related_tests
# plus 1 point if issue_term_hits >= 2

# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def _normalise(s: str) -> str:
    return s.lower().strip("`./\\ ")


def _module_name_from_path(path: str) -> str:
    return path.replace("/", ".").replace("\\", ".").removesuffix(".py")


def _path_tokens(path: str) -> set[str]:
    """Lowercase tokens from path segments and the filename stem."""
    parts = path.replace("\\", "/").split("/")
    stem = os.path.splitext(parts[-1])[0]
    tokens = parts[:-1] + [stem] + stem.replace("-", "_").split("_")
    return {t.lower() for t in tokens if t}


# Error class name pattern for extracting type names from error_messages strings
_ERROR_CLASS_RE = re.compile(
    r"\b[A-Z][a-zA-Z]*(?:Error|Exception|Failure|Fault|Warning)\b"
)


def _error_class_names(signals: IssueSignals) -> set[str]:
    """Return lowercase set of exception class names from error_messages + class_names."""
    names: set[str] = set()
    for msg in signals.error_messages:
        for m in _ERROR_CLASS_RE.finditer(msg):
            names.add(m.group(0).lower())
    for cn in signals.class_names:
        cn_lower = cn.lower()
        if any(cn_lower.endswith(s) for s in ("error", "exception", "failure", "fault")):
            names.add(cn_lower)
    return names


# ---------------------------------------------------------------------------
# Index builders (computed once per collect_evidence call)
# ---------------------------------------------------------------------------

def _build_callers_index(
    intelligence: RepoIntelligence,
) -> dict[str, list[FileSymbols]]:
    """Return {module_name: [FileSymbols of files that import it]}."""
    index: dict[str, list[FileSymbols]] = defaultdict(list)
    fs_by_path: dict[str, FileSymbols] = {fs.path: fs for fs in intelligence.files}
    for fs in intelligence.files:
        for imp in fs.imports:
            index[imp.module].append(fs)
    return index


def _build_fs_index(intelligence: RepoIntelligence) -> dict[str, FileSymbols]:
    return {fs.path: fs for fs in intelligence.files}


# ---------------------------------------------------------------------------
# Individual evidence collectors
# ---------------------------------------------------------------------------

def _collect_symbol_evidence(
    fs: FileSymbols,
    signals: IssueSignals,
    error_names: set[str],
) -> list[SymbolEvidence]:
    """Return SymbolEvidence for every symbol that matches an issue signal."""
    func_lower = {_normalise(n): n for n in signals.function_names}
    class_lower = {_normalise(n): n for n in signals.class_names}
    kw_lower = {_normalise(k): k for k in signals.keywords if len(k) > 2}

    found: list[SymbolEvidence] = []
    seen: set[str] = set()

    for sym in fs.symbols:
        name_l = sym.name.lower()
        key = f"{sym.name}:{sym.start_line}"
        if key in seen:
            continue

        matched_signal: Optional[str] = None
        matched_term: Optional[str] = None

        if sym.type in (
            SymbolType.function, SymbolType.async_function,
            SymbolType.method, SymbolType.async_method,
        ):
            if name_l in func_lower:
                matched_signal = "function_name"
                matched_term = func_lower[name_l]
            elif name_l in kw_lower:
                matched_signal = "keyword"
                matched_term = kw_lower[name_l]

        elif sym.type == SymbolType.class_:
            if name_l in error_names:
                matched_signal = "error_type"
                matched_term = sym.name
            elif name_l in class_lower:
                matched_signal = "class_name"
                matched_term = class_lower[name_l]
            elif name_l in kw_lower:
                matched_signal = "keyword"
                matched_term = kw_lower[name_l]

        if matched_signal and matched_term:
            seen.add(key)
            found.append(SymbolEvidence(
                name=sym.name,
                kind=sym.type.value,
                start_line=sym.start_line,
                end_line=sym.end_line,
                parent=sym.parent,
                matched_signal=matched_signal,
                matched_term=matched_term,
            ))

    return found


def _collect_import_evidence(
    fs: FileSymbols,
    signals: IssueSignals,
) -> list[ImportEvidence]:
    """Return ImportEvidence for imports whose module names overlap with issue signals."""
    # Build a combined set of issue terms to match against module names
    all_terms: list[str] = (
        signals.function_names
        + signals.class_names
        + [k for k in signals.keywords if len(k) > 2]
    )
    terms_lower = {_normalise(t): t for t in all_terms}

    found: list[ImportEvidence] = []
    seen: set[str] = set()

    for imp in fs.imports:
        module_l = imp.module.lower()
        # Check each segment of the dotted module name
        segments = [s.lower() for s in module_l.replace("-", "_").split(".")]
        for seg in segments:
            if seg in terms_lower and module_l not in seen:
                seen.add(module_l)
                found.append(ImportEvidence(
                    module=imp.module,
                    line=imp.line,
                    matched_term=terms_lower[seg],
                ))
                break

    return found


def _collect_callers(
    fs: FileSymbols,
    callers_index: dict[str, list[FileSymbols]],
    *,
    limit: int = 10,
) -> list[CallerEvidence]:
    """Return files that import this file's module."""
    module = _module_name_from_path(fs.path)
    callers = callers_index.get(module, [])
    return [
        CallerEvidence(path=c.path, is_test=c.is_test)
        for c in callers[:limit]
        if c.path != fs.path  # exclude self-import edge
    ]


def _collect_related_tests(
    fs: FileSymbols,
    callers_index: dict[str, list[FileSymbols]],
    fs_index: dict[str, FileSymbols],
    *,
    limit: int = 5,
) -> list[TestEvidence]:
    """Return test files that import this file's module."""
    module = _module_name_from_path(fs.path)
    callers = callers_index.get(module, [])
    found: list[TestEvidence] = []
    for caller in callers:
        if not caller.is_test:
            continue
        test_syms = [
            s.name for s in caller.symbols
            if s.type in (SymbolType.function, SymbolType.async_function)
            and s.name.startswith("test")
        ]
        found.append(TestEvidence(path=caller.path, test_symbol_names=test_syms))
        if len(found) >= limit:
            break
    return found


def _count_issue_term_hits(
    fs: FileSymbols,
    signals: IssueSignals,
) -> int:
    """Count how many distinct issue terms appear in path tokens or symbol names."""
    all_terms_lower: set[str] = set()
    for t in signals.function_names + signals.class_names:
        all_terms_lower.add(_normalise(t))
    for k in signals.keywords:
        if len(k) > 2:
            all_terms_lower.add(_normalise(k))

    path_toks = _path_tokens(fs.path)
    sym_names_lower = {s.name.lower() for s in fs.symbols}
    available = path_toks | sym_names_lower

    return sum(1 for t in all_terms_lower if t in available)


def _compute_confidence(
    matched_symbols: list[SymbolEvidence],
    relevant_imports: list[ImportEvidence],
    callers: list[CallerEvidence],
    related_tests: list[TestEvidence],
    issue_term_hits: int,
) -> tuple[Confidence, Optional[str]]:
    """Derive confidence level and an optional weak_reason explanation."""
    # Count distinct non-empty evidence buckets
    buckets = sum([
        bool(matched_symbols),
        bool(relevant_imports),
        bool(callers),
        bool(related_tests),
        issue_term_hits >= 2,
    ])

    if buckets >= _STRONG_THRESHOLD:
        return Confidence.strong, None

    if buckets >= _MODERATE_THRESHOLD:
        return Confidence.moderate, None

    # Weak — explain why the file is still included
    reasons = []
    if callers:
        reasons.append(f"imported by {len(callers)} other file(s)")
    if issue_term_hits == 1:
        reasons.append("one issue term matches a path token or symbol name")
    weak_reason = (
        "Surfaced by structural proximity only: " + "; ".join(reasons)
        if reasons
        else "Surfaced by import-graph proximity; no direct term or symbol match"
    )
    return Confidence.weak, weak_reason


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_evidence_for_file(
    candidate: CandidateFile,
    intelligence: RepoIntelligence,
    signals: IssueSignals,
    *,
    callers_index: Optional[dict[str, list[FileSymbols]]] = None,
    fs_index: Optional[dict[str, FileSymbols]] = None,
) -> EvidenceReport:
    """Build an EvidenceReport for a single candidate file.

    Parameters
    ----------
    candidate:
        The candidate file to examine.
    intelligence:
        Full repository index.
    signals:
        Extracted issue signals.
    callers_index:
        Pre-built caller index (pass in when processing multiple files to
        avoid rebuilding it for each call).
    fs_index:
        Pre-built {path: FileSymbols} index.

    Returns
    -------
    EvidenceReport
        Structured evidence; confidence is always set.
        No field is fabricated — only real data from the index.
    """
    if callers_index is None:
        callers_index = _build_callers_index(intelligence)
    if fs_index is None:
        fs_index = _build_fs_index(intelligence)

    fs = fs_index.get(candidate.path)
    if fs is None:
        # File listed in candidates but absent from index (shouldn't happen normally)
        return EvidenceReport(
            path=candidate.path,
            confidence=Confidence.weak,
            weak_reason="File not found in repository index.",
        )

    error_names = _error_class_names(signals)

    matched_symbols  = _collect_symbol_evidence(fs, signals, error_names)
    relevant_imports = _collect_import_evidence(fs, signals)
    callers          = _collect_callers(fs, callers_index)
    related_tests    = _collect_related_tests(fs, callers_index, fs_index)
    issue_term_hits  = _count_issue_term_hits(fs, signals)

    confidence, weak_reason = _compute_confidence(
        matched_symbols, relevant_imports, callers, related_tests, issue_term_hits
    )

    return EvidenceReport(
        path=candidate.path,
        confidence=confidence,
        matched_symbols=matched_symbols,
        relevant_imports=relevant_imports,
        callers=callers,
        related_tests=related_tests,
        issue_term_hits=issue_term_hits,
        weak_reason=weak_reason,
    )


def collect_evidence(
    candidates: list[CandidateFile],
    intelligence: RepoIntelligence,
    signals: IssueSignals,
) -> EvidenceSet:
    """Collect evidence for every candidate in *candidates*.

    Builds shared indexes once and reuses them across all candidates.

    Parameters
    ----------
    candidates:
        Ordered list of candidates from :func:`app.search.candidate_search.search_candidates`.
    intelligence:
        Full repository index.
    signals:
        Extracted issue signals.

    Returns
    -------
    EvidenceSet
        Reports in the same order as *candidates*.
    """
    if not candidates:
        return EvidenceSet(reports=[], candidates_evaluated=0)

    # Build shared indexes once
    callers_index = _build_callers_index(intelligence)
    fs_index = _build_fs_index(intelligence)

    reports = [
        collect_evidence_for_file(
            candidate, intelligence, signals,
            callers_index=callers_index,
            fs_index=fs_index,
        )
        for candidate in candidates
    ]

    return EvidenceSet(reports=reports, candidates_evaluated=len(candidates))
