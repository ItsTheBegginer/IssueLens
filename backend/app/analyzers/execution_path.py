"""Execution path and test relationship analyzer.

Traces the execution/dependency flow around the localized bug location
and identifies the test files and test cases that exercise it.
Deterministic — purely structural based on AST and RepoIntelligence.
"""
from __future__ import annotations

import os
from typing import Optional

from app.models.analysis import ExecutionPathStep, RelevantTest
from app.models.evidence import EvidenceReport, EvidenceSet
from app.models.intelligence import FileSymbols, RepoIntelligence, SymbolType
from app.models.issue_signals import IssueSignals
from app.models.localization import LocalizationResult


def _module_name_from_path(path: str) -> str:
    """Convert file path to dotted module name."""
    return path.replace("/", ".").replace("\\", ".").removesuffix(".py")


def build_execution_path(
    localization: LocalizationResult,
    intelligence: RepoIntelligence,
    evidence_set: EvidenceSet,
) -> tuple[list[str], list[ExecutionPathStep]]:
    """Trace the execution/dependency path for the localized primary location.

    Returns:
        (text_summary_steps, structured_steps)
    """
    if localization.is_uncertain or not localization.primary_file:
        return ([], [])

    primary_file = localization.primary_file
    primary_module = _module_name_from_path(primary_file)

    # Find the primary file's FileSymbols
    fs_map = {f.path: f for f in intelligence.files}
    primary_fs = fs_map.get(primary_file)

    evidence_map: dict[str, EvidenceReport] = {
        r.path: r for r in evidence_set.reports
    }
    primary_ev = evidence_map.get(primary_file)

    steps: list[ExecutionPathStep] = []
    step_num = 1

    # 1. Upstream callers / entry points (non-test files that import primary_module)
    if primary_ev and primary_ev.callers:
        non_test_callers = [c for c in primary_ev.callers if not c.is_test]
        # Sort callers so candidate files appear first
        candidate_paths = {r.path for r in evidence_set.reports}
        non_test_callers.sort(key=lambda c: (c.path not in candidate_paths, c.path))

        for caller in non_test_callers[:2]:  # cap at top 2 upstream callers
            steps.append(
                ExecutionPathStep(
                    step_number=step_num,
                    step_type="caller",
                    file=caller.path,
                    detail=f"Upstream entry/caller '{caller.path}' imports '{primary_module}'",
                )
            )
            step_num += 1

    # 2. Primary target location
    sym_desc = f"::{localization.primary_symbol}" if localization.primary_symbol else ""
    kind_desc = f" ({localization.primary_symbol_kind})" if localization.primary_symbol_kind else ""
    lines_desc = (
        f" [lines {localization.line_range.start}–{localization.line_range.end}]"
        if localization.line_range
        else ""
    )

    steps.append(
        ExecutionPathStep(
            step_number=step_num,
            step_type="target",
            file=primary_file,
            symbol=localization.primary_symbol,
            line_range=localization.line_range,
            detail=f"Target location '{primary_file}{sym_desc}'{kind_desc}{lines_desc} — suspected root cause",
        )
    )
    step_num += 1

    # 3. Downstream dependencies (internal repo modules imported by primary_file)
    if primary_fs and primary_fs.imports:
        module_to_path = {_module_name_from_path(f.path): f.path for f in intelligence.files if not f.is_test}
        downstream_paths: list[tuple[str, str]] = []

        for imp in primary_fs.imports:
            # Check if this import matches an internal module
            matched_path = module_to_path.get(imp.module)
            if not matched_path:
                # Check prefix match for submodule imports
                for m_name, f_path in module_to_path.items():
                    if imp.module.startswith(m_name) and f_path != primary_file:
                        matched_path = f_path
                        break

            if matched_path and matched_path != primary_file:
                downstream_paths.append((imp.module, matched_path))

        # Deduplicate and cap at top 2 downstream dependencies
        seen_deps = set()
        for mod, dep_path in downstream_paths:
            if dep_path not in seen_deps:
                seen_deps.add(dep_path)
                steps.append(
                    ExecutionPathStep(
                        step_number=step_num,
                        step_type="dependency",
                        file=dep_path,
                        detail=f"Downstream dependency: '{primary_file}' imports '{mod}' ({dep_path})",
                    )
                )
                step_num += 1
                if len(seen_deps) >= 2:
                    break

    # Format human-readable text strings
    text_steps: list[str] = []
    for s in steps:
        if s.step_type == "caller":
            text_steps.append(f"{s.step_number}. [Caller] {s.file} → imports {primary_module}")
        elif s.step_type == "target":
            loc = f" lines {s.line_range.start}–{s.line_range.end}" if s.line_range else ""
            sym = f"::{s.symbol}" if s.symbol else ""
            text_steps.append(f"{s.step_number}. [Target] {s.file}{sym}{loc}")
        elif s.step_type == "dependency":
            text_steps.append(f"{s.step_number}. [Dependency] {primary_file} → {s.file}")

    return (text_steps, steps)


def build_relevant_tests(
    localization: LocalizationResult,
    intelligence: RepoIntelligence,
    evidence_set: EvidenceSet,
    signals: IssueSignals,
) -> list[RelevantTest]:
    """Identify test files and test functions covering the localized code."""
    evidence_map: dict[str, EvidenceReport] = {
        r.path: r for r in evidence_set.reports
    }
    tests_found: dict[str, RelevantTest] = {}

    primary_file = localization.primary_file
    primary_module = _module_name_from_path(primary_file) if primary_file else None

    # 1. Tests from primary evidence
    if primary_file and primary_file in evidence_map:
        ev = evidence_map[primary_file]
        for t in ev.related_tests:
            tests_found[t.path] = RelevantTest(
                path=t.path,
                test_symbols=t.test_symbol_names[:8],  # limit to top 8 test functions
                reason=f"Exercises candidate module '{primary_module}' directly via imports.",
            )

    # 2. Tests from related candidates
    for rel in localization.related[:3]:
        if rel.path in evidence_map:
            rel_ev = evidence_map[rel.path]
            for t in rel_ev.related_tests:
                if t.path not in tests_found:
                    rel_module = _module_name_from_path(rel.path)
                    tests_found[t.path] = RelevantTest(
                        path=t.path,
                        test_symbols=t.test_symbol_names[:5],
                        reason=f"Exercises related module '{rel_module}'.",
                    )

    # 3. Match test files or test functions in intelligence that mention issue keywords or symbols
    target_terms: set[str] = set()
    if localization.primary_symbol:
        target_terms.add(localization.primary_symbol.lower())
    for fn in signals.function_names:
        target_terms.add(fn.lower())
    for kw in signals.keywords[:5]:
        if len(kw) > 3:
            target_terms.add(kw.lower())

    if target_terms:
        for fs in intelligence.files:
            if not fs.is_test or fs.path in tests_found:
                continue
            matching_syms: list[str] = []
            for sym in fs.symbols:
                if sym.type in (SymbolType.function, SymbolType.async_function) and sym.name.startswith("test"):
                    sym_lower = sym.name.lower()
                    if any(term in sym_lower for term in target_terms):
                        matching_syms.append(sym.name)

            if matching_syms:
                tests_found[fs.path] = RelevantTest(
                    path=fs.path,
                    test_symbols=matching_syms[:8],
                    reason=f"Contains test cases matching issue terms: {', '.join(matching_syms[:3])}",
                )

    return list(tests_found.values())[:10]  # cap at top 10 relevant tests
