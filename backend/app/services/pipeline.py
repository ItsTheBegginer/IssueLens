"""End-to-end backend analysis pipeline.

Connects all IssueLens components into a single deterministic localization flow:
GitHub issue → issue signals → repository analysis → candidate search → evidence → localization → execution/tests → optional AI explanation.

Zero AI API requirement for core localization. Fully grounded in real repo data.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.analyzers.evidence import collect_evidence
from app.analyzers.execution_path import build_execution_path, build_relevant_tests
from app.analyzers.issue_signals import extract_signals
from app.analyzers.localization import localize
from app.analyzers.repo_intelligence import build_repo_intelligence
from app.github.service import (
    GitHubError,
    fetch_issue,
    fetch_repo_file_tree,
    fetch_repo_metadata,
    parse_github_url,
)
from app.models.analysis import AnalysisResponse
from app.models.localization import UncertaintyReason
from app.search.candidate_search import search_candidates
from app.services.ai_explanation import generate_ai_explanation

logger = logging.getLogger(__name__)


async def run_pipeline(
    repo_url: str,
    issue_number: int,
    *,
    client: Optional[httpx.AsyncClient] = None,
    include_ai_explanation: bool = True,
    max_analyzed_files: int = 150,
) -> AnalysisResponse:
    """Execute the complete end-to-end IssueLens analysis.

    Parameters
    ----------
    repo_url:
        GitHub repository URL, e.g. 'https://github.com/psf/requests'.
    issue_number:
        Positive issue number.
    client:
        Optional existing httpx.AsyncClient (for connection pooling / tests).
    include_ai_explanation:
        Whether to attempt optional AI explanation if configured.
    max_analyzed_files:
        Maximum number of source files to download & analyze in large repos.

    Returns
    -------
    AnalysisResponse
        Structured localization result containing all evidence, execution path,
        and test information.

    Raises
    ------
    ValueError
        On invalid repository URL or issue number.
    GitHubError
        On repository not found (404), issue not found (404), pull request (422),
        or rate limit (403).
    """
    # 1. Parse and validate GitHub URL
    coords = parse_github_url(repo_url)
    if issue_number < 1:
        raise ValueError(f"Issue number must be a positive integer, got {issue_number}.")

    # 2. Fetch repository metadata, issue, and file tree
    async def _fetch_and_run(c: httpx.AsyncClient) -> AnalysisResponse:
        repo = await fetch_repo_metadata(c, coords)
        issue = await fetch_issue(c, coords, issue_number, include_comments=True)
        files = await fetch_repo_file_tree(c, coords, branch=repo.default_branch)

        # 3. Extract deterministic issue signals
        signals = extract_signals(issue)

        # 4. Repository analysis (AST parsing & relationships)
        intelligence = await build_repo_intelligence(
            repo.full_name,
            files,
            signals=signals,
            max_files=max_analyzed_files,
            client=c,
        )

        # 5. Candidate search
        candidate_set = search_candidates(signals, intelligence, top_n=20)

        # 6. Evidence collection
        evidence_set = collect_evidence(candidate_set.candidates, intelligence, signals)

        # 7. Primary localization
        localization = localize(candidate_set, evidence_set, signals)

        # If repo had no analyzable files, provide a clear uncertainty detail
        if intelligence.analyzed_count == 0 and localization.is_uncertain:
            localization.uncertainty_detail = (
                f"Repository '{repo.full_name}' contains no supported source files for AST analysis "
                "(currently supported: Python)."
            )

        # 8. Execution path and relevant tests
        exec_path, exec_steps = build_execution_path(localization, intelligence, evidence_set)
        relevant_tests = build_relevant_tests(localization, intelligence, evidence_set, signals)

        # 9. Optional AI explanation layer (non-blocking)
        ai_explanation: Optional[str] = None
        if include_ai_explanation and not localization.is_uncertain:
            ai_explanation = await generate_ai_explanation(
                issue_title=issue.title,
                signals=signals,
                localization=localization,
                execution_path=exec_path,
            )

        # 10. Assemble complete response
        return AnalysisResponse(
            repo_url=repo_url,
            issue_number=issue_number,
            issue_title=issue.title,
            is_uncertain=localization.is_uncertain,
            uncertainty_reason=localization.uncertainty_reason.value if localization.uncertainty_reason else None,
            uncertainty_detail=localization.uncertainty_detail,
            primary_file=localization.primary_file,
            primary_symbol=localization.primary_symbol,
            primary_symbol_kind=localization.primary_symbol_kind,
            line_range=localization.line_range,
            evidence_score=localization.evidence_score,
            evidence_confidence=localization.evidence_confidence,
            explanation=localization.explanation,
            ai_explanation=ai_explanation,
            related_files=localization.related,
            execution_path=exec_path,
            execution_path_steps=exec_steps,
            relevant_tests=relevant_tests,
            total_files_analyzed=intelligence.analyzed_count,
        )

    if client is not None:
        return await _fetch_and_run(client)
    else:
        async with httpx.AsyncClient(timeout=40.0, follow_redirects=True) as new_client:
            return await _fetch_and_run(new_client)
