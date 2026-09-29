"""Optional AI Explanation Layer for IssueLens.

Provides a model-powered explanation around the deterministic evidence
already produced by the IssueLens pipeline.

Grounded entirely in deterministic evidence — never invents code locations.
If no confirmed AI inference endpoint is configured or reachable, falls back
gracefully without blocking the core application.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from app.models.analysis import ExecutionPathStep
from app.models.evidence import EvidenceReport
from app.models.issue_signals import IssueSignals
from app.models.localization import LocalizationResult
from app.services.model import ModelConfigError, generate

logger = logging.getLogger(__name__)


def is_ai_available() -> bool:
    """Return True if valid AI inference credentials appear to be configured."""
    api_key = os.environ.get("BOB_API_KEY", "").strip()
    api_url = os.environ.get("BOB_API_URL", "").strip()

    if not api_key or not api_url:
        return False

    # Guard against default placeholder URLs in .env.example
    if "your-instance" in api_url or "example.com" in api_url:
        return False

    return True


def _build_explanation_prompt(
    issue_title: str,
    signals: IssueSignals,
    localization: LocalizationResult,
    execution_path: list[str],
) -> str:
    """Construct a strictly grounded prompt explaining existing evidence."""
    lines: list[str] = [
        "You are an expert software engineer reviewing a bug localization report.",
        "Your task is to explain to a developer why the identified location was flagged as the likely root cause.",
        "",
        "CRITICAL RULES:",
        "1. Do NOT invent new file paths, function names, line numbers, or evidence.",
        "2. Base your explanation ONLY on the provided issue summary, primary candidate, evidence, and execution path.",
        "3. Keep the explanation concise, technical, and developer-friendly (2-3 paragraphs max).",
        "",
        "## Issue",
        f"Title: {issue_title}",
    ]

    if signals.observed_behavior:
        lines.append(f"Observed: {signals.observed_behavior}")
    if signals.expected_behavior:
        lines.append(f"Expected: {signals.expected_behavior}")
    if signals.error_messages:
        lines.append(f"Error messages: {', '.join(signals.error_messages[:3])}")

    lines.extend([
        "",
        "## Deterministic Localization",
        f"Primary File: {localization.primary_file}",
        f"Primary Symbol: {localization.primary_symbol or 'None'}",
        f"Symbol Kind: {localization.primary_symbol_kind or 'N/A'}",
        f"Line Range: {localization.line_range.start}–{localization.line_range.end}" if localization.line_range else "Line Range: N/A",
        f"Confidence Score: {localization.evidence_score:.2f} ({localization.evidence_confidence or 'moderate'})",
        f"Deterministic Grounding: {localization.explanation}",
    ])

    if execution_path:
        lines.extend([
            "",
            "## Execution / Dependency Flow",
            *execution_path[:5],
        ])

    if localization.related:
        lines.extend([
            "",
            "## Related Candidates",
            *[f"- {r.path}: {r.reason}" for r in localization.related[:3]],
        ])

    lines.extend([
        "",
        "## Developer Explanation:",
    ])

    return "\n".join(lines)


async def generate_ai_explanation(
    issue_title: str,
    signals: IssueSignals,
    localization: LocalizationResult,
    execution_path: list[str],
) -> Optional[str]:
    """Generate an optional developer-friendly explanation using the configured model.

    Returns None if:
    - Localization is uncertain
    - Model credentials are not configured or are placeholder values
    - Inference call fails or times out
    """
    if localization.is_uncertain or not localization.primary_file:
        return None

    if not is_ai_available():
        return None

    try:
        prompt = _build_explanation_prompt(
            issue_title=issue_title,
            signals=signals,
            localization=localization,
            execution_path=execution_path,
        )
        explanation = await generate(prompt, max_new_tokens=400)
        return explanation.strip() if explanation else None
    except (ModelConfigError, Exception) as exc:
        logger.warning("AI explanation failed or unavailable: %s", exc)
        return None
