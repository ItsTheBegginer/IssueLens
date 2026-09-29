"""Issue understanding agent.

Converts a GitHub issue into structured :class:`IssueUnderstanding` output
by prompting the configured watsonx.ai model and parsing the JSON response.

This component only analyses the *issue text*.  It does not inspect source
files or decide which file or function is responsible.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import ValidationError

from app.models.github import IssueDetail
from app.models.issue_understanding import IssueUnderstanding
from app.services.model import generate

__all__ = ["IssueUnderstandingError", "understand_issue"]


class IssueUnderstandingError(ValueError):
    """Raised when the model response cannot be parsed into IssueUnderstanding."""

    def __init__(self, message: str, raw_response: str = ""):
        super().__init__(message)
        self.raw_response = raw_response


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SCHEMA = """{
  "problem_summary": "<one or two sentence description of the core problem>",
  "expected_behavior": "<what the reporter expected, or null>",
  "observed_behavior": "<what actually happened, or null>",
  "components": ["<module, class, subsystem, or concept name>", "..."],
  "keywords": ["<technical term that might appear in source code or file names>", "..."],
  "error_messages": ["<exact error text, traceback line, or exception name>", "..."]
}"""


def _build_prompt(issue: IssueDetail) -> str:
    """Return the prompt string to send to the model."""
    lines: list[str] = []

    lines.append("You are a software engineering assistant.")
    lines.append(
        "Analyse the GitHub issue below and extract structured information "
        "that will help locate the relevant source code."
    )
    lines.append("")
    lines.append("## Issue")
    lines.append(f"Title: {issue.title}")

    if issue.labels:
        lines.append(f"Labels: {', '.join(issue.labels)}")

    if issue.body and issue.body.strip():
        lines.append("")
        lines.append("Body:")
        lines.append(issue.body.strip())

    if issue.comment_list:
        lines.append("")
        lines.append("## Top comments")
        for comment in issue.comment_list[:3]:  # cap at first 3 comments
            author = comment.user.login if comment.user else "unknown"
            lines.append(f"[{author}]: {comment.body.strip()}")

    lines.append("")
    lines.append("## Task")
    lines.append(
        "Respond with ONLY a JSON object matching this exact schema — "
        "no markdown fences, no explanation, no extra keys:"
    )
    lines.append(_SCHEMA)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_json(text: str) -> str:
    """Return the first JSON object found in *text*.

    The model sometimes wraps its response in a markdown code fence even when
    told not to.  This function strips the fence if present, otherwise returns
    the raw text for direct parsing.
    """
    # Try a fenced block first
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    # Fall back to the first { … } span
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()


def _parse_response(raw: str) -> IssueUnderstanding:
    """Parse *raw* model output into an :class:`IssueUnderstanding`.

    Raises
    ------
    IssueUnderstandingError
        If the text cannot be decoded as JSON or fails Pydantic validation.
    """
    candidate = _extract_json(raw)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise IssueUnderstandingError(
            f"Model response is not valid JSON: {exc}",
            raw_response=raw,
        ) from exc

    if not isinstance(data, dict):
        raise IssueUnderstandingError(
            f"Expected a JSON object, got {type(data).__name__}",
            raw_response=raw,
        )

    try:
        return IssueUnderstanding(**data)
    except (ValidationError, TypeError) as exc:
        raise IssueUnderstandingError(
            f"Model response does not match expected schema: {exc}",
            raw_response=raw,
        ) from exc


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def understand_issue(
    issue: IssueDetail,
    *,
    max_new_tokens: int = 512,
) -> IssueUnderstanding:
    """Analyse *issue* and return structured :class:`IssueUnderstanding`.

    Parameters
    ----------
    issue:
        The GitHub issue to analyse.
    max_new_tokens:
        Token budget passed to the model.  512 is sufficient for the JSON
        output this prompt requests.

    Raises
    ------
    IssueUnderstandingError
        If the model returns a response that cannot be parsed.
    app.services.model.ModelConfigError
        If watsonx credentials are not configured.
    httpx.HTTPError
        On network or HTTP-level failures.
    """
    prompt = _build_prompt(issue)
    raw = await generate(prompt, max_new_tokens=max_new_tokens)
    return _parse_response(raw)
