"""Tests for the issue understanding agent.

All model calls are mocked — no real network I/O.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.issue_understanding import (
    IssueUnderstandingError,
    _build_prompt,
    _extract_json,
    _parse_response,
    understand_issue,
)
from app.models.github import IssueComment, IssueDetail, GitHubUser
from app.models.issue_understanding import IssueUnderstanding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_issue(
    *,
    title: str = "App crashes on startup",
    body: str = "When I run `python app.py` I get a TypeError: 'NoneType' object is not iterable.",
    labels: list[str] | None = None,
    comments: list[IssueComment] | None = None,
) -> IssueDetail:
    return IssueDetail(
        number=1,
        title=title,
        body=body,
        state="open",
        user=GitHubUser(login="reporter"),
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        html_url="https://github.com/owner/repo/issues/1",
        comments=len(comments or []),
        labels=labels or [],
        comment_list=comments or [],
    )


_VALID_RESPONSE = json.dumps({
    "problem_summary": "The app crashes on startup with a TypeError.",
    "expected_behavior": "App starts without errors.",
    "observed_behavior": "TypeError: 'NoneType' object is not iterable.",
    "components": ["startup", "initializer"],
    "keywords": ["startup", "TypeError", "NoneType"],
    "error_messages": ["TypeError: 'NoneType' object is not iterable"],
})


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_title_included(self):
        issue = _make_issue(title="Widget fails on save")
        prompt = _build_prompt(issue)
        assert "Widget fails on save" in prompt

    def test_body_included(self):
        issue = _make_issue(body="Saving a widget raises ValueError.")
        prompt = _build_prompt(issue)
        assert "Saving a widget raises ValueError." in prompt

    def test_labels_included(self):
        issue = _make_issue(labels=["bug", "regression"])
        prompt = _build_prompt(issue)
        assert "bug" in prompt
        assert "regression" in prompt

    def test_comments_included(self):
        comment = IssueComment(
            id=1,
            user=GitHubUser(login="alice"),
            body="I can reproduce this on v2.0.",
            created_at="2024-01-02T00:00:00Z",
        )
        issue = _make_issue(comments=[comment])
        prompt = _build_prompt(issue)
        assert "I can reproduce this on v2.0." in prompt
        assert "alice" in prompt

    def test_only_first_three_comments_included(self):
        comments = [
            IssueComment(
                id=i,
                user=GitHubUser(login=f"user{i}"),
                body=f"comment body {i}",
                created_at="2024-01-01T00:00:00Z",
            )
            for i in range(5)
        ]
        issue = _make_issue(comments=comments)
        prompt = _build_prompt(issue)
        assert "comment body 0" in prompt
        assert "comment body 2" in prompt
        assert "comment body 3" not in prompt
        assert "comment body 4" not in prompt

    def test_schema_included(self):
        prompt = _build_prompt(_make_issue())
        assert "problem_summary" in prompt
        assert "keywords" in prompt
        assert "error_messages" in prompt

    def test_empty_body_handled(self):
        issue = _make_issue(body="")
        prompt = _build_prompt(issue)
        assert "App crashes on startup" in prompt  # title still present

    def test_none_body_handled(self):
        issue = _make_issue(body=None)  # type: ignore[arg-type]
        prompt = _build_prompt(issue)
        assert "problem_summary" in prompt  # schema still present


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_bare_json_returned_as_is(self):
        text = '{"a": 1}'
        assert _extract_json(text) == '{"a": 1}'

    def test_fenced_json_block_extracted(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_json(text) == '{"a": 1}'

    def test_fenced_block_without_language_extracted(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_json(text) == '{"a": 1}'

    def test_json_surrounded_by_prose(self):
        text = 'Here is the answer:\n{"a": 1}\nThat is all.'
        result = _extract_json(text)
        assert result == '{"a": 1}'

    def test_no_json_object_returns_stripped_text(self):
        text = "  no braces here  "
        assert _extract_json(text) == "no braces here"

    def test_multiline_json_extracted(self):
        text = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = _extract_json(text)
        assert json.loads(result) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# _parse_response — valid inputs
# ---------------------------------------------------------------------------

class TestParseResponseValid:
    def test_full_response_parsed(self):
        result = _parse_response(_VALID_RESPONSE)
        assert isinstance(result, IssueUnderstanding)
        assert result.problem_summary == "The app crashes on startup with a TypeError."
        assert result.expected_behavior == "App starts without errors."
        assert result.observed_behavior == "TypeError: 'NoneType' object is not iterable."
        assert "startup" in result.components
        assert "TypeError" in result.keywords
        assert len(result.error_messages) == 1

    def test_null_optional_fields_allowed(self):
        data = {
            "problem_summary": "Something broke.",
            "expected_behavior": None,
            "observed_behavior": None,
            "components": [],
            "keywords": [],
            "error_messages": [],
        }
        result = _parse_response(json.dumps(data))
        assert result.expected_behavior is None
        assert result.observed_behavior is None
        assert result.components == []

    def test_fenced_response_parsed(self):
        fenced = f"```json\n{_VALID_RESPONSE}\n```"
        result = _parse_response(fenced)
        assert result.problem_summary == "The app crashes on startup with a TypeError."

    def test_extra_keys_ignored(self):
        """Pydantic v2 ignores extra keys by default."""
        data = json.loads(_VALID_RESPONSE)
        data["unexpected_key"] = "ignored"
        result = _parse_response(json.dumps(data))
        assert result.problem_summary == "The app crashes on startup with a TypeError."


# ---------------------------------------------------------------------------
# _parse_response — malformed inputs
# ---------------------------------------------------------------------------

class TestParseResponseMalformed:
    def test_empty_string_raises(self):
        with pytest.raises(IssueUnderstandingError, match="not valid JSON"):
            _parse_response("")

    def test_plain_prose_raises(self):
        with pytest.raises(IssueUnderstandingError):
            _parse_response("The model forgot to output JSON.")

    def test_json_array_raises(self):
        with pytest.raises(IssueUnderstandingError, match="JSON object"):
            _parse_response("[1, 2, 3]")

    def test_missing_required_field_raises(self):
        """problem_summary is required; omitting it must raise."""
        data = {
            "expected_behavior": "It works.",
            "components": [],
            "keywords": [],
            "error_messages": [],
        }
        with pytest.raises(IssueUnderstandingError, match="schema"):
            _parse_response(json.dumps(data))

    def test_wrong_type_for_list_field_raises(self):
        data = json.loads(_VALID_RESPONSE)
        data["keywords"] = "not-a-list"
        with pytest.raises(IssueUnderstandingError, match="schema"):
            _parse_response(json.dumps(data))

    def test_raw_response_preserved_on_error(self):
        raw = "this is not json at all"
        with pytest.raises(IssueUnderstandingError) as exc_info:
            _parse_response(raw)
        assert exc_info.value.raw_response == raw

    def test_truncated_json_raises(self):
        with pytest.raises(IssueUnderstandingError, match="not valid JSON"):
            _parse_response('{"problem_summary": "half')


# ---------------------------------------------------------------------------
# understand_issue — integration (mocked generate)
# ---------------------------------------------------------------------------

class TestUnderstandIssue:
    @pytest.mark.asyncio
    async def test_success(self):
        issue = _make_issue()
        with patch(
            "app.agents.issue_understanding.generate",
            new=AsyncMock(return_value=_VALID_RESPONSE),
        ):
            result = await understand_issue(issue)
        assert isinstance(result, IssueUnderstanding)
        assert "TypeError" in result.keywords

    @pytest.mark.asyncio
    async def test_model_returns_fenced_json(self):
        issue = _make_issue()
        fenced = f"```json\n{_VALID_RESPONSE}\n```"
        with patch(
            "app.agents.issue_understanding.generate",
            new=AsyncMock(return_value=fenced),
        ):
            result = await understand_issue(issue)
        assert isinstance(result, IssueUnderstanding)

    @pytest.mark.asyncio
    async def test_malformed_response_raises_understanding_error(self):
        issue = _make_issue()
        with patch(
            "app.agents.issue_understanding.generate",
            new=AsyncMock(return_value="Sorry, I cannot help with that."),
        ):
            with pytest.raises(IssueUnderstandingError):
                await understand_issue(issue)

    @pytest.mark.asyncio
    async def test_generate_called_once(self):
        issue = _make_issue()
        mock_generate = AsyncMock(return_value=_VALID_RESPONSE)
        with patch("app.agents.issue_understanding.generate", new=mock_generate):
            await understand_issue(issue)
        mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_prompt_contains_issue_title(self):
        """Verify the prompt actually passes the issue content to the model."""
        issue = _make_issue(title="Unique title XYZ-987")
        captured_prompts: list[str] = []

        async def capture(prompt: str, **kwargs: object) -> str:
            captured_prompts.append(prompt)
            return _VALID_RESPONSE

        with patch("app.agents.issue_understanding.generate", new=capture):
            await understand_issue(issue)

        assert len(captured_prompts) == 1
        assert "Unique title XYZ-987" in captured_prompts[0]
