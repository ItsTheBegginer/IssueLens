"""Tests for the GitHub service layer.

All GitHub API calls are mocked — no real network I/O.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.github.service import (
    GitHubError,
    _should_include_file,
    fetch_issue,
    fetch_issue_with_context,
    fetch_repo_metadata,
    parse_github_url,
)
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

class TestParseGitHubUrl:
    def test_standard_https_url(self):
        c = parse_github_url("https://github.com/pallets/flask")
        assert c.owner == "pallets"
        assert c.repo == "flask"

    def test_url_with_git_suffix(self):
        c = parse_github_url("https://github.com/pallets/flask.git")
        assert c.repo == "flask"

    def test_url_with_tree_path(self):
        c = parse_github_url("https://github.com/pallets/flask/tree/main/src")
        assert c.owner == "pallets"
        assert c.repo == "flask"

    def test_url_without_scheme(self):
        c = parse_github_url("github.com/owner/repo")
        assert c.owner == "owner"
        assert c.repo == "repo"

    def test_trailing_whitespace_stripped(self):
        c = parse_github_url("  https://github.com/owner/repo  ")
        assert c.owner == "owner"

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="not a GitHub URL"):
            parse_github_url("https://gitlab.com/owner/repo")

    def test_empty_url_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            parse_github_url("")

    def test_missing_repo_raises(self):
        with pytest.raises(ValueError, match="Cannot extract"):
            parse_github_url("https://github.com/only-owner")

    def test_invalid_owner_character_raises(self):
        with pytest.raises(ValueError):
            parse_github_url("https://github.com/owner with spaces/repo")


# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------

class TestShouldIncludeFile:
    def test_python_file_included(self):
        assert _should_include_file("src/app.py", 1000) is True

    def test_node_modules_excluded(self):
        assert _should_include_file("node_modules/express/index.js", 500) is False

    def test_venv_excluded(self):
        assert _should_include_file("venv/lib/python3.11/site.py", 500) is False

    def test_pycache_excluded(self):
        assert _should_include_file("src/__pycache__/module.cpython-311.pyc", 500) is False

    def test_binary_png_excluded(self):
        assert _should_include_file("assets/logo.png", 5000) is False

    def test_large_file_excluded(self):
        assert _should_include_file("data/huge.py", 300_000) is False

    def test_lock_file_excluded(self):
        assert _should_include_file("yarn.lock", 10_000) is False

    def test_dist_excluded(self):
        assert _should_include_file("dist/bundle.js", 5_000) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, data: dict | list) -> httpx.Response:
    return httpx.Response(status_code=status_code, json=data)


REPO_PAYLOAD = {
    "full_name": "pallets/flask",
    "description": "The Python micro framework",
    "default_branch": "main",
    "language": "Python",
    "stargazers_count": 60000,
    "forks_count": 15000,
    "open_issues_count": 5,
    "html_url": "https://github.com/pallets/flask",
    "topics": ["flask", "python"],
}

ISSUE_PAYLOAD = {
    "number": 42,
    "title": "Bug: Something is wrong",
    "body": "This does not work.",
    "state": "open",
    "user": {"login": "bob", "html_url": "https://github.com/bob"},
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-02T00:00:00Z",
    "html_url": "https://github.com/pallets/flask/issues/42",
    "comments": 0,
    "labels": [{"name": "bug"}],
}

GIT_TREE_PAYLOAD = {
    "tree": [
        {"type": "blob", "path": "src/flask/app.py", "size": 5000},
        {"type": "blob", "path": "src/flask/__init__.py", "size": 200},
        {"type": "blob", "path": "node_modules/foo/bar.js", "size": 100},
        {"type": "tree", "path": "src/flask", "size": 0},
    ]
}


# ---------------------------------------------------------------------------
# fetch_repo_metadata
# ---------------------------------------------------------------------------

class TestFetchRepoMetadata:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(200, REPO_PAYLOAD))
        from app.github.service import RepoCoords
        coords = RepoCoords(owner="pallets", repo="flask")
        meta = await fetch_repo_metadata(mock_client, coords)
        assert meta.full_name == "pallets/flask"
        assert meta.stars == 60000
        assert meta.default_branch == "main"

    @pytest.mark.asyncio
    async def test_not_found_raises(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(404, {"message": "Not Found"}))
        from app.github.service import RepoCoords
        coords = RepoCoords(owner="nobody", repo="ghost")
        with pytest.raises(GitHubError) as exc_info:
            await fetch_repo_metadata(mock_client, coords)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_rate_limit_raises(self):
        mock_client = AsyncMock()
        resp = httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0"},
        )
        mock_client.get = AsyncMock(return_value=resp)
        from app.github.service import RepoCoords
        coords = RepoCoords(owner="owner", repo="repo")
        with pytest.raises(GitHubError) as exc_info:
            await fetch_repo_metadata(mock_client, coords)
        assert "rate limit" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# fetch_issue
# ---------------------------------------------------------------------------

class TestFetchIssue:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(200, ISSUE_PAYLOAD))
        from app.github.service import RepoCoords
        coords = RepoCoords(owner="pallets", repo="flask")
        issue = await fetch_issue(mock_client, coords, 42, include_comments=False)
        assert issue.number == 42
        assert issue.title == "Bug: Something is wrong"
        assert issue.state == "open"
        assert "bug" in issue.labels

    @pytest.mark.asyncio
    async def test_invalid_issue_number(self):
        mock_client = AsyncMock()
        from app.github.service import RepoCoords
        coords = RepoCoords(owner="pallets", repo="flask")
        with pytest.raises(ValueError, match="positive integer"):
            await fetch_issue(mock_client, coords, 0)

    @pytest.mark.asyncio
    async def test_issue_not_found(self):
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(404, {"message": "Not Found"}))
        from app.github.service import RepoCoords
        coords = RepoCoords(owner="pallets", repo="flask")
        with pytest.raises(GitHubError) as exc_info:
            await fetch_issue(mock_client, coords, 9999, include_comments=False)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_pull_request_rejected(self):
        pr_payload = {**ISSUE_PAYLOAD, "pull_request": {"url": "..."}}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=_make_response(200, pr_payload))
        from app.github.service import RepoCoords
        coords = RepoCoords(owner="pallets", repo="flask")
        with pytest.raises(GitHubError, match="pull request"):
            await fetch_issue(mock_client, coords, 42, include_comments=False)


# ---------------------------------------------------------------------------
# API endpoint — integration (TestClient, mocks GitHub service)
# ---------------------------------------------------------------------------

class TestFetchIssueEndpoint:
    def test_valid_request(self):
        mock_response = MagicMock()
        mock_response.repo = MagicMock(
            full_name="pallets/flask",
            description="Flask",
            default_branch="main",
            language="Python",
            stars=60000,
            forks=15000,
            open_issues=5,
            html_url="https://github.com/pallets/flask",
            topics=["python"],
        )
        mock_response.issue = MagicMock(
            number=42,
            title="A bug",
            body="Details",
            state="open",
            user=None,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            html_url="https://github.com/pallets/flask/issues/42",
            comments=0,
            labels=[],
            comment_list=[],
        )
        mock_response.files = []

        with patch(
            "app.api.github.fetch_issue_with_context",
            new=AsyncMock(return_value=mock_response),
        ):
            resp = client.post(
                "/github/fetch-issue",
                json={"repo_url": "https://github.com/pallets/flask", "issue_number": 42},
            )
        assert resp.status_code == 200

    def test_invalid_url_returns_422(self):
        with patch(
            "app.api.github.fetch_issue_with_context",
            new=AsyncMock(side_effect=ValueError("not a GitHub URL")),
        ):
            resp = client.post(
                "/github/fetch-issue",
                json={"repo_url": "https://gitlab.com/owner/repo", "issue_number": 1},
            )
        assert resp.status_code == 422

    def test_repo_not_found_returns_404(self):
        with patch(
            "app.api.github.fetch_issue_with_context",
            new=AsyncMock(side_effect=GitHubError("Not found.", status_code=404)),
        ):
            resp = client.post(
                "/github/fetch-issue",
                json={"repo_url": "https://github.com/nobody/ghost", "issue_number": 1},
            )
        assert resp.status_code == 404

    def test_github_error_returns_502(self):
        with patch(
            "app.api.github.fetch_issue_with_context",
            new=AsyncMock(side_effect=GitHubError("Server error", status_code=500)),
        ):
            resp = client.post(
                "/github/fetch-issue",
                json={"repo_url": "https://github.com/owner/repo", "issue_number": 1},
            )
        assert resp.status_code == 502

    def test_missing_fields_returns_422(self):
        resp = client.post("/github/fetch-issue", json={"repo_url": "https://github.com/owner/repo"})
        assert resp.status_code == 422
