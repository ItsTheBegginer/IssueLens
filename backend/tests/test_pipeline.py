"""End-to-end backend pipeline tests for IssueLens.

Tests the full flow:
GitHub issue → issue signals → repository analysis → candidate search → evidence → localization → execution/tests.

Validates normal localization, edge cases, and failure scenarios.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.github.service import GitHubError
from app.main import app
from app.models.analysis import AnalysisResponse
from app.services.pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Test Fixtures & Mock Sources
# ---------------------------------------------------------------------------

MODELS_PY_SOURCE = '''"""Request models."""

class PreparedRequest:
    def __init__(self):
        self.url = None

    def prepare_url(self, url, params):
        """Prepare given HTTP URL."""
        if not url:
            raise ValueError("URL cannot be empty")
        self.url = url
        return self.url
'''

SESSIONS_PY_SOURCE = '''"""Session handler."""
from requests.models import PreparedRequest

class Session:
    def send(self, request):
        prep = PreparedRequest()
        return prep.prepare_url(request.url, {})
'''

TEST_MODELS_PY_SOURCE = '''"""Tests for models."""
from requests.models import PreparedRequest

def test_prepare_url():
    p = PreparedRequest()
    assert p.prepare_url("https://example.com", {}) == "https://example.com"

def test_prepare_url_empty():
    p = PreparedRequest()
    try:
        p.prepare_url("", {})
    except ValueError:
        pass
'''


def _mock_transport_factory(
    *,
    repo_status: int = 200,
    issue_status: int = 200,
    is_pr: bool = False,
    is_empty_issue: bool = False,
    is_rate_limited: bool = False,
    empty_tree: bool = False,
    no_python_files: bool = False,
    syntax_error_file: bool = False,
):
    """Create an httpx.MockTransport simulating GitHub API and raw content downloads."""

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)

        # 1. Rate limit simulation
        if is_rate_limited:
            return httpx.Response(
                403,
                json={"message": "API rate limit exceeded"},
                headers={"X-RateLimit-Remaining": "0"},
            )

        # 2. Repo metadata
        if "/repos/testowner/testrepo" in url_str and not any(k in url_str for k in ["/issues", "/git/trees"]):
            if repo_status != 200:
                return httpx.Response(repo_status, json={"message": "Not Found"})
            return httpx.Response(
                200,
                json={
                    "full_name": "testowner/testrepo",
                    "description": "A test repository",
                    "default_branch": "main",
                    "language": "Python",
                    "stargazers_count": 42,
                    "forks_count": 5,
                    "open_issues_count": 3,
                    "html_url": "https://github.com/testowner/testrepo",
                    "topics": ["http", "client"],
                },
            )

        # 3. Issue endpoint
        if "/issues/100/comments" in url_str:
            return httpx.Response(200, json=[])

        if "/issues/100" in url_str:
            if issue_status != 200:
                return httpx.Response(issue_status, json={"message": "Not Found"})
            payload = {
                "number": 100,
                "title": "Minimal title" if is_empty_issue else "Error in prepare_url when passing invalid url params",
                "body": "" if is_empty_issue else "Calling `prepare_url` in `requests/models.py` fails with ValueError on invalid input.",
                "state": "open",
                "user": {"login": "reporter", "html_url": "https://github.com/reporter"},
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "html_url": "https://github.com/testowner/testrepo/issues/100",
                "comments": 0,
                "labels": [{"name": "bug"}],
            }
            if is_pr:
                payload["pull_request"] = {"url": "https://api.github.com/repos/testowner/testrepo/pulls/100"}
            return httpx.Response(200, json=payload)

        # 4. Git Trees endpoint
        if "/git/trees/main" in url_str:
            if empty_tree:
                return httpx.Response(200, json={"tree": []})
            if no_python_files:
                return httpx.Response(
                    200,
                    json={
                        "tree": [
                            {"type": "blob", "path": "README.md", "size": 100},
                            {"type": "blob", "path": "package.json", "size": 200},
                        ]
                    },
                )
            tree_items = [
                {"type": "blob", "path": "requests/models.py", "size": len(MODELS_PY_SOURCE)},
                {"type": "blob", "path": "requests/sessions.py", "size": len(SESSIONS_PY_SOURCE)},
                {"type": "blob", "path": "tests/test_models.py", "size": len(TEST_MODELS_PY_SOURCE)},
            ]
            if syntax_error_file:
                tree_items.append({"type": "blob", "path": "broken.py", "size": 50})
            return httpx.Response(200, json={"tree": tree_items})

        # 5. Raw file downloads
        if "raw.githubusercontent.com" in url_str:
            if "requests/models.py" in url_str:
                return httpx.Response(200, text=MODELS_PY_SOURCE)
            if "requests/sessions.py" in url_str:
                return httpx.Response(200, text=SESSIONS_PY_SOURCE)
            if "tests/test_models.py" in url_str:
                return httpx.Response(200, text=TEST_MODELS_PY_SOURCE)
            if "broken.py" in url_str:
                return httpx.Response(200, text="def unclosed_syntax_error(\n")
            return httpx.Response(404, text="Not found")

        return httpx.Response(404, text="Not Found")

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pipeline_end_to_end_success():
    """Verify complete flow from GitHub issue to primary localization, execution path, and tests."""
    transport = _mock_transport_factory()
    async with httpx.AsyncClient(transport=transport) as client:
        res = await run_pipeline(
            "https://github.com/testowner/testrepo",
            100,
            client=client,
            include_ai_explanation=False,
        )

    assert isinstance(res, AnalysisResponse)
    assert res.repo_url == "https://github.com/testowner/testrepo"
    assert res.issue_number == 100
    assert "prepare_url" in res.issue_title
    assert res.is_uncertain is False
    assert res.primary_file == "requests/models.py"
    assert res.primary_symbol == "prepare_url"
    assert res.primary_symbol_kind in ("function", "method")
    assert res.line_range is not None
    assert res.line_range.start >= 1
    assert res.evidence_score > 0.0
    assert "prepare_url" in res.explanation

    # Verify execution path
    assert len(res.execution_path) >= 1
    assert any("requests/models.py" in p for p in res.execution_path)
    assert any(s.file == "requests/models.py" for s in res.execution_path_steps)

    # Verify relevant tests
    assert len(res.relevant_tests) >= 1
    assert any(t.path == "tests/test_models.py" for t in res.relevant_tests)
    test_obj = next(t for t in res.relevant_tests if t.path == "tests/test_models.py")
    assert "test_prepare_url" in test_obj.test_symbols

    # Verify metadata
    assert res.total_files_analyzed == 3


@pytest.mark.asyncio
async def test_run_pipeline_invalid_repo_url():
    """Invalid repo URL raises ValueError."""
    with pytest.raises(ValueError, match="URL is not a GitHub URL"):
        await run_pipeline("https://notgithub.com/owner/repo", 1)


@pytest.mark.asyncio
async def test_run_pipeline_invalid_issue_number():
    """Issue number < 1 raises ValueError."""
    with pytest.raises(ValueError, match="positive integer"):
        await run_pipeline("https://github.com/owner/repo", 0)


@pytest.mark.asyncio
async def test_run_pipeline_repo_not_found():
    """Non-existent repo raises GitHubError 404."""
    transport = _mock_transport_factory(repo_status=404)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(GitHubError) as exc_info:
            await run_pipeline("https://github.com/testowner/testrepo", 100, client=client)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_run_pipeline_issue_not_found():
    """Non-existent issue raises GitHubError 404."""
    transport = _mock_transport_factory(issue_status=404)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(GitHubError) as exc_info:
            await run_pipeline("https://github.com/testowner/testrepo", 100, client=client)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_run_pipeline_pull_request_rejected():
    """PR numbers return GitHubError 422."""
    transport = _mock_transport_factory(is_pr=True)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(GitHubError) as exc_info:
            await run_pipeline("https://github.com/testowner/testrepo", 100, client=client)
        assert exc_info.value.status_code == 422
        assert "pull request" in str(exc_info.value)


@pytest.mark.asyncio
async def test_run_pipeline_rate_limit_exceeded():
    """GitHub 403 rate limit is translated cleanly."""
    transport = _mock_transport_factory(is_rate_limited=True)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(GitHubError) as exc_info:
            await run_pipeline("https://github.com/testowner/testrepo", 100, client=client)
        assert exc_info.value.status_code == 403
        assert "rate limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_run_pipeline_unsupported_repo_no_python_files():
    """Repositories with 0 Python files return an uncertain response without crashing."""
    transport = _mock_transport_factory(no_python_files=True)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await run_pipeline(
            "https://github.com/testowner/testrepo",
            100,
            client=client,
            include_ai_explanation=False,
        )

    assert res.is_uncertain is True
    assert res.primary_file is None
    assert "no supported source files" in res.uncertainty_detail.lower()
    assert res.total_files_analyzed == 0


@pytest.mark.asyncio
async def test_run_pipeline_empty_issue():
    """Issues with minimal title and empty body do not cause crashes."""
    transport = _mock_transport_factory(is_empty_issue=True)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await run_pipeline(
            "https://github.com/testowner/testrepo",
            100,
            client=client,
            include_ai_explanation=False,
        )

    assert isinstance(res, AnalysisResponse)
    assert res.issue_title == "Minimal title"


@pytest.mark.asyncio
async def test_run_pipeline_handles_syntax_errors():
    """Files with AST syntax errors do not stop analysis of valid files."""
    transport = _mock_transport_factory(syntax_error_file=True)
    async with httpx.AsyncClient(transport=transport) as client:
        res = await run_pipeline(
            "https://github.com/testowner/testrepo",
            100,
            client=client,
            include_ai_explanation=False,
        )

    assert res.is_uncertain is False
    assert res.primary_file == "requests/models.py"


# ---------------------------------------------------------------------------
# FastAPI Endpoint Tests (/analyze)
# ---------------------------------------------------------------------------

def test_api_analyze_endpoint_success(monkeypatch):
    """POST /analyze returns 200 with structured JSON."""
    transport = _mock_transport_factory()

    # Monkeypatch httpx.AsyncClient inside pipeline to use mock transport
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "app.services.pipeline.httpx.AsyncClient",
        lambda *args, **kwargs: orig_async_client(transport=transport, follow_redirects=True),
    )

    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={"repo_url": "https://github.com/testowner/testrepo", "issue_number": 100},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["primary_file"] == "requests/models.py"
    assert data["primary_symbol"] == "prepare_url"
    assert "line_range" in data
    assert "execution_path" in data
    assert "relevant_tests" in data


def test_api_analyze_endpoint_invalid_url():
    """POST /analyze returns 422 for malformed GitHub URLs."""
    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={"repo_url": "not-a-valid-url", "issue_number": 100},
    )
    assert response.status_code == 422
    assert "detail" in response.json()


def test_api_analyze_endpoint_404_repo(monkeypatch):
    """POST /analyze returns 404 if repo doesn't exist."""
    transport = _mock_transport_factory(repo_status=404)
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        "app.services.pipeline.httpx.AsyncClient",
        lambda *args, **kwargs: orig_async_client(transport=transport, follow_redirects=True),
    )

    client = TestClient(app)
    response = client.post(
        "/analyze",
        json={"repo_url": "https://github.com/testowner/testrepo", "issue_number": 100},
    )
    assert response.status_code == 404
