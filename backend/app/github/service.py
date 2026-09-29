"""GitHub API service.

Handles URL parsing, validation, and all GitHub REST API calls.
All I/O is done via httpx (async).
"""
from __future__ import annotations

import os
import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from app.models.github import (
    FetchIssueResponse,
    GitHubUser,
    IssueComment,
    IssueDetail,
    RepoCoords,
    RepoFile,
    RepoMetadata,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"

# Directories / files we never want to traverse or return.
_IGNORED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "coverage",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "site-packages",
    ".eggs",
    "egg-info",
}

_BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".whl", ".pyd",
    ".db", ".sqlite", ".sqlite3",
    ".pyc", ".pyo",
    ".min.js", ".min.css",
    ".lock",   # yarn.lock / poetry.lock can be huge
}

# Soft cap: skip files larger than this (bytes) to avoid flooding LLM context.
_MAX_FILE_SIZE = 200_000  # 200 KB


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def parse_github_url(url: str) -> RepoCoords:
    """Parse a GitHub repository URL into owner and repo.

    Accepts:
        https://github.com/owner/repo
        https://github.com/owner/repo.git
        https://github.com/owner/repo/tree/branch
        github.com/owner/repo  (no scheme)

    Raises:
        ValueError on any invalid / non-GitHub URL.
    """
    url = url.strip()
    if not url:
        raise ValueError("Repository URL must not be empty.")

    # Add scheme if missing so urlparse works
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    if parsed.netloc.lower().rstrip("/") not in ("github.com", "www.github.com"):
        raise ValueError(f"URL is not a GitHub URL: {url!r}")

    # Strip leading slash and split path segments
    parts = [p for p in parsed.path.split("/") if p]

    if len(parts) < 2:
        raise ValueError(
            f"Cannot extract owner/repo from URL: {url!r}. "
            "Expected https://github.com/owner/repo"
        )

    owner = parts[0]
    repo = parts[1].removesuffix(".git")

    _validate_identifier(owner, "owner")
    _validate_identifier(repo, "repository name")

    return RepoCoords(owner=owner, repo=repo)


_IDENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validate_identifier(value: str, label: str) -> None:
    if not value:
        raise ValueError(f"GitHub {label} must not be empty.")
    if not _IDENT_RE.match(value):
        raise ValueError(
            f"Invalid GitHub {label} {value!r}: "
            "only alphanumeric characters, hyphens, underscores, and dots are allowed."
        )


# ---------------------------------------------------------------------------
# GitHub client
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class GitHubError(Exception):
    """Wraps GitHub API errors with a human-readable message."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _raise_for_status(response: httpx.Response, context: str) -> None:
    """Translate httpx HTTP errors into GitHubError."""
    if response.status_code == 200:
        return
    if response.status_code == 403:
        # Could be rate-limit or forbidden
        if "rate limit" in response.text.lower() or response.headers.get("X-RateLimit-Remaining") == "0":
            raise GitHubError(
                "GitHub API rate limit exceeded. "
                "Set GITHUB_TOKEN to increase your rate limit.",
                status_code=403,
            )
        raise GitHubError(f"GitHub API forbidden for {context}.", status_code=403)
    if response.status_code == 404:
        raise GitHubError(f"Not found: {context}.", status_code=404)
    if response.status_code == 422:
        raise GitHubError(
            f"GitHub rejected the request for {context}: {response.text}",
            status_code=422,
        )
    raise GitHubError(
        f"Unexpected GitHub API response {response.status_code} for {context}: {response.text[:200]}",
        status_code=response.status_code,
    )


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

async def fetch_repo_metadata(client: httpx.AsyncClient, coords: RepoCoords) -> RepoMetadata:
    url = f"{GITHUB_API_BASE}/repos/{coords.owner}/{coords.repo}"
    resp = await client.get(url, headers=_build_headers())
    _raise_for_status(resp, f"repository {coords.owner}/{coords.repo}")
    data = resp.json()
    return RepoMetadata(
        full_name=data["full_name"],
        description=data.get("description"),
        default_branch=data.get("default_branch", "main"),
        language=data.get("language"),
        stars=data.get("stargazers_count", 0),
        forks=data.get("forks_count", 0),
        open_issues=data.get("open_issues_count", 0),
        html_url=data["html_url"],
        topics=data.get("topics", []),
    )


async def fetch_issue(
    client: httpx.AsyncClient,
    coords: RepoCoords,
    issue_number: int,
    *,
    include_comments: bool = True,
) -> IssueDetail:
    if issue_number < 1:
        raise ValueError(f"Issue number must be a positive integer, got {issue_number}.")

    url = f"{GITHUB_API_BASE}/repos/{coords.owner}/{coords.repo}/issues/{issue_number}"
    resp = await client.get(url, headers=_build_headers())
    _raise_for_status(resp, f"issue #{issue_number} in {coords.owner}/{coords.repo}")
    data = resp.json()

    # GitHub returns pull-requests via the issues endpoint too — distinguish them.
    if "pull_request" in data:
        raise GitHubError(
            f"#{issue_number} is a pull request, not an issue.",
            status_code=422,
        )

    user_data = data.get("user") or {}
    labels = [lbl["name"] for lbl in (data.get("labels") or []) if lbl.get("name")]

    comment_list: list[IssueComment] = []
    if include_comments and data.get("comments", 0) > 0:
        comment_list = await _fetch_issue_comments(client, coords, issue_number)

    return IssueDetail(
        number=data["number"],
        title=data["title"],
        body=data.get("body"),
        state=data["state"],
        user=GitHubUser(login=user_data.get("login", ""), html_url=user_data.get("html_url")) if user_data else None,
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        html_url=data["html_url"],
        comments=data.get("comments", 0),
        labels=labels,
        comment_list=comment_list,
    )


async def _fetch_issue_comments(
    client: httpx.AsyncClient,
    coords: RepoCoords,
    issue_number: int,
) -> list[IssueComment]:
    url = f"{GITHUB_API_BASE}/repos/{coords.owner}/{coords.repo}/issues/{issue_number}/comments"
    resp = await client.get(url, headers=_build_headers(), params={"per_page": 30})
    _raise_for_status(resp, f"comments for issue #{issue_number}")
    comments = []
    for c in resp.json():
        user_data = c.get("user") or {}
        comments.append(
            IssueComment(
                id=c["id"],
                user=GitHubUser(login=user_data.get("login", ""), html_url=user_data.get("html_url")) if user_data else None,
                body=c.get("body", ""),
                created_at=c["created_at"],
            )
        )
    return comments


async def fetch_repo_file_tree(
    client: httpx.AsyncClient,
    coords: RepoCoords,
    branch: str = "HEAD",
) -> list[RepoFile]:
    """Return a filtered flat list of source files in the repository.

    Uses the Git Trees API (recursive) to avoid many round-trips.
    """
    url = f"{GITHUB_API_BASE}/repos/{coords.owner}/{coords.repo}/git/trees/{branch}"
    resp = await client.get(url, headers=_build_headers(), params={"recursive": "1"})
    _raise_for_status(resp, f"git tree for {coords.owner}/{coords.repo}@{branch}")
    data = resp.json()

    files: list[RepoFile] = []
    for item in data.get("tree", []):
        if item.get("type") != "blob":
            continue
        path: str = item["path"]
        size: int = item.get("size", 0)

        if not _should_include_file(path, size):
            continue

        download_url = (
            f"https://raw.githubusercontent.com/{coords.owner}/{coords.repo}/{branch}/{path}"
        )
        files.append(RepoFile(path=path, size=size, download_url=download_url))

    return files


def _should_include_file(path: str, size: int) -> bool:
    """Return True if this file should be included in analysis."""
    if size > _MAX_FILE_SIZE:
        return False

    parts = path.split("/")
    # Ignore any path component that matches ignored dirs
    for part in parts[:-1]:  # directory components
        if part in _IGNORED_DIRS or part.endswith(".egg-info"):
            return False

    filename = parts[-1]

    # Check extension
    for ext in _BINARY_EXTENSIONS:
        if filename.endswith(ext):
            return False

    return True


# ---------------------------------------------------------------------------
# Orchestrated fetch
# ---------------------------------------------------------------------------

async def fetch_issue_with_context(
    repo_url: str,
    issue_number: int,
) -> FetchIssueResponse:
    """Parse URL, validate, then fetch repo metadata + issue + file tree."""
    coords = parse_github_url(repo_url)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        repo = await fetch_repo_metadata(client, coords)
        issue = await fetch_issue(client, coords, issue_number)
        files = await fetch_repo_file_tree(client, coords, branch=repo.default_branch)

    return FetchIssueResponse(repo=repo, issue=issue, files=files)
