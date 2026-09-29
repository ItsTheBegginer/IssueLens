from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class RepoCoords(BaseModel):
    """Parsed owner + repo from a GitHub URL."""
    owner: str
    repo: str


class GitHubUser(BaseModel):
    login: str
    html_url: Optional[str] = None


class IssueComment(BaseModel):
    id: int
    user: Optional[GitHubUser] = None
    body: str
    created_at: str


class IssueDetail(BaseModel):
    number: int
    title: str
    body: Optional[str] = None
    state: str
    user: Optional[GitHubUser] = None
    created_at: str
    updated_at: str
    html_url: str
    comments: int
    labels: list[str] = []
    comment_list: list[IssueComment] = []


class RepoFile(BaseModel):
    path: str
    size: int  # bytes
    download_url: Optional[str] = None


class RepoMetadata(BaseModel):
    full_name: str
    description: Optional[str] = None
    default_branch: str
    language: Optional[str] = None
    stars: int
    forks: int
    open_issues: int
    html_url: str
    topics: list[str] = []


class FetchIssueRequest(BaseModel):
    repo_url: str
    issue_number: int


class FetchIssueResponse(BaseModel):
    repo: RepoMetadata
    issue: IssueDetail
    files: list[RepoFile]
