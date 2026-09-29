from fastapi import APIRouter, HTTPException

from app.github.service import GitHubError, fetch_issue_with_context
from app.models.github import FetchIssueRequest, FetchIssueResponse

router = APIRouter(prefix="/github", tags=["github"])


@router.post("/fetch-issue", response_model=FetchIssueResponse)
async def fetch_issue_endpoint(body: FetchIssueRequest) -> FetchIssueResponse:
    """Fetch repository metadata, issue details, and file tree from GitHub."""
    try:
        return await fetch_issue_with_context(body.repo_url, body.issue_number)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GitHubError as exc:
        status = exc.status_code or 502
        # Map GitHub 404 → our 404, others → 502 (bad gateway)
        if status == 404:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if status == 422:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc
