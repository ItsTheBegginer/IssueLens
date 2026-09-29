"""Analysis API endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.github.service import GitHubError
from app.models.analysis import AnalysisRequest, AnalysisResponse
from app.services.pipeline import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
@router.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(body: AnalysisRequest) -> AnalysisResponse:
    """Analyze a GitHub issue and localize the responsible source code."""
    try:
        return await run_pipeline(body.repo_url, body.issue_number)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GitHubError as exc:
        status = exc.status_code or 502
        if status == 404:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if status == 403:
            code = 429 if "rate limit" in str(exc).lower() else 403
            raise HTTPException(status_code=code, detail=str(exc)) from exc
        if status == 422:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during analysis: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {exc}") from exc
