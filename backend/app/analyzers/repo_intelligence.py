"""Repository Intelligence Engine.

Downloads source files from GitHub and runs the AST analyzer on each one.
Results are assembled into a :class:`RepoIntelligence` object.
"""
from __future__ import annotations

import asyncio

import httpx

from app.analyzers.python_ast import analyze_file
from app.analyzers.relationships import build_relationships
from app.models.intelligence import FileSymbols, RepoIntelligence
from app.github.service import _build_headers

# Only analyze these extensions for now (Python-only phase)
_ANALYZABLE_EXTENSIONS = {".py"}

# Max concurrent downloads
_CONCURRENCY = 10


async def _download_file(
    client: httpx.AsyncClient,
    path: str,
    download_url: str,
) -> tuple[str, str | None]:
    """Return (path, source_text_or_None)."""
    try:
        resp = await client.get(download_url, headers=_build_headers())
        if resp.status_code == 200:
            return path, resp.text
        return path, None
    except Exception:
        return path, None


async def build_repo_intelligence(
    repo_full_name: str,
    files: list,  # list[RepoFile]
    *,
    signals: Optional[object] = None,
    max_files: Optional[int] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> RepoIntelligence:
    """Analyze all Python files in *files* and build a :class:`RepoIntelligence`."""
    # Filter to analyzable files
    analyzable = [f for f in files if any(f.path.endswith(ext) for ext in _ANALYZABLE_EXTENSIONS)]
    
    if max_files and len(analyzable) > max_files:
        # Prioritize files based on signals if available
        def _file_priority(f):
            score = 0
            p_lower = f.path.lower()
            if signals and hasattr(signals, "file_references"):
                for ref in signals.file_references:
                    if ref.lower() in p_lower or p_lower in ref.lower():
                        score += 10
            if signals and hasattr(signals, "keywords"):
                for kw in signals.keywords:
                    if len(kw) > 2 and kw.lower() in p_lower:
                        score += 2
            if not f.path.startswith("tests/") and not "/tests/" in f.path:
                score += 1
            return -score

        analyzable = sorted(analyzable, key=_file_priority)[:max_files]

    skipped = len(files) - len(analyzable)

    sem = asyncio.Semaphore(_CONCURRENCY)
    file_symbols: list[FileSymbols] = []

    async def _fetch_all(c: httpx.AsyncClient):
        async def bounded_download(f):
            async with sem:
                return await _download_file(c, f.path, f.download_url)
        tasks = [bounded_download(f) for f in analyzable]
        return await asyncio.gather(*tasks, return_exceptions=False)

    if client is not None:
        results = await _fetch_all(client)
    else:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as new_client:
            results = await _fetch_all(new_client)

    for path, source in results:
        if source is None:
            file_symbols.append(FileSymbols(path=path, parse_error="Download failed"))
            continue
        fs = analyze_file(path, source)
        file_symbols.append(fs)

    relationships = build_relationships(file_symbols)

    return RepoIntelligence(
        repo_full_name=repo_full_name,
        file_count=len(files),
        analyzed_count=len(analyzable),
        skipped_count=skipped,
        files=file_symbols,
        relationships=relationships,
    )
