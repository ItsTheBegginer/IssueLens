"""Tests for optional AI explanation layer (Task 15).

Verifies:
1. When AI inference is not configured or placeholder, returns None cleanly.
2. When localization is uncertain, returns None.
3. When AI is mocked/configured, returns generated explanation.
4. When AI endpoint errors out, falls back gracefully without breaking localization.
"""
from __future__ import annotations

import pytest

from app.models.issue_signals import IssueSignals
from app.models.localization import LineRange, LocalizationResult
from app.services.ai_explanation import (
    _build_explanation_prompt,
    generate_ai_explanation,
    is_ai_available,
)


def _sample_localization(is_uncertain: bool = False) -> LocalizationResult:
    if is_uncertain:
        return LocalizationResult(
            is_uncertain=True,
            uncertainty_reason="all_weak",
            uncertainty_detail="Insufficient signals",
            explanation="Uncertain localization.",
        )
    return LocalizationResult(
        is_uncertain=False,
        primary_file="requests/models.py",
        primary_symbol="prepare_url",
        primary_symbol_kind="method",
        line_range=LineRange(start=10, end=25),
        evidence_score=0.85,
        evidence_confidence="strong",
        explanation="Direct symbol match in models.py",
    )


def test_is_ai_available_default(monkeypatch):
    """Default placeholder env returns False."""
    monkeypatch.setenv("BOB_API_KEY", "")
    monkeypatch.setenv("BOB_API_URL", "")
    assert is_ai_available() is False

    monkeypatch.setenv("BOB_API_KEY", "dummy_key")
    monkeypatch.setenv("BOB_API_URL", "https://your-instance.bob.ibm.com")
    assert is_ai_available() is False


def test_is_ai_available_valid(monkeypatch):
    """Valid non-placeholder credentials return True."""
    monkeypatch.setenv("BOB_API_KEY", "real_key_123")
    monkeypatch.setenv("BOB_API_URL", "https://prod-inference.bob.cloud")
    assert is_ai_available() is True


@pytest.mark.asyncio
async def test_generate_ai_explanation_uncertain_returns_none():
    """Uncertain localization returns None without calling AI."""
    signals = IssueSignals(title="Bug title")
    loc = _sample_localization(is_uncertain=True)
    res = await generate_ai_explanation("Bug title", signals, loc, [])
    assert res is None


@pytest.mark.asyncio
async def test_generate_ai_explanation_unconfigured_returns_none(monkeypatch):
    """Unconfigured environment returns None cleanly without error."""
    monkeypatch.delenv("BOB_API_KEY", raising=False)
    monkeypatch.delenv("BOB_API_URL", raising=False)
    signals = IssueSignals(title="Bug title")
    loc = _sample_localization(is_uncertain=False)
    res = await generate_ai_explanation("Bug title", signals, loc, ["1. [Target] requests/models.py"])
    assert res is None


@pytest.mark.asyncio
async def test_generate_ai_explanation_mocked_success(monkeypatch):
    """When configured and model responds, returns the generated explanation text."""
    monkeypatch.setenv("BOB_API_KEY", "valid_key")
    monkeypatch.setenv("BOB_API_URL", "https://valid.inference.host")

    async def mock_generate(prompt: str, *, max_new_tokens: int = 512) -> str:
        assert "requests/models.py" in prompt
        assert "prepare_url" in prompt
        return "This issue occurs because prepare_url does not validate empty parameters."

    monkeypatch.setattr("app.services.ai_explanation.generate", mock_generate)

    signals = IssueSignals(title="Bug title", observed_behavior="Fails on empty params")
    loc = _sample_localization(is_uncertain=False)
    res = await generate_ai_explanation(
        "Bug title",
        signals,
        loc,
        ["1. [Target] requests/models.py::prepare_url lines 10-25"],
    )
    assert res is not None
    assert "prepare_url" in res


@pytest.mark.asyncio
async def test_generate_ai_explanation_error_falls_back_gracefully(monkeypatch):
    """When inference fails, returns None instead of crashing."""
    monkeypatch.setenv("BOB_API_KEY", "valid_key")
    monkeypatch.setenv("BOB_API_URL", "https://valid.inference.host")

    async def mock_generate_fail(prompt: str, *, max_new_tokens: int = 512) -> str:
        raise RuntimeError("Inference connection failed")

    monkeypatch.setattr("app.services.ai_explanation.generate", mock_generate_fail)

    signals = IssueSignals(title="Bug title")
    loc = _sample_localization(is_uncertain=False)
    res = await generate_ai_explanation("Bug title", signals, loc, [])
    assert res is None
