"""Tests for the model service layer.

All network I/O is mocked — no real API calls.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.model import ModelConfigError, generate, _require_env


# ---------------------------------------------------------------------------
# _require_env
# ---------------------------------------------------------------------------

class TestRequireEnv:
    def test_present_variable_returned(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "test-key")
        assert _require_env("BOB_API_KEY") == "test-key"

    def test_missing_variable_raises(self, monkeypatch):
        monkeypatch.delenv("BOB_API_KEY", raising=False)
        with pytest.raises(ModelConfigError, match="BOB_API_KEY"):
            _require_env("BOB_API_KEY")

    def test_empty_variable_raises(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "   ")
        with pytest.raises(ModelConfigError, match="BOB_API_KEY"):
            _require_env("BOB_API_KEY")


# ---------------------------------------------------------------------------
# generate — missing credentials
# ---------------------------------------------------------------------------

class TestGenerateMissingConfig:
    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("BOB_API_KEY", raising=False)
        monkeypatch.setenv("BOB_API_URL", "https://bob.example.com")
        with pytest.raises(ModelConfigError, match="BOB_API_KEY"):
            await generate("hello")

    @pytest.mark.asyncio
    async def test_missing_api_url_raises(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "key-abc")
        monkeypatch.delenv("BOB_API_URL", raising=False)
        with pytest.raises(ModelConfigError, match="BOB_API_URL"):
            await generate("hello")


# ---------------------------------------------------------------------------
# generate — successful round-trip (mocked HTTP)
# ---------------------------------------------------------------------------

class TestGenerateSuccess:
    @pytest.mark.asyncio
    async def test_returns_generated_text(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "key-abc")
        monkeypatch.setenv("BOB_API_URL", "https://bob.example.com")

        chat_response = MagicMock(spec=httpx.Response)
        chat_response.status_code = 200
        chat_response.json.return_value = {
            "choices": [{"message": {"content": "The answer is 42."}}]
        }

        mock_post = AsyncMock(return_value=chat_response)

        with patch("app.services.model.httpx.AsyncClient") as mock_client_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.post = mock_post
            mock_client_cls.return_value = mock_ctx

            result = await generate("What is the answer?")

        assert result == "The answer is 42."

    @pytest.mark.asyncio
    async def test_correct_endpoint_called(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "key-abc")
        monkeypatch.setenv("BOB_API_URL", "https://bob.example.com")

        chat_response = MagicMock(spec=httpx.Response)
        chat_response.status_code = 200
        chat_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        mock_post = AsyncMock(return_value=chat_response)

        with patch("app.services.model.httpx.AsyncClient") as mock_client_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.post = mock_post
            mock_client_cls.return_value = mock_ctx

            await generate("hello")

        called_url = mock_post.call_args[0][0]
        assert called_url == "https://bob.example.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_bearer_token_sent(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "my-secret-key")
        monkeypatch.setenv("BOB_API_URL", "https://bob.example.com")

        chat_response = MagicMock(spec=httpx.Response)
        chat_response.status_code = 200
        chat_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        mock_post = AsyncMock(return_value=chat_response)

        with patch("app.services.model.httpx.AsyncClient") as mock_client_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.post = mock_post
            mock_client_cls.return_value = mock_ctx

            await generate("hello")

        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret-key"

    @pytest.mark.asyncio
    async def test_non_200_raises_http_status_error(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "key-abc")
        monkeypatch.setenv("BOB_API_URL", "https://bob.example.com")

        bad_response = MagicMock(spec=httpx.Response)
        bad_response.status_code = 401
        bad_response.text = "Unauthorized"
        bad_response.request = MagicMock()

        mock_post = AsyncMock(return_value=bad_response)

        with patch("app.services.model.httpx.AsyncClient") as mock_client_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.post = mock_post
            mock_client_cls.return_value = mock_ctx

            with pytest.raises(httpx.HTTPStatusError):
                await generate("hello")

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped_from_url(self, monkeypatch):
        monkeypatch.setenv("BOB_API_KEY", "key-abc")
        monkeypatch.setenv("BOB_API_URL", "https://bob.example.com/")

        chat_response = MagicMock(spec=httpx.Response)
        chat_response.status_code = 200
        chat_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        mock_post = AsyncMock(return_value=chat_response)

        with patch("app.services.model.httpx.AsyncClient") as mock_client_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value.post = mock_post
            mock_client_cls.return_value = mock_ctx

            await generate("hello")

        called_url = mock_post.call_args[0][0]
        assert called_url == "https://bob.example.com/v1/chat/completions"
