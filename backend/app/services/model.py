"""Minimal model interface for IssueLens.

Calls the IBM Bob inference proxy, which exposes an OpenAI-compatible
``POST /v1/chat/completions`` endpoint backed by LiteLLM.

No additional SDK or framework is required — only ``httpx``, which is
already in requirements.txt.

Required environment variables
--------------------------------
BOB_API_KEY  — Inference API key from bob.ibm.com
               (create one at bob.ibm.com → your instance → API keys,
               choose type "Inference")
BOB_API_URL  — Base URL of the Bob LiteLLM proxy, e.g.
               https://<your-instance>.bob.ibm.com
               (do NOT include /v1/chat/completions)

Optional
--------
BOB_MODEL_ID — Model name as listed in the Bob proxy, e.g.
               ibm/granite-3-3-8b-instruct
               (default: ibm/granite-3-3-8b-instruct)

The application raises ``ModelConfigError`` at call time if any required
variable is missing so the failure is clear and immediate.
"""
from __future__ import annotations

import os

import httpx

__all__ = ["ModelConfigError", "generate"]

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "ibm/granite-3-3-8b-instruct"


class ModelConfigError(RuntimeError):
    """Raised when required model credentials/configuration are missing."""


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ModelConfigError(
            f"Required environment variable '{name}' is not set. "
            "Set BOB_API_KEY and BOB_API_URL to use the AI model features."
        )
    return value


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def generate(prompt: str, *, max_new_tokens: int = 512) -> str:
    """Send *prompt* to the configured Bob model and return the generated text.

    Uses the OpenAI-compatible ``POST /v1/chat/completions`` endpoint exposed
    by Bob's LiteLLM inference proxy.

    Parameters
    ----------
    prompt:
        The full prompt to send as the user message.
    max_new_tokens:
        Maximum number of tokens the model may generate.

    Raises
    ------
    ModelConfigError
        If BOB_API_KEY or BOB_API_URL are absent.
    httpx.HTTPStatusError
        On a non-200 response from the inference endpoint.
    httpx.HTTPError
        On network-level failures.
    """
    api_key = _require_env("BOB_API_KEY")
    base_url = _require_env("BOB_API_URL").rstrip("/")
    model_id = os.environ.get("BOB_MODEL_ID", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        if resp.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Bob inference returned HTTP {resp.status_code}: {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
        # OpenAI-compatible response shape:
        # {"choices": [{"message": {"content": "..."}}]}
        return data["choices"][0]["message"]["content"]
