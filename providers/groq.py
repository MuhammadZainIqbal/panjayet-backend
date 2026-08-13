"""
Groq provider — OpenAI-compatible REST API, direct httpx calls.
Used by The Pragmatist for high-speed Llama inference (500+ tok/s).
Falls back to OpenRouter if Groq itself fails (non-429 errors).
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import httpx

from providers.base import BaseProvider
from utils.retry import RateLimitError, ProviderError

BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    async def complete(
        self,
        system: str,
        user: str,
        api_key: str,
        model: str = "meta-llama/llama-4-maverick-17b-128e",
        max_tokens: int = 1024,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Single attempt against Groq.
        NOTE: Verify the model ID at https://console.groq.com/docs/models
        before first run — Groq IDs don't always match OpenRouter slugs.
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                BASE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )

        if resp.status_code == 429:
            raise RateLimitError(f"Groq 429: {resp.text[:200]}")

        if resp.status_code >= 400:
            raise ProviderError(
                f"Groq HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content, model

    def resolve_key(self, user_key: Optional[str]) -> str:
        key = user_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError(
                "No Groq API key available. "
                "Set GROQ_API_KEY in .env or pass a user key."
            )
        return key


# Module-level singleton
groq_provider = GroqProvider()
