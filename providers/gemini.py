"""
Gemini provider — Google AI Studio REST API, direct httpx calls.
Used by: Pre-Processor + Judge (1,500 req/day free tier).
Model: gemini-2.5-flash
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import httpx

from providers.base import BaseProvider
from utils.retry import RateLimitError, ProviderError

# Direct REST endpoint — no SDK dependency
BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent"
)
DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiProvider(BaseProvider):
    async def complete(
        self,
        system: str,
        user: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
        json_mode: bool = False,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Call Gemini Flash via the REST generateContent endpoint.
        Set json_mode=True for Pre-Processor and Judge calls to
        enforce application/json MIME type in the response.
        """
        url = BASE_URL.format(model=model)

        generation_config: dict = {"maxOutputTokens": max_tokens}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation_config,
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                url,
                json=payload,
                params={"key": api_key},
                headers={"Content-Type": "application/json"},
            )

        if resp.status_code == 429:
            raise RateLimitError(f"Gemini 429: {resp.text[:200]}")

        if resp.status_code >= 400:
            raise ProviderError(
                f"Gemini HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return content, model

    def resolve_key(self, user_key: Optional[str]) -> str:
        key = user_key or os.getenv("GOOGLE_AI_KEY", "")
        if not key:
            raise RuntimeError(
                "No Google AI key available. "
                "Set GOOGLE_AI_KEY in .env or pass a user key."
            )
        return key


# Module-level singleton
gemini = GeminiProvider()
