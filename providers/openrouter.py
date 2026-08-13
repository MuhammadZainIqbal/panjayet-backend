"""
OpenRouter provider — with the Revolver Magazine key rotation.

Key rotation logic:
  1. On HTTP 429, rotate to the next server key and retry.
  2. OpenRouter itself handles model-level fallback via the models[] array.
  3. So we handle key-level 429s here; model-level failures are handled by OR.

The module-level `server_revolver` is initialized once from OPENROUTER_KEYS
on first use. User-supplied keys bypass the revolver and are used directly.
"""
from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Tuple

import httpx

from providers.base import BaseProvider
from utils.retry import RateLimitError, ProviderError

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
SITE_URL = "https://panjayet.app"
SITE_NAME = "Panjayet"


# ─── Revolver Magazine ───────────────────────────────────────────

class KeyRevolver:
    """
    Thread-safe, async-safe round-robin key rotation.
    On 429, call rotate() to advance to the next key.
    """

    def __init__(self, keys: List[str]) -> None:
        if not keys:
            raise ValueError("KeyRevolver requires at least one API key.")
        self._keys = keys
        self._index = 0
        self._lock = asyncio.Lock()

    async def get_next(self) -> str:
        """Return the current key and advance the index proactively."""
        async with self._lock:
            key = self._keys[self._index % len(self._keys)]
            self._index = (self._index + 1) % len(self._keys)
            return key

    def __len__(self) -> int:
        return len(self._keys)


_server_revolver: Optional[KeyRevolver] = None


def get_server_revolver() -> KeyRevolver:
    global _server_revolver
    if _server_revolver is None:
        raw = os.getenv("OPENROUTER_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise RuntimeError(
                "No OPENROUTER_KEYS found in environment. "
                "Add at least one key to your .env file."
            )
        _server_revolver = KeyRevolver(keys)
    return _server_revolver


# ─── Provider ────────────────────────────────────────────────────

class OpenRouterProvider(BaseProvider):
    """
    Calls OpenRouter with a model priority array.
    OpenRouter routes: primary model first, fallback if primary unavailable.
    We handle key-level 429s by rotating the server revolver.
    """

    async def complete(
        self,
        system: str,
        user: str,
        api_key: str,
        models: Optional[List[str]] = None,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Tuple[str, str]:
        """Single attempt. Raises RateLimitError on 429."""
        if not models:
            raise ValueError("OpenRouterProvider.complete() requires models list.")

        payload = {
            "models": models,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                BASE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME,
                    "Content-Type": "application/json",
                },
            )

        if resp.status_code == 429:
            raise RateLimitError(f"OpenRouter 429: {resp.text[:200]}")

        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        model_used = data.get("model", models[0])
        return content, model_used

    async def complete_with_revolver(
        self,
        system: str,
        user: str,
        models: List[str],
        user_key: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> Tuple[str, str]:
        """
        Complete with key rotation on 429.
        - If user_key provided: check if comma-separated and rotate, else use directly.
        - If not: use the server revolver.
        """
        if user_key:
            user_keys_list = [k.strip() for k in user_key.split(",") if k.strip()]
            if not user_keys_list:
                # Fallback to server revolver if empty after stripping
                pass
            elif len(user_keys_list) == 1:
                return await self.complete(system, user, user_keys_list[0], models, max_tokens)
            else:
                user_revolver = KeyRevolver(user_keys_list)
                max_rot = len(user_revolver)
                for attempt in range(max_rot):
                    key = await user_revolver.get_next()
                    try:
                        return await self.complete(system, user, key, models, max_tokens)
                    except RateLimitError:
                        if attempt < max_rot - 1:
                            continue
                        raise


        revolver = get_server_revolver()
        max_rotations = len(revolver)

        for attempt in range(max_rotations):
            key = await revolver.get_next()
            try:
                return await self.complete(system, user, key, models, max_tokens)
            except RateLimitError:
                if attempt < max_rotations - 1:
                    continue
                raise  # All keys exhausted — propagate


# Module-level singleton
openrouter = OpenRouterProvider()
