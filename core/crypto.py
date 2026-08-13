"""
Key resolution utility.
The frontend decrypts API keys in the browser and sends plaintext to the backend.
This module resolves which key to use: user-supplied > server env.
Keys are NEVER logged. NEVER persisted. Used and discarded per request.
"""
from __future__ import annotations
from typing import Optional


def resolve_openrouter_key(user_key: Optional[str]) -> Optional[str]:
    """
    Returns the user-supplied key if present, else None (signals to use the revolver).
    The revolver reads server env keys — caller should call
    openrouter.complete_with_revolver() with user_key=None to engage it.
    """
    return user_key.strip() if user_key and user_key.strip() else None


def resolve_groq_key(user_key: Optional[str]) -> Optional[str]:
    return user_key.strip() if user_key and user_key.strip() else None


def resolve_gemini_key(user_key: Optional[str]) -> Optional[str]:
    return user_key.strip() if user_key and user_key.strip() else None


def sanitize_key(raw: str) -> str:
    """Strip whitespace. Never log the return value."""
    return raw.strip()
