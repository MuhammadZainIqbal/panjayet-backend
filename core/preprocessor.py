"""
Pre-Processor — single Gemini Flash call before Round 1.
Parses the query and returns structured metadata to guide agent prompts.

Failure policy: if this call fails for ANY reason, return a default
PreprocessorOutput and let the session continue. Never block on this.

JSON safety: Gemini Flash occasionally wraps output in markdown fences
(```json ... ```) even when responseMimeType=application/json is set.
_strip_json_fences() handles this defensively and logs a warning.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from models.session import PreprocessorOutput
from providers.gemini import gemini

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research query analyzer. Your job is to classify
and decompose the user's research question to guide a multi-agent debate.

Return ONLY a valid JSON object. No preamble. No markdown. No explanation.
Exact schema:
{
  "category": "string — one-word domain (e.g. economics, ethics, technology)",
  "depth_level": "surface | medium | deep",
  "debate_focus": "the single most contested dimension of this question",
  "devil_advocate_angle": "the strongest counterargument to the most obvious answer",
  "confidence_required": "low | medium | high",
  "estimated_complexity": integer between 1 and 10
}"""

_DEFAULT = PreprocessorOutput(
    category="general",
    depth_level="medium",
    debate_focus="the core assumptions underlying the question",
    devil_advocate_angle="the opposite of what most people assume",
    confidence_required="medium",
    estimated_complexity=5,
)


from utils.json_parser import extract_and_parse_json

async def run_preprocessor(
    query: str,
    gemini_key: str,
) -> PreprocessorOutput:
    """
    Fire a single Gemini Flash call to classify the query.
    Returns _DEFAULT on any failure — session must continue regardless.
    """
    try:
        content, _ = await gemini.complete(
            system=SYSTEM_PROMPT,
            user=query,
            api_key=gemini_key,
            max_tokens=4000,
            json_mode=True,
        )

        data = extract_and_parse_json(content)
        return PreprocessorOutput(**data)

    except Exception as exc:
        logger.warning(f"Pre-processor failed ({type(exc).__name__}: {exc}). Using defaults. Raw: {content[:100] if 'content' in locals() else 'None'}")
        return _DEFAULT
