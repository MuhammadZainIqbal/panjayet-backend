"""
Round 1 — Parallel independent agent dispatch via asyncio.gather().
ZERO cross-talk. No agent sees another's answer.
All 5 fire simultaneously. Results are pushed to the SSE queue as they arrive.

Failure policy (from spec):
  - HTTP 429 → retry with backoff (tenacity, 3 attempts).
  - All retries exhausted → mark agent as failed, continue with others.
  - Session with 4/5 agents is still valid.
  - agent_failed SSE event is emitted for failed slots.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import List, Optional

from models.roster import AgentDef, AGENT_BY_SLUG
from models.session import AgentResponse, PreprocessorOutput
from providers.openrouter import openrouter
from providers.groq import groq_provider
from utils.retry import RateLimitError, ProviderError, agent_retry

logger = logging.getLogger(__name__)

_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d+)", re.IGNORECASE)
_ASSUMPTIONS_RE = re.compile(r"ASSUMPTIONS:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def _parse_confidence(text: str) -> int:
    m = _CONFIDENCE_RE.search(text)
    if m:
        val = int(m.group(1))
        return max(1, min(10, val))
    return 5  # neutral default


def _parse_assumptions(text: str) -> List[str]:
    m = _ASSUMPTIONS_RE.search(text)
    if not m:
        return []
    raw = m.group(1).strip()
    # Split on common list separators
    parts = re.split(r"[;,\n]|\d+\.", raw)
    return [p.strip() for p in parts if p.strip()][:2]


def _build_r1_user_prompt(query: str, pre: PreprocessorOutput) -> str:
    return (
        f"{query}\n\n"
        f"Focus your analysis on: {pre.debate_focus}\n\n"
        "End your response with:\n"
        "CONFIDENCE: [1-10]\n"
        "ASSUMPTIONS: [your top 2 unstated assumptions]"
    )


@agent_retry
async def _call_openrouter_agent(
    agent: AgentDef,
    user_prompt: str,
    openrouter_key: Optional[str],
) -> tuple:
    content, model_used = await openrouter.complete_with_revolver(
        system=agent.persona,
        user=user_prompt,
        models=agent.models,
        user_key=openrouter_key,
        max_tokens=4000,
    )
    return content, model_used


@agent_retry
async def _call_groq_agent(
    agent: AgentDef,
    user_prompt: str,
    groq_key: Optional[str],
    openrouter_key: Optional[str],
) -> tuple:
    """Groq primary → OpenRouter fallback on ProviderError."""
    resolved_key = groq_provider.resolve_key(groq_key)
    try:
        content, model = await groq_provider.complete(
            system=agent.persona,
            user=user_prompt,
            api_key=resolved_key,
            model=agent.models[0],
            max_tokens=4000,
        )
        return content, model
    except (ProviderError, Exception) as exc:
        if isinstance(exc, RateLimitError):
            raise  # Let tenacity handle 429s
        # Non-429 Groq failure → fall back to OpenRouter
        logger.warning(
            f"Groq failed for {agent.slug} ({type(exc).__name__}). "
            f"Falling back to OpenRouter."
        )
        if agent.openrouter_fallback:
            content, model_used = await openrouter.complete_with_revolver(
                system=agent.persona,
                user=user_prompt,
                models=agent.openrouter_fallback,
                user_key=openrouter_key,
                max_tokens=4000,
            )
            return content, model_used
        raise ProviderError(f"Groq failed and no OpenRouter fallback for {agent.slug}")


async def _run_single_agent(
    agent: AgentDef,
    user_prompt: str,
    openrouter_key: Optional[str],
    groq_key: Optional[str],
) -> AgentResponse:
    start = time.monotonic()
    last_exc = None
    
    for attempt in range(3):
        try:
            if agent.route == "groq":
                content, model_used = await _call_groq_agent(
                    agent, user_prompt, groq_key, openrouter_key
                )
            else:
                content, model_used = await _call_openrouter_agent(
                    agent, user_prompt, openrouter_key
                )

            if not content or len(content) < 10:
                raise ValueError("Generated content too short or empty (hallucination/truncation)")

            elapsed_ms = int((time.monotonic() - start) * 1000)
            return AgentResponse(
                agent_slug=agent.slug,
                agent_name=agent.name,
                content=content,
                confidence=_parse_confidence(content),
                assumptions=_parse_assumptions(content),
                model_used=model_used,
                elapsed_ms=elapsed_ms,
                failed=False,
            )

        except Exception as exc:
            last_exc = exc
            logger.warning(f"Agent {agent.slug} failed attempt {attempt+1}/3: {exc}")
            await asyncio.sleep(2 ** attempt)

    # Exhausted retries -> strict enforceability hard fail
    raise RuntimeError(f"Agent {agent.slug} failed all 3 attempts. Last error: {last_exc}")


async def run_round1(
    query: str,
    pre: PreprocessorOutput,
    active_slugs: List[str],
    openrouter_key: Optional[str],
    groq_key: Optional[str],
    sse_queue: asyncio.Queue,
) -> List[AgentResponse]:
    """
    Fire all active agents in parallel. Push agent_r1_done or agent_failed
    events to the SSE queue as each one completes.
    Returns all AgentResponse objects (including failed ones).
    """
    user_prompt = _build_r1_user_prompt(query, pre)

    agents = [AGENT_BY_SLUG[slug] for slug in active_slugs if slug in AGENT_BY_SLUG]

    async def _run_and_emit(agent: AgentDef) -> AgentResponse:
        response = await _run_single_agent(
            agent, user_prompt, openrouter_key, groq_key
        )
        if response.failed:
            await sse_queue.put((
                "agent_failed",
                {"agent_name": agent.name, "agent_slug": agent.slug, "error": response.error},
            ))
        else:
            await sse_queue.put((
                "agent_r1_done",
                {
                    "agent_name": agent.name,
                    "agent_slug": agent.slug,
                    "summary": response.content,
                    "confidence": response.confidence,
                    "model_used": response.model_used,
                },
            ))
        return response

    results = await asyncio.gather(*[_run_and_emit(a) for a in agents])
    return list(results)
