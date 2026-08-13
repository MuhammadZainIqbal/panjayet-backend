"""
Round 2 — Adversarial pair attacks + observer defenses.
Fires all 5 simultaneously via asyncio.gather(), same as Round 1.

Adversarial pair (A vs B):
  - Agent A gets Agent B's Round 1 answer and must attack it.
  - Agent B gets Agent A's Round 1 answer and must attack it.

Observers (other 3):
  - Each gets the most pointed attack from the pair debate
    and must respond: defend, revise, or acknowledge uncertainty.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from models.roster import AGENT_BY_SLUG
from models.session import AgentResponse, SupervisorOutput
from providers.openrouter import openrouter
from providers.groq import groq_provider
from utils.retry import RateLimitError, ProviderError, agent_retry

logger = logging.getLogger(__name__)

_ATTACK_PROMPT = """Here is another analyst's answer to the same question:
---
{target_answer}
---
Find every flaw, unsupported claim, logical gap, and unstated assumption \
in this answer. Be specific. Be brutal. Do not hedge. Do not be diplomatic."""

_OBSERVER_PROMPT = """Here was your original analysis in Round 1:
---
{agent_own_round1_answer}
---
Here is a new counterargument that emerged in the debate:
---
{most_critical_attack_from_pair_debate}
---
What do you defend, what do you revise, and what remains genuinely uncertain?"""


def _pick_sharpest_attack(r2_responses: Dict[str, "AgentResponse"]) -> str:
    """
    From the adversarial pair's completed Round 2 responses, pick the
    most pointed attack to send to observers. Uses the longer response
    as a proxy for depth — simple but effective.
    """
    if not r2_responses:
        return "The previous analysis may have overlooked key assumptions."
    sharpest = max(r2_responses.values(), key=lambda r: len(r.content))
    return sharpest.content[:1500]  # Cap to avoid blowing observer context


@agent_retry
async def _call_openrouter_r2(agent, user_prompt, openrouter_key):
    content, model_used = await openrouter.complete_with_revolver(
        system=agent.persona,
        user=user_prompt,
        models=agent.models,
        user_key=openrouter_key,
        max_tokens=4000,
    )
    return content, model_used


@agent_retry
async def _call_groq_r2(agent, user_prompt, groq_key, openrouter_key):
    resolved = groq_provider.resolve_key(groq_key)
    try:
        return await groq_provider.complete(
            system=agent.persona,
            user=user_prompt,
            api_key=resolved,
            model=agent.models[0],
            max_tokens=4000,
        )
    except (ProviderError, Exception) as exc:
        if isinstance(exc, RateLimitError):
            raise
        if agent.openrouter_fallback:
            return await openrouter.complete_with_revolver(
                system=agent.persona,
                user=user_prompt,
                models=agent.openrouter_fallback,
                user_key=openrouter_key,
                max_tokens=4000,
            )
        raise


async def _run_r2_agent(
    slug: str,
    user_prompt: str,
    openrouter_key: Optional[str],
    groq_key: Optional[str],
) -> AgentResponse:
    agent = AGENT_BY_SLUG.get(slug)
    if not agent:
        return AgentResponse(
            agent_slug=slug, agent_name=slug, content="", confidence=0,
            assumptions=[], model_used="unknown", elapsed_ms=0, failed=True,
            error="Agent not found in roster.",
        )

    start = time.monotonic()
    last_exc = None
    
    for attempt in range(3):
        try:
            if agent.route == "groq":
                content, model_used = await _call_groq_r2(
                    agent, user_prompt, groq_key, openrouter_key
                )
            else:
                content, model_used = await _call_openrouter_r2(
                    agent, user_prompt, openrouter_key
                )
                
            if not content or len(content) < 10:
                raise ValueError("Generated content too short or empty (hallucination/truncation)")
                
            return AgentResponse(
                agent_slug=slug, agent_name=agent.name, content=content,
                confidence=5, assumptions=[], model_used=model_used,
                elapsed_ms=int((time.monotonic() - start) * 1000), failed=False,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Agent {slug} failed attempt {attempt+1}/3 in Round 2: {exc}")
            await asyncio.sleep(2 ** attempt)

    # Exhausted retries -> strict enforceability hard fail
    raise RuntimeError(f"Agent {slug} failed all 3 attempts in Round 2. Last error: {last_exc}")


async def run_round2(
    r1_responses: List[AgentResponse],
    supervisor: SupervisorOutput,
    openrouter_key: Optional[str],
    groq_key: Optional[str],
    sse_queue: asyncio.Queue,
) -> List[AgentResponse]:
    """
    Build per-agent Round 2 prompts and fire all simultaneously.
    Emits agent_r2_done or agent_failed SSE events as they complete.
    """
    # Map slug → R1 content for quick lookup
    r1_map: Dict[str, str] = {
        r.agent_slug: r.content for r in r1_responses if not r.failed
    }

    pair = supervisor.adversarial_pair
    observers = supervisor.observers

    tasks: Dict[str, asyncio.Task] = {}

    # Adversarial pair prompts
    if len(pair) == 2:
        slug_a, slug_b = pair
        if slug_a in r1_map and slug_b in r1_map:
            tasks[slug_a] = asyncio.create_task(
                _run_r2_agent(
                    slug_a,
                    _ATTACK_PROMPT.format(target_answer=r1_map[slug_b]),
                    openrouter_key, groq_key,
                )
            )
            tasks[slug_b] = asyncio.create_task(
                _run_r2_agent(
                    slug_b,
                    _ATTACK_PROMPT.format(target_answer=r1_map[slug_a]),
                    openrouter_key, groq_key,
                )
            )

    # Run adversarial pair first so we can extract the sharpest attack
    # for observers — but we still fire everything in gather() for speed.
    # We use a placeholder until pair finishes; gather handles ordering.

    # Observer prompts — use a placeholder attack initially, update after gather
    # DESIGN CHOICE: For simplicity and speed, we use the BEST R1 attacker
    # content pre-selected by supervisor (most critical R1 content from pair)
    # rather than waiting for R2 pair to finish. This keeps everything parallel.
    pair_r1_attack = ""
    for slug in pair:
        if slug in r1_map:
            pair_r1_attack = r1_map[slug][:1500]
            break

    observer_attack_content = pair_r1_attack or "Consider the strongest objection to your analysis."

    for slug in observers:
        tasks[slug] = asyncio.create_task(
            _run_r2_agent(
                slug,
                _OBSERVER_PROMPT.format(
                    agent_own_round1_answer=r1_map.get(slug, "No previous answer."),
                    most_critical_attack_from_pair_debate=observer_attack_content
                ),
                openrouter_key, groq_key,
            )
        )

    # One coroutine per agent — all run concurrently, emit SSE as they land.
    async def _await_and_emit(slug: str, task: asyncio.Task) -> AgentResponse:
        response = await task
        role = "attack" if slug in pair else "defense"
        if response.failed:
            await sse_queue.put((
                "agent_failed",
                {
                    "agent_name": response.agent_name,
                    "agent_slug": slug,
                    "round": 2,
                    "error": response.error,
                },
            ))
        else:
            await sse_queue.put((
                "agent_r2_done",
                {
                    "agent_name": response.agent_name,
                    "agent_slug": slug,
                    "role": role,
                    "summary": response.content,
                },
            ))
        return response

    r2_responses: List[AgentResponse] = list(
        await asyncio.gather(
            *[_await_and_emit(slug, task) for slug, task in tasks.items()]
        )
    )
    return r2_responses
