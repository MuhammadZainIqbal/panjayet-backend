"""
Agent roster — the 5 personas, their model priority arrays, routes,
and system prompts. This is the single source of truth.
NEVER hardcode a model ID anywhere else. Always pull from here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentDef:
    name: str
    slug: str
    # For OpenRouter agents: [primary_model, fallback_model]
    # OpenRouter's priority routing fires primary first, falls back automatically.
    models: List[str]
    # "openrouter" | "groq"
    # Groq agents fall back to OpenRouter if Groq itself fails.
    route: str
    persona: str
    # Only relevant for Groq-routed agents: which OpenRouter models to use
    # as a fallback if Groq returns a non-429 failure.
    openrouter_fallback: List[str] = field(default_factory=list)


AGENTS: List[AgentDef] = [
    AgentDef(
        name="The Architect",
        slug="architect",
        models=[
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ],
        route="openrouter",
        persona=(
            "You are The Architect. Analyze this question from a "
            "systems-thinking perspective. Focus on structure, "
            "dependencies, and long-term implications. Be "
            "comprehensive and precise."
        ),
    ),
    AgentDef(
        name="The Pragmatist",
        slug="pragmatist",
        # Groq model: openai/gpt-oss-120b (Llama 4 Maverick was deprecated on Groq in early 2026)
        # Verify current IDs at: GET https://api.groq.com/openai/v1/models
        models=["openai/gpt-oss-120b"],
        route="groq",
        persona=(
            "You are The Pragmatist. Evaluate this purely on "
            "real-world feasibility. What actually works in "
            "practice? What sounds good in theory but fails in "
            "execution? Be blunt and practical."
        ),
        openrouter_fallback=["openai/gpt-oss-20b:free"],
    ),
    AgentDef(
        name="The Contrarian",
        slug="contrarian",
        models=[
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ],
        route="openrouter",
        persona=(
            "You are The Contrarian. Your job is to challenge "
            "the most obvious answer to this question. Steelman "
            "the minority position. Find what everyone else is "
            "ignoring or assuming without justification."
        ),
    ),
    AgentDef(
        name="The Technician",
        slug="technician",
        models=[
            "poolside/laguna-s-2.1:free",
            "cohere/north-mini-code:free",
        ],
        route="openrouter",
        persona=(
            "You are The Technician. Evaluate the technical "
            "specifics. Focus on implementation details, edge "
            "cases, failure modes, and what the technical "
            "reality is versus what is being assumed."
        ),
    ),
    AgentDef(
        name="The Critic",
        slug="critic",
        models=[
            "openai/gpt-oss-20b:free",
            "google/gemma-4-26b-a4b-it:free",
        ],
        route="openrouter",
        persona=(
            "You are The Critic. Identify gaps, missing "
            "context, unstated assumptions, and what has "
            "been left unsaid. What is the most important "
            "thing nobody is addressing about this question?"
        ),
    ),
]

# Lookup helpers
AGENT_BY_SLUG: dict[str, AgentDef] = {a.slug: a for a in AGENTS}
DEFAULT_SLUGS: List[str] = [a.slug for a in AGENTS]
