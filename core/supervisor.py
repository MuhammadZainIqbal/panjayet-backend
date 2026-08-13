"""
Supervisor — pure Python disagreement scorer. NO LLM calls here.

Algorithm:
  1. Tokenize each agent's Round 1 response.
  2. Lowercase, strip punctuation, remove stop words.
  3. Compute pairwise Jaccard similarity for all agent pairs.
  4. Average the similarity scores per agent (lower avg = more divergent).
  5. Identify the pair with the LOWEST similarity = highest disagreement.
  6. The other 3 agents are observers.

Jaccard similarity: |A ∩ B| / |A ∪ B|
Range: 0.0 (completely different) → 1.0 (identical)
"""
from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, List, Tuple

from models.session import AgentResponse, SupervisorOutput

# ~50 common English stop words — no NLTK, no spaCy
STOP_WORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "its", "be",
    "as", "are", "was", "were", "been", "has", "have", "had",
    "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "shall", "can", "not", "no", "this", "that",
    "these", "those", "i", "we", "you", "he", "she", "they",
    "which", "who", "what", "when", "where", "how", "all", "each",
    "any", "more", "also", "than", "so", "if", "then", "about",
    "up", "out", "into", "just", "because", "while", "although",
})

_PUNCT_RE = re.compile(r"[^\w\s]")


def _tokenize(text: str) -> frozenset:
    """Lowercase → strip punctuation → split → remove stop words → return set."""
    cleaned = _PUNCT_RE.sub("", text.lower())
    tokens = cleaned.split()
    return frozenset(t for t in tokens if t not in STOP_WORDS and len(t) > 1)


def _jaccard(set_a: frozenset, set_b: frozenset) -> float:
    """Jaccard similarity. Returns 0.0 if both sets empty."""
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def run_supervisor(responses: List[AgentResponse]) -> SupervisorOutput:
    """
    Given a list of Round 1 AgentResponse objects, return a SupervisorOutput
    identifying the adversarial pair and observers.

    Handles edge cases:
      - Only 1 or 2 agents available: adversarial pair is those agents.
      - Failed agents are excluded from pair selection.
    """
    # Filter out failed agents
    alive = [r for r in responses if not r.failed]

    if len(alive) < 2:
        # Can't form a pair — treat all as observers
        slugs = [r.agent_slug for r in responses]
        return SupervisorOutput(
            disagreement_scores={s: 0.0 for s in slugs},
            adversarial_pair=[],
            observers=slugs,
            low_confidence_agents=[
                r.agent_slug for r in responses if r.confidence < 5
            ],
        )

    # Tokenize each agent's response
    token_sets: Dict[str, frozenset] = {
        r.agent_slug: _tokenize(r.content) for r in alive
    }

    # Pairwise Jaccard similarity
    pair_scores: Dict[Tuple[str, str], float] = {}
    for slug_a, slug_b in combinations(token_sets.keys(), 2):
        sim = _jaccard(token_sets[slug_a], token_sets[slug_b])
        pair_scores[(slug_a, slug_b)] = sim

    # Per-agent average similarity vs. all others (lower = more divergent)
    avg_similarity: Dict[str, float] = {}
    for slug in token_sets:
        scores = [
            v for (a, b), v in pair_scores.items() if slug in (a, b)
        ]
        avg_similarity[slug] = sum(scores) / len(scores) if scores else 0.0

    # Disagreement score = 1 - avg_similarity (higher = more divergent)
    disagreement_scores = {s: round(1.0 - v, 4) for s, v in avg_similarity.items()}

    # Adversarial pair: the pair with LOWEST similarity score
    most_divergent_pair = min(pair_scores, key=pair_scores.get)
    slug_a, slug_b = most_divergent_pair

    all_alive_slugs = list(token_sets.keys())
    observers = [s for s in all_alive_slugs if s not in (slug_a, slug_b)]

    # Flag agents with self-reported confidence < 5
    low_confidence = [r.agent_slug for r in responses if r.confidence < 5]

    return SupervisorOutput(
        disagreement_scores=disagreement_scores,
        adversarial_pair=[slug_a, slug_b],
        observers=observers,
        low_confidence_agents=low_confidence,
    )
