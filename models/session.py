"""
Pydantic models for the full Panjayet session lifecycle.
Covers: inbound request, per-agent responses, supervisor output,
judge report, and the complete JSON schema contract from Part 5.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Inbound ─────────────────────────────────────────────────────

class APIKeys(BaseModel):
    """
    User-supplied API keys, decrypted in the browser before sending.
    Each field is optional — if absent, the server falls back to its
    own environment keys (OPENROUTER_KEYS / GROQ_API_KEY / GOOGLE_AI_KEY).
    """
    openrouter: Optional[str] = None
    groq: Optional[str] = None
    gemini: Optional[str] = None


class SessionRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=4000)
    api_keys: Optional[APIKeys] = None
    # Optional subset — pass agent slugs to restrict which agents run.
    # None = run all 5 defaults.
    selected_agents: Optional[List[str]] = None


# ─── Pre-Processor ───────────────────────────────────────────────

class PreprocessorOutput(BaseModel):
    category: str
    depth_level: str            # surface | medium | deep
    debate_focus: str
    devil_advocate_angle: str
    confidence_required: str    # low | medium | high
    estimated_complexity: float   # 1–10


# ─── Per-Agent ───────────────────────────────────────────────────

class AgentResponse(BaseModel):
    agent_slug: str
    agent_name: str
    content: str
    confidence: float             # 1–10, parsed from CONFIDENCE: line
    assumptions: List[str]      # parsed from ASSUMPTIONS: line
    model_used: str
    elapsed_ms: int
    failed: bool = False
    error: Optional[str] = None


# ─── Supervisor ──────────────────────────────────────────────────

class SupervisorOutput(BaseModel):
    # Jaccard similarity scores: lower = more divergent from the group
    disagreement_scores: Dict[str, float]
    adversarial_pair: List[str]     # [slug_a, slug_b] — highest disagreers
    observers: List[str]            # remaining 3 slugs
    low_confidence_agents: List[str]


# ─── Judge Report — exact Part 5 contract ────────────────────────

class VerdictSchema(BaseModel):
    text: str
    confidence: str     # HIGH | MEDIUM | LOW
    one_liner: str


class ConsensusItem(BaseModel):
    claim: str
    agents_agreed: List[str]
    agreement_count: float


class ContestedSide(BaseModel):
    claim: str
    agents: List[str]


class ContestedItem(BaseModel):
    topic: str
    side_a: ContestedSide
    side_b: ContestedSide
    why_it_matters: str
    judge_lean: str     # side_a | side_b | unresolved


class AttackItem(BaseModel):
    attacker: str
    target: str
    attack_summary: str
    was_defended: bool
    defense_quality: str    # strong | weak | conceded


class AgentScorecard(BaseModel):
    agent: str
    round1_summary: str
    round1_confidence: float
    round2_summary: str
    round2_confidence: float
    position_delta: float   # negative = retreated, positive = doubled down
    key_assumption: str


class JudgeReport(BaseModel):
    session_id: str
    query: str
    category: str
    timestamp: str          # ISO8601
    total_time_seconds: float
    agents_used: List[str]
    verdict: VerdictSchema
    consensus_zone: List[ConsensusItem]
    contested_zone: List[ContestedItem]
    attacks_landed: List[AttackItem]
    open_questions: List[str]
    agent_scorecards: List[AgentScorecard]


# ─── SSE Event Payloads ──────────────────────────────────────────

class SSEEvent(BaseModel):
    event: str
    data: Any
