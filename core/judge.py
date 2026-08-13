"""
Judge — final Gemini Flash synthesis call.
Reads all Round 1 + Round 2 responses and produces a structured JSON report
matching the exact Part 5 contract.

CRITICAL: The judge does NOT blend answers. It MAPS agreement and disagreement.
The Contested Zone is the output.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from models.session import (
    AgentResponse, JudgeReport, PreprocessorOutput, SupervisorOutput,
    VerdictSchema, ConsensusItem, ContestedItem, ContestedSide,
    AttackItem, AgentScorecard,
)
from providers.gemini import gemini
from providers.openrouter import openrouter
from utils.json_parser import extract_and_parse_json

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """You are a research judge evaluating a multi-agent debate.

Your job is NOT to blend these answers or produce a consensus summary.
Your job is to:
1. Map where agents agreed (Consensus Zone)
2. Map where agents disagreed and WHY (Contested Zone)
3. Identify which Round 2 attacks landed vs failed
4. List genuinely open questions nobody resolved
5. Give a final verdict with confidence level

Output ONLY valid JSON matching the schema provided.
Do NOT output any text outside the JSON object.
Do NOT use markdown code fences."""

_SCHEMA_HINT = """
Required JSON schema:
{
  "session_id": "string",
  "query": "string",
  "category": "string",
  "timestamp": "ISO8601 string",
  "total_time_seconds": number,
  "agents_used": ["string"],
  "verdict": {
    "text": "one paragraph — actual judgment, not summary",
    "confidence": "HIGH | MEDIUM | LOW",
    "one_liner": "max 15 words"
  },
  "consensus_zone": [
    {"claim": "string", "agents_agreed": ["string"], "agreement_count": number}
  ],
  "contested_zone": [
    {
      "topic": "string",
      "side_a": {"claim": "string", "agents": ["string"]},
      "side_b": {"claim": "string", "agents": ["string"]},
      "why_it_matters": "string",
      "judge_lean": "side_a | side_b | unresolved"
    }
  ],
  "attacks_landed": [
    {
      "attacker": "string",
      "target": "string",
      "attack_summary": "string",
      "was_defended": boolean,
      "defense_quality": "strong | weak | conceded"
    }
  ],
  "open_questions": ["string"],
  "agent_scorecards": [
    {
      "agent": "string",
      "round1_summary": "string",
      "round1_confidence": number,
      "round2_summary": "string",
      "round2_confidence": number,
      "position_delta": number,
      "key_assumption": "string"
    }
  ]
}
"""


def _build_judge_input(
    query: str,
    pre: PreprocessorOutput,
    r1_responses: List[AgentResponse],
    r2_responses: List[AgentResponse],
    supervisor: SupervisorOutput,
    session_id: str,
    total_time_seconds: float,
) -> str:
    lines = [
        f"QUERY: {query}",
        f"CATEGORY: {pre.category}",
        f"DEBATE FOCUS: {pre.debate_focus}",
        f"ADVERSARIAL PAIR: {', '.join(supervisor.adversarial_pair)}",
        "",
        "═══ ROUND 1 RESPONSES ═══",
    ]
    for r in r1_responses:
        if not r.failed:
            lines.append(f"\n[{r.agent_name}] (confidence: {r.confidence}/10)")
            lines.append(f"ASSUMPTIONS: {'; '.join(r.assumptions)}")
            lines.append(r.content)

    lines.append("\n═══ ROUND 2 RESPONSES ═══")
    for r in r2_responses:
        if not r.failed:
            role = "ATTACK" if r.agent_slug in supervisor.adversarial_pair else "DEFENSE"
            lines.append(f"\n[{r.agent_name}] ({role})")
            lines.append(r.content)

    lines.append(_SCHEMA_HINT)
    lines.append(f"\nFill session_id with: {session_id}")
    lines.append(f"Fill total_time_seconds with: {total_time_seconds:.1f}")
    lines.append(f"Fill timestamp with: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Fill agents_used with: {[r.agent_name for r in r1_responses if not r.failed]}")

    return "\n".join(lines)


async def run_judge(
    query: str,
    pre: PreprocessorOutput,
    r1_responses: List[AgentResponse],
    r2_responses: List[AgentResponse],
    supervisor: SupervisorOutput,
    gemini_key: str,
    total_time_seconds: float,
    session_id: Optional[str] = None,
    openrouter_key: Optional[str] = None,
) -> JudgeReport:
    if not session_id:
        session_id = str(uuid.uuid4())

    user_content = _build_judge_input(
        query, pre, r1_responses, r2_responses, supervisor,
        session_id, total_time_seconds,
    )

    try:
        logger.info("Attempting Judge synthesis via Gemini...")
        content, _ = await gemini.complete(
            system=_JUDGE_SYSTEM,
            user=user_content,
            api_key=gemini_key,
            max_tokens=8192,
            json_mode=True,
        )
    except Exception as e:
        logger.warning(f"Gemini Judge failed: {e}. Falling back to OpenRouter Nemotron.")
        content, _ = await openrouter.complete_with_revolver(
            system=_JUDGE_SYSTEM,
            user=user_content,
            models=["nvidia/nemotron-3-ultra-550b-a55b:free"],
            user_key=openrouter_key,
            max_tokens=16000,
            # Note: json_mode is handled via system prompt for OpenRouter/Nemotron natively
        )

    try:
        data = extract_and_parse_json(content)
    except Exception as exc:
        logger.error(f"Judge returned invalid JSON: {exc}. Raw content: {content[:500]}...")
        # Fallback to an empty/error report instead of crashing the pipeline
        return JudgeReport(
            session_id=session_id,
            query=query,
            category=pre.category,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_time_seconds=total_time_seconds,
            agents_used=[],
            verdict=VerdictSchema(
                text=f"Pipeline error during Judge synthesis: {exc}",
                confidence="LOW",
                one_liner="Error generating verdict."
            ),
            consensus_zone=[],
            contested_zone=[],
            attacks_landed=[],
            open_questions=[],
            agent_scorecards=[]
        )

    # Parse into typed models
    return JudgeReport(
        session_id=data.get("session_id", session_id),
        query=data.get("query", query),
        category=data.get("category", pre.category),
        timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        total_time_seconds=data.get("total_time_seconds", total_time_seconds),
        agents_used=data.get("agents_used", []),
        verdict=VerdictSchema(**data.get("verdict", {"text":"Empty","confidence":"LOW","one_liner":""})),
        consensus_zone=[ConsensusItem(**c) for c in data.get("consensus_zone", [])],
        contested_zone=[
            ContestedItem(
                topic=c.get("topic", "Unknown"),
                side_a=ContestedSide(**c.get("side_a", {"claim":"", "agents":[]})),
                side_b=ContestedSide(**c.get("side_b", {"claim":"", "agents":[]})),
                why_it_matters=c.get("why_it_matters", ""),
                judge_lean=c.get("judge_lean", "unresolved"),
            )
            for c in data.get("contested_zone", [])
        ],
        attacks_landed=[AttackItem(**a) for a in data.get("attacks_landed", [])],
        open_questions=data.get("open_questions", []),
        agent_scorecards=[AgentScorecard(**s) for s in data.get("agent_scorecards", [])],
    )
