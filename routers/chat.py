"""
Gatekeeper Router — the semantic triage layer.

POST /chat
  - Accepts {query, chat_history, api_keys}
  - Runs the Gatekeeper LLM (ultralight model) to classify intent
  - If is_worth_fighting_for: false → streams standard_chat_reply token-by-token (SSE)
  - If is_worth_fighting_for: true  → emits a single SSE 'escalate' event with primary_conflict

Model priority:
  1. Groq key present → llama-3.1-8b-instant (fastest, sub-1s latency)
  2. OpenRouter key   → nvidia/nemotron-nano-9b-v2:free
  Fallback on any error → degrade to standard chat mode, inform the user.

SSE event contract (client must handle):
  event: token          → data: {"text": "<chunk>"}
  event: chat_done      → data: {"full_reply": "<full text>"}
  event: escalate       → data: {"primary_conflict": "<str>", "confidence_score": 0.95}
  event: degraded       → data: {"reason": "<str>"}  (gatekeeper failed, chat mode only)
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

import httpx
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.crypto import resolve_openrouter_key, resolve_groq_key
from models.session import APIKeys

logger = logging.getLogger(__name__)
router = APIRouter()

GROQ_BASE   = "https://api.groq.com/openai/v1/chat/completions"
OR_BASE     = "https://openrouter.ai/api/v1/chat/completions"
SITE_URL    = "https://panjayet.app"
SITE_NAME   = "Panjayet"

GATEKEEPER_MODEL_GROQ = "llama-3.1-8b-instant"
GATEKEEPER_MODEL_OR   = "nvidia/nemotron-nano-9b-v2:free"


# ─── Pydantic Models ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []
    api_keys: Optional[APIKeys] = None


class IntentClassification(BaseModel):
    is_worth_fighting_for: bool
    confidence_score: float
    primary_conflict: Optional[str] = None
    standard_chat_reply: Optional[str] = None


# ─── Gatekeeper System Prompt ────────────────────────────────────

GATEKEEPER_SYSTEM = """You are the Panjayet Gatekeeper. Your sole function is to triage incoming queries with brutal precision.

Evaluate the FULL conversation history and the latest query. Decide immediately:

ESCALATE (is_worth_fighting_for: true) ONLY for:
- High-stakes architectural or engineering trade-offs with multiple valid conflicting positions
- Highly contested socio-technical paradigms (e.g., AI replacing jobs, microservices vs monolith at scale)
- Philosophical or ethical dilemmas with no single correct answer
- Scientific or geopolitical debates where domain experts genuinely disagree
- Strategic decisions where organizational context meaningfully changes the answer

REJECT (is_worth_fighting_for: false) for:
- Factual lookups with a single correct answer
- Simple debugging or how-to questions
- Subjective fluff ("what's the best X") with no real stakes
- Basic greetings or meta questions about the system
- Opinion requests that don't warrant adversarial pressure-testing

IMPORTANT CONTEXT AWARENESS:
- A conversation may START simple and ESCALATE. Re-evaluate the entire history.
- If the latest query builds on prior chat into complex territory, escalate now.

Respond with ONLY valid JSON. No markdown. No explanation. No preamble.

Schema:
{
  "is_worth_fighting_for": bool,
  "confidence_score": float (0.0–1.0),
  "primary_conflict": "string describing the core debate axis" | null,
  "standard_chat_reply": "your helpful conversational reply" | null
}

Rules:
- If is_worth_fighting_for is true:  fill primary_conflict, set standard_chat_reply to null
- If is_worth_fighting_for is false: fill standard_chat_reply, set primary_conflict to null
- Keep standard_chat_reply concise, direct, and genuinely useful (2-4 sentences max)
"""


# ─── Helpers ─────────────────────────────────────────────────────

def _build_messages(history: List[ChatMessage], latest_query: str) -> list:
    """Convert chat history + latest query into OpenAI message format."""
    msgs = [{"role": "system", "content": GATEKEEPER_SYSTEM}]
    for msg in history[-10:]:  # Cap at last 10 turns to prevent context overflow
        msgs.append({"role": msg.role, "content": msg.content})
    msgs.append({"role": "user", "content": latest_query})
    return msgs


async def _classify_intent(
    messages: list,
    groq_key: Optional[str],
    or_key: Optional[str],
) -> IntentClassification:
    """
    Call the ultralight gatekeeper model and parse the JSON response.
    Priority: Groq → OpenRouter.
    Raises on both failing.
    """
    if not groq_key and not or_key:
        raise ValueError("NO_KEYS")

    payload_base = {
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1,  # Low temp for deterministic classification
    }

    last_error_code = None

    # --- Try Groq first (fastest) ---
    if groq_key:
        try:
            payload = {**payload_base, "model": GATEKEEPER_MODEL_GROQ}
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    GROQ_BASE,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json",
                    },
                )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"]
                if not raw:
                    raise ValueError("CONTENT_FILTER")
                return IntentClassification.model_validate_json(raw.strip())
            
            last_error_code = resp.status_code
            if resp.status_code == 429:
                raise ValueError("RATE_LIMIT")
            elif resp.status_code in (400, 403):
                raise ValueError("CONTENT_FILTER")
            
            logger.warning("Groq gatekeeper HTTP %s — trying OpenRouter", resp.status_code)
        except ValueError:
            raise
        except Exception as e:
            logger.warning("Groq gatekeeper failed: %s — trying OpenRouter", e)

    # --- Try OpenRouter ---
    if or_key:
        try:
            payload = {
                **payload_base,
                "model": GATEKEEPER_MODEL_OR,
            }
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    OR_BASE,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {or_key}",
                        "HTTP-Referer": SITE_URL,
                        "X-Title": SITE_NAME,
                        "Content-Type": "application/json",
                    },
                )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"].get("content", "")
                if not raw:
                    raise ValueError("CONTENT_FILTER")
                return IntentClassification.model_validate_json(raw.strip())
            
            last_error_code = resp.status_code
            if resp.status_code == 429:
                raise ValueError("RATE_LIMIT")
            elif resp.status_code in (400, 403):
                raise ValueError("CONTENT_FILTER")
                
            logger.warning("OpenRouter gatekeeper HTTP %s", resp.status_code)
        except ValueError:
            raise
        except Exception as e:
            logger.warning("OpenRouter gatekeeper failed: %s", e)

    if last_error_code == 401:
        raise ValueError("INVALID_KEY")

    raise RuntimeError("All gatekeeper providers failed.")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─── Route ───────────────────────────────────────────────────────

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Semantic router. Classifies intent then either:
    - Streams chat reply token-by-token (SSE)
    - Signals escalation to the Panjayet pipeline
    """
    groq_key = resolve_groq_key(request.api_keys.groq if request.api_keys else None)
    or_key   = resolve_openrouter_key(request.api_keys.openrouter if request.api_keys else None)

    messages = _build_messages(request.chat_history, request.query)

    async def event_stream():
        # ── Step 1: Classify intent ───────────────────────────────
        try:
            intent = await _classify_intent(messages, groq_key, or_key)
        except ValueError as e:
            err_type = str(e)
            if err_type == "NO_KEYS":
                reply = "I cannot process your request. Please configure an API key (Groq or OpenRouter) in the settings first."
            elif err_type == "RATE_LIMIT":
                reply = "Your API key limit seems to be exhausted. Please recharge or renew your keys to continue."
            elif err_type == "INVALID_KEY":
                reply = "The provided API key is invalid or unauthorized. Please check your settings."
            elif err_type == "CONTENT_FILTER":
                reply = "I cannot process that query. Please avoid controversial, unsafe, or filtered content."
            else:
                reply = "An unexpected configuration error occurred."
            
            yield _sse("degraded", {"reason": "API Key or Policy Error"})
            for word in reply.split(" "):
                yield _sse("token", {"text": word + " "})
                import asyncio
                await asyncio.sleep(0.04)
            yield _sse("chat_done", {"full_reply": reply})
            return
        except Exception as e:
            logger.error("Gatekeeper classification failed entirely: %s", e)
            # Graceful degradation for server crashes
            reply = "I'm running in degraded mode right now. The semantic routing layer is unavailable, but I can still help directly. What would you like to know?"
            yield _sse("degraded", {"reason": "System degraded: running in standard chat mode."})
            for word in reply.split(" "):
                yield _sse("token", {"text": word + " "})
                import asyncio
                await asyncio.sleep(0.04)
            yield _sse("chat_done", {"full_reply": reply})
            return

        # ── Step 2: Route ─────────────────────────────────────────
        if intent.is_worth_fighting_for:
            # Escalate — fire the Panjayet pipeline signal
            yield _sse("escalate", {
                "primary_conflict": intent.primary_conflict or request.query,
                "confidence_score": intent.confidence_score,
            })
            return

        # ── Step 3: Stream chat reply token-by-token ──────────────
        reply = intent.standard_chat_reply or "I'm not sure how to answer that."

        # Simulate streaming by yielding word-by-word chunks
        # This gives the "alive" feel without requiring a streaming LLM call
        # for the gatekeeper layer (which must be fast and cheap)
        words = reply.split(" ")
        full_text = ""
        chunk = ""
        for i, word in enumerate(words):
            chunk += word + (" " if i < len(words) - 1 else "")
            # Yield chunks of ~3-5 words for natural rhythm
            if (i + 1) % 4 == 0 or i == len(words) - 1:
                yield _sse("token", {"text": chunk})
                full_text += chunk
                chunk = ""
                # Small async yield to prevent blocking
                import asyncio
                await asyncio.sleep(0.04)

        yield _sse("chat_done", {"full_reply": reply})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
