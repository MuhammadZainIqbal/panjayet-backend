"""
Session router — the main pipeline orchestrator.

Routes:
  POST /session/start   — validates request, kicks off the full pipeline
  GET  /session/stream/{session_id} — SSE stream for live updates
  GET  /health          — liveness probe

Pipeline flow (per spec Part 3):
  preprocessor → round1 → supervisor → round2 → judge → session_complete

asyncio.Queue() buffers all SSE events. The SSE consumer drains it.
This prevents payload collisions when multiple agents finish simultaneously.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.crypto import resolve_openrouter_key, resolve_groq_key, resolve_gemini_key
from core.preprocessor import run_preprocessor
from core.round1 import run_round1
from core.round2 import run_round2
from core.supervisor import run_supervisor
from core.judge import run_judge
from models.roster import DEFAULT_SLUGS
from models.session import SessionRequest
from providers.gemini import gemini
from utils.sse import format_sse, sse_queue_consumer

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session store: session_id → asyncio.Queue
# Stateless per spec — queues are garbage-collected after stream ends.
_session_queues: Dict[str, asyncio.Queue] = {}
_session_tasks: Dict[str, asyncio.Task] = {}


# ─── Pipeline ────────────────────────────────────────────────────

async def _run_pipeline(
    session_id: str,
    request: SessionRequest,
    queue: asyncio.Queue,
) -> None:
    """
    Full pipeline. Runs in a background task.
    Pushes SSE events to the queue throughout.
    Puts None sentinel when done to close the stream.
    """
    start_time = time.monotonic()

    # Key resolution
    or_key = resolve_openrouter_key(
        request.api_keys.openrouter if request.api_keys else None
    )
    groq_key = resolve_groq_key(
        request.api_keys.groq if request.api_keys else None
    )
    gemini_key_raw = resolve_gemini_key(
        request.api_keys.gemini if request.api_keys else None
    )
    gemini_key = gemini.resolve_key(gemini_key_raw)

    active_slugs = request.selected_agents or DEFAULT_SLUGS

    try:
        # ── STEP 1: Pre-Processor ─────────────────────────────────
        pre = await run_preprocessor(request.query, gemini_key)
        await queue.put((
            "preprocessor_done",
            {
                "category": pre.category,
                "debate_focus": pre.debate_focus,
                "depth_level": pre.depth_level,
                "estimated_complexity": pre.estimated_complexity,
            },
        ))

        # ── STEP 2: Round 1 ──────────────────────────────────────
        r1_responses = await run_round1(
            query=request.query,
            pre=pre,
            active_slugs=active_slugs,
            openrouter_key=or_key,
            groq_key=groq_key,
            sse_queue=queue,
        )

        alive_r1 = [r for r in r1_responses if not r.failed]
        if not alive_r1:
            await queue.put((
                "session_error",
                {"message": "All agents failed in Round 1. Cannot continue."},
            ))
            await queue.put(None)
            return

        # ── STEP 3: Supervisor ────────────────────────────────────
        supervisor = run_supervisor(r1_responses)
        await queue.put((
            "supervisor_done",
            {
                "adversarial_pair": supervisor.adversarial_pair,
                "disagreement_scores": supervisor.disagreement_scores,
                "observers": supervisor.observers,
            },
        ))

        # ── STEP 4: Round 2 ──────────────────────────────────────
        r2_responses = await run_round2(
            r1_responses=r1_responses,
            supervisor=supervisor,
            openrouter_key=or_key,
            groq_key=groq_key,
            sse_queue=queue,
        )

        # ── STEP 5: Judge ─────────────────────────────────────────
        total_so_far = time.monotonic() - start_time
        report = await run_judge(
            query=request.query,
            pre=pre,
            r1_responses=r1_responses,
            r2_responses=r2_responses,
            supervisor=supervisor,
            gemini_key=gemini_key,
            total_time_seconds=total_so_far,
            session_id=session_id,
            openrouter_key=or_key,
        )

        total_ms = int((time.monotonic() - start_time) * 1000)
        await queue.put(("judge_done", report.model_dump()))
        await queue.put(("session_complete", {"total_time_ms": total_ms}))

    except Exception as exc:
        logger.exception(f"Pipeline error in session {session_id}: {exc}")
        await queue.put((
            "session_error",
            {"message": f"Pipeline error: {type(exc).__name__}: {str(exc)[:200]}"},
        ))
    finally:
        # Sentinel to close the SSE stream
        await queue.put(None)
        # Clean up queue reference after a delay
        await asyncio.sleep(5)
        _session_queues.pop(session_id, None)


# ─── Routes ──────────────────────────────────────────────────────

@router.post("/session/start")
async def start_session(request: SessionRequest):
    """
    Validate the request, create a session ID and queue,
    kick off the pipeline as a background task,
    return the session ID immediately so the client can open the SSE stream.
    """
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    _session_queues[session_id] = queue

    # Fire pipeline in background — client streams from /session/stream/{id}
    task = asyncio.create_task(_run_pipeline(session_id, request, queue))
    _session_tasks[session_id] = task

    return {"session_id": session_id, "status": "started"}


@router.get("/session/stream/{session_id}")
async def stream_session(session_id: str, request: Request):
    """
    SSE stream for a running session.
    Drains the asyncio.Queue and yields SSE-formatted events until None sentinel.
    """
    queue = _session_queues.get(session_id)
    if queue is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or already completed.",
        )

    async def _monitor_disconnect():
        while True:
            if await request.is_disconnected():
                logger.warning(f"Client disconnected via monitor for session {session_id}. Cancelling tasks.")
                task = _session_tasks.get(session_id)
                if task and not task.done():
                    task.cancel()
                break
            await asyncio.sleep(1)

    async def event_generator():
        monitor_task = asyncio.create_task(_monitor_disconnect())
        try:
            # Yield a comment immediately so the client knows the connection is live
            yield ": connected\n\n"
            async for event_str in sse_queue_consumer(queue, None):
                yield event_str
        except asyncio.CancelledError:
            logger.warning(f"Client disconnected for session {session_id}. Cancelling tasks.")
            task = _session_tasks.get(session_id)
            if task and not task.done():
                task.cancel()
            raise
        finally:
            monitor_task.cancel()
            _session_tasks.pop(session_id, None)
            _session_queues.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "panjayet-api",
        "version": "1.0.0",
        "active_sessions": len(_session_queues),
    }
