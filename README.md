# Panjayet Backend

A stateless, multi-agent adversarial deliberation API. Accepts a research query, routes it through a five-stage LLM pipeline, and streams the entire deliberation as Server-Sent Events. Zero database. Zero auth. You bring the keys.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| Production Server | Gunicorn (`UvicornWorker`) |
| HTTP Client | httpx (async) |
| Validation | Pydantic v2 |
| Retry Logic | tenacity |
| LLM Providers | OpenRouter, Groq, Google Gemini Flash |
| Streaming | Server-Sent Events (SSE) via `StreamingResponse` |
| Deployment | Render (Blueprint via `render.yaml`) |

---

## Architecture

### BYOK and the Revolver Magazine

The service supports two key modes simultaneously.

**Server keys** are loaded from environment variables at startup. `OPENROUTER_KEYS` accepts a comma-separated list. These are loaded into a `KeyRevolver` instance — a thread-safe, async-safe round-robin rotator. On every HTTP `429` from OpenRouter, the revolver advances to the next key and retries. This sustains throughput across multiple concurrent sessions without manual intervention.

**User-supplied keys** arrive per-request in the `api_keys` JSON payload (`openrouter`, `groq`, `gemini`). These keys are resolved in `core/crypto.py` via `resolve_openrouter_key`, `resolve_groq_key`, and `resolve_gemini_key`. A user-supplied key **always takes priority** over the server key. If the user passes multiple comma-separated keys, a per-request `KeyRevolver` is instantiated for that call alone.

The backend stores **no keys** between requests. Every call is independent.

---

### The Five-Stage Pipeline

Triggered by `POST /session/start`. The pipeline runs as a background `asyncio.Task`. The client connects to `GET /session/stream/{session_id}` to consume the live SSE event stream.

```
POST /session/start
       │
       └─► asyncio.Task: _run_pipeline()
                 │
          [1] Preprocessor  ──► event: preprocessor_done
                 │
          [2] Round 1       ──► event: agent_r1_done  (×N, parallel)
                 │                   event: agent_failed (on failure)
          [3] Supervisor    ──► event: supervisor_done
                 │
          [4] Round 2       ──► event: agent_r2_done  (×2, adversarial pair)
                 │
          [5] Judge         ──► event: judge_done
                 │
                 └──────────────► event: session_complete
```

**Stage 1 — Preprocessor** (`core/preprocessor.py`)

Single Gemini Flash call. Classifies the query into a structured `PreprocessorOutput`: category, depth level, debate focus, devil's advocate angle, and estimated complexity. Failure policy: returns a hardcoded default and **never blocks** the pipeline.

**Stage 2 — Round 1** (`core/round1.py`)

All five agents fire simultaneously via `asyncio.gather()`. Zero cross-talk — no agent sees another's response. Each agent runs against its assigned provider (OpenRouter or Groq) with the persona defined in `models/roster.py`. HTTP `429`s are retried up to 3 times with exponential backoff via `tenacity`. Groq agents fall back to OpenRouter on non-`429` failures. A session with 4 of 5 agents is still valid.

**Stage 3 — Supervisor** (`core/supervisor.py`)

Pure Python. Zero LLM calls. Tokenizes each Round 1 response, strips stop words, then computes **pairwise Jaccard similarity** across all agent pairs. The pair with the lowest similarity score (highest lexical divergence) is designated the adversarial pair for Round 2. The remaining agents become observers.

```
Jaccard similarity: |A ∩ B| / |A ∪ B|
Range: 0.0 (completely different) → 1.0 (identical)
Adversarial pair = argmin(pair_scores)
```

**Stage 4 — Round 2** (`core/round2.py`)

Only the adversarial pair responds. Each sees the other's Round 1 output and is instructed to attack, probe for weaknesses, and defend its own position. Observers are not called.

**Stage 5 — Judge** (`core/judge.py`)

Single Gemini Flash call with an 8192-token budget. Reads all Round 1 and Round 2 responses. Does **not** produce a consensus summary. Produces a structured `JudgeReport`: consensus zone, contested zone, attacks landed (with defense quality ratings), open questions, per-agent scorecards, and a final verdict with confidence level. Falls back to OpenRouter Nemotron on Gemini failure.

---

### The Gatekeeper (`routers/chat.py`)

`POST /chat` is a separate, lightweight semantic triage layer that fronts the pipeline. It is not part of the five-stage pipeline itself.

On every incoming message, the Gatekeeper calls an ultralight model (Groq `llama-3.1-8b-instant` preferred; OpenRouter `nvidia/nemotron-nano-9b-v2:free` as fallback) with a strict JSON-only system prompt. The model returns an `IntentClassification`:

- `is_worth_fighting_for: false` → streams a standard chat reply token-by-token as SSE `token` events. No pipeline is invoked.
- `is_worth_fighting_for: true` → emits a single `escalate` event. The frontend calls `POST /session/start` and opens the SSE stream.

On Gatekeeper failure, the service degrades gracefully: emits `event: degraded` and falls through to chat-only mode. The session continues.

---

### The Agent Roster (`models/roster.py`)

Five agents with fixed personas and model priority arrays. This is the single source of truth for all model IDs.

| Agent | Route | Primary Model | Fallback |
|---|---|---|---|
| The Architect | OpenRouter | `nvidia/nemotron-3-ultra-550b-a55b:free` | `nvidia/nemotron-3-super-120b-a12b:free` |
| The Pragmatist | Groq | `openai/gpt-oss-120b` | `openai/gpt-oss-20b:free` (OpenRouter) |
| The Contrarian | OpenRouter | `google/gemma-4-26b-a4b-it:free` | `nvidia/nemotron-3-super-120b-a12b:free` |
| The Technician | OpenRouter | `poolside/laguna-s-2.1:free` | `cohere/north-mini-code:free` |
| The Critic | OpenRouter | `openai/gpt-oss-20b:free` | `google/gemma-4-26b-a4b-it:free` |

---

### SSE Event Contract

The full event stream contract for `GET /session/stream/{session_id}`:

```
: connected                     ← heartbeat comment on connect

event: preprocessor_done
data: {"category": "...", "debate_focus": "...", "depth_level": "...", "estimated_complexity": N}

event: agent_r1_done
data: {"agent_name": "...", "agent_slug": "...", "summary": "...", "confidence": N, "model_used": "..."}

event: agent_failed
data: {"agent_name": "...", "agent_slug": "...", "error": "..."}

event: supervisor_done
data: {"adversarial_pair": ["...", "..."], "disagreement_scores": {...}, "observers": [...]}

event: agent_r2_done
data: {"agent_name": "...", "agent_slug": "...", "content": "...", "model_used": "..."}

event: judge_done
data: { <full JudgeReport JSON> }

event: session_complete
data: {"total_time_ms": N}

event: session_error
data: {"message": "..."}
```

The `/chat` SSE contract:

```
event: token          data: {"text": "<chunk>"}
event: chat_done      data: {"full_reply": "<full text>"}
event: escalate       data: {"primary_conflict": "...", "confidence_score": 0.95}
event: degraded       data: {"reason": "..."}
```

All `StreamingResponse` objects set `Cache-Control: no-cache` and `X-Accel-Buffering: no`.

---

### Log Security

A `SecureLogFilter` intercepts all log records at the root logger and masks any `?key=` or `&key=` query parameters with `***MASKED***`. Applied to both the root logger and `httpx` to prevent key leakage in Render's log stream.

---

## Environment Variables

Copy `.env.example` to `.env`. Never commit `.env`.

```env
# OpenRouter — The Revolver Magazine
# Comma-separated list. On HTTP 429, the backend rotates to the next key.
# More keys = higher sustained throughput under concurrent sessions. Minimum: 1.
OPENROUTER_KEYS=sk-or-v1-...

# Groq — High-speed inference for The Pragmatist (Gatekeeper primary)
GROQ_API_KEY=gsk_...

# Google AI Studio — Gemini Flash for Preprocessor and Judge
# Free tier: 1,500 req/day
GOOGLE_AI_KEY=AI...

# CORS allowed origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,https://your-frontend.domain
```

**Notes:**
- `OPENROUTER_KEYS` and `GROQ_API_KEY` are used by the server-side Revolver. If a user supplies their own key in the request body, the server key is bypassed for that request.
- `GOOGLE_AI_KEY` has no Revolver. A single key is expected. The free tier is sufficient for Preprocessor + Judge volume at normal usage.
- In production, set all secrets as **environment variables in the Render dashboard**. Do not use `render.yaml` for actual secret values — the `sync: false` flag in the blueprint marks them as manually-provisioned.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Gatekeeper triage. Returns SSE stream. |
| `POST` | `/session/start` | Starts a pipeline session. Returns `session_id`. |
| `GET` | `/session/stream/{session_id}` | SSE stream for a running session. |
| `GET` | `/health` | Liveness probe. Returns active session count. |
| `GET` | `/docs` | Swagger UI. |
| `GET` | `/redoc` | ReDoc UI. |

All routes are served under the `root_path` of `/panjayet` when deployed behind a reverse proxy.

---

## Local Development

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Copy and populate the environment file
cp .env.example .env

# Run the development server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Deployment

The repository includes a `render.yaml` Blueprint for one-command deployment.

**Steps:**

1. Push this repository to GitHub.
2. In the Render Dashboard: **New → Blueprint** → connect the repository.
3. Render reads `render.yaml` and provisions the web service automatically.
4. Navigate to **Environment** in the service settings and set the following secrets manually (marked `sync: false` in the blueprint):
   - `OPENROUTER_API_KEY`
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`
5. Set `FRONTEND_URL` to the deployed frontend origin (no trailing slash).

**Production start command** (defined in `render.yaml`):

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --timeout 120 --keep-alive 5 --log-level info
```

- `-w 4` — 4 Uvicorn worker processes. Tune to `(2 * CPU cores) + 1`.
- `-k uvicorn.workers.UvicornWorker` — required for async/SSE correctness. Standard sync workers will not handle SSE streams.
- `--timeout 120` — safety net for long-running pipeline sessions. SSE keepalive prevents premature worker termination.
- Health check path: `/health`
