"""
test_pipeline.py -- Panjayet Week 2 Integration Test
====================================================
Simulates the frontend. No pytest required. Run directly with Python.

Tests (in order):
  [1] SUPERVISOR UNIT TEST     -- offline, pure Python, no network
  [2] PREPROCESSOR JSON FORMAT -- live Gemini call, checks for fence hallucination
  [3] REVOLVER MAGAZINE UNIT   -- mocks a 429 and asserts key rotation
  [4] FULL PIPELINE SSE STREAM -- hits the live server, prints events in real-time

Prerequisites:
  - Server must be running:  uvicorn main:app --reload --port 8000
  - .env must be populated with real keys
  - Run from inside panjayet-backend/ with the venv active

Usage:
  .venv/Scripts/python.exe test_pipeline.py
  .venv/Scripts/python.exe test_pipeline.py --section 1   (run only one section)
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
from typing import Optional
from unittest.mock import AsyncMock, patch

# Force UTF-8 output on Windows terminals (cp1252 can't handle box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Path setup so we can import backend modules directly ─────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── ANSI color codes ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

BASE_URL   = "http://127.0.0.1:8000"
# Hardcoded research question -- picked to guarantee genuine agent disagreement
TEST_QUERY = (
    "Is remote work net positive or net negative for long-term "
    "organizational productivity and innovation?"
)

RESULTS: dict[str, str] = {}   # section_name → PASS / FAIL / WARN


# ── Helpers ──────────────────────────────────────────────────────

def header(title: str) -> None:
    bar = "─" * 60
    print(f"\n{BOLD}{CYAN}{bar}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{bar}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}·{RESET}  {msg}")


def event_line(event: str, data: dict) -> None:
    tag = f"{CYAN}[{event}]{RESET}"
    # Pretty-print the payload -- truncate long values
    short = {
        k: (v[:120] + "..." if isinstance(v, str) and len(v) > 120 else v)
        for k, v in data.items()
    }
    print(f"  {tag} {json.dumps(short, indent=None, ensure_ascii=False)}")


def assert_server_up() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            ok(f"Server is up -- {r.json()}")
            return True
        fail(f"Health check returned {r.status_code}")
        return False
    except httpx.ConnectError:
        fail(
            f"Cannot reach {BASE_URL}. "
            "Start the server first:\n"
            "    .venv\\Scripts\\uvicorn.exe main:app --reload --port 8000"
        )
        return False


# ════════════════════════════════════════════════════════════════
# TEST 1 -- SUPERVISOR UNIT TEST (offline, no API)
# ════════════════════════════════════════════════════════════════

def test_supervisor() -> None:
    header("TEST 1/4 -- SUPERVISOR UNIT TEST (offline)")

    from core.supervisor import run_supervisor, _tokenize, _jaccard
    from models.session import AgentResponse

    # --- 1a. Tokenizer sanity check ---
    tokens = _tokenize("The quick brown fox. THE QUICK!")
    assert "the" not in tokens, "Stop words should be removed"
    assert "quick" in tokens
    assert "fox" in tokens
    ok("Tokenizer: lowercases, strips punctuation, removes stop words")

    # --- 1b. Jaccard edge cases ---
    assert _jaccard(frozenset(), frozenset()) == 0.0
    assert _jaccard(frozenset({"a"}), frozenset({"a"})) == 1.0
    assert _jaccard(frozenset({"a", "b"}), frozenset({"c", "d"})) == 0.0
    ok("Jaccard: edge cases (empty sets, identical sets, zero-overlap sets)")

    # --- 1c. Pair selection ---
    # Architect and Pragmatist talk about completely different things.
    # The other three agents say nearly the same thing.
    # We expect the adversarial pair to be architect & pragmatist.

    def _make(slug: str, name: str, content: str) -> AgentResponse:
        return AgentResponse(
            agent_slug=slug, agent_name=name, content=content,
            confidence=7, assumptions=[], model_used="test", elapsed_ms=100,
        )

    responses = [
        _make(
            "architect",
            "The Architect",
            "Systems infrastructure dependencies structural long-term architecture "
            "scalability enterprise planning roadmap technical debt reduction",
        ),
        _make(
            "pragmatist",
            "The Pragmatist",
            "Revenue growth quarterly earnings market share customer acquisition "
            "competitive advantage profit margin sales pipeline conversion rate",
        ),
        # The following three say similar things -- should NOT be the pair
        _make(
            "contrarian",
            "The Contrarian",
            "Remote work productivity collaboration teamwork innovation culture "
            "employees engagement retention hybrid office environment",
        ),
        _make(
            "technician",
            "The Technician",
            "Remote work employees productivity collaboration office hybrid "
            "culture retention engagement teamwork innovation",
        ),
        _make(
            "critic",
            "The Critic",
            "Remote work productivity office hybrid employees collaboration "
            "culture retention teamwork engagement innovation results",
        ),
    ]

    result = run_supervisor(responses)

    info(f"Disagreement scores: {result.disagreement_scores}")
    info(f"Adversarial pair selected: {result.adversarial_pair}")
    info(f"Observers: {result.observers}")

    expected_pair = {"architect", "pragmatist"}
    actual_pair   = set(result.adversarial_pair)

    if actual_pair == expected_pair:
        ok(f"Correct adversarial pair identified: {result.adversarial_pair}")
        RESULTS["supervisor"] = "PASS"
    else:
        fail(
            f"Wrong pair selected: {result.adversarial_pair}. "
            f"Expected {list(expected_pair)}."
        )
        warn(
            "This may indicate the stop-word list needs tuning "
            "or the test content overlaps more than expected."
        )
        RESULTS["supervisor"] = "FAIL"

    # --- 1d. Failed agents are excluded from pair selection ---
    failed_response = _make("architect", "The Architect", "")
    failed_response.failed = True
    failed_response.content = ""
    partial = [failed_response] + responses[1:]
    partial_result = run_supervisor(partial)
    assert "architect" not in partial_result.adversarial_pair, (
        "Failed agent should not be in adversarial pair"
    )
    ok("Failed agents correctly excluded from pair selection")


# ════════════════════════════════════════════════════════════════
# TEST 2 -- PREPROCESSOR JSON FORMAT (live API, single call)
# ════════════════════════════════════════════════════════════════

async def test_preprocessor() -> None:
    header("TEST 2/4 -- PREPROCESSOR JSON FORMAT (live Gemini call)")

    gemini_key = os.getenv("GOOGLE_AI_KEY", "")
    if not gemini_key:
        fail("GOOGLE_AI_KEY not set in .env -- skipping preprocessor test.")
        RESULTS["preprocessor"] = "SKIP"
        return

    from core.preprocessor import run_preprocessor, _strip_json_fences
    from providers.gemini import gemini

    info(f"Query: '{TEST_QUERY[:80]}...'")
    info("Calling Gemini Flash directly (json_mode=True)...")

    t0 = time.monotonic()
    # Call the raw provider first so we can inspect the raw response
    try:
        raw_content, model = await gemini.complete(
            system=(
                "You are a research query analyzer. Return ONLY valid JSON. "
                "No preamble. No markdown. No explanation."
            ),
            user=TEST_QUERY,
            api_key=gemini_key,
            max_tokens=512,
            json_mode=True,
        )
        elapsed = time.monotonic() - t0
        info(f"Raw Gemini response received in {elapsed:.2f}s")

        # Check for fence hallucination
        cleaned, was_fenced = _strip_json_fences(raw_content)

        if was_fenced:
            warn(
                "Gemini hallucinated markdown fences (```json...```) despite json_mode=True.\n"
                f"    Raw prefix: {raw_content[:100]!r}\n"
                "    → _strip_json_fences() handled it. No session impact."
            )
            RESULTS["preprocessor_fences"] = "WARN"
        else:
            ok("Gemini returned clean JSON -- no markdown fences. json_mode=True is working.")
            RESULTS["preprocessor_fences"] = "PASS"

        # Now parse the full preprocessor output
        output = await run_preprocessor(TEST_QUERY, gemini_key)
        info(f"category:             {output.category}")
        info(f"depth_level:          {output.depth_level}")
        info(f"debate_focus:         {output.debate_focus}")
        info(f"devil_advocate_angle: {output.devil_advocate_angle}")
        info(f"confidence_required:  {output.confidence_required}")
        info(f"estimated_complexity: {output.estimated_complexity}/10")
        ok("PreprocessorOutput parsed cleanly into Pydantic model")
        RESULTS["preprocessor"] = "PASS"

    except Exception as exc:
        fail(f"Preprocessor test failed: {type(exc).__name__}: {exc}")
        RESULTS["preprocessor"] = "FAIL"


# ════════════════════════════════════════════════════════════════
# TEST 3 -- REVOLVER MAGAZINE UNIT TEST (mock 429, no real API call)
# ════════════════════════════════════════════════════════════════

async def test_revolver() -> None:
    header("TEST 3/4 -- REVOLVER MAGAZINE UNIT TEST (mocked 429s)")

    from providers.openrouter import KeyRevolver, OpenRouterProvider
    from utils.retry import RateLimitError

    # --- 3a. Basic rotation ---
    rev = KeyRevolver(["key-A", "key-B", "key-C"])
    assert await rev.current() == "key-A"
    await rev.rotate()
    assert await rev.current() == "key-B"
    await rev.rotate()
    assert await rev.current() == "key-C"
    await rev.rotate()
    assert await rev.current() == "key-A"   # wraps around
    ok("KeyRevolver: round-robin rotation works correctly, wraps around")

    # --- 3b. Single-key revolver (edge case) ---
    single = KeyRevolver(["only-key"])
    assert await single.current() == "only-key"
    await single.rotate()
    assert await single.current() == "only-key"   # still the only key
    ok("KeyRevolver: single-key revolver stays stable on rotate()")

    # --- 3c. Simulate 429 → rotation → success ---
    rev2 = KeyRevolver(["bad-key", "good-key"])
    provider = OpenRouterProvider()

    call_count = 0
    used_keys: list[str] = []

    async def _fake_complete(self_inner, system, user, api_key, models, max_tokens=1024):
        nonlocal call_count
        used_keys.append(api_key)
        call_count += 1
        # Any key starting with "bad" simulates a 429.
        # This covers both "bad-key" (3c) and "bad-1"/"bad-2" (3d).
        if api_key.startswith("bad"):
            raise RateLimitError(f"Simulated 429 on {api_key}")
        return "This is a valid response from the model.", "test-model"

    # Patch the low-level complete() and inject our two-key revolver
    with patch.object(OpenRouterProvider, "complete", _fake_complete):
        # Temporarily inject our revolver
        import providers.openrouter as or_module
        original_revolver = or_module._server_revolver
        or_module._server_revolver = rev2
        try:
            content, model = await provider.complete_with_revolver(
                system="test",
                user="test",
                models=["test-model:free"],
                user_key=None,   # forces server revolver
            )
        finally:
            or_module._server_revolver = original_revolver

    info(f"Keys tried in order: {used_keys}")

    if used_keys == ["bad-key", "good-key"]:
        ok("Revolver: detected 429 on key-1, rotated to key-2, succeeded")
        ok(f"Response received: {content[:60]!r}")
        RESULTS["revolver"] = "PASS"
    else:
        fail(f"Unexpected key sequence: {used_keys}. Expected ['bad-key', 'good-key']")
        RESULTS["revolver"] = "FAIL"

    # --- 3d. All keys exhausted -- should raise, not silently hang ---
    all_bad_rev = KeyRevolver(["bad-1", "bad-2"])
    or_module._server_revolver = all_bad_rev
    raised = False
    try:
        with patch.object(OpenRouterProvider, "complete", _fake_complete):
            or_module._server_revolver = all_bad_rev
            await provider.complete_with_revolver(
                system="test", user="test",
                models=["test-model:free"], user_key=None,
            )
    except RateLimitError:
        raised = True
    finally:
        or_module._server_revolver = original_revolver

    if raised:
        ok("Revolver: all keys exhausted → raises RateLimitError (not silent hang)")
    else:
        fail("Revolver: all keys exhausted but no exception raised -- silent failure risk")
        RESULTS["revolver"] = "FAIL"


# ════════════════════════════════════════════════════════════════
# TEST 4 -- FULL PIPELINE SSE STREAM (live server + real API keys)
# ════════════════════════════════════════════════════════════════

async def test_full_pipeline() -> None:
    header("TEST 4/4 -- FULL PIPELINE SSE STREAM (live server, real keys)")

    if not assert_server_up():
        RESULTS["pipeline"] = "SKIP"
        return

    # Check at least one key is set
    has_keys = any([
        os.getenv("OPENROUTER_KEYS"),
        os.getenv("GOOGLE_AI_KEY"),
    ])
    if not has_keys:
        fail("No API keys found in .env. Populate OPENROUTER_KEYS and GOOGLE_AI_KEY first.")
        RESULTS["pipeline"] = "SKIP"
        return

    # ── Step 1: POST /session/start ───────────────────────────────
    info(f"Query: {TEST_QUERY!r}")
    info("POSTing to /session/start...")

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{BASE_URL}/session/start",
            json={"query": TEST_QUERY},
        )

    if resp.status_code != 200:
        fail(f"/session/start returned HTTP {resp.status_code}: {resp.text[:200]}")
        RESULTS["pipeline"] = "FAIL"
        return

    session_data = resp.json()
    session_id   = session_data.get("session_id")
    ok(f"Session started -- ID: {session_id}")

    # ── Step 2: GET /session/stream/{id} ─────────────────────────
    print()
    info("Connecting to SSE stream... (this will take 40-75s)")
    print()

    events_received: list[str] = []
    t_stream_start = time.monotonic()
    pipeline_error = False

    try:
        async with httpx.AsyncClient(timeout=None) as stream_client:
            async with stream_client.stream(
                "GET",
                f"{BASE_URL}/session/stream/{session_id}",
                timeout=180.0,
            ) as response:
                event_name: Optional[str] = None
                data_buffer: list[str] = []

                async for line in response.aiter_lines():
                    line = line.strip()

                    if line.startswith("event:"):
                        event_name = line[len("event:"):].strip()

                    elif line.startswith("data:"):
                        data_buffer.append(line[len("data:"):].strip())

                    elif line == "" and event_name:
                        # End of one SSE message
                        raw_data = " ".join(data_buffer)
                        try:
                            payload = json.loads(raw_data) if raw_data else {}
                        except json.JSONDecodeError:
                            payload = {"raw": raw_data}

                        events_received.append(event_name)
                        elapsed = time.monotonic() - t_stream_start
                        ts = f"{DIM}[{elapsed:>6.1f}s]{RESET}"

                        # ── Print each event ──────────────────────
                        if event_name == "preprocessor_done":
                            print(f"\n  {ts} {BOLD}PRE-PROCESSOR DONE{RESET}")
                            event_line(event_name, payload)

                        elif event_name == "agent_r1_done":
                            agent = payload.get("agent_name", "?")
                            conf  = payload.get("confidence", "?")
                            model = payload.get("model_used", "?")
                            print(
                                f"  {ts} {GREEN}R1 DONE{RESET}  "
                                f"{BOLD}{agent:<20}{RESET}  "
                                f"conf={conf}/10  model={DIM}{model}{RESET}"
                            )
                            # Print 2-line summary
                            summary = payload.get("summary", "")
                            if summary:
                                for s_line in summary[:200].split("\n")[:2]:
                                    print(f"              {DIM}{s_line.strip()}{RESET}")

                        elif event_name == "supervisor_done":
                            print(f"\n  {ts} {BOLD}SUPERVISOR DONE{RESET}")
                            pair = payload.get("adversarial_pair", [])
                            scores = payload.get("disagreement_scores", {})
                            print(f"             Adversarial pair: {CYAN}{pair}{RESET}")
                            print(f"             Disagreement scores:")
                            for slug, score in sorted(scores.items(), key=lambda x: -x[1]):
                                bar_len = int(score * 20)
                                bar = "█" * bar_len + "░" * (20 - bar_len)
                                print(f"               {slug:<14} {bar} {score:.4f}")

                        elif event_name == "agent_r2_done":
                            agent = payload.get("agent_name", "?")
                            role  = payload.get("role", "?").upper()
                            role_color = RED if role == "ATTACK" else YELLOW
                            print(
                                f"  {ts} {role_color}R2 {role:<8}{RESET}  "
                                f"{BOLD}{agent}{RESET}"
                            )
                            summary = payload.get("summary", "")
                            if summary:
                                print(f"              {DIM}{summary[:180].strip()}{RESET}")

                        elif event_name == "agent_failed":
                            agent = payload.get("agent_name", "?")
                            err   = payload.get("error", "unknown")
                            print(f"  {ts} {RED}AGENT FAILED{RESET}  {agent}  → {err[:80]}")

                        elif event_name == "judge_done":
                            print(f"\n  {ts} {BOLD}JUDGE DONE{RESET}")
                            verdict = payload.get("verdict", {})
                            one_liner = verdict.get("one_liner", "")
                            confidence = verdict.get("confidence", "")
                            contested = payload.get("contested_zone", [])
                            consensus = payload.get("consensus_zone", [])
                            print(f"             Verdict:  {BOLD}{one_liner}{RESET}")
                            print(f"             Confidence: {confidence}")
                            print(f"             Contested Zone items: {len(contested)}")
                            print(f"             Consensus Zone items: {len(consensus)}")
                            if contested:
                                print(f"             First contested topic: {contested[0].get('topic','')}")

                        elif event_name == "session_complete":
                            total_ms = payload.get("total_time_ms", 0)
                            print(
                                f"\n  {ts} {BOLD}{GREEN}SESSION COMPLETE{RESET}  "
                                f"total={total_ms/1000:.1f}s"
                            )
                            break

                        elif event_name == "session_error":
                            msg = payload.get("message", "unknown error")
                            print(f"\n  {ts} {RED}SESSION ERROR:{RESET} {msg}")
                            pipeline_error = True
                            break

                        # Reset for next message
                        event_name = None
                        data_buffer = []

    except httpx.ReadTimeout:
        fail("SSE stream timed out (180s). The pipeline is hanging somewhere.")
        RESULTS["pipeline"] = "FAIL"
        return
    except Exception as exc:
        fail(f"SSE stream error: {type(exc).__name__}: {exc}")
        RESULTS["pipeline"] = "FAIL"
        return

    # ── Assertions ────────────────────────────────────────────────
    print()
    total_elapsed = time.monotonic() - t_stream_start

    required_events = {
        "preprocessor_done", "agent_r1_done", "supervisor_done",
        "agent_r2_done", "judge_done", "session_complete",
    }
    received_set = set(events_received)
    missing = required_events - received_set

    if pipeline_error:
        fail("Pipeline terminated with session_error.")
        RESULTS["pipeline"] = "FAIL"
    elif missing:
        fail(f"Missing expected events: {missing}")
        RESULTS["pipeline"] = "FAIL"
    elif total_elapsed > 75:
        warn(f"Pipeline completed but took {total_elapsed:.1f}s (spec limit: 75s)")
        RESULTS["pipeline"] = "WARN"
    else:
        ok(f"All required events received. Total time: {total_elapsed:.1f}s (limit: 75s)")
        r1_count = events_received.count("agent_r1_done")
        r2_count = events_received.count("agent_r2_done")
        ok(f"Round 1: {r1_count}/5 agents responded")
        ok(f"Round 2: {r2_count}/5 agents responded")
        RESULTS["pipeline"] = "PASS"


# ════════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ════════════════════════════════════════════════════════════════

def print_summary() -> None:
    header("RESULTS SUMMARY")
    all_passed = True
    labels = {
        "supervisor":         "Supervisor (Jaccard pair selection)",
        "preprocessor":       "Preprocessor (JSON parse)",
        "preprocessor_fences":"Preprocessor (fence hallucination check)",
        "revolver":           "Revolver Magazine (429 key rotation)",
        "pipeline":           "Full Pipeline SSE Stream",
    }
    for key, label in labels.items():
        result = RESULTS.get(key, "-")
        if result == "PASS":
            print(f"  {GREEN}✓  PASS{RESET}  {label}")
        elif result == "FAIL":
            print(f"  {RED}✗  FAIL{RESET}  {label}")
            all_passed = False
        elif result == "WARN":
            print(f"  {YELLOW}⚠  WARN{RESET}  {label}")
        elif result == "SKIP":
            print(f"  {DIM}-  SKIP{RESET}  {label}")
        else:
            print(f"  {DIM}-  N/A {RESET}  {label}")

    print()
    if all_passed and "FAIL" not in RESULTS.values():
        print(f"  {BOLD}{GREEN}All tests passed. API contract verified. Safe to build frontend.{RESET}")
    else:
        print(f"  {BOLD}{RED}One or more tests failed. Fix before proceeding.{RESET}")
    print()


# ════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════

async def main(section: Optional[int]) -> None:
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  PANJAYET PIPELINE TEST -- Week 2 Integration{RESET}")
    print(f"{BOLD}  Query: '{TEST_QUERY[:55]}...'{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")

    run_all = section is None

    if run_all or section == 1:
        try:
            test_supervisor()
        except AssertionError as e:
            fail(f"Assertion failed: {e}")
            RESULTS["supervisor"] = "FAIL"
        except Exception as e:
            fail(f"Unexpected error in supervisor test: {e}")
            RESULTS["supervisor"] = "FAIL"

    if run_all or section == 2:
        await test_preprocessor()

    if run_all or section == 3:
        await test_revolver()

    if run_all or section == 4:
        await test_full_pipeline()

    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panjayet pipeline integration test")
    parser.add_argument(
        "--section", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run only a specific test section (1=supervisor, 2=preprocessor, 3=revolver, 4=pipeline)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.section))
