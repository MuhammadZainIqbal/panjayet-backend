"""
SSE (Server-Sent Events) event formatter.
FastAPI StreamingResponse consumes the async generator produced here.
"""
import json
from typing import Any


def format_sse(event: str, data: Any) -> str:
    """
    Format a single SSE message. Returns a string with:
        event: <event_name>
        data: <json_payload>
        (blank line terminator)
    """
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def sse_queue_consumer(queue, session_complete_event):
    """
    Async generator that drains an asyncio.Queue and yields SSE-formatted
    strings. Stops when it receives the sentinel value None.

    Usage in FastAPI:
        return StreamingResponse(
            sse_queue_consumer(queue, done_event),
            media_type="text/event-stream"
        )
    """
    while True:
        item = await queue.get()
        if item is None:
            # Sentinel: pipeline is finished
            yield format_sse("session_complete", {"status": "done"})
            break
        event_name, payload = item
        yield format_sse(event_name, payload)
