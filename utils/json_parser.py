import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_and_parse_json(raw: str) -> dict | list:
    """
    Robustly extract a JSON object or array from a string that might contain
    markdown fences, conversational filler, or both.
    """
    # 1. Try to parse directly first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Try to find markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass # Fall back to bracket extraction

    # 3. Extract first valid JSON object or array using regex
    # Match outermost {} or [] using a greedy match from first { or [ to last } or ]
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse extracted JSON: {exc}") from exc

    raise ValueError("No valid JSON found in raw string")
