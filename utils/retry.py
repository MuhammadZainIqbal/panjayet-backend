"""
Retry utilities and shared exception types.
All provider call sites import RateLimitError and agent_retry from here.
"""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)


class RateLimitError(Exception):
    """Raised when any provider returns HTTP 429."""
    pass


class ProviderError(Exception):
    """Raised for non-recoverable provider errors (4xx non-429, 5xx after retries)."""
    pass


def is_transient_error(exception):
    # Always retry on explicit 429 rate limits
    if isinstance(exception, RateLimitError):
        return True
    # Retry on transient 5xx server errors
    if isinstance(exception, ProviderError) and any(code in str(exception) for code in ["500", "502", "503", "504"]):
        return True
    return False


# Standard decorator for all provider call sites.
# 3 attempts. Waits: 3s → 6s → 12s. Retries on 429 and 5xx.
agent_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=3, max=12),
    retry=retry_if_exception(is_transient_error),
    reraise=True,
)
