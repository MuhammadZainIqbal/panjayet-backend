"""
Abstract provider interface.
Every provider (OpenRouter, Groq, Gemini) implements this contract.
"""
from abc import ABC, abstractmethod
from typing import Tuple


class BaseProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        api_key: str,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Call the model and return (content: str, model_used: str).
        Raises:
            RateLimitError — on HTTP 429
            ProviderError  — on non-recoverable errors
        """
        ...
