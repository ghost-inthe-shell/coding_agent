"""Model provider contracts and adapters."""

from .base import CompletionRequest, LLMProvider

__all__ = ["CompletionRequest", "LLMProvider"]
"""Model-provider boundary."""

from .base import CompletionRequest, LLMProvider, ProviderError
from .openai_compatible import OpenAICompatibleProvider

__all__ = [
    "CompletionRequest",
    "LLMProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
]
