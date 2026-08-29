"""Model provider contracts and adapters."""

from .base import CompletionRequest, LLMProvider

__all__ = ["CompletionRequest", "LLMProvider"]
"""Model-provider boundary."""

from .base import CompletionRequest, LLMProvider, ProviderError

__all__ = ["CompletionRequest", "LLMProvider", "ProviderError"]
