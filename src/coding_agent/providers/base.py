"""Provider boundary consumed by the agent runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from coding_agent.core.messages import AssistantMessage, Message
from coding_agent.tools.base import ToolSpec

DEFAULT_MAX_OUTPUT_TOKENS = 16_384


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[Message, ...]
    system_prompt: str
    tools: tuple[ToolSpec, ...] = ()


class ProviderError(RuntimeError):
    """An expected provider/API failure safe to report as a failed run."""


class LLMProvider(ABC):
    @property
    def max_output_tokens(self) -> int | None:
        """Configured per-request output limit, when the provider exposes one."""

        return None

    @abstractmethod
    def complete(self, request: CompletionRequest) -> AssistantMessage:
        """Return one normalized assistant message."""
