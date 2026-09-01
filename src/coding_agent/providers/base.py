"""Provider boundary consumed by the agent runtime."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from coding_agent.core.messages import AssistantMessage, Message
from coding_agent.tools.base import ToolSpec

from .reasoning import ReasoningLevel

DEFAULT_MAX_OUTPUT_TOKENS = 16_384


@dataclass(frozen=True, slots=True)
class CompletionTextDelta:
    text: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("completion text delta must not be empty")


@dataclass(frozen=True, slots=True)
class CompletionThinkingDelta:
    thinking: str

    def __post_init__(self) -> None:
        if not self.thinking:
            raise ValueError("completion thinking delta must not be empty")


CompletionEvent = CompletionTextDelta | CompletionThinkingDelta
CompletionEventSink = Callable[[CompletionEvent], None]


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[Message, ...]
    system_prompt: str
    tools: tuple[ToolSpec, ...] = ()
    max_output_tokens: int | None = None
    reasoning: ReasoningLevel = ReasoningLevel.DEFAULT

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when provided")
        if not isinstance(self.reasoning, ReasoningLevel):
            raise TypeError("reasoning must be a ReasoningLevel")


class ProviderError(RuntimeError):
    """An expected provider/API failure safe to report as a failed run."""


class LLMProvider(ABC):
    @property
    def max_output_tokens(self) -> int | None:
        """Configured per-request output limit, when the provider exposes one."""

        return None

    @abstractmethod
    def complete(
        self,
        request: CompletionRequest,
        *,
        event_sink: CompletionEventSink | None = None,
    ) -> AssistantMessage:
        """Return one normalized assistant message."""
