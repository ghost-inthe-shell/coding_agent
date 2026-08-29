"""Provider boundary consumed by the agent runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from coding_agent.core.messages import AssistantMessage, Message
from coding_agent.tools.base import ToolSpec


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[Message, ...]
    system_prompt: str = ""
    tools: tuple[ToolSpec, ...] = ()
    max_output_tokens: int | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, request: CompletionRequest) -> AssistantMessage:
        """Return one normalized assistant message."""

