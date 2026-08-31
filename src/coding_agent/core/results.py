"""Structured results returned by tools and the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from .json_types import JsonObject
from .messages import TextBlock, ToolCall, ToolResultMessage, timestamp_ms
from .types import RunStatus, StopReason, ToolResultStatus
from .usage import Usage


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        status: ToolResultStatus = ToolResultStatus.SUCCESS,
        metadata: JsonObject | None = None,
    ) -> ToolResult:
        return cls(content=text, status=status, metadata=metadata or {})

    @classmethod
    def error(cls, message: str, *, metadata: JsonObject | None = None) -> ToolResult:
        return cls.from_text(message, status=ToolResultStatus.ERROR, metadata=metadata)

    @classmethod
    def denied(cls, message: str, *, metadata: JsonObject | None = None) -> ToolResult:
        return cls.from_text(message, status=ToolResultStatus.DENIED, metadata=metadata)

    @property
    def is_error(self) -> bool:
        return self.status is not ToolResultStatus.SUCCESS

    def to_message(self, call: ToolCall, *, timestamp: int | None = None) -> ToolResultMessage:
        return ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=(TextBlock(self.content),),
            status=self.status,
            metadata=self.metadata,
            timestamp=timestamp if timestamp is not None else timestamp_ms(),
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    final_text: str
    usage: Usage = field(default_factory=Usage)
    model_turns: int = 0
    tool_calls: int = 0
    max_output_tokens: int | None = None
    stop_reason: StopReason | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.model_turns < 0 or self.tool_calls < 0:
            raise ValueError("run counters must be non-negative")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when provided")


@dataclass(frozen=True, slots=True)
class CompactionResult:
    compacted: bool
    usage: Usage = field(default_factory=Usage)
    summarized_messages: int = 0
    tokens_before: int | None = None
    tokens_after: int | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.summarized_messages < 0:
            raise ValueError("summarized_messages must be non-negative")
        for name, value in (
            ("tokens_before", self.tokens_before),
            ("tokens_after", self.tokens_after),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when provided")
