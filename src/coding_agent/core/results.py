"""Structured results returned by tools and the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from .json_types import JsonObject
from .messages import TextBlock, ToolCall, ToolResultContent, ToolResultMessage, timestamp_ms
from .types import RunStatus, StopReason, ToolResultStatus
from .usage import Usage


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: tuple[ToolResultContent, ...]
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    metadata: JsonObject = field(default_factory=dict)
    artifact_path: str | None = None
    duration_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        status: ToolResultStatus = ToolResultStatus.SUCCESS,
        metadata: JsonObject | None = None,
    ) -> ToolResult:
        return cls(content=(TextBlock(text),), status=status, metadata=metadata or {})

    @property
    def is_error(self) -> bool:
        return self.status is not ToolResultStatus.SUCCESS

    def to_message(self, call: ToolCall, *, timestamp: int | None = None) -> ToolResultMessage:
        metadata = dict(self.metadata)
        if self.artifact_path is not None:
            metadata["artifact_path"] = self.artifact_path
        if self.duration_ms is not None:
            metadata["duration_ms"] = self.duration_ms
        return ToolResultMessage(
            tool_call_id=call.id,
            tool_name=call.name,
            content=self.content,
            status=self.status,
            metadata=metadata,
            timestamp=timestamp if timestamp is not None else timestamp_ms(),
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    final_text: str
    usage: Usage = field(default_factory=Usage)
    model_turns: int = 0
    tool_calls: int = 0
    stop_reason: StopReason | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.model_turns < 0 or self.tool_calls < 0:
            raise ValueError("run counters must be non-negative")

