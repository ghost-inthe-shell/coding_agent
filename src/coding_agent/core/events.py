"""The six synchronous events emitted by one runtime turn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Union

from .messages import AssistantMessage, ToolCall, ToolResultMessage, timestamp_ms
from .results import RunResult


@dataclass(frozen=True, slots=True)
class TurnStarted:
    session_id: str
    timestamp: int = field(default_factory=timestamp_ms)


@dataclass(frozen=True, slots=True)
class ModelRequested:
    session_id: str
    model_call: int
    timestamp: int = field(default_factory=timestamp_ms)


@dataclass(frozen=True, slots=True)
class ModelResponded:
    session_id: str
    model_call: int
    message: AssistantMessage
    timestamp: int = field(default_factory=timestamp_ms)


@dataclass(frozen=True, slots=True)
class ToolStarted:
    session_id: str
    call: ToolCall
    timestamp: int = field(default_factory=timestamp_ms)


@dataclass(frozen=True, slots=True)
class ToolFinished:
    session_id: str
    result: ToolResultMessage
    timestamp: int = field(default_factory=timestamp_ms)


@dataclass(frozen=True, slots=True)
class TurnFinished:
    session_id: str
    result: RunResult
    timestamp: int = field(default_factory=timestamp_ms)


RuntimeEvent = Union[
    TurnStarted,
    ModelRequested,
    ModelResponded,
    ToolStarted,
    ToolFinished,
    TurnFinished,
]
EventSink = Callable[[RuntimeEvent], None]
