"""Small event envelope; payloads stay JSON-compatible and provider-independent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from coding_agent.core.json_types import JsonObject
from coding_agent.core.messages import timestamp_ms


class EventType(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_MESSAGE = "user_message"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PERMISSION_DECISION = "permission_decision"
    PROVIDER_RETRY = "provider_retry"
    CONTEXT_COMPACT_START = "context_compact_start"
    CONTEXT_COMPACT_END = "context_compact_end"
    SESSION_CHECKPOINT = "session_checkpoint"
    INTERRUPT = "interrupt"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True, slots=True)
class Event:
    type: EventType
    session_id: str
    sequence: int
    payload: JsonObject = field(default_factory=dict)
    timestamp: int = field(default_factory=timestamp_ms)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.sequence <= 0:
            raise ValueError("sequence must be positive")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> JsonObject:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Event:
        payload = data.get("payload", {})
        if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
            raise ValueError("payload must be an object with string keys")
        return cls(
            type=EventType(_required_string(data, "type")),
            session_id=_required_string(data, "session_id"),
            sequence=_required_integer(data, "sequence"),
            timestamp=_required_integer(data, "timestamp"),
            payload=dict(payload),  # type: ignore[arg-type]
        )


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _required_integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value

