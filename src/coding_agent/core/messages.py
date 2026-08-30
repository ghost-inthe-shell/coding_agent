"""Normalized conversation messages and tool-call protocol validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time_ns
from typing import Mapping, Sequence, Union

from .json_types import JsonObject
from .types import StopReason, ToolResultStatus
from .usage import Usage


def timestamp_ms() -> int:
    return time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str

    @property
    def type(self) -> str:
        return "text"

    def to_dict(self) -> JsonObject:
        return {"type": self.type, "text": self.text}


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    thinking: str
    replay_field: str | None = None

    def __post_init__(self) -> None:
        supported_fields = {"reasoning_content", "reasoning", "reasoning_text"}
        if self.replay_field is not None and self.replay_field not in supported_fields:
            raise ValueError(f"unsupported thinking replay field: {self.replay_field!r}")

    @property
    def type(self) -> str:
        return "thinking"

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"type": self.type, "thinking": self.thinking}
        if self.replay_field is not None:
            result["replay_field"] = self.replay_field
        return result


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: JsonObject = field(default_factory=dict)
    raw_arguments: str | None = None
    parse_error: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("tool call id must not be empty")
        if not self.name:
            raise ValueError("tool call name must not be empty")
        object.__setattr__(self, "arguments", dict(self.arguments))

    @property
    def type(self) -> str:
        return "tool_call"

    def to_dict(self) -> JsonObject:
        result: JsonObject = {
            "type": self.type,
            "id": self.id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.raw_arguments is not None:
            result["raw_arguments"] = self.raw_arguments
        if self.parse_error is not None:
            result["parse_error"] = self.parse_error
        return result


UserContent = TextBlock
AssistantContent = Union[TextBlock, ThinkingBlock, ToolCall]
ToolResultContent = TextBlock


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: tuple[UserContent, ...]
    timestamp: int = field(default_factory=timestamp_ms)

    @property
    def role(self) -> str:
        return "user"

    @classmethod
    def from_text(cls, text: str, *, timestamp: int | None = None) -> UserMessage:
        return cls(
            content=(TextBlock(text),),
            timestamp=timestamp if timestamp is not None else timestamp_ms(),
        )


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: tuple[AssistantContent, ...]
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = StopReason.STOP
    response_id: str | None = None
    raw_stop_reason: str | None = None
    error_message: str | None = None
    timestamp: int = field(default_factory=timestamp_ms)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")
        has_tool_calls = any(isinstance(block, ToolCall) for block in self.content)
        if has_tool_calls and self.stop_reason not in {StopReason.TOOL_USE, StopReason.LENGTH}:
            raise ValueError(
                "assistant messages with tool calls must use stop_reason=tool_use or length"
            )
        if self.stop_reason is StopReason.TOOL_USE and not has_tool_calls:
            raise ValueError("stop_reason=tool_use requires at least one tool call")
        if self.stop_reason in {StopReason.ERROR, StopReason.ABORTED} and not self.error_message:
            raise ValueError("error_message is required for error and aborted responses")

    @property
    def role(self) -> str:
        return "assistant"

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    @property
    def thinking(self) -> str:
        return "".join(
            block.thinking for block in self.content if isinstance(block, ThinkingBlock)
        )

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(block for block in self.content if isinstance(block, ToolCall))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: tuple[ToolResultContent, ...]
    status: ToolResultStatus = ToolResultStatus.SUCCESS
    metadata: JsonObject = field(default_factory=dict)
    timestamp: int = field(default_factory=timestamp_ms)

    def __post_init__(self) -> None:
        if not self.tool_call_id:
            raise ValueError("tool_call_id must not be empty")
        if not self.tool_name:
            raise ValueError("tool_name must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def role(self) -> str:
        return "tool_result"

    @property
    def is_error(self) -> bool:
        return self.status is not ToolResultStatus.SUCCESS

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


class ProtocolError(ValueError):
    """Raised when a persisted conversation violates tool-call pairing rules."""


def validate_message_sequence(
    messages: Sequence[Message],
    *,
    allow_pending_tail: bool = False,
) -> None:
    pending: list[ToolCall] = []
    seen_tool_call_ids: set[str] = set()

    for index, message in enumerate(messages):
        if isinstance(message, ToolResultMessage):
            if not pending:
                raise ProtocolError(f"message {index}: tool result has no pending tool call")
            expected = pending[0]
            if message.tool_call_id != expected.id:
                raise ProtocolError(
                    f"message {index}: expected result for {expected.id!r}, "
                    f"got {message.tool_call_id!r}"
                )
            if message.tool_name != expected.name:
                raise ProtocolError(
                    f"message {index}: tool name {message.tool_name!r} does not match "
                    f"call {expected.name!r}"
                )
            pending.pop(0)
            continue

        if pending:
            missing = ", ".join(call.id for call in pending)
            raise ProtocolError(f"message {index}: missing tool results for {missing}")

        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.id in seen_tool_call_ids:
                    raise ProtocolError(f"message {index}: duplicate tool call id {call.id!r}")
                seen_tool_call_ids.add(call.id)
                pending.append(call)

    if pending and not allow_pending_tail:
        missing = ", ".join(call.id for call in pending)
        raise ProtocolError(f"conversation ends with pending tool calls: {missing}")


def message_to_dict(message: Message) -> JsonObject:
    if isinstance(message, UserMessage):
        return {
            "role": message.role,
            "content": [block.to_dict() for block in message.content],
            "timestamp": message.timestamp,
        }
    if isinstance(message, AssistantMessage):
        result: JsonObject = {
            "role": message.role,
            "content": [block.to_dict() for block in message.content],
            "provider": message.provider,
            "model": message.model,
            "usage": message.usage.to_dict(),
            "stop_reason": message.stop_reason.value,
            "timestamp": message.timestamp,
        }
        for key, value in (
            ("response_id", message.response_id),
            ("raw_stop_reason", message.raw_stop_reason),
            ("error_message", message.error_message),
        ):
            if value is not None:
                result[key] = value
        return result
    return {
        "role": message.role,
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "content": [block.to_dict() for block in message.content],
        "status": message.status.value,
        "metadata": dict(message.metadata),
        "timestamp": message.timestamp,
    }


def message_from_dict(data: Mapping[str, object]) -> Message:
    role = _required_string(data, "role")
    timestamp = _required_integer(data, "timestamp")
    raw_content = data.get("content")
    if not isinstance(raw_content, list):
        raise ValueError("content must be a list")

    if role == "user":
        content = tuple(_user_content_from_dict(item) for item in raw_content)
        return UserMessage(content=content, timestamp=timestamp)
    if role == "assistant":
        content = tuple(_assistant_content_from_dict(item) for item in raw_content)
        usage = data.get("usage", {})
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be an object")
        return AssistantMessage(
            content=content,
            provider=_required_string(data, "provider"),
            model=_required_string(data, "model"),
            usage=Usage.from_dict(usage),
            stop_reason=StopReason(_required_string(data, "stop_reason")),
            response_id=_optional_string(data, "response_id"),
            raw_stop_reason=_optional_string(data, "raw_stop_reason"),
            error_message=_optional_string(data, "error_message"),
            timestamp=timestamp,
        )
    if role == "tool_result":
        content = tuple(_tool_result_content_from_dict(item) for item in raw_content)
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict) or not all(isinstance(key, str) for key in metadata):
            raise ValueError("metadata must be an object with string keys")
        return ToolResultMessage(
            tool_call_id=_required_string(data, "tool_call_id"),
            tool_name=_required_string(data, "tool_name"),
            content=content,
            status=ToolResultStatus(_required_string(data, "status")),
            metadata=dict(metadata),  # type: ignore[arg-type]
            timestamp=timestamp,
        )
    raise ValueError(f"unsupported message role: {role!r}")


def _user_content_from_dict(value: object) -> UserContent:
    block = _content_mapping(value)
    if block.get("type") == "text":
        return TextBlock(_required_string(block, "text"))
    raise ValueError(f"unsupported user content type: {block.get('type')!r}")


def _assistant_content_from_dict(value: object) -> AssistantContent:
    block = _content_mapping(value)
    if block.get("type") == "text":
        return TextBlock(_required_string(block, "text"))
    if block.get("type") == "thinking":
        return ThinkingBlock(
            thinking=_required_string(block, "thinking"),
            replay_field=_optional_string(block, "replay_field"),
        )
    if block.get("type") == "tool_call":
        arguments = block.get("arguments", {})
        if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
            raise ValueError("tool call arguments must be an object with string keys")
        return ToolCall(
            id=_required_string(block, "id"),
            name=_required_string(block, "name"),
            arguments=dict(arguments),  # type: ignore[arg-type]
            raw_arguments=_optional_string(block, "raw_arguments"),
            parse_error=_optional_string(block, "parse_error"),
        )
    raise ValueError(f"unsupported assistant content type: {block.get('type')!r}")


def _tool_result_content_from_dict(value: object) -> ToolResultContent:
    return _user_content_from_dict(value)


def _content_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("content block must be an object")
    return value


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null")
    return value


def _required_integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value
