"""Serializable session state; provider clients and raw responses never belong here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .json_types import JsonObject
from .messages import (
    Message,
    message_from_dict,
    message_to_dict,
    timestamp_ms,
    validate_message_sequence,
)
from .types import SessionStatus
from .usage import Usage


@dataclass(slots=True)
class SessionState:
    session_id: str
    workspace_root: str
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    status: SessionStatus = SessionStatus.CREATED
    created_at: int = field(default_factory=timestamp_ms)
    updated_at: int = field(default_factory=timestamp_ms)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.workspace_root:
            raise ValueError("workspace_root must not be empty")
        if not self.system_prompt:
            raise ValueError("system_prompt must not be empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported session schema version: {self.schema_version}")

    @classmethod
    def create(cls, session_id: str, workspace_root: str | Path) -> SessionState:
        from coding_agent.prompts.loader import load_system_prompt

        return cls(
            session_id=session_id,
            workspace_root=str(Path(workspace_root).expanduser().resolve()),
            system_prompt=load_system_prompt(),
        )

    def validate(self, *, allow_pending_tool_calls: bool = False) -> None:
        validate_message_sequence(self.messages, allow_pending_tail=allow_pending_tool_calls)

    def touch(self) -> None:
        self.updated_at = timestamp_ms()

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "system_prompt": self.system_prompt,
            "messages": [message_to_dict(message) for message in self.messages],
            "usage": self.usage.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SessionState:
        messages = data.get("messages", [])
        usage = data.get("usage", {})
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be an object")

        parsed_messages = []
        for item in messages:
            if not isinstance(item, Mapping):
                raise ValueError("each message must be an object")
            parsed_messages.append(message_from_dict(item))

        state = cls(
            schema_version=_required_integer(data, "schema_version"),
            session_id=_required_string(data, "session_id"),
            workspace_root=_required_string(data, "workspace_root"),
            system_prompt=_required_string(data, "system_prompt"),
            messages=parsed_messages,
            usage=Usage.from_dict(usage),
            status=SessionStatus(_required_string(data, "status")),
            created_at=_required_integer(data, "created_at"),
            updated_at=_required_integer(data, "updated_at"),
        )
        state.validate(allow_pending_tool_calls=True)
        return state


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
