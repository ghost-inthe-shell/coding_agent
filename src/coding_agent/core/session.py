"""Serializable session state; provider clients and raw responses never belong here."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class FileVersion:
    mtime_ns: int
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.mtime_ns < 0 or self.size < 0:
            raise ValueError("file version values must be non-negative")
        if len(self.sha256) != 64:
            raise ValueError("sha256 must contain 64 hexadecimal characters")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("sha256 must contain 64 hexadecimal characters") from exc

    def to_dict(self) -> JsonObject:
        return {"mtime_ns": self.mtime_ns, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FileVersion:
        return cls(
            mtime_ns=_required_integer(data, "mtime_ns"),
            size=_required_integer(data, "size"),
            sha256=_required_string(data, "sha256"),
        )


@dataclass(slots=True)
class SessionState:
    session_id: str
    workspace_root: str
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    read_file_versions: dict[str, FileVersion] = field(default_factory=dict)
    modified_files: list[str] = field(default_factory=list)
    status: SessionStatus = SessionStatus.CREATED
    created_at: int = field(default_factory=timestamp_ms)
    updated_at: int = field(default_factory=timestamp_ms)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.workspace_root:
            raise ValueError("workspace_root must not be empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported session schema version: {self.schema_version}")

    def validate(self, *, allow_pending_tool_calls: bool = False) -> None:
        validate_message_sequence(self.messages, allow_pending_tail=allow_pending_tool_calls)

    def touch(self) -> None:
        self.updated_at = timestamp_ms()

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "messages": [message_to_dict(message) for message in self.messages],
            "usage": self.usage.to_dict(),
            "read_file_versions": {
                path: version.to_dict() for path, version in self.read_file_versions.items()
            },
            "modified_files": list(self.modified_files),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SessionState:
        messages = data.get("messages", [])
        usage = data.get("usage", {})
        versions = data.get("read_file_versions", {})
        modified_files = data.get("modified_files", [])
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be an object")
        if not isinstance(versions, Mapping):
            raise ValueError("read_file_versions must be an object")
        if not isinstance(modified_files, list) or not all(
            isinstance(path, str) for path in modified_files
        ):
            raise ValueError("modified_files must be a list of strings")

        parsed_messages = []
        for item in messages:
            if not isinstance(item, Mapping):
                raise ValueError("each message must be an object")
            parsed_messages.append(message_from_dict(item))

        parsed_versions: dict[str, FileVersion] = {}
        for path, value in versions.items():
            if not isinstance(path, str) or not isinstance(value, Mapping):
                raise ValueError("file versions must map paths to objects")
            parsed_versions[path] = FileVersion.from_dict(value)

        state = cls(
            schema_version=_required_integer(data, "schema_version"),
            session_id=_required_string(data, "session_id"),
            workspace_root=_required_string(data, "workspace_root"),
            messages=parsed_messages,
            usage=Usage.from_dict(usage),
            read_file_versions=parsed_versions,
            modified_files=list(modified_files),
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
