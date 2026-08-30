"""Serializable session state; provider clients and raw responses never belong here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .file_state import FileVersion
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

CURRENT_SESSION_SCHEMA_VERSION = 3


@dataclass(slots=True)
class SessionState:
    session_id: str
    workspace_root: str
    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    read_file_versions: dict[str, FileVersion] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    status: SessionStatus = SessionStatus.CREATED
    created_at: int = field(default_factory=timestamp_ms)
    updated_at: int = field(default_factory=timestamp_ms)
    schema_version: int = CURRENT_SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if not self.workspace_root:
            raise ValueError("workspace_root must not be empty")
        if not self.system_prompt:
            raise ValueError("system_prompt must not be empty")
        if self.schema_version != CURRENT_SESSION_SCHEMA_VERSION:
            raise ValueError(f"unsupported session schema version: {self.schema_version}")
        versions = dict(self.read_file_versions)
        for path, version in versions.items():
            _validate_version_path(path)
            if not isinstance(version, FileVersion):
                raise ValueError("read_file_versions values must be FileVersion instances")
        self.read_file_versions = versions

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
            "read_file_versions": {
                path: version.to_dict() for path, version in self.read_file_versions.items()
            },
            "usage": self.usage.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SessionState:
        messages = data.get("messages", [])
        usage = data.get("usage", {})
        schema_version = _required_integer(data, "schema_version")
        if schema_version not in {1, 2, CURRENT_SESSION_SCHEMA_VERSION}:
            raise ValueError(f"unsupported session schema version: {schema_version}")
        if schema_version >= 2 and "read_file_versions" not in data:
            raise ValueError(
                f"read_file_versions is required in session schema version {schema_version}"
            )
        versions = data.get("read_file_versions", {})
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be an object")
        if not isinstance(versions, Mapping):
            raise ValueError("read_file_versions must be an object")

        parsed_messages = []
        for item in messages:
            if not isinstance(item, Mapping):
                raise ValueError("each message must be an object")
            parsed_messages.append(message_from_dict(item))

        parsed_versions: dict[str, FileVersion] = {}
        for path, version in versions.items():
            if not isinstance(path, str):
                raise ValueError("read_file_versions keys must be strings")
            if not isinstance(version, Mapping):
                raise ValueError("each file version must be an object")
            parsed_versions[path] = FileVersion.from_dict(version)

        state = cls(
            schema_version=CURRENT_SESSION_SCHEMA_VERSION,
            session_id=_required_string(data, "session_id"),
            workspace_root=_required_string(data, "workspace_root"),
            system_prompt=_required_string(data, "system_prompt"),
            messages=parsed_messages,
            read_file_versions=parsed_versions,
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


def _validate_version_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or value != path.as_posix()
        or ".." in path.parts
    ):
        raise ValueError(f"invalid workspace-relative file version path: {value!r}")
