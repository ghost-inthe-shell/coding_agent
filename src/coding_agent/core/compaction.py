"""Serializable pointer from full history to its active compacted view."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .json_types import JsonObject
from .messages import timestamp_ms


@dataclass(frozen=True, slots=True)
class CompactionCheckpoint:
    summary: str
    first_kept_message_index: int
    tokens_before: int
    created_at: int = field(default_factory=timestamp_ms)

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("compaction summary must not be empty")
        for name, value in (
            ("first_kept_message_index", self.first_kept_message_index),
            ("tokens_before", self.tokens_before),
            ("created_at", self.created_at),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.first_kept_message_index == 0:
            raise ValueError("compaction checkpoint must summarize at least one message")

    def to_dict(self) -> JsonObject:
        return {
            "summary": self.summary,
            "first_kept_message_index": self.first_kept_message_index,
            "tokens_before": self.tokens_before,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CompactionCheckpoint:
        return cls(
            summary=_required_string(data, "summary"),
            first_kept_message_index=_required_integer(data, "first_kept_message_index"),
            tokens_before=_required_integer(data, "tokens_before"),
            created_at=_required_integer(data, "created_at"),
        )


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _required_integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return value
