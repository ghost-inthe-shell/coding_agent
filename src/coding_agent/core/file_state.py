"""Serializable file versions used by the read-before-edit guard."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from string import hexdigits

from .json_types import JsonObject


@dataclass(frozen=True, slots=True)
class FileVersion:
    mtime_ns: int
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.mtime_ns, int) or isinstance(self.mtime_ns, bool):
            raise ValueError("mtime_ns must be an integer")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError("size must be a non-negative integer")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in hexdigits for character in self.sha256)
        ):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "sha256", self.sha256.lower())

    def to_dict(self) -> JsonObject:
        return {
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> FileVersion:
        return cls(
            mtime_ns=_required_integer(data, "mtime_ns"),
            size=_required_integer(data, "size"),
            sha256=_required_string(data, "sha256"),
        )


def _required_integer(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_string(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value
