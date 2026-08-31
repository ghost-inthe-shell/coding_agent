"""Task domain model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Task:
    title: str
    completed: bool = False

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Task:
        return cls(title=value["title"], completed=value.get("completed", False))
