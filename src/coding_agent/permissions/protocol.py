"""Synchronous permission request protocol shared by runtimes and clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PermissionOperation(str, Enum):
    READ = "read"


class PermissionDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PermissionRequest:
    operation: PermissionOperation
    target: str

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("permission target must not be empty")


class PermissionHandler(Protocol):
    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        """Return one decision for one request without persisting a grant."""

        ...
