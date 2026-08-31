"""A tiny in-memory TTL cache."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class TTLCache:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, tuple[Any, float]] = {}

    def put(self, key: str, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._entries[key] = (value, self._clock() + ttl_seconds)

    def get(self, key: str) -> Any:
        value, expires_at = self._entries[key]
        if expires_at < self._clock():
            del self._entries[key]
            raise KeyError(key)
        return value

    def __len__(self) -> int:
        now = self._clock()
        expired = [key for key, (_, expires_at) in self._entries.items() if expires_at < now]
        for key in expired:
            del self._entries[key]
        return len(self._entries)
