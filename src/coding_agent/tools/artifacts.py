"""Session-scoped storage for tool output that is too large for model context."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re


DEFAULT_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: Path
    stored_bytes: int
    incomplete: bool


class ArtifactStore:
    def __init__(self, session_id: str, *, state_home: Path | None = None) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id) or session_id in {".", ".."}:
            raise ValueError("session_id contains unsafe path characters")
        base = state_home or _default_state_home()
        self.root = (base / "coding-agent" / "sessions" / session_id / "tool-results").resolve()

    def write(
        self,
        tool_call_id: str,
        content: str,
        *,
        max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    ) -> ArtifactRecord:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        encoded = content.encode("utf-8")
        incomplete = len(encoded) > max_bytes
        if incomplete:
            encoded = encoded[:max_bytes].decode("utf-8", errors="ignore").encode("utf-8")

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / _artifact_name(tool_call_id)
        with path.open("xb") as artifact:
            artifact.write(encoded)
        return ArtifactRecord(path=path, stored_bytes=len(encoded), incomplete=incomplete)


def _default_state_home() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "state"


def _artifact_name(tool_call_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_call_id).strip("._") or "tool-call"
    digest = sha256(tool_call_id.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:80]}-{digest}.txt"
