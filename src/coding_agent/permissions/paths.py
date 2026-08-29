"""Read-path boundary for workspace files and agent-owned artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coding_agent.tools.base import ToolContext


class PathAccessDenied(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class ReadPathPolicy:
    """Auto-allow only the workspace and this session's artifact directory."""

    def resolve(self, requested_path: str, context: ToolContext) -> Path:
        candidate = Path(requested_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(context.cwd) / candidate
        resolved = candidate.resolve()

        allowed_roots = (
            Path(context.workspace_root).resolve(),
            Path(context.artifact_root).resolve(),
        )
        if any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
            return resolved

        # TODO: replace denial with an interactive ask/confirm decision.
        raise PathAccessDenied(
            f"reading outside the workspace requires confirmation: {requested_path}"
        )
