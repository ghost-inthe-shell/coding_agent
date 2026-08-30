"""Read-path boundary for workspace files and agent-owned artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coding_agent.tools.base import ToolContext

from .protocol import PermissionDecision, PermissionOperation, PermissionRequest


class PathAccessDenied(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class ReadPathPolicy:
    """Auto-allow owned roots and ask once for each outside read."""

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

        if context.permission_handler is not None:
            decision = context.permission_handler(
                PermissionRequest(
                    operation=PermissionOperation.READ,
                    target=str(resolved),
                )
            )
            if decision is PermissionDecision.ALLOW:
                return resolved

        raise PathAccessDenied(
            f"reading outside the workspace was not approved: {requested_path}"
        )
