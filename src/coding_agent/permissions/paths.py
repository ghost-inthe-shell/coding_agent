"""Read and write path boundaries for workspace files and agent artifacts."""

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
        resolved = _resolve(requested_path, context)

        allowed_roots = (
            Path(context.workspace_root).resolve(),
            Path(context.artifact_root).resolve(),
        )
        if any(_contains(root, resolved) for root in allowed_roots):
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


@dataclass(frozen=True, slots=True)
class WritePathPolicy:
    """Reject forbidden targets before asking about one workspace write."""

    def resolve(self, requested_path: str, context: ToolContext) -> Path:
        resolved = _resolve(requested_path, context)
        workspace = Path(context.workspace_root).resolve()
        artifact = Path(context.artifact_root).resolve()

        if _contains(artifact, resolved):
            raise PathAccessDenied(
                f"writing agent-owned artifacts is not allowed: {requested_path}"
            )
        if not _contains(workspace, resolved):
            raise PathAccessDenied(
                f"writing outside the workspace is not allowed: {requested_path}"
            )
        return resolved

    def authorize(self, resolved_path: Path, context: ToolContext) -> Path:
        """Recheck the boundary and ask once immediately before mutation."""

        resolved = self.resolve(str(resolved_path), context)
        if context.permission_handler is not None:
            decision = context.permission_handler(
                PermissionRequest(
                    operation=PermissionOperation.WRITE,
                    target=str(resolved),
                )
            )
            if decision is PermissionDecision.ALLOW:
                return resolved
        raise PathAccessDenied(f"writing in the workspace was not approved: {resolved}")


def _resolve(requested_path: str, context: ToolContext) -> Path:
    candidate = Path(requested_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(context.cwd) / candidate
    return candidate.resolve()


def _contains(root: Path, path: Path) -> bool:
    return path == root or path.is_relative_to(root)
