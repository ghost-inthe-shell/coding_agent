"""Per-call approval policy for model-requested shell commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coding_agent.tools.base import ToolContext

from .protocol import PermissionDecision, PermissionOperation, PermissionRequest


class ExecutionDenied(PermissionError):
    pass


@dataclass(frozen=True, slots=True)
class ExecutePermissionPolicy:
    """Require one explicit approval for every shell command."""

    def authorize(self, command: str, context: ToolContext) -> None:
        if context.permission_handler is not None:
            decision = context.permission_handler(
                PermissionRequest(
                    operation=PermissionOperation.EXECUTE,
                    target=command,
                )
            )
            if decision is PermissionDecision.ALLOW:
                return
        raise ExecutionDenied("shell command was not approved")
