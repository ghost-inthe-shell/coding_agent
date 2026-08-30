"""Run one approved Bash command in the workspace with bounded execution."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from coding_agent.core.json_types import JsonObject
from coding_agent.core.results import ToolResult
from coding_agent.core.types import ToolResultStatus
from coding_agent.permissions import ExecutePermissionPolicy, ExecutionDenied

from .base import Tool, ToolContext, ToolInput
from .process import ProcessOutput, run_limited_process

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
SHELL_PATH = Path("/bin/bash")
_PROVIDER_SECRET_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")


class RunShellInput(ToolInput):
    command: str = Field(min_length=1)
    timeout_seconds: int | None = Field(default=None, ge=1, le=MAX_TIMEOUT_SECONDS)


class RunShellTool(Tool[RunShellInput]):
    name = "run_shell"
    description = (
        "Run a Bash command in the workspace and return its exit code, stdout, and stderr. "
        "Commands require approval and time out after 120 seconds by default."
    )
    input_model = RunShellInput

    def __init__(self, permission_policy: ExecutePermissionPolicy | None = None) -> None:
        self._permission_policy = permission_policy or ExecutePermissionPolicy()

    def execute(self, arguments: RunShellInput, context: ToolContext) -> ToolResult:
        validation_error = _validate_command(arguments.command)
        if validation_error is not None:
            return ToolResult.error(validation_error)
        try:
            workspace = Path(context.workspace_root).resolve()
        except (OSError, RuntimeError) as exc:
            return ToolResult.error(f"could not resolve workspace: {exc}")
        if not workspace.exists():
            return ToolResult.error(f"workspace does not exist: {workspace}")
        if not workspace.is_dir():
            return ToolResult.error(f"workspace is not a directory: {workspace}")
        if not SHELL_PATH.is_file() or not os.access(SHELL_PATH, os.X_OK):
            return ToolResult.error(f"Bash is not executable: {SHELL_PATH}")

        timeout = arguments.timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        try:
            self._permission_policy.authorize(arguments.command, context)
        except ExecutionDenied as exc:
            return ToolResult.denied(str(exc))

        try:
            rechecked_workspace = Path(context.workspace_root).resolve()
        except (OSError, RuntimeError):
            return ToolResult.denied("workspace changed during shell confirmation")
        if rechecked_workspace != workspace or not rechecked_workspace.is_dir():
            return ToolResult.denied("workspace changed during shell confirmation")

        environment = os.environ.copy()
        for name in _PROVIDER_SECRET_NAMES:
            environment.pop(name, None)

        try:
            output = run_limited_process(
                [str(SHELL_PATH), "-c", arguments.command],
                cwd=workspace,
                timeout_seconds=timeout,
                env=environment,
            )
        except OSError as exc:
            return ToolResult.error(
                f"could not start shell command: {exc}",
                metadata=_metadata(arguments.command, workspace, timeout),
            )

        metadata = _metadata(arguments.command, workspace, timeout, output)
        content = _format_result(arguments.command, workspace, timeout, output)
        if output.timed_out:
            return ToolResult.from_text(
                content,
                status=ToolResultStatus.TIMEOUT,
                metadata=metadata,
            )
        if output.incomplete:
            return ToolResult.error(content, metadata=metadata)
        if output.returncode != 0:
            return ToolResult.error(content, metadata=metadata)
        return ToolResult.from_text(content, metadata=metadata)


def _validate_command(command: str) -> str | None:
    if not command.strip():
        return "shell command must contain a non-whitespace character"
    if "\x00" in command:
        return "shell command must not contain NUL bytes"
    try:
        command.encode("utf-8")
    except UnicodeEncodeError as exc:
        return f"shell command is not valid UTF-8 text: {exc}"
    return None


def _metadata(
    command: str,
    workspace: Path,
    timeout: int,
    output: ProcessOutput | None = None,
) -> JsonObject:
    return {
        "command": command,
        "cwd": str(workspace),
        "timeout_seconds": timeout,
        "exit_code": output.returncode if output is not None else None,
        "duration_ms": output.duration_ms if output is not None else 0,
        "timed_out": output.timed_out if output is not None else False,
        "artifact_incomplete": output.incomplete if output is not None else False,
    }


def _format_result(
    command: str,
    workspace: Path,
    timeout: int,
    output: ProcessOutput,
) -> str:
    if output.timed_out:
        outcome = f"Timed out after {timeout} seconds."
    elif output.incomplete:
        outcome = "Stopped after reaching the raw output limit."
    else:
        outcome = f"Exited with code {output.returncode}."
    header = (
        f"Command:\n{command}\n\n"
        f"Working directory: {workspace}\n"
        f"Duration: {output.duration_ms} ms\n"
        f"{outcome}"
    )
    return "\n\n".join(
        (
            header,
            _format_stream("stdout", output.stdout),
            _format_stream("stderr", output.stderr),
        )
    )


def _format_stream(name: str, content: str) -> str:
    if not content:
        return f"{name}: (empty)"
    return f"{name}:\n{content}"
