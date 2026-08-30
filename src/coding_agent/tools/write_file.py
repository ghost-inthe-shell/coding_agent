"""Create one new UTF-8 file without overwriting or creating directories."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from pydantic import Field

from coding_agent.core.file_state import FileVersion
from coding_agent.core.results import ToolResult
from coding_agent.permissions import PathAccessDenied, WritePathPolicy

from .base import Tool, ToolContext, ToolInput
from .path_helpers import display_path


class WriteFileInput(ToolInput):
    path: str = Field(min_length=1)
    content: str


class WriteFileTool(Tool[WriteFileInput]):
    name = "write_file"
    description = "Create a new UTF-8 text file. Never overwrite or create parent directories."
    input_model = WriteFileInput

    def __init__(self, path_policy: WritePathPolicy | None = None) -> None:
        self._path_policy = path_policy or WritePathPolicy()

    def execute(self, arguments: WriteFileInput, context: ToolContext) -> ToolResult:
        try:
            path = self._path_policy.resolve(arguments.path, context)
        except PathAccessDenied as exc:
            return ToolResult.denied(str(exc))

        requested = _requested_path(arguments.path, context)
        validation_error = _validate_new_file(requested, path, arguments.path)
        if validation_error is not None:
            return ToolResult.error(validation_error)
        try:
            encoded = arguments.content.encode("utf-8")
        except UnicodeEncodeError as exc:
            return ToolResult.error(f"content is not valid UTF-8 text: {exc}")

        try:
            path = self._path_policy.authorize(path, context)
        except PathAccessDenied as exc:
            return ToolResult.denied(str(exc))

        validation_error = _validate_new_file(requested, path, arguments.path)
        if validation_error is not None:
            return ToolResult.error(validation_error)

        try:
            with path.open("xb") as destination:
                destination.write(encoded)
        except FileExistsError:
            return ToolResult.error(
                f"path already exists; use edit_file for existing files: {arguments.path}"
            )
        except OSError as exc:
            return ToolResult.error(f"could not create {arguments.path}: {exc}")

        try:
            stat = path.stat()
        except OSError as exc:
            return ToolResult.error(f"file was created but could not be stat'ed: {exc}")
        version = FileVersion(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            sha256=sha256(encoded).hexdigest(),
        )
        relative_path = path.relative_to(Path(context.workspace_root).resolve()).as_posix()
        if context.read_file_versions is not None:
            context.read_file_versions[relative_path] = version

        return ToolResult.from_text(
            f"Created {display_path(path, context.workspace_root)} ({len(encoded)} bytes).",
            metadata={
                "path": relative_path,
                "bytes": len(encoded),
                "file_version": version.to_dict(),
            },
        )


def _requested_path(requested_path: str, context: ToolContext) -> Path:
    candidate = Path(requested_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(context.cwd) / candidate
    return candidate.absolute()


def _validate_new_file(requested: Path, resolved: Path, display: str) -> str | None:
    if requested.is_symlink():
        return f"path already exists as a symbolic link: {display}"
    if os.path.lexists(resolved):
        return f"path already exists; use edit_file for existing files: {display}"
    if not resolved.parent.exists():
        return f"parent directory does not exist: {display}"
    if not resolved.parent.is_dir():
        return f"parent path is not a directory: {display}"
    return None
