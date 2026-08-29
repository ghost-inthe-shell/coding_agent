"""Read a numbered range from one UTF-8 text file."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import Field

from coding_agent.core.results import ToolResult
from coding_agent.permissions import PathAccessDenied, ReadPathPolicy

from .base import Tool, ToolContext, ToolInput
from .path_helpers import display_path


class ReadFileInput(ToolInput):
    path: str = Field(min_length=1)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=200, ge=1, le=2000)


class ReadFileTool(Tool[ReadFileInput]):
    name = "read_file"
    description = "Read a UTF-8 text file with 1-based line numbers."
    input_model = ReadFileInput

    def __init__(self, path_policy: ReadPathPolicy | None = None) -> None:
        self._path_policy = path_policy or ReadPathPolicy()

    def execute(self, arguments: ReadFileInput, context: ToolContext) -> ToolResult:
        try:
            path = self._path_policy.resolve(arguments.path, context)
        except PathAccessDenied as exc:
            return ToolResult.denied(str(exc))

        if not path.exists():
            return ToolResult.error(f"file does not exist: {arguments.path}")
        if not path.is_file():
            return ToolResult.error(f"path is not a file: {arguments.path}")

        digest = sha256()
        selected: list[tuple[int, str]] = []
        total_lines = 0
        last_requested = arguments.offset + arguments.limit - 1
        try:
            with path.open("rb") as source:
                for total_lines, raw_line in enumerate(source, start=1):
                    digest.update(raw_line)
                    if b"\x00" in raw_line:
                        return ToolResult.error(f"binary file is not supported: {arguments.path}")
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        return ToolResult.error(f"file is not valid UTF-8: {arguments.path}")
                    if arguments.offset <= total_lines <= last_requested:
                        selected.append((total_lines, line.rstrip("\r\n")))
        except OSError as exc:
            return ToolResult.error(f"could not read {arguments.path}: {exc}")

        try:
            stat = path.stat()
        except OSError as exc:
            return ToolResult.error(f"could not stat {arguments.path}: {exc}")
        actual_start = selected[0][0] if selected else 0
        actual_end = selected[-1][0] if selected else 0
        content = "\n".join(f"{number}: {line}" for number, line in selected)
        if not selected:
            content = "(file is empty)" if total_lines == 0 else "(offset is beyond end of file)"

        return ToolResult.from_text(
            content,
            metadata={
                "path": display_path(path, context.workspace_root),
                "total_lines": total_lines,
                "actual_start": actual_start,
                "actual_end": actual_end,
                "truncated": actual_end < total_lines,
                "file_version": {
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "sha256": digest.hexdigest(),
                },
            },
        )
