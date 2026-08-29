"""List files matching a glob without imposing an item-count limit."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import shutil

from pydantic import Field

from coding_agent.core.results import ToolResult
from coding_agent.permissions import PathAccessDenied, ReadPathPolicy

from .artifacts import DEFAULT_MAX_ARTIFACT_BYTES
from .base import Tool, ToolContext, ToolInput
from .path_helpers import IGNORED_DIRECTORY_NAMES, RG_IGNORE_GLOBS, display_path
from .process import run_limited_process


class GlobFilesInput(ToolInput):
    pattern: str = Field(min_length=1)
    path: str = Field(default=".", min_length=1)


class GlobFilesTool(Tool[GlobFilesInput]):
    name = "glob_files"
    description = "List files matching a glob under a directory."
    input_model = GlobFilesInput

    def __init__(self, path_policy: ReadPathPolicy | None = None) -> None:
        self._path_policy = path_policy or ReadPathPolicy()

    def execute(self, arguments: GlobFilesInput, context: ToolContext) -> ToolResult:
        try:
            root = self._path_policy.resolve(arguments.path, context)
        except PathAccessDenied as exc:
            return ToolResult.denied(str(exc))
        if not root.exists():
            return ToolResult.error(f"path does not exist: {arguments.path}")
        if not root.is_dir():
            return ToolResult.error(f"path is not a directory: {arguments.path}")

        rg = shutil.which("rg")
        if rg is not None:
            return self._with_ripgrep(rg, root, arguments.pattern, context)
        return self._with_walk(root, arguments.pattern, context)

    def _with_ripgrep(
        self,
        rg: str,
        root: Path,
        pattern: str,
        context: ToolContext,
    ) -> ToolResult:
        target = display_path(root, context.workspace_root)
        command = [rg, "--files", "--glob", pattern]
        for ignored in RG_IGNORE_GLOBS:
            command.extend(("--glob", ignored))
        command.extend(("--", target))
        try:
            output = run_limited_process(command, cwd=Path(context.workspace_root))
        except OSError as exc:
            return ToolResult.error(f"could not run ripgrep: {exc}")
        if output.returncode not in (0, 1) and not output.incomplete:
            return ToolResult.error(output.stderr.strip() or "ripgrep failed")

        matches = sorted(line.removeprefix("./") for line in output.stdout.splitlines())
        return ToolResult.from_text(
            "\n".join(matches) if matches else "No files matched.",
            metadata={
                "engine": "rg",
                "artifact_incomplete": output.incomplete,
            },
        )

    def _with_walk(self, root: Path, pattern: str, context: ToolContext) -> ToolResult:
        lines: list[str] = []
        size = 0
        incomplete = False
        for directory, dirs, files in os.walk(root):
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in IGNORED_DIRECTORY_NAMES and not name.startswith(".")
            )
            for name in sorted(files):
                if name.startswith("."):
                    continue
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if not fnmatch.fnmatch(relative, pattern) and not fnmatch.fnmatch(name, pattern):
                    continue
                line = display_path(path, context.workspace_root)
                line_size = len((line + "\n").encode("utf-8"))
                if size + line_size > DEFAULT_MAX_ARTIFACT_BYTES:
                    incomplete = True
                    break
                lines.append(line)
                size += line_size
            if incomplete:
                break

        return ToolResult.from_text(
            "\n".join(sorted(lines)) if lines else "No files matched.",
            metadata={
                "engine": "python",
                "gitignore_honored": False,
                "artifact_incomplete": incomplete,
            },
        )
