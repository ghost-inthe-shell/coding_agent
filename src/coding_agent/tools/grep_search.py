"""Search UTF-8 text with regular expressions and no match-count limit."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import re
import shutil

from pydantic import Field

from coding_agent.core.results import ToolResult
from coding_agent.permissions import PathAccessDenied, ReadPathPolicy

from .artifacts import DEFAULT_MAX_ARTIFACT_BYTES
from .base import Tool, ToolContext, ToolInput
from .path_helpers import IGNORED_DIRECTORY_NAMES, RG_IGNORE_GLOBS, display_path
from .process import run_limited_process


class GrepSearchInput(ToolInput):
    pattern: str = Field(min_length=1)
    path: str = Field(default=".", min_length=1)
    glob: str | None = Field(default=None, min_length=1)


class GrepSearchTool(Tool[GrepSearchInput]):
    name = "grep_search"
    description = "Search text with a regular expression and return path:line:text matches."
    input_model = GrepSearchInput

    def __init__(self, path_policy: ReadPathPolicy | None = None) -> None:
        self._path_policy = path_policy or ReadPathPolicy()

    def execute(self, arguments: GrepSearchInput, context: ToolContext) -> ToolResult:
        try:
            target = self._path_policy.resolve(arguments.path, context)
        except PathAccessDenied as exc:
            return ToolResult.denied(str(exc))
        if not target.exists():
            return ToolResult.error(f"path does not exist: {arguments.path}")

        rg = shutil.which("rg")
        if rg is not None:
            return self._with_ripgrep(rg, target, arguments, context)
        return self._with_python(target, arguments, context)

    def _with_ripgrep(
        self,
        rg: str,
        target: Path,
        arguments: GrepSearchInput,
        context: ToolContext,
    ) -> ToolResult:
        command = [rg, "--line-number", "--no-heading", "--color", "never"]
        if arguments.glob is not None:
            command.extend(("--glob", arguments.glob))
        for ignored in RG_IGNORE_GLOBS:
            command.extend(("--glob", ignored))
        command.extend(("--", arguments.pattern, display_path(target, context.workspace_root)))
        try:
            output = run_limited_process(command, cwd=Path(context.workspace_root))
        except OSError as exc:
            return ToolResult.error(f"could not run ripgrep: {exc}")

        if output.returncode == 1 and not output.incomplete:
            return ToolResult.from_text("No matches found.", metadata={"engine": "rg"})
        if output.returncode not in (0, 1) and not output.incomplete:
            return ToolResult.error(output.stderr.strip() or "ripgrep failed")
        content = "\n".join(line.removeprefix("./") for line in output.stdout.splitlines())
        return ToolResult.from_text(
            content,
            metadata={
                "engine": "rg",
                "artifact_incomplete": output.incomplete,
            },
        )

    def _with_python(
        self,
        target: Path,
        arguments: GrepSearchInput,
        context: ToolContext,
    ) -> ToolResult:
        try:
            expression = re.compile(arguments.pattern)
        except re.error as exc:
            return ToolResult.error(f"invalid regular expression: {exc}")

        files = [target] if target.is_file() else _walk_files(target)
        lines: list[str] = []
        size = 0
        incomplete = False
        for path in files:
            relative = path.name if target.is_file() else path.relative_to(target).as_posix()
            if arguments.glob is not None and not (
                fnmatch.fnmatch(relative, arguments.glob)
                or fnmatch.fnmatch(path.name, arguments.glob)
            ):
                continue
            try:
                with path.open("r", encoding="utf-8") as source:
                    for number, text in enumerate(source, start=1):
                        text = text.rstrip("\r\n")
                        if expression.search(text) is None:
                            continue
                        line = f"{display_path(path, context.workspace_root)}:{number}:{text}"
                        line_size = len((line + "\n").encode("utf-8"))
                        if size + line_size > DEFAULT_MAX_ARTIFACT_BYTES:
                            incomplete = True
                            break
                        lines.append(line)
                        size += line_size
            except (OSError, UnicodeDecodeError):
                continue
            if incomplete:
                break

        return ToolResult.from_text(
            "\n".join(lines) if lines else "No matches found.",
            metadata={
                "engine": "python",
                "gitignore_honored": False,
                "artifact_incomplete": incomplete,
            },
        )


def _walk_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in IGNORED_DIRECTORY_NAMES and not name.startswith(".")
        )
        paths.extend(Path(directory) / name for name in sorted(files) if not name.startswith("."))
    return paths
