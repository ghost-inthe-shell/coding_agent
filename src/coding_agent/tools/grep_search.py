"""Search UTF-8 text with regular expressions and no match-count limit."""

from __future__ import annotations

from pathlib import Path
import shutil

from pydantic import Field

from coding_agent.core.results import ToolResult
from coding_agent.permissions import PathAccessDenied, ReadPathPolicy

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
        grep = shutil.which("grep")
        if grep is not None:
            return self._with_grep(grep, target, arguments, context)
        return ToolResult.error("grep_search requires either rg or grep")

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

    def _with_grep(
        self,
        grep: str,
        target: Path,
        arguments: GrepSearchInput,
        context: ToolContext,
    ) -> ToolResult:
        command = [
            grep,
            "--recursive",
            "--line-number",
            "--with-filename",
            "--extended-regexp",
            "--binary-files=without-match",
            "--color=never",
        ]
        if arguments.glob is not None:
            command.append(f"--include={arguments.glob}")
        command.extend(("--exclude=.*", "--exclude-dir=.?", "--exclude-dir=.??*"))
        command.extend(
            f"--exclude-dir={directory}" for directory in sorted(IGNORED_DIRECTORY_NAMES)
        )
        command.extend(("--", arguments.pattern, display_path(target, context.workspace_root)))
        try:
            output = run_limited_process(command, cwd=Path(context.workspace_root))
        except OSError as exc:
            return ToolResult.error(f"could not run grep: {exc}")

        if output.returncode == 1 and not output.incomplete:
            return ToolResult.from_text(
                "No matches found.",
                metadata={"engine": "grep", "gitignore_honored": False},
            )
        if output.returncode not in (0, 1) and not output.incomplete:
            return ToolResult.error(output.stderr.strip() or "grep failed")
        content = "\n".join(line.removeprefix("./") for line in output.stdout.splitlines())
        return ToolResult.from_text(
            content,
            metadata={
                "engine": "grep",
                "gitignore_honored": False,
                "artifact_incomplete": output.incomplete,
            },
        )
