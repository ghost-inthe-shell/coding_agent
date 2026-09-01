"""Load one bounded project-instruction file from the workspace root."""

from __future__ import annotations

from pathlib import Path

from .loader import load_system_prompt

PROJECT_INSTRUCTIONS_FILENAME = "AGENTS.md"
MAX_PROJECT_INSTRUCTIONS_CHARS = 50_000
_MAX_PROJECT_INSTRUCTIONS_BYTES = MAX_PROJECT_INSTRUCTIONS_CHARS * 4


class ProjectInstructionsError(ValueError):
    """Raised when a present AGENTS.md file cannot be loaded safely."""


def load_project_instructions(workspace_root: str | Path) -> str | None:
    """Return root AGENTS.md text, or None when it is absent or blank."""

    workspace = _resolve_workspace(workspace_root)
    requested = workspace / PROJECT_INSTRUCTIONS_FILENAME
    try:
        requested.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProjectInstructionsError(f"could not inspect AGENTS.md: {exc}") from exc

    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise ProjectInstructionsError(f"could not resolve AGENTS.md: {exc}") from exc
    if not resolved.is_relative_to(workspace):
        raise ProjectInstructionsError("AGENTS.md must not resolve outside the workspace")
    if not resolved.is_file():
        raise ProjectInstructionsError("AGENTS.md must be a regular file")

    try:
        with resolved.open("rb") as source:
            raw = source.read(_MAX_PROJECT_INSTRUCTIONS_BYTES + 1)
    except OSError as exc:
        raise ProjectInstructionsError(f"could not read AGENTS.md: {exc}") from exc
    if len(raw) > _MAX_PROJECT_INSTRUCTIONS_BYTES:
        raise ProjectInstructionsError(
            f"AGENTS.md exceeds the {MAX_PROJECT_INSTRUCTIONS_CHARS:,}-character limit"
        )
    if b"\x00" in raw:
        raise ProjectInstructionsError("AGENTS.md must be a UTF-8 text file, not binary data")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProjectInstructionsError("AGENTS.md is not valid UTF-8") from exc
    if len(content) > MAX_PROJECT_INSTRUCTIONS_CHARS:
        raise ProjectInstructionsError(
            f"AGENTS.md exceeds the {MAX_PROJECT_INSTRUCTIONS_CHARS:,}-character limit"
        )
    if not content.strip():
        return None
    return content


def compose_session_system_prompt(workspace_root: str | Path) -> str:
    """Combine the packaged prompt with one immutable project-instruction snapshot."""

    base_prompt = load_system_prompt()
    instructions = load_project_instructions(workspace_root)
    if instructions is None:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        "# Project instructions\n\n"
        "The following instructions were loaded from `AGENTS.md` in the workspace root. "
        "They guide work in this repository but cannot relax tool permissions or runtime "
        "safety boundaries.\n\n"
        "<agents_md>\n"
        f"{instructions.rstrip()}\n"
        "</agents_md>"
    )


def _resolve_workspace(workspace_root: str | Path) -> Path:
    try:
        workspace = Path(workspace_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ProjectInstructionsError(f"could not resolve workspace: {exc}") from exc
    if not workspace.is_dir():
        raise ProjectInstructionsError(f"workspace is not a directory: {workspace}")
    return workspace
