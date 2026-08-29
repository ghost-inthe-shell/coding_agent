"""Shared display and traversal helpers for read-only tools."""

from __future__ import annotations

from pathlib import Path


IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)

RG_IGNORE_GLOBS = tuple(f"!**/{name}/**" for name in sorted(IGNORED_DIRECTORY_NAMES))


def display_path(path: Path, workspace_root: str) -> str:
    workspace = Path(workspace_root).resolve()
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return str(path)
