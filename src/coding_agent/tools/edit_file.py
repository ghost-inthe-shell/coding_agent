"""Replace one unique text block in an existing, previously read file."""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import Field

from coding_agent.core.file_state import FileVersion
from coding_agent.core.results import ToolResult
from coding_agent.permissions import PathAccessDenied, WritePathPolicy

from .base import Tool, ToolContext, ToolInput
from .path_helpers import display_path


class EditFileInput(ToolInput):
    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    content: str
    version: FileVersion
    mode: int


class EditFileTool(Tool[EditFileInput]):
    name = "edit_file"
    description = (
        "Replace one exact, unique text block in an existing UTF-8 file. "
        "The file must be read first."
    )
    input_model = EditFileInput

    def __init__(self, path_policy: WritePathPolicy | None = None) -> None:
        self._path_policy = path_policy or WritePathPolicy()

    def execute(self, arguments: EditFileInput, context: ToolContext) -> ToolResult:
        try:
            path = self._path_policy.resolve(arguments.path, context)
        except PathAccessDenied as exc:
            return ToolResult.denied(str(exc))

        if not path.exists():
            return ToolResult.error(f"file does not exist: {arguments.path}")
        if not path.is_file():
            return ToolResult.error(f"path is not a file: {arguments.path}")

        relative_path = path.relative_to(Path(context.workspace_root).resolve()).as_posix()
        expected_version = _expected_version(relative_path, context)
        if expected_version is None:
            return ToolResult.error(
                f"file must be read before editing; use read_file first: {arguments.path}"
            )

        snapshot, error = _read_snapshot(path, arguments.path)
        if error is not None:
            return ToolResult.error(error)
        assert snapshot is not None
        validation_error = _validate_edit(snapshot, expected_version, arguments)
        if validation_error is not None:
            return ToolResult.error(validation_error)
        _, encoding_error = _encode_updated_content(snapshot, arguments)
        if encoding_error is not None:
            return ToolResult.error(encoding_error)

        try:
            path = self._path_policy.authorize(path, context)
        except PathAccessDenied as exc:
            return ToolResult.denied(str(exc))

        snapshot, error = _read_snapshot(path, arguments.path)
        if error is not None:
            return ToolResult.error(error)
        assert snapshot is not None
        validation_error = _validate_edit(snapshot, expected_version, arguments)
        if validation_error is not None:
            return ToolResult.error(validation_error)
        encoded, encoding_error = _encode_updated_content(snapshot, arguments)
        if encoding_error is not None:
            return ToolResult.error(encoding_error)
        assert encoded is not None

        write_error = _atomic_replace(path, encoded, snapshot.mode)
        if write_error is not None:
            return ToolResult.error(f"could not edit {arguments.path}: {write_error}")

        try:
            current_stat = path.stat()
        except OSError as exc:
            return ToolResult.error(f"file was edited but could not be stat'ed: {exc}")
        version = FileVersion(
            mtime_ns=current_stat.st_mtime_ns,
            size=current_stat.st_size,
            sha256=sha256(encoded).hexdigest(),
        )
        if context.read_file_versions is not None:
            context.read_file_versions[relative_path] = version

        return ToolResult.from_text(
            f"Edited {display_path(path, context.workspace_root)} (1 replacement).",
            metadata={
                "path": relative_path,
                "replacements": 1,
                "bytes": len(encoded),
                "file_version": version.to_dict(),
            },
        )


def _expected_version(path: str, context: ToolContext) -> FileVersion | None:
    if context.read_file_versions is None:
        return None
    return context.read_file_versions.get(path)


def _read_snapshot(path: Path, display: str) -> tuple[_FileSnapshot | None, str | None]:
    try:
        with path.open("rb") as source:
            encoded = source.read()
            current_stat = os.fstat(source.fileno())
    except FileNotFoundError:
        return None, f"file does not exist: {display}"
    except OSError as exc:
        return None, f"could not read {display}: {exc}"

    if not stat.S_ISREG(current_stat.st_mode):
        return None, f"path is not a file: {display}"
    if b"\x00" in encoded:
        return None, f"binary file is not supported: {display}"
    try:
        content = encoded.decode("utf-8")
    except UnicodeDecodeError:
        return None, f"file is not valid UTF-8: {display}"

    return (
        _FileSnapshot(
            content=content,
            version=FileVersion(
                mtime_ns=current_stat.st_mtime_ns,
                size=current_stat.st_size,
                sha256=sha256(encoded).hexdigest(),
            ),
            mode=stat.S_IMODE(current_stat.st_mode),
        ),
        None,
    )


def _validate_edit(
    snapshot: _FileSnapshot,
    expected_version: FileVersion,
    arguments: EditFileInput,
) -> str | None:
    if snapshot.version != expected_version:
        return f"file changed since it was read; use read_file again: {arguments.path}"
    occurrences = _count_occurrences(snapshot.content, arguments.old_text)
    if occurrences == 0:
        return f"old_text was not found exactly in {arguments.path}"
    if occurrences > 1:
        return (
            f"old_text occurs {occurrences} times in {arguments.path}; "
            "provide more context so it is unique"
        )
    if arguments.old_text == arguments.new_text:
        return f"replacement would not change {arguments.path}"
    return None


def _count_occurrences(content: str, old_text: str) -> int:
    occurrences = 0
    start = 0
    while (index := content.find(old_text, start)) >= 0:
        occurrences += 1
        start = index + 1
    return occurrences


def _encode_updated_content(
    snapshot: _FileSnapshot,
    arguments: EditFileInput,
) -> tuple[bytes | None, str | None]:
    updated = snapshot.content.replace(arguments.old_text, arguments.new_text, 1)
    try:
        return updated.encode("utf-8"), None
    except UnicodeEncodeError as exc:
        return None, f"new content is not valid UTF-8 text: {exc}"


def _atomic_replace(path: Path, content: bytes, mode: int) -> str | None:
    descriptor = -1
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.coding-agent-",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as destination:
            descriptor = -1
            os.fchmod(destination.fileno(), mode)
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except OSError as exc:
        return str(exc)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
    return None
