import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.core.messages import ToolCall
from coding_agent.core.results import ToolResult
from coding_agent.core.types import ToolResultStatus
from coding_agent.permissions import PermissionDecision, PermissionOperation, PermissionRequest
from coding_agent.tools import (
    ArtifactStore,
    EditFileTool,
    ReadFileTool,
    ToolContext,
    ToolExecutor,
    ToolResultProcessor,
)


class RecordingPermissionHandler:
    def __init__(self, decision: PermissionDecision) -> None:
        self.decision = decision
        self.requests: list[PermissionRequest] = []

    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.requests.append(request)
        return self.decision


class EditFileToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.store = ArtifactStore("session-1", state_home=self.root / "state")
        self.handler = RecordingPermissionHandler(PermissionDecision.ALLOW)
        self.versions = {}
        self.context = ToolContext(
            session_id="session-1",
            workspace_root=str(self.workspace),
            artifact_root=str(self.store.root),
            cwd=str(self.workspace),
            permission_handler=self.handler,
            read_file_versions=self.versions,
        )
        self.executor = ToolExecutor(
            (ReadFileTool(), EditFileTool()),
            ToolResultProcessor(self.store),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def read(self, path: str) -> ToolResult:
        return self.executor.execute(
            ToolCall(id="read-call", name="read_file", arguments={"path": path}),
            self.context,
        )

    def edit(self, path: str, old_text: str, new_text: str) -> ToolResult:
        return self.executor.execute(
            ToolCall(
                id="edit-call",
                name="edit_file",
                arguments={"path": path, "old_text": old_text, "new_text": new_text},
            ),
            self.context,
        )

    def test_replaces_one_exact_block_preserves_mode_and_updates_version(self) -> None:
        path = self.workspace / "sample.txt"
        path.write_bytes(b"first\r\nold block\r\nlast\r\n")
        path.chmod(0o640)
        self.assertEqual(self.read("sample.txt").status, ToolResultStatus.SUCCESS)
        previous_version = self.versions["sample.txt"]

        result = self.edit("sample.txt", "old block", "new \u5185\u5bb9")

        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(path.read_bytes(), "first\r\nnew \u5185\u5bb9\r\nlast\r\n".encode())
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        self.assertEqual(len(self.handler.requests), 1)
        self.assertIs(self.handler.requests[0].operation, PermissionOperation.WRITE)
        self.assertEqual(self.handler.requests[0].target, str(path))
        self.assertEqual(result.metadata["replacements"], 1)
        self.assertNotEqual(self.versions["sample.txt"], previous_version)
        self.assertEqual(
            self.versions["sample.txt"].to_dict(),
            result.metadata["file_version"],
        )

    def test_requires_a_trusted_version_before_asking(self) -> None:
        path = self.workspace / "unread.txt"
        path.write_text("old", encoding="utf-8")

        result = self.edit("unread.txt", "old", "new")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("read_file first", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "old")
        self.assertEqual(self.handler.requests, [])

    def test_missing_file_and_directory_are_rejected_before_read_guard(self) -> None:
        directory = self.workspace / "directory"
        directory.mkdir()

        missing = self.edit("missing.txt", "old", "new")
        not_file = self.edit("directory", "old", "new")

        self.assertEqual(missing.status, ToolResultStatus.ERROR)
        self.assertIn("does not exist", missing.content)
        self.assertEqual(not_file.status, ToolResultStatus.ERROR)
        self.assertIn("not a file", not_file.content)
        self.assertEqual(self.handler.requests, [])

    def test_sha_detects_change_even_with_the_same_size_and_mtime(self) -> None:
        path = self.workspace / "changed.txt"
        path.write_text("alpha", encoding="utf-8")
        self.read("changed.txt")
        recorded = self.versions["changed.txt"]
        path.write_text("bravo", encoding="utf-8")
        os.utime(path, ns=(recorded.mtime_ns, recorded.mtime_ns))

        result = self.edit("changed.txt", "alpha", "omega")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("changed since it was read", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "bravo")
        self.assertEqual(self.handler.requests, [])

    def test_text_must_match_exactly_once_and_make_a_change(self) -> None:
        path = self.workspace / "matches.txt"
        path.write_text("same\nsame\n", encoding="utf-8")
        self.read("matches.txt")

        duplicate = self.edit("matches.txt", "same", "other")
        missing = self.edit("matches.txt", "Same", "other")
        unchanged = self.edit("matches.txt", "same\nsame", "same\nsame")

        self.assertEqual(duplicate.status, ToolResultStatus.ERROR)
        self.assertIn("occurs 2 times", duplicate.content)
        self.assertEqual(missing.status, ToolResultStatus.ERROR)
        self.assertIn("not found exactly", missing.content)
        self.assertEqual(unchanged.status, ToolResultStatus.ERROR)
        self.assertIn("would not change", unchanged.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "same\nsame\n")
        self.assertEqual(self.handler.requests, [])

    def test_overlapping_matches_are_not_considered_unique(self) -> None:
        path = self.workspace / "overlap.txt"
        path.write_text("aaa", encoding="utf-8")
        self.read("overlap.txt")

        result = self.edit("overlap.txt", "aa", "b")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("occurs 2 times", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "aaa")
        self.assertEqual(self.handler.requests, [])

    def test_empty_old_text_is_rejected_by_the_tool_schema(self) -> None:
        path = self.workspace / "empty-old.txt"
        path.write_text("content", encoding="utf-8")
        self.read("empty-old.txt")

        result = self.edit("empty-old.txt", "", "prefix")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("invalid arguments", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "content")
        self.assertEqual(self.handler.requests, [])

    def test_invalid_replacement_text_is_rejected_before_asking(self) -> None:
        path = self.workspace / "encoding.txt"
        path.write_text("old", encoding="utf-8")
        self.read("encoding.txt")

        result = self.edit("encoding.txt", "old", "\ud800")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("not valid UTF-8", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "old")
        self.assertEqual(self.handler.requests, [])

    def test_declined_edit_does_not_change_the_file(self) -> None:
        path = self.workspace / "declined.txt"
        path.write_text("old", encoding="utf-8")
        self.read("declined.txt")
        self.handler.decision = PermissionDecision.DENY

        result = self.edit("declined.txt", "old", "new")

        self.assertEqual(result.status, ToolResultStatus.DENIED)
        self.assertEqual(path.read_text(encoding="utf-8"), "old")
        self.assertEqual(len(self.handler.requests), 1)

    def test_change_during_confirmation_is_not_overwritten(self) -> None:
        path = self.workspace / "race.txt"
        path.write_text("old", encoding="utf-8")
        self.read("race.txt")

        class RacingHandler:
            def __call__(self, request: PermissionRequest) -> PermissionDecision:
                Path(request.target).write_text("user change", encoding="utf-8")
                return PermissionDecision.ALLOW

        self.context = ToolContext(
            session_id=self.context.session_id,
            workspace_root=self.context.workspace_root,
            artifact_root=self.context.artifact_root,
            cwd=self.context.cwd,
            permission_handler=RacingHandler(),
            read_file_versions=self.versions,
        )

        result = self.edit("race.txt", "old", "agent change")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("changed since it was read", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "user change")

    def test_failed_atomic_replace_preserves_original_and_removes_temporary_file(self) -> None:
        path = self.workspace / "failure.txt"
        path.write_text("old", encoding="utf-8")
        self.read("failure.txt")

        with patch("coding_agent.tools.edit_file.os.replace", side_effect=OSError("blocked")):
            result = self.edit("failure.txt", "old", "new")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("blocked", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(self.workspace.iterdir()), [path])

    def test_outside_path_is_denied_without_asking(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("old", encoding="utf-8")

        result = self.edit(str(outside), "old", "new")

        self.assertEqual(result.status, ToolResultStatus.DENIED)
        self.assertEqual(outside.read_text(encoding="utf-8"), "old")
        self.assertEqual(self.handler.requests, [])


if __name__ == "__main__":
    unittest.main()
