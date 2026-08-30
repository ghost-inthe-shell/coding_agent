import os
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from coding_agent.core.messages import ToolCall
from coding_agent.core.results import ToolResult
from coding_agent.core.types import ToolResultStatus
from coding_agent.permissions import PermissionDecision, PermissionRequest
from coding_agent.tools import (
    ArtifactStore,
    GlobFilesTool,
    GrepSearchTool,
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


class ReadOnlyToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.store = ArtifactStore("session-1", state_home=self.root / "state")
        self.context = ToolContext(
            session_id="session-1",
            workspace_root=str(self.workspace),
            artifact_root=str(self.store.root),
            cwd=str(self.workspace),
        )
        self.executor = ToolExecutor(
            (ReadFileTool(), GlobFilesTool(), GrepSearchTool()),
            ToolResultProcessor(self.store),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(
        self,
        name: str,
        arguments: dict,
        call_id: str = "call-1",
        *,
        context: ToolContext | None = None,
    ) -> ToolResult:
        return self.executor.execute(
            ToolCall(id=call_id, name=name, arguments=arguments),
            context or self.context,
        )

    def test_read_file_returns_numbered_range_and_version(self) -> None:
        (self.workspace / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

        result = self.execute("read_file", {"path": "notes.txt", "offset": 2, "limit": 1})

        self.assertEqual(result.content, "2: two")
        self.assertEqual(result.metadata["total_lines"], 3)
        self.assertEqual(result.metadata["actual_start"], 2)
        self.assertEqual(result.metadata["actual_end"], 2)
        self.assertTrue(result.metadata["truncated"])
        self.assertEqual(len(result.metadata["file_version"]["sha256"]), 64)

    def test_read_file_rejects_binary_and_invalid_utf8(self) -> None:
        (self.workspace / "binary").write_bytes(b"a\x00b")
        (self.workspace / "invalid").write_bytes(b"\xff")

        binary = self.execute("read_file", {"path": "binary"}, "binary-call")
        invalid = self.execute("read_file", {"path": "invalid"}, "invalid-call")

        self.assertEqual(binary.status, ToolResultStatus.ERROR)
        self.assertIn("binary", binary.content)
        self.assertEqual(invalid.status, ToolResultStatus.ERROR)
        self.assertIn("UTF-8", invalid.content)

    def test_workspace_escape_and_symlink_escape_are_denied(self) -> None:
        outside = self.root / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        os.symlink(outside, self.workspace / "link.txt")

        direct = self.execute("read_file", {"path": str(outside)}, "direct-call")
        symlink = self.execute("read_file", {"path": "link.txt"}, "symlink-call")

        self.assertEqual(direct.status, ToolResultStatus.DENIED)
        self.assertEqual(symlink.status, ToolResultStatus.DENIED)

    def test_outside_read_tools_ask_for_each_call_and_can_be_allowed(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        note = outside / "note.txt"
        note.write_text("needle\n", encoding="utf-8")
        os.symlink(note, self.workspace / "outside-link.txt")
        handler = RecordingPermissionHandler(PermissionDecision.ALLOW)
        context = replace(self.context, permission_handler=handler)

        read_result = self.execute(
            "read_file", {"path": "outside-link.txt"}, "read-outside", context=context
        )
        glob_result = self.execute(
            "glob_files",
            {"path": str(outside), "pattern": "*.txt"},
            "glob-outside",
            context=context,
        )
        grep_result = self.execute(
            "grep_search",
            {"path": str(outside), "pattern": "needle"},
            "grep-outside",
            context=context,
        )

        self.assertEqual(read_result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(glob_result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(grep_result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(len(handler.requests), 3)
        self.assertEqual(handler.requests[0].target, str(note.resolve()))
        self.assertEqual(handler.requests[1].target, str(outside.resolve()))
        self.assertEqual(handler.requests[2].target, str(outside.resolve()))

    def test_outside_read_is_denied_when_user_declines(self) -> None:
        outside = self.root / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        handler = RecordingPermissionHandler(PermissionDecision.DENY)
        context = replace(self.context, permission_handler=handler)

        result = self.execute("read_file", {"path": str(outside)}, context=context)

        self.assertEqual(result.status, ToolResultStatus.DENIED)
        self.assertEqual(len(handler.requests), 1)

    def test_workspace_read_does_not_call_permission_handler(self) -> None:
        (self.workspace / "notes.txt").write_text("inside\n", encoding="utf-8")
        handler = RecordingPermissionHandler(PermissionDecision.DENY)
        context = replace(self.context, permission_handler=handler)

        result = self.execute("read_file", {"path": "notes.txt"}, context=context)

        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(handler.requests, [])

    def test_agent_owned_artifact_is_auto_readable(self) -> None:
        artifact = self.store.write("source-call", "stored output")

        result = self.execute("read_file", {"path": str(artifact.path)}, "read-artifact")

        self.assertEqual(result.content, "1: stored output")

    def test_glob_and_grep_have_no_small_item_cap(self) -> None:
        for index in range(230):
            (self.workspace / f"item-{index:03}.txt").write_text(
                f"needle {index}\n", encoding="utf-8"
            )

        glob_result = self.execute("glob_files", {"pattern": "*.txt"}, "glob-call")
        grep_result = self.execute(
            "grep_search", {"pattern": "needle", "glob": "*.txt"}, "grep-call"
        )

        self.assertEqual(len(glob_result.content.splitlines()), 230)
        self.assertEqual(len(grep_result.content.splitlines()), 230)

    def test_large_result_is_persisted_with_head_and_tail(self) -> None:
        processor = ToolResultProcessor(self.store, max_result_chars=120, max_artifact_bytes=1000)
        original = "A" * 1000 + "middle" + "Z" * 1000
        result = processor.process(
            ToolCall(id="large-call", name="fake"),
            ToolResult.from_text(original),
        )

        self.assertLessEqual(len(result.content), 120)
        self.assertIn("A", result.content)
        self.assertIn("Z", result.content)
        self.assertTrue(result.metadata["truncated"])
        artifact = Path(result.metadata["artifact_path"])
        self.assertLessEqual(artifact.stat().st_size, 1000)
        self.assertTrue(result.metadata["artifact_incomplete"])

    def test_grep_command_is_used_when_ripgrep_is_missing(self) -> None:
        grep = shutil.which("grep")
        self.assertIsNotNone(grep)
        (self.workspace / "match.py").write_text("needle\n", encoding="utf-8")
        (self.workspace / "skip.txt").write_text("needle\n", encoding="utf-8")
        (self.workspace / ".hidden.py").write_text("needle\n", encoding="utf-8")

        with patch(
            "coding_agent.tools.grep_search.shutil.which",
            side_effect=lambda command: None if command == "rg" else grep,
        ):
            result = self.execute(
                "grep_search",
                {"pattern": "needle", "glob": "*.py"},
                "grep-fallback-call",
            )

        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(result.metadata["engine"], "grep")
        self.assertFalse(result.metadata["gitignore_honored"])
        self.assertIn("match.py:1:needle", result.content)
        self.assertNotIn("skip.txt", result.content)
        self.assertNotIn(".hidden.py", result.content)

    def test_grep_command_distinguishes_no_match_and_invalid_regex(self) -> None:
        grep = shutil.which("grep")
        self.assertIsNotNone(grep)
        (self.workspace / "sample.txt").write_text("text\n", encoding="utf-8")

        with patch(
            "coding_agent.tools.grep_search.shutil.which",
            side_effect=lambda command: None if command == "rg" else grep,
        ):
            no_match = self.execute(
                "grep_search", {"pattern": "missing"}, "grep-no-match-call"
            )
            invalid = self.execute(
                "grep_search", {"pattern": "["}, "grep-invalid-call"
            )

        self.assertEqual(no_match.status, ToolResultStatus.SUCCESS)
        self.assertEqual(no_match.content, "No matches found.")
        self.assertEqual(invalid.status, ToolResultStatus.ERROR)

    def test_grep_search_fails_clearly_when_no_search_command_exists(self) -> None:
        with patch("coding_agent.tools.grep_search.shutil.which", return_value=None):
            result = self.execute(
                "grep_search", {"pattern": "anything"}, "no-search-command-call"
            )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("requires either rg or grep", result.content)


if __name__ == "__main__":
    unittest.main()
