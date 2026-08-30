import os
import tempfile
import unittest
from pathlib import Path

from coding_agent.core.messages import ToolCall
from coding_agent.core.results import ToolResult
from coding_agent.core.types import ToolResultStatus
from coding_agent.permissions import PermissionDecision, PermissionOperation, PermissionRequest
from coding_agent.tools import (
    ArtifactStore,
    ToolContext,
    ToolExecutor,
    ToolResultProcessor,
    WriteFileTool,
)


class RecordingPermissionHandler:
    def __init__(self, decision: PermissionDecision) -> None:
        self.decision = decision
        self.requests: list[PermissionRequest] = []

    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.requests.append(request)
        return self.decision


class WriteFileToolTests(unittest.TestCase):
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
            (WriteFileTool(),),
            ToolResultProcessor(self.store),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(self, path: str, content: str = "content") -> ToolResult:
        return self.executor.execute(
            ToolCall(
                id="write-call",
                name="write_file",
                arguments={"path": path, "content": content},
            ),
            self.context,
        )

    def test_creates_exact_utf8_content_and_records_a_trusted_version(self) -> None:
        content = "first\r\n第二行"

        result = self.execute("new.txt", content)

        path = self.workspace / "new.txt"
        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual(path.read_bytes(), content.encode("utf-8"))
        self.assertEqual(result.metadata["bytes"], len(content.encode("utf-8")))
        self.assertEqual(len(self.handler.requests), 1)
        self.assertIs(self.handler.requests[0].operation, PermissionOperation.WRITE)
        self.assertEqual(self.handler.requests[0].target, str(path))
        self.assertEqual(set(self.versions), {"new.txt"})
        self.assertEqual(
            self.versions["new.txt"].to_dict(),
            result.metadata["file_version"],
        )

    def test_empty_file_is_a_valid_new_file(self) -> None:
        result = self.execute("empty.txt", "")

        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertEqual((self.workspace / "empty.txt").read_bytes(), b"")
        self.assertEqual(result.metadata["bytes"], 0)

    def test_existing_path_is_never_overwritten_or_confirmed(self) -> None:
        path = self.workspace / "existing.txt"
        path.write_text("original", encoding="utf-8")

        result = self.execute("existing.txt", "replacement")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("edit_file", result.content)
        self.assertEqual(path.read_text(encoding="utf-8"), "original")
        self.assertEqual(self.handler.requests, [])
        self.assertEqual(self.versions, {})

    def test_missing_parent_is_rejected_without_creating_directories(self) -> None:
        result = self.execute("missing/new.txt")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("parent directory does not exist", result.content)
        self.assertFalse((self.workspace / "missing").exists())
        self.assertEqual(self.handler.requests, [])

    def test_declined_write_does_not_create_a_file(self) -> None:
        self.handler.decision = PermissionDecision.DENY

        result = self.execute("declined.txt")

        self.assertEqual(result.status, ToolResultStatus.DENIED)
        self.assertFalse((self.workspace / "declined.txt").exists())
        self.assertEqual(len(self.handler.requests), 1)

    def test_outside_artifact_and_broken_symlink_are_rejected_without_asking(self) -> None:
        outside = self.root / "outside.txt"
        artifact = self.store.root / "agent.txt"
        broken_target = self.workspace / "target.txt"
        os.symlink(broken_target, self.workspace / "link.txt")

        outside_result = self.execute(str(outside))
        artifact_result = self.execute(str(artifact))
        symlink_result = self.execute("link.txt")

        self.assertEqual(outside_result.status, ToolResultStatus.DENIED)
        self.assertEqual(artifact_result.status, ToolResultStatus.DENIED)
        self.assertEqual(symlink_result.status, ToolResultStatus.ERROR)
        self.assertFalse(outside.exists())
        self.assertFalse(artifact.exists())
        self.assertFalse(broken_target.exists())
        self.assertEqual(self.handler.requests, [])

    def test_exclusive_create_prevents_a_confirmation_race_from_overwriting(self) -> None:
        class RacingHandler:
            def __call__(self, request: PermissionRequest) -> PermissionDecision:
                Path(request.target).write_text("racer", encoding="utf-8")
                return PermissionDecision.ALLOW

        self.context = ToolContext(
            session_id=self.context.session_id,
            workspace_root=self.context.workspace_root,
            artifact_root=self.context.artifact_root,
            cwd=self.context.cwd,
            permission_handler=RacingHandler(),
            read_file_versions=self.versions,
        )

        result = self.execute("race.txt", "agent")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("already exists", result.content)
        self.assertEqual((self.workspace / "race.txt").read_text(encoding="utf-8"), "racer")
        self.assertEqual(self.versions, {})


if __name__ == "__main__":
    unittest.main()
