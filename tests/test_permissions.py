import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from coding_agent.permissions import (
    PathAccessDenied,
    PermissionDecision,
    PermissionOperation,
    PermissionRequest,
    WritePathPolicy,
)
from coding_agent.tools import ToolContext


class RecordingPermissionHandler:
    def __init__(self, decision: PermissionDecision) -> None:
        self.decision = decision
        self.requests: list[PermissionRequest] = []

    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.requests.append(request)
        return self.decision


class WritePathPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.artifact = self.root / "artifacts"
        self.workspace.mkdir()
        self.artifact.mkdir()
        self.handler = RecordingPermissionHandler(PermissionDecision.ALLOW)
        self.context = ToolContext(
            session_id="session-1",
            workspace_root=str(self.workspace),
            artifact_root=str(self.artifact),
            cwd=str(self.workspace),
            permission_handler=self.handler,
        )
        self.policy = WritePathPolicy()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_workspace_write_is_resolved_before_each_delayed_confirmation(self) -> None:
        target = self.policy.resolve("new.py", self.context)

        self.assertEqual(target, self.workspace / "new.py")
        self.assertEqual(self.handler.requests, [])

        first = self.policy.authorize(target, self.context)
        second = self.policy.authorize(target, self.context)

        self.assertEqual(first, target)
        self.assertEqual(second, target)
        self.assertEqual(len(self.handler.requests), 2)
        self.assertTrue(
            all(
                request.operation is PermissionOperation.WRITE
                for request in self.handler.requests
            )
        )
        self.assertEqual(self.handler.requests[0].target, str(target))

    def test_workspace_write_fails_closed_without_approval(self) -> None:
        target = self.policy.resolve("new.py", self.context)
        without_handler = replace(self.context, permission_handler=None)
        declining_handler = RecordingPermissionHandler(PermissionDecision.DENY)
        declining = replace(self.context, permission_handler=declining_handler)

        with self.assertRaisesRegex(PathAccessDenied, "not approved"):
            self.policy.authorize(target, without_handler)
        with self.assertRaisesRegex(PathAccessDenied, "not approved"):
            self.policy.authorize(target, declining)

        self.assertEqual(len(declining_handler.requests), 1)

    def test_outside_and_symlink_escape_are_denied_without_asking(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        outside_file = outside / "file.txt"
        outside_file.write_text("outside\n", encoding="utf-8")
        os.symlink(outside_file, self.workspace / "file-link.txt")
        os.symlink(outside, self.workspace / "dir-link")

        for requested in (
            str(outside_file),
            "file-link.txt",
            "dir-link/new.txt",
        ):
            with self.subTest(requested=requested), self.assertRaisesRegex(
                PathAccessDenied, "outside the workspace"
            ):
                self.policy.resolve(requested, self.context)

        self.assertEqual(self.handler.requests, [])

    def test_authorize_rechecks_parent_symlinks_before_asking(self) -> None:
        directory = self.workspace / "directory"
        directory.mkdir()
        target = self.policy.resolve("directory/new.txt", self.context)
        outside = self.root / "outside-after-resolve"
        outside.mkdir()
        directory.rmdir()
        os.symlink(outside, directory)

        with self.assertRaisesRegex(PathAccessDenied, "outside the workspace"):
            self.policy.authorize(target, self.context)

        self.assertEqual(self.handler.requests, [])

    def test_authorize_rechecks_the_target_after_approval(self) -> None:
        directory = self.workspace / "during-confirmation"
        directory.mkdir()
        outside = self.root / "outside-during-confirmation"
        outside.mkdir()
        target = self.policy.resolve("during-confirmation/new.txt", self.context)

        class RetargetingHandler:
            def __call__(self, request: PermissionRequest) -> PermissionDecision:
                directory.rmdir()
                os.symlink(outside, directory)
                return PermissionDecision.ALLOW

        context = replace(self.context, permission_handler=RetargetingHandler())

        with self.assertRaisesRegex(PathAccessDenied, "outside the workspace"):
            self.policy.authorize(target, context)

    def test_artifact_is_denied_even_when_nested_in_workspace(self) -> None:
        nested_artifact = self.workspace / ".agent-artifacts"
        nested_artifact.mkdir()
        nested_context = replace(self.context, artifact_root=str(nested_artifact))

        for requested, context in (
            (self.artifact / "result.txt", self.context),
            (nested_artifact / "result.txt", nested_context),
        ):
            with self.subTest(requested=requested), self.assertRaisesRegex(
                PathAccessDenied, "agent-owned artifacts"
            ):
                self.policy.resolve(str(requested), context)

        self.assertEqual(self.handler.requests, [])


if __name__ == "__main__":
    unittest.main()
