import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from coding_agent.core.messages import ToolCall
from coding_agent.core.results import ToolResult
from coding_agent.core.types import ToolResultStatus
from coding_agent.permissions import PermissionDecision, PermissionOperation, PermissionRequest
from coding_agent.tools import (
    ArtifactStore,
    RunShellTool,
    ToolContext,
    ToolExecutor,
    ToolResultProcessor,
)
from coding_agent.tools.process import ProcessOutput


class RecordingPermissionHandler:
    def __init__(self, decision: PermissionDecision) -> None:
        self.decision = decision
        self.requests: list[PermissionRequest] = []

    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        self.requests.append(request)
        return self.decision


class RunShellToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.store = ArtifactStore("session-1", state_home=self.root / "state")
        self.handler = RecordingPermissionHandler(PermissionDecision.ALLOW)
        self.context = ToolContext(
            session_id="session-1",
            workspace_root=str(self.workspace),
            artifact_root=str(self.store.root),
            cwd=str(self.workspace),
            permission_handler=self.handler,
        )
        self.executor = ToolExecutor(
            (RunShellTool(),),
            ToolResultProcessor(self.store),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute(self, command: str, timeout_seconds=None, *, context=None) -> ToolResult:
        arguments = {"command": command}
        if timeout_seconds is not None:
            arguments["timeout_seconds"] = timeout_seconds
        return self.executor.execute(
            ToolCall(id="shell-call", name="run_shell", arguments=arguments),
            context or self.context,
        )

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_executes_approved_command_with_fixed_cwd_and_sanitized_env(self, runner) -> None:
        runner.return_value = _output(stdout="hello\n", stderr="warning\n", duration_ms=12)

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-secret",
                "ANTHROPIC_API_KEY": "anthropic-secret",
                "VISIBLE_SETTING": "visible",
            },
        ):
            result = self.execute("printf hello")

        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertIn("Exited with code 0", result.content)
        self.assertIn("stdout:\nhello\n", result.content)
        self.assertIn("stderr:\nwarning\n", result.content)
        self.assertEqual(result.metadata["cwd"], str(self.workspace))
        self.assertEqual(result.metadata["timeout_seconds"], 120)
        self.assertEqual(result.metadata["duration_ms"], 12)
        self.assertEqual(len(self.handler.requests), 1)
        self.assertIs(self.handler.requests[0].operation, PermissionOperation.EXECUTE)
        self.assertEqual(self.handler.requests[0].target, "printf hello")

        call = runner.call_args
        self.assertEqual(call.args[0], ["/bin/bash", "-c", "printf hello"])
        self.assertEqual(call.kwargs["cwd"], self.workspace)
        self.assertEqual(call.kwargs["timeout_seconds"], 120)
        environment = call.kwargs["env"]
        self.assertEqual(environment["VISIBLE_SETTING"], "visible")
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("ANTHROPIC_API_KEY", environment)

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_each_command_requires_a_new_approval(self, runner) -> None:
        runner.return_value = _output()

        first = self.execute("true")
        second = self.execute("true")

        self.assertEqual(first.status, ToolResultStatus.SUCCESS)
        self.assertEqual(second.status, ToolResultStatus.SUCCESS)
        self.assertEqual(len(self.handler.requests), 2)
        self.assertEqual(runner.call_count, 2)

    def test_real_bash_uses_workspace_and_does_not_receive_provider_keys(self) -> None:
        command = (
            "printf '%s|%s|%s|%s' "
            '"$PWD" "$VISIBLE_SETTING" '
            '"${OPENAI_API_KEY-unset}" "${ANTHROPIC_API_KEY-unset}"'
        )
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-secret",
                "ANTHROPIC_API_KEY": "anthropic-secret",
                "VISIBLE_SETTING": "visible",
            },
        ):
            result = self.execute(command)

        self.assertEqual(result.status, ToolResultStatus.SUCCESS)
        self.assertIn(
            f"stdout:\n{self.workspace}|visible|unset|unset",
            result.content,
        )

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_explicit_timeout_is_bounded_by_the_tool_schema(self, runner) -> None:
        runner.return_value = _output()

        maximum = self.execute("true", 600)
        zero = self.execute("true", 0)
        too_large = self.execute("true", 601)
        fractional = self.execute("true", 1.5)

        self.assertEqual(maximum.status, ToolResultStatus.SUCCESS)
        self.assertEqual(runner.call_args.kwargs["timeout_seconds"], 600)
        for invalid in (zero, too_large, fractional):
            self.assertEqual(invalid.status, ToolResultStatus.ERROR)
            self.assertIn("invalid arguments", invalid.content)
        self.assertEqual(len(self.handler.requests), 1)
        self.assertEqual(runner.call_count, 1)

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_decline_or_missing_handler_fails_closed(self, runner) -> None:
        self.handler.decision = PermissionDecision.DENY
        declined = self.execute("true")
        no_handler = self.execute("true", context=replace(self.context, permission_handler=None))

        self.assertEqual(declined.status, ToolResultStatus.DENIED)
        self.assertEqual(no_handler.status, ToolResultStatus.DENIED)
        self.assertEqual(len(self.handler.requests), 1)
        runner.assert_not_called()

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_timeout_nonzero_exit_and_output_limit_are_distinct_errors(self, runner) -> None:
        runner.side_effect = (
            _output(stdout="partial\n", returncode=-9, timed_out=True),
            _output(stderr="failed\n", returncode=7),
            _output(stdout="x" * 100, returncode=-15, incomplete=True),
        )

        timed_out = self.execute("sleep 999")
        failed = self.execute("false")
        incomplete = self.execute("yes")

        self.assertEqual(timed_out.status, ToolResultStatus.TIMEOUT)
        self.assertTrue(timed_out.metadata["timed_out"])
        self.assertIn("Timed out after 120 seconds", timed_out.content)
        self.assertEqual(failed.status, ToolResultStatus.ERROR)
        self.assertEqual(failed.metadata["exit_code"], 7)
        self.assertIn("Exited with code 7", failed.content)
        self.assertEqual(incomplete.status, ToolResultStatus.ERROR)
        self.assertTrue(incomplete.metadata["artifact_incomplete"])
        self.assertIn("raw output limit", incomplete.content)

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_spawn_failure_is_returned_as_a_tool_error(self, runner) -> None:
        runner.side_effect = OSError("spawn blocked")

        result = self.execute("true")

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("spawn blocked", result.content)
        self.assertEqual(result.metadata["exit_code"], None)

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_workspace_change_during_confirmation_is_denied(self, runner) -> None:
        outside = self.root / "outside"
        outside.mkdir()

        class RetargetingHandler:
            def __call__(self, request: PermissionRequest) -> PermissionDecision:
                self_workspace = Path(self_context.workspace_root)
                self_workspace.rename(self_workspace.with_name("workspace-old"))
                os.symlink(outside, self_workspace)
                return PermissionDecision.ALLOW

        self_context = replace(self.context, permission_handler=RetargetingHandler())

        result = self.execute("true", context=self_context)

        self.assertEqual(result.status, ToolResultStatus.DENIED)
        self.assertIn("workspace changed", result.content)
        runner.assert_not_called()

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_invalid_workspace_is_rejected_without_asking(self, runner) -> None:
        missing = self.root / "missing"
        context = replace(
            self.context,
            workspace_root=str(missing),
            cwd=str(missing),
        )

        result = self.execute("true", context=context)

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("workspace does not exist", result.content)
        self.assertEqual(self.handler.requests, [])
        runner.assert_not_called()

    @patch("coding_agent.tools.run_shell.run_limited_process")
    def test_invalid_command_text_is_rejected_without_asking(self, runner) -> None:
        whitespace = self.execute(" \n\t")
        nul = self.execute("printf ok\x00")
        invalid_utf8 = self.execute("printf \ud800")

        for result in (whitespace, nul, invalid_utf8):
            self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("non-whitespace", whitespace.content)
        self.assertIn("NUL", nul.content)
        self.assertIn("invalid arguments", invalid_utf8.content)
        self.assertEqual(self.handler.requests, [])
        runner.assert_not_called()


def _output(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    incomplete: bool = False,
    timed_out: bool = False,
    duration_ms: int = 1,
) -> ProcessOutput:
    return ProcessOutput(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        incomplete=incomplete,
        timed_out=timed_out,
        duration_ms=duration_ms,
    )


if __name__ == "__main__":
    unittest.main()
