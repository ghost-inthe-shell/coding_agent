import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from coding_agent import cli
from coding_agent.cli import InteractivePermissionHandler, run_repl
from coding_agent.core.results import RunResult
from coding_agent.core.session import SessionState
from coding_agent.core.types import RunStatus
from coding_agent.permissions import PermissionDecision, PermissionOperation, PermissionRequest


class RecordingRuntime:
    def __init__(self, results: list[RunResult]) -> None:
        self.results = iter(results)
        self.calls: list[tuple[SessionState, str]] = []

    def run_turn(self, state: SessionState, user_input: str) -> RunResult:
        self.calls.append((state, user_input))
        return next(self.results)


class InteractiveStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class ReplTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.state = SessionState(
            session_id="session-1",
            workspace_root=str(self.workspace),
            system_prompt="System prompt.",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_repl_reuses_one_session_for_multiple_turns(self) -> None:
        runtime = RecordingRuntime(
            [
                RunResult(status=RunStatus.COMPLETED, final_text="first answer"),
                RunResult(status=RunStatus.COMPLETED, final_text="second answer"),
            ]
        )
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            input_stream=StringIO("first question\n\nsecond question\n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual([call[1] for call in runtime.calls], ["first question", "second question"])
        self.assertIs(runtime.calls[0][0], self.state)
        self.assertIs(runtime.calls[1][0], self.state)
        self.assertIn("assistant> first answer", output.getvalue())
        self.assertIn("assistant> second answer", output.getvalue())

    def test_help_unknown_command_and_eof_do_not_call_runtime(self) -> None:
        runtime = RecordingRuntime([])
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            input_stream=StringIO("/help\n/unknown\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(runtime.calls, [])
        self.assertIn("/exit", output.getvalue())
        self.assertIn("Unknown command: /unknown", output.getvalue())

    def test_non_completed_result_reports_status_and_error(self) -> None:
        runtime = RecordingRuntime(
            [
                RunResult(
                    status=RunStatus.PROVIDER_ERROR,
                    final_text="",
                    error_message="offline",
                )
            ]
        )
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            input_stream=StringIO("hello\n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("[provider_error] offline", output.getvalue())

    def test_real_tty_uses_terminal_input_editor(self) -> None:
        runtime = RecordingRuntime(
            [RunResult(status=RunStatus.COMPLETED, final_text="answer")]
        )
        input_stream = InteractiveStringIO()
        output_stream = InteractiveStringIO()

        with (
            patch.object(cli.sys, "stdin", input_stream),
            patch.object(cli.sys, "stdout", output_stream),
            patch("builtins.input", side_effect=["edited question", "/exit"]) as terminal_input,
        ):
            exit_code = run_repl(
                runtime,
                self.state,
                input_stream=input_stream,
                output_stream=output_stream,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual([call[1] for call in runtime.calls], ["edited question"])
        self.assertEqual(
            [call.args[0] for call in terminal_input.call_args_list],
            ["> ", "> "],
        )

    def test_terminal_eof_uses_normal_repl_exit(self) -> None:
        runtime = RecordingRuntime([])
        input_stream = InteractiveStringIO()
        output_stream = InteractiveStringIO()

        with (
            patch.object(cli.sys, "stdin", input_stream),
            patch.object(cli.sys, "stdout", output_stream),
            patch("builtins.input", side_effect=EOFError),
        ):
            exit_code = run_repl(
                runtime,
                self.state,
                input_stream=input_stream,
                output_stream=output_stream,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(runtime.calls, [])


class InteractivePermissionHandlerTests(unittest.TestCase):
    def test_accepts_yes_for_one_request(self) -> None:
        output = StringIO()
        handler = InteractivePermissionHandler(
            input_stream=StringIO("yes\n"),
            output_stream=output,
        )

        decision = handler(
            PermissionRequest(PermissionOperation.READ, "/outside/file.txt")
        )

        self.assertEqual(decision, PermissionDecision.ALLOW)
        self.assertIn("/outside/file.txt", output.getvalue())
        self.assertIn("Approve once", output.getvalue())
        self.assertIn("Approved", output.getvalue())

    def test_decline_and_eof_fail_closed(self) -> None:
        request = PermissionRequest(PermissionOperation.READ, "/outside/file.txt")
        for answer in ("n\n", ""):
            with self.subTest(answer=answer):
                output = StringIO()
                handler = InteractivePermissionHandler(
                    input_stream=StringIO(answer),
                    output_stream=output,
                )

                decision = handler(request)

                self.assertEqual(decision, PermissionDecision.DENY)
                self.assertIn("Denied", output.getvalue())

    def test_write_request_uses_workspace_prompt(self) -> None:
        output = StringIO()
        handler = InteractivePermissionHandler(
            input_stream=StringIO("y\n"),
            output_stream=output,
        )

        decision = handler(
            PermissionRequest(PermissionOperation.WRITE, "/workspace/new.py")
        )

        self.assertEqual(decision, PermissionDecision.ALLOW)
        self.assertIn("write in the workspace", output.getvalue())

    def test_execute_request_escapes_control_characters_in_command(self) -> None:
        output = StringIO()
        handler = InteractivePermissionHandler(
            input_stream=StringIO("y\n"),
            output_stream=output,
        )

        decision = handler(
            PermissionRequest(
                PermissionOperation.EXECUTE,
                "printf '\x1b[31mred'\nnext",
            )
        )

        rendered = output.getvalue()
        self.assertEqual(decision, PermissionDecision.ALLOW)
        self.assertIn("shell command in the workspace", rendered)
        self.assertIn(r"\u001b[31mred", rendered)
        self.assertIn(r"\nnext", rendered)
        self.assertNotIn("\x1b", rendered)


if __name__ == "__main__":
    unittest.main()
