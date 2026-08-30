from io import StringIO
from pathlib import Path
import tempfile
import unittest

from coding_agent.cli import run_repl
from coding_agent.core.results import RunResult
from coding_agent.core.session import SessionState
from coding_agent.core.types import RunStatus


class RecordingRuntime:
    def __init__(self, results: list[RunResult]) -> None:
        self.results = iter(results)
        self.calls: list[tuple[SessionState, str]] = []

    def run_turn(self, state: SessionState, user_input: str) -> RunResult:
        self.calls.append((state, user_input))
        return next(self.results)


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


if __name__ == "__main__":
    unittest.main()
