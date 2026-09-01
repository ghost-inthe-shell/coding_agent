import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, Mock, patch

from coding_agent import cli
from coding_agent.cli import InteractivePermissionHandler, build_parser, run_repl
from coding_agent.core.results import CompactionResult, RunResult
from coding_agent.core.session import SessionState
from coding_agent.core.session_store import SessionNotFoundError, SessionSaveError
from coding_agent.core.types import RunStatus
from coding_agent.core.usage import Usage
from coding_agent.permissions import PermissionDecision, PermissionOperation, PermissionRequest
from coding_agent.providers import ApiDialect, ReasoningLevel
from coding_agent.ui import ColorMode, TerminalRenderer


class RecordingRuntime:
    def __init__(
        self,
        results: list[RunResult],
        compaction_results: list[CompactionResult] | None = None,
    ) -> None:
        self.results = iter(results)
        self.compaction_results = iter(compaction_results or ())
        self.calls: list[tuple[SessionState, str]] = []
        self.compaction_calls: list[SessionState] = []

    def run_turn(self, state: SessionState, user_input: str) -> RunResult:
        self.calls.append((state, user_input))
        return next(self.results)

    def compact(self, state: SessionState) -> CompactionResult:
        self.compaction_calls.append(state)
        return next(self.compaction_results)


class RecordingSessionStore:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.saved: list[SessionState] = []

    def save(self, state: SessionState) -> Path:
        self.saved.append(state)
        if self.error is not None:
            raise self.error
        return Path("/state/session.json")


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
        self.assertIn("Session: session-1", output.getvalue())
        self.assertIn("assistant> first answer", output.getvalue())
        self.assertIn("assistant> second answer", output.getvalue())

    def test_trailing_backslash_builds_one_multiline_prompt(self) -> None:
        runtime = RecordingRuntime([RunResult(status=RunStatus.COMPLETED, final_text="answer")])
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            input_stream=StringIO("  first line\\\n\\\n  second line  \n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call[1] for call in runtime.calls],
            ["  first line\n\n  second line  "],
        )
        self.assertIn("> ... ... ", output.getvalue())

    def test_eof_during_continuation_does_not_send_partial_prompt(self) -> None:
        runtime = RecordingRuntime([])
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            input_stream=StringIO("partial\\\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(runtime.calls, [])
        self.assertIn("> ... ", output.getvalue())

    def test_even_trailing_backslashes_are_submitted_literally(self) -> None:
        runtime = RecordingRuntime([RunResult(status=RunStatus.COMPLETED, final_text="answer")])

        exit_code = run_repl(
            runtime,
            self.state,
            input_stream=StringIO("literal\\\\\n/exit\n"),
            output_stream=StringIO(),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual([call[1] for call in runtime.calls], ["literal\\\\"])

    def test_repl_checkpoints_after_each_completed_turn(self) -> None:
        runtime = RecordingRuntime(
            [
                RunResult(status=RunStatus.COMPLETED, final_text="first answer"),
                RunResult(
                    status=RunStatus.PROVIDER_ERROR,
                    final_text="",
                    error_message="offline",
                ),
            ]
        )
        store = RecordingSessionStore()

        exit_code = run_repl(
            runtime,
            self.state,
            session_store=store,
            input_stream=StringIO("first\nsecond\n/exit\n"),
            output_stream=StringIO(),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(store.saved, [self.state, self.state])

    def test_checkpoint_failure_terminates_before_another_turn(self) -> None:
        runtime = RecordingRuntime(
            [
                RunResult(status=RunStatus.COMPLETED, final_text="answer"),
                RunResult(status=RunStatus.COMPLETED, final_text="must not run"),
            ]
        )
        store = RecordingSessionStore(SessionSaveError("disk full"))
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            session_store=store,
            input_stream=StringIO("first\nsecond\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual([call[1] for call in runtime.calls], ["first"])
        self.assertIn("[session_error]", output.getvalue())
        self.assertIn("REPL terminated", output.getvalue())

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
        self.assertEqual(runtime.compaction_calls, [])
        self.assertIn("/compact", output.getvalue())
        self.assertIn("/exit", output.getvalue())
        self.assertIn("Unknown command: /unknown", output.getvalue())

    def test_compact_runs_at_repl_boundary_and_saves_changed_state(self) -> None:
        runtime = RecordingRuntime(
            [],
            [
                CompactionResult(
                    compacted=True,
                    usage=Usage(input_tokens=40, output_tokens=8),
                    summarized_messages=3,
                    tokens_before=1_200,
                    tokens_after=450,
                )
            ],
        )
        store = RecordingSessionStore()
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            session_store=store,
            input_stream=StringIO("/compact\n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(runtime.compaction_calls, [self.state])
        self.assertEqual(store.saved, [self.state])
        self.assertIn("[compacted] summarized_messages=3", output.getvalue())
        self.assertIn("tokens_before=1200", output.getvalue())
        self.assertIn("tokens_after=450", output.getvalue())

    def test_compact_noop_does_not_write_an_identical_checkpoint(self) -> None:
        runtime = RecordingRuntime([], [CompactionResult(compacted=False)])
        store = RecordingSessionStore()
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            session_store=store,
            input_stream=StringIO("/compact\n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(store.saved, [])
        self.assertIn("[compact] nothing to compact", output.getvalue())

    def test_compact_without_token_reduction_reports_metrics_and_saves_usage(self) -> None:
        runtime = RecordingRuntime(
            [],
            [
                CompactionResult(
                    compacted=False,
                    usage=Usage(input_tokens=20, output_tokens=10),
                    tokens_before=120,
                    tokens_after=180,
                )
            ],
        )
        store = RecordingSessionStore()
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            session_store=store,
            input_stream=StringIO("/compact\n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(store.saved, [self.state])
        self.assertIn("[compact] not applied", output.getvalue())
        self.assertIn("tokens_before=120", output.getvalue())
        self.assertIn("tokens_after=180", output.getvalue())

    def test_invalid_compaction_is_reported_and_usage_is_checkpointed(self) -> None:
        runtime = RecordingRuntime(
            [],
            [
                CompactionResult(
                    compacted=False,
                    usage=Usage(input_tokens=20, output_tokens=2),
                    error_message="summary was truncated",
                )
            ],
        )
        store = RecordingSessionStore()
        output = StringIO()

        exit_code = run_repl(
            runtime,
            self.state,
            session_store=store,
            input_stream=StringIO("/compact\n/exit\n"),
            output_stream=output,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(store.saved, [self.state])
        self.assertIn("[compact_error] summary was truncated", output.getvalue())

    def test_non_completed_result_reports_status_and_error(self) -> None:
        runtime = RecordingRuntime(
            [
                RunResult(
                    status=RunStatus.PROVIDER_ERROR,
                    final_text="",
                    usage=Usage(output_tokens=12, reasoning_tokens=7),
                    model_turns=2,
                    tool_calls=1,
                    max_output_tokens=16_384,
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
        rendered = output.getvalue()
        self.assertIn("[provider_error] offline", rendered)
        self.assertIn("model_turns=2", rendered)
        self.assertIn("tool_calls=1", rendered)
        self.assertIn("output_tokens=12", rendered)
        self.assertIn("reasoning_tokens=7", rendered)
        self.assertIn("max_output_tokens_per_call=16384", rendered)

    def test_parser_uses_shared_default_output_limit(self) -> None:
        arguments = build_parser().parse_args(["--model", "model"])

        self.assertEqual(arguments.max_tokens, 16_384)
        self.assertEqual(arguments.max_turns, 32)
        self.assertEqual(arguments.context_window, 128_000)
        self.assertEqual(arguments.api_dialect, "generic")
        self.assertEqual(arguments.reasoning, "default")
        self.assertEqual(arguments.color, "auto")
        self.assertEqual(arguments.thinking_display, "brief")
        self.assertTrue(arguments.stream)
        self.assertTrue(build_parser().parse_args(["--model", "model", "--stream"]).stream)
        self.assertFalse(
            build_parser().parse_args(["--model", "model", "--stream", "--no-stream"]).stream
        )

    def test_parser_does_not_expose_internal_minimal_reasoning(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--model", "model", "--reasoning", "minimal"])

    def test_main_rejects_openai_reasoning_options_for_anthropic(self) -> None:
        invalid_options = (
            ("--api-dialect", "deepseek"),
            ("--reasoning", "low"),
        )
        for option, value in invalid_options:
            with self.subTest(option=option), self.assertRaises(SystemExit):
                cli.main(
                    [
                        "--provider",
                        "anthropic",
                        "--model",
                        "model",
                        option,
                        value,
                    ]
                )

    def test_create_provider_passes_explicit_dialect_and_reasoning(self) -> None:
        provider = object()
        with patch.object(
            cli,
            "OpenAICompatibleProvider",
            return_value=provider,
        ) as provider_class:
            result = cli._create_provider(
                "openai-compatible",
                "model",
                base_url="https://api.example.test",
                max_tokens=8_192,
                api_dialect=ApiDialect.DEEPSEEK,
                reasoning=ReasoningLevel.LOW,
                stream=True,
            )

        self.assertIs(result, provider)
        provider_class.assert_called_once_with(
            "model",
            base_url="https://api.example.test",
            max_tokens=8_192,
            dialect=ApiDialect.DEEPSEEK,
            reasoning=ReasoningLevel.LOW,
            stream=True,
        )

    def test_workspace_and_resume_are_mutually_exclusive(self) -> None:
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--model",
                    "model",
                    "--workspace",
                    str(self.workspace),
                    "--resume",
                    "session-1",
                ]
            )

    def test_main_rejects_context_window_without_output_reserve(self) -> None:
        with (
            patch.object(cli, "_create_provider") as create_provider,
            self.assertRaises(SystemExit),
        ):
            cli.main(
                [
                    "--model",
                    "model",
                    "--context-window",
                    "16000",
                    "--max-tokens",
                    "15000",
                ]
            )

        create_provider.assert_not_called()

    def test_main_loads_explicit_session_before_starting_repl(self) -> None:
        store = Mock()
        store.load.return_value = self.state
        runtime = object()

        with (
            patch.object(cli, "SessionStore", return_value=store),
            patch.object(cli, "_create_provider", return_value=object()),
            patch.object(cli, "Runtime", return_value=runtime),
            patch.object(cli, "run_repl", return_value=0) as repl,
        ):
            exit_code = cli.main(["--model", "model", "--resume", "session-1"])

        self.assertEqual(exit_code, 0)
        store.load.assert_called_once_with("session-1")
        store.save.assert_not_called()
        repl.assert_called_once_with(
            runtime,
            self.state,
            session_store=store,
            renderer=ANY,
        )
        self.assertIsInstance(repl.call_args.kwargs["renderer"], TerminalRenderer)

    def test_main_saves_new_session_before_starting_repl(self) -> None:
        store = Mock()
        runtime = object()

        with (
            patch.object(cli, "SessionStore", return_value=store),
            patch.object(cli, "_create_provider", return_value=object()),
            patch.object(cli, "Runtime", return_value=runtime) as runtime_class,
            patch.object(cli, "run_repl", return_value=0) as repl,
        ):
            exit_code = cli.main(
                [
                    "--model",
                    "model",
                    "--workspace",
                    str(self.workspace),
                    "--max-turns",
                    "64",
                ]
            )

        self.assertEqual(exit_code, 0)
        state = store.save.call_args.args[0]
        self.assertEqual(state.workspace_root, str(self.workspace))
        self.assertEqual(len(state.session_id), 32)
        self.assertEqual(runtime_class.call_args.kwargs["limits"].max_model_calls, 64)
        self.assertIs(
            runtime_class.call_args.kwargs["event_sink"],
            repl.call_args.kwargs["renderer"],
        )
        repl.assert_called_once_with(runtime, state, session_store=store, renderer=ANY)
        self.assertIsInstance(repl.call_args.kwargs["renderer"], TerminalRenderer)

    def test_main_fails_before_provider_creation_when_resume_cannot_load(self) -> None:
        store = Mock()
        store.load.side_effect = SessionNotFoundError("session not found: missing")
        error_output = StringIO()

        with (
            patch.object(cli, "SessionStore", return_value=store),
            patch.object(cli, "_create_provider") as create_provider,
            patch.object(cli.sys, "stderr", error_output),
        ):
            exit_code = cli.main(["--model", "model", "--resume", "missing"])

        self.assertEqual(exit_code, 1)
        create_provider.assert_not_called()
        self.assertIn("session not found: missing", error_output.getvalue())

    def test_main_does_not_enter_repl_when_initial_checkpoint_fails(self) -> None:
        store = Mock()
        store.save.side_effect = SessionSaveError("disk full")
        error_output = StringIO()

        with (
            patch.object(cli, "SessionStore", return_value=store),
            patch.object(cli, "_create_provider", return_value=object()),
            patch.object(cli, "Runtime"),
            patch.object(cli, "run_repl") as repl,
            patch.object(cli.sys, "stderr", error_output),
        ):
            exit_code = cli.main(["--model", "model", "--workspace", str(self.workspace)])

        self.assertEqual(exit_code, 1)
        repl.assert_not_called()
        self.assertIn("disk full", error_output.getvalue())

    def test_real_tty_uses_terminal_input_editor(self) -> None:
        runtime = RecordingRuntime([RunResult(status=RunStatus.COMPLETED, final_text="answer")])
        input_stream = InteractiveStringIO()
        output_stream = InteractiveStringIO()
        readline = Mock()

        with (
            patch.object(cli, "_readline", readline),
            patch.object(cli.sys, "stdin", input_stream),
            patch.object(cli.sys, "stdout", output_stream),
            patch(
                "builtins.input",
                side_effect=["edited\r\nquestion", "/exit"],
            ) as terminal_input,
        ):
            exit_code = run_repl(
                runtime,
                self.state,
                input_stream=input_stream,
                output_stream=output_stream,
                renderer=TerminalRenderer(
                    output_stream,
                    color=ColorMode.NEVER,
                ),
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual([call[1] for call in runtime.calls], ["edited\nquestion"])
        readline.parse_and_bind.assert_called_once_with("set enable-bracketed-paste on")
        self.assertEqual(
            [call.args[0] for call in terminal_input.call_args_list],
            ["> ", "> "],
        )

    def test_empty_terminal_line_is_not_treated_as_eof(self) -> None:
        runtime = RecordingRuntime([])
        input_stream = InteractiveStringIO()
        output_stream = InteractiveStringIO()

        with (
            patch.object(cli.sys, "stdin", input_stream),
            patch.object(cli.sys, "stdout", output_stream),
            patch("builtins.input", side_effect=["", "/exit"]) as terminal_input,
        ):
            exit_code = run_repl(
                runtime,
                self.state,
                input_stream=input_stream,
                output_stream=output_stream,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(terminal_input.call_count, 2)

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

        decision = handler(PermissionRequest(PermissionOperation.READ, "/outside/file.txt"))

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

        decision = handler(PermissionRequest(PermissionOperation.WRITE, "/workspace/new.py"))

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

    def test_execute_permission_keeps_the_complete_long_command(self) -> None:
        output = StringIO()
        handler = InteractivePermissionHandler(
            input_stream=StringIO("n\n"),
            output_stream=output,
        )
        command = "printf " + "x" * 300

        decision = handler(PermissionRequest(PermissionOperation.EXECUTE, command))

        self.assertEqual(decision, PermissionDecision.DENY)
        lines = output.getvalue().splitlines()
        command_index = next(index for index, line in enumerate(lines) if "Command:" in line)
        target_lines = []
        for line in lines[command_index + 1 :]:
            if line.startswith("╰"):
                break
            target_lines.append(line[4:-2].rstrip())
        self.assertEqual("".join(target_lines), json.dumps(command))
        self.assertNotIn("chars]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
