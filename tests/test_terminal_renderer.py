import unittest
from io import StringIO

from coding_agent.core.results import CompactionResult, RunResult
from coding_agent.core.types import RunStatus
from coding_agent.core.usage import Usage
from coding_agent.ui import ColorMode, TerminalRenderer


class TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class TerminalRendererTests(unittest.TestCase):
    def test_auto_color_is_plain_for_non_terminal_output(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, environment={})

        renderer.banner("session-1", "/workspace")
        renderer.run_result(RunResult(status=RunStatus.COMPLETED, final_text="answer"))

        self.assertNotIn("\x1b", output.getvalue())
        self.assertIn("Coding Agent\n", output.getvalue())
        self.assertIn("assistant> answer\n", output.getvalue())

    def test_auto_color_styles_terminal_output(self) -> None:
        output = TTYStringIO()
        renderer = TerminalRenderer(output, environment={})

        renderer.assistant("answer")

        self.assertTrue(renderer.color_enabled)
        self.assertIn("\x1b[32m", output.getvalue())
        self.assertIn("assistant>\x1b[0m answer", output.getvalue())

    def test_no_color_environment_disables_auto_color(self) -> None:
        output = TTYStringIO()
        renderer = TerminalRenderer(output, environment={"NO_COLOR": ""})

        renderer.assistant("answer")

        self.assertFalse(renderer.color_enabled)
        self.assertEqual(output.getvalue(), "assistant> answer\n")

    def test_explicit_color_modes_override_terminal_detection(self) -> None:
        plain_terminal = TerminalRenderer(
            TTYStringIO(),
            color=ColorMode.NEVER,
            environment={},
        )
        colored_pipe = TerminalRenderer(
            StringIO(),
            color=ColorMode.ALWAYS,
            environment={"NO_COLOR": ""},
        )

        self.assertFalse(plain_terminal.color_enabled)
        self.assertTrue(colored_pipe.color_enabled)

    def test_readline_prompt_marks_escape_sequences_as_non_printing(self) -> None:
        renderer = TerminalRenderer(
            TTYStringIO(),
            color=ColorMode.ALWAYS,
            environment={},
        )

        prompt = renderer.input_prompt(continuation=False, readline=True)
        continuation = renderer.input_prompt(continuation=True, readline=True)

        self.assertEqual(prompt, "\001\x1b[1m\x1b[36m\002> \001\x1b[0m\002")
        self.assertIn("... ", continuation)

    def test_non_completed_result_keeps_diagnostics(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)

        renderer.run_result(
            RunResult(
                status=RunStatus.PROVIDER_ERROR,
                final_text="",
                usage=Usage(output_tokens=12, reasoning_tokens=7),
                model_turns=2,
                tool_calls=1,
                max_output_tokens=16_384,
                error_message="offline",
            )
        )

        rendered = output.getvalue()
        self.assertIn("[provider_error] offline", rendered)
        self.assertIn("model_turns=2", rendered)
        self.assertIn("max_output_tokens_per_call=16384", rendered)

    def test_compaction_result_uses_existing_plain_text_protocol(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)

        renderer.compaction_result(
            CompactionResult(
                compacted=True,
                summarized_messages=3,
                tokens_before=1_200,
                tokens_after=450,
            )
        )

        self.assertEqual(
            output.getvalue(),
            "[compacted] summarized_messages=3, tokens_before=1200, tokens_after=450\n",
        )


if __name__ == "__main__":
    unittest.main()
