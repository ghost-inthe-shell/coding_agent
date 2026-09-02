import unittest
from io import StringIO

from coding_agent.core.events import (
    ModelRequestPurpose,
    ModelRequested,
    ModelResponded,
    ModelTextDelta,
    ModelThinkingDelta,
    ToolFinished,
    ToolStarted,
    TurnStarted,
)
from coding_agent.core.messages import AssistantMessage, TextBlock, ToolCall
from coding_agent.core.results import CompactionResult, RunResult, ToolResult
from coding_agent.core.types import RunStatus, StopReason
from coding_agent.core.usage import Usage
from coding_agent.ui import ColorMode, TerminalRenderer, ThinkingDisplay


class TTYStringIO(StringIO):
    def isatty(self) -> bool:
        return True


class TerminalRendererTests(unittest.TestCase):
    def test_auto_compaction_request_is_visible(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)

        renderer(
            ModelRequested(
                "session-1",
                2,
                purpose=ModelRequestPurpose.COMPACTION,
            )
        )

        self.assertEqual(output.getvalue(), "context> Compacting conversation...\n")

    def test_runtime_deltas_render_once_and_keep_thinking_separate(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(
            output,
            color=ColorMode.NEVER,
            thinking_display=ThinkingDisplay.FULL,
        )
        renderer(TurnStarted("session-1"))
        renderer(ModelRequested("session-1", 1))
        renderer(ModelThinkingDelta("session-1", 1, "inspect "))
        renderer(ModelThinkingDelta("session-1", 1, "first"))
        renderer(ModelTextDelta("session-1", 1, "hel"))
        renderer(ModelTextDelta("session-1", 1, "lo"))
        message = AssistantMessage(
            content=(TextBlock("hello"),),
            provider="test",
            model="model",
        )
        renderer(ModelResponded("session-1", 1, message))

        renderer.run_result(RunResult(status=RunStatus.COMPLETED, final_text="hello"))

        self.assertEqual(
            output.getvalue(),
            "thinking> inspect first\nassistant> hello\n",
        )

    def test_brief_thinking_shows_activity_and_hidden_character_count(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)

        renderer(ModelRequested("session-1", 1))
        renderer(ModelThinkingDelta("session-1", 1, "inspect "))
        renderer(ModelThinkingDelta("session-1", 1, "first"))
        renderer(ModelTextDelta("session-1", 1, "answer"))

        self.assertEqual(
            output.getvalue(),
            "thinking> … (13 chars hidden)\nassistant> answer",
        )

    def test_hidden_thinking_emits_no_terminal_content(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(
            output,
            color=ColorMode.NEVER,
            thinking_display=ThinkingDisplay.HIDDEN,
        )

        renderer(ModelThinkingDelta("session-1", 1, "private reasoning"))
        renderer(ModelTextDelta("session-1", 1, "answer"))

        self.assertEqual(output.getvalue(), "assistant> answer")

    def test_thinking_display_mode_is_strict(self) -> None:
        with self.assertRaises(TypeError):
            TerminalRenderer(StringIO(), thinking_display="full")  # type: ignore[arg-type]

    def test_non_streamed_result_still_renders_at_turn_boundary(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)
        renderer(TurnStarted("session-1"))
        renderer(ModelRequested("session-1", 1))
        message = AssistantMessage(
            content=(TextBlock("answer"),),
            provider="test",
            model="model",
        )
        renderer(ModelResponded("session-1", 1, message))

        renderer.run_result(RunResult(status=RunStatus.COMPLETED, final_text="answer"))

        self.assertEqual(output.getvalue(), "assistant> answer\n")

    def test_tool_events_pair_by_call_id_without_rendering_result_body(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)
        call = ToolCall(
            id="call-1",
            name="read_file",
            arguments={"path": "README.md"},
        )
        assistant = AssistantMessage(
            content=(call,),
            provider="test",
            model="model",
            stop_reason=StopReason.TOOL_USE,
        )
        renderer(ModelResponded("session-1", 1, assistant))
        renderer(ToolStarted("session-1", call))
        renderer(ToolFinished("session-1", ToolResult.from_text("secret").to_message(call)))

        self.assertEqual(
            output.getvalue(),
            'tool> read_file "README.md"\ntool> read_file success\n',
        )
        self.assertNotIn("secret", output.getvalue())

    def test_shell_tool_summary_is_bounded_and_escapes_control_characters(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)
        command = "first\nsecond\n" + "x" * 300 + "\x1b[31msecret"
        call = ToolCall(
            id="call-shell",
            name="run_shell",
            arguments={"command": command, "timeout_seconds": 30},
        )

        renderer(ToolStarted("session-1", call))

        rendered = output.getvalue()
        self.assertIn("tool> run_shell $ first\\nsecond…", rendered)
        self.assertIn(f"[{len(command)} chars]", rendered)
        self.assertIn("(timeout=30s)", rendered)
        self.assertNotIn("secret", rendered)
        self.assertNotIn("\x1b[31m", rendered)

    def test_write_and_edit_summaries_never_render_file_content(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)
        write = ToolCall(
            id="call-write",
            name="write_file",
            arguments={"path": "new.py", "content": "private contents"},
        )
        edit = ToolCall(
            id="call-edit",
            name="edit_file",
            arguments={
                "path": "old.py",
                "old_text": "private old",
                "new_text": "private new",
            },
        )

        renderer(ToolStarted("session-1", write))
        renderer(ToolStarted("session-1", edit))

        self.assertEqual(
            output.getvalue(),
            'tool> write_file "new.py"\ntool> edit_file "old.py"\n',
        )

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
        self.assertIn("\x1b[34m", output.getvalue())
        self.assertIn("assistant>\x1b[0m answer", output.getvalue())
        self.assertNotIn("\x1b[32manswer", output.getvalue())

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

    def test_skill_error_uses_a_distinct_plain_text_label(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)

        renderer.skill_error("project skill not found: review")

        self.assertEqual(
            output.getvalue(),
            "[skill_error] project skill not found: review\n",
        )

    def test_permission_request_uses_a_fixed_width_box_and_wraps_target(self) -> None:
        output = StringIO()
        renderer = TerminalRenderer(output, color=ColorMode.NEVER)

        renderer.permission_request(
            "Run this shell command in the workspace?",
            "Command",
            '"' + "x" * 100 + '"',
        )

        lines = output.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("╭─ Permission "))
        self.assertTrue(lines[0].endswith("╮"))
        self.assertEqual(len(lines[0]), 80)
        self.assertTrue(lines[-1].startswith("╰"))
        self.assertTrue(lines[-1].endswith("╯"))
        self.assertGreater(len(lines), 5)
        self.assertTrue(all(len(line) == 80 for line in lines))


if __name__ == "__main__":
    unittest.main()
