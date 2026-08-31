import tempfile
import unittest
from collections import deque
from pathlib import Path

from pydantic import Field

from coding_agent.core.compaction import CompactionCheckpoint
from coding_agent.core.events import ModelRequested, TurnFinished
from coding_agent.core.messages import (
    AssistantMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.results import ToolResult
from coding_agent.core.runtime import Runtime, RuntimeLimits
from coding_agent.core.session import SessionState
from coding_agent.core.types import (
    RunStatus,
    SessionStatus,
    StopReason,
    ToolResultStatus,
)
from coding_agent.core.usage import Usage
from coding_agent.permissions import PermissionDecision, PermissionRequest
from coding_agent.providers import (
    CompletionRequest,
    LLMProvider,
    ProviderError,
    ReasoningLevel,
)
from coding_agent.tools import ReadFileTool, Tool, ToolContext, ToolInput


class EchoInput(ToolInput):
    text: str = Field(min_length=1)


class EchoTool(Tool[EchoInput]):
    name = "echo"
    description = "Echo text."
    input_model = EchoInput

    def __init__(self) -> None:
        self.executions: list[str] = []
        self.permission_handlers = []

    def execute(self, arguments: EchoInput, context: ToolContext) -> ToolResult:
        self.executions.append(arguments.text)
        self.permission_handlers.append(context.permission_handler)
        return ToolResult.from_text(arguments.text)


class SequenceProvider(LLMProvider):
    def __init__(self, messages, *, max_output_tokens=2048):
        self.messages = deque(messages)
        self.requests: list[CompletionRequest] = []
        self._max_output_tokens = max_output_tokens

    @property
    def max_output_tokens(self) -> int | None:
        return self._max_output_tokens

    def complete(self, request: CompletionRequest) -> AssistantMessage:
        self.requests.append(request)
        return self.messages.popleft()


def tool_message(*calls: ToolCall) -> AssistantMessage:
    return AssistantMessage(
        content=tuple(calls),
        provider="fake",
        model="fake",
        stop_reason=StopReason.TOOL_USE,
    )


def text_message(text: str, *, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text),),
        provider="fake",
        model="fake",
        usage=usage or Usage(),
        stop_reason=StopReason.STOP,
    )


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def state(self) -> SessionState:
        return SessionState(
            session_id="session-1",
            workspace_root=str(self.workspace),
            system_prompt="Original prompt snapshot.",
        )

    def test_runtime_uses_state_prompt_and_pairs_tool_result(self) -> None:
        provider = SequenceProvider(
            [
                tool_message(ToolCall(id="call-1", name="echo", arguments={"text": "hello"})),
                text_message("done"),
            ]
        )
        events = []
        state = self.state()
        tool = EchoTool()

        def permission_handler(request: PermissionRequest) -> PermissionDecision:
            return PermissionDecision.ALLOW

        runtime = Runtime(
            provider,
            (tool,),
            event_sink=events.append,
            state_home=self.root,
            permission_handler=permission_handler,
        )

        result = runtime.run_turn(state, "work")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.final_text, "done")
        self.assertEqual(result.max_output_tokens, 2048)
        self.assertEqual(provider.requests[0].system_prompt, "Original prompt snapshot.")
        self.assertIsInstance(state.messages[2], ToolResultMessage)
        self.assertEqual(tool.permission_handlers, [permission_handler])
        state.validate()
        self.assertIsInstance(events[-1], TurnFinished)

    def test_tool_budget_pairs_skipped_calls_then_requests_final_text(self) -> None:
        provider = SequenceProvider(
            [
                tool_message(
                    ToolCall(id="call-1", name="echo", arguments={"text": "first"}),
                    ToolCall(id="call-2", name="echo", arguments={"text": "second"}),
                ),
                text_message("budget summary"),
            ]
        )
        state = self.state()
        runtime = Runtime(
            provider,
            (EchoTool(),),
            limits=RuntimeLimits(max_model_calls=3, max_tool_calls=1),
            state_home=self.root,
        )

        result = runtime.run_turn(state, "work")

        self.assertEqual(result.status, RunStatus.LIMIT_REACHED)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.max_output_tokens, 2048)
        self.assertEqual(result.final_text, "budget summary")
        self.assertEqual(provider.requests[-1].tools, ())
        tool_results = [m for m in state.messages if isinstance(m, ToolResultMessage)]
        self.assertEqual(len(tool_results), 2)
        self.assertEqual(tool_results[1].status, ToolResultStatus.ERROR)
        state.validate()

    def test_expected_provider_error_returns_failed_run(self) -> None:
        class FailingProvider(LLMProvider):
            def complete(self, request: CompletionRequest) -> AssistantMessage:
                raise ProviderError("offline")

        result = Runtime(FailingProvider(), (), state_home=self.root).run_turn(self.state(), "work")

        self.assertEqual(result.status, RunStatus.PROVIDER_ERROR)
        self.assertEqual(result.error_message, "offline")

    def test_runtime_passes_permission_decision_to_outside_read(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("approved content\n", encoding="utf-8")
        provider = SequenceProvider(
            [
                tool_message(
                    ToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"path": str(outside)},
                    )
                ),
                text_message("done"),
            ]
        )
        requests: list[PermissionRequest] = []

        def allow(request: PermissionRequest) -> PermissionDecision:
            requests.append(request)
            return PermissionDecision.ALLOW

        state = self.state()
        runtime = Runtime(
            provider,
            (ReadFileTool(),),
            state_home=self.root / "state",
            permission_handler=allow,
        )

        result = runtime.run_turn(state, "read the outside file")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(len(requests), 1)
        tool_result = next(
            message for message in state.messages if isinstance(message, ToolResultMessage)
        )
        self.assertEqual(tool_result.status, ToolResultStatus.SUCCESS)
        self.assertIn("approved content", tool_result.text)
        self.assertEqual(state.read_file_versions, {})

    def test_runtime_records_workspace_read_in_session_state(self) -> None:
        path = self.workspace / "notes.txt"
        path.write_text("tracked content\n", encoding="utf-8")
        provider = SequenceProvider(
            [
                tool_message(
                    ToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"path": "notes.txt"},
                    )
                ),
                text_message("done"),
            ]
        )
        state = self.state()
        runtime = Runtime(provider, (ReadFileTool(),), state_home=self.root / "state")

        result = runtime.run_turn(state, "read notes")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(set(state.read_file_versions), {"notes.txt"})
        self.assertEqual(state.read_file_versions["notes.txt"].size, path.stat().st_size)

    def test_truncated_tool_calls_are_paired_but_never_executed(self) -> None:
        truncated = AssistantMessage(
            content=(
                TextBlock("partial"),
                ToolCall(id="call-1", name="echo", arguments={"text": "unsafe"}),
            ),
            provider="fake",
            model="fake",
            stop_reason=StopReason.LENGTH,
        )
        provider = SequenceProvider((truncated, text_message("recovered")))
        tool = EchoTool()
        state = self.state()

        result = Runtime(provider, (tool,), state_home=self.root).run_turn(state, "work")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.final_text, "recovered")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(tool.executions, [])
        tool_results = [
            message for message in state.messages if isinstance(message, ToolResultMessage)
        ]
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0].status, ToolResultStatus.ERROR)
        self.assertEqual(tool_results[0].metadata["reason"], "truncated_model_response")
        state.validate()

    def test_truncated_text_returns_partial_limit_result(self) -> None:
        partial = AssistantMessage(
            content=(TextBlock("partial answer"),),
            provider="fake",
            model="fake",
            stop_reason=StopReason.LENGTH,
        )

        result = Runtime(
            SequenceProvider((partial,), max_output_tokens=16),
            (),
            state_home=self.root,
        ).run_turn(self.state(), "work")

        self.assertEqual(result.status, RunStatus.LIMIT_REACHED)
        self.assertEqual(result.final_text, "partial answer")
        self.assertEqual(result.stop_reason, StopReason.LENGTH)
        self.assertEqual(result.max_output_tokens, 16)

    def test_model_call_budget_has_a_distinct_result(self) -> None:
        provider = SequenceProvider(
            (
                tool_message(ToolCall(id="call-1", name="echo", arguments={"text": "x"})),
                text_message("budget summary"),
            )
        )
        runtime = Runtime(
            provider,
            (EchoTool(),),
            limits=RuntimeLimits(max_model_calls=2, max_tool_calls=2),
            state_home=self.root,
        )

        result = runtime.run_turn(self.state(), "work")

        self.assertEqual(result.status, RunStatus.LIMIT_REACHED)
        self.assertEqual(result.error_message, "turn model-call limit reached")
        self.assertEqual(result.final_text, "budget summary")
        self.assertEqual(result.model_turns, 2)
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(result.max_output_tokens, 2048)
        self.assertNotEqual(provider.requests[0].tools, ())
        self.assertEqual(provider.requests[1].tools, ())
        self.assertIn("final model call", provider.requests[1].system_prompt)

    def test_auto_compaction_uses_summary_then_sends_only_active_history(self) -> None:
        old_marker = "old-marker-" + "x" * 17_000
        state = self.state()
        state.messages.extend(
            (
                UserMessage.from_text(old_marker, timestamp=1),
                text_message("recent-" + "y" * 3_000),
            )
        )
        provider = SequenceProvider(
            (
                text_message(
                    "rolling summary",
                    usage=Usage(input_tokens=50, output_tokens=5),
                ),
                text_message("done", usage=Usage(input_tokens=60, output_tokens=4)),
            )
        )
        events = []
        runtime = Runtime(
            provider,
            (),
            event_sink=events.append,
            state_home=self.root,
            context_window=8_000,
        )

        result = runtime.run_turn(state, "continue")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.model_turns, 1)
        self.assertEqual(result.usage.input_tokens, 110)
        self.assertEqual(result.usage.output_tokens, 9)
        self.assertIsNotNone(state.compaction)
        assert state.compaction is not None
        self.assertEqual(state.compaction.first_kept_message_index, 1)
        self.assertIn(old_marker, state.messages[0].content[0].text)
        self.assertEqual(provider.requests[0].tools, ())
        self.assertEqual(provider.requests[0].max_output_tokens, 2_048)
        self.assertEqual(provider.requests[0].reasoning, ReasoningLevel.MINIMAL)
        self.assertIn("rolling checkpoint", provider.requests[0].system_prompt)
        active_text = "\n".join(
            block.text
            for message in provider.requests[1].messages
            if isinstance(message, (UserMessage, AssistantMessage))
            for block in message.content
            if isinstance(block, TextBlock)
        )
        self.assertIn("rolling summary", active_text)
        self.assertNotIn(old_marker, active_text)
        model_requests = [event for event in events if isinstance(event, ModelRequested)]
        self.assertEqual([event.model_call for event in model_requests], [1, 2])
        state.validate()

    def test_invalid_compaction_response_fails_without_replacing_checkpoint(self) -> None:
        state = self.state()
        state.messages.extend(
            (
                UserMessage.from_text("x" * 17_000, timestamp=1),
                text_message("y" * 3_000),
            )
        )
        tool_response = tool_message(ToolCall(id="call-1", name="echo"))
        invalid_summary = AssistantMessage(
            content=tool_response.content,
            provider=tool_response.provider,
            model=tool_response.model,
            usage=Usage(input_tokens=50, output_tokens=2),
            stop_reason=tool_response.stop_reason,
        )
        runtime = Runtime(
            SequenceProvider((invalid_summary,)),
            (),
            state_home=self.root,
            context_window=8_000,
        )

        result = runtime.run_turn(state, "continue")

        self.assertEqual(result.status, RunStatus.PROVIDER_ERROR)
        self.assertEqual(result.model_turns, 0)
        self.assertEqual(result.usage.input_tokens, 50)
        self.assertIn("unexpectedly contained tool calls", result.error_message)
        self.assertIsNone(state.compaction)
        state.validate()

    def test_compaction_does_not_consume_the_agent_model_call_budget(self) -> None:
        state = self.state()
        state.messages.extend(
            (
                UserMessage.from_text("x" * 17_000, timestamp=1),
                text_message("y" * 3_000),
            )
        )
        runtime = Runtime(
            SequenceProvider((text_message("summary"), text_message("budget summary"))),
            (),
            limits=RuntimeLimits(max_model_calls=1),
            state_home=self.root,
            context_window=8_000,
        )

        result = runtime.run_turn(state, "continue")

        self.assertEqual(result.status, RunStatus.LIMIT_REACHED)
        self.assertEqual(result.model_turns, 1)
        self.assertEqual(result.final_text, "budget summary")
        self.assertEqual(result.error_message, "turn model-call limit reached")
        self.assertIsNotNone(state.compaction)

    def test_auto_compaction_rejects_a_checkpoint_without_token_reduction(self) -> None:
        state = self.state()
        state.messages.extend(
            (
                UserMessage.from_text("x" * 17_000, timestamp=1),
                text_message("y" * 3_000),
            )
        )
        runtime = Runtime(
            SequenceProvider(
                (
                    text_message(
                        "summary " + "z" * 18_000,
                        usage=Usage(input_tokens=40, output_tokens=20),
                    ),
                )
            ),
            (),
            state_home=self.root,
            context_window=8_000,
        )

        result = runtime.run_turn(state, "continue")

        self.assertEqual(result.status, RunStatus.PROVIDER_ERROR)
        self.assertEqual(result.model_turns, 0)
        self.assertEqual(result.usage, Usage(input_tokens=40, output_tokens=20))
        self.assertIn("did not reduce estimated context", result.error_message)
        self.assertIsNone(state.compaction)
        self.assertEqual(state.usage, result.usage)

    def test_manual_compaction_forces_one_rolling_checkpoint_below_threshold(self) -> None:
        state = self.state()
        state.messages.extend(
            (
                UserMessage.from_text("old question " + "x" * 2_000, timestamp=1),
                text_message("old answer " + "y" * 2_000),
                UserMessage.from_text("recent question " + "z" * 2_000, timestamp=2),
                text_message("recent answer " + "w" * 2_000),
            )
        )
        provider = SequenceProvider(
            (
                text_message(
                    "manual rolling summary",
                    usage=Usage(input_tokens=30, output_tokens=4),
                ),
            )
        )
        runtime = Runtime(provider, (), state_home=self.root)

        result = runtime.compact(state)

        self.assertTrue(result.compacted)
        self.assertEqual(result.summarized_messages, 4)
        self.assertIsNotNone(result.tokens_before)
        self.assertIsNotNone(result.tokens_after)
        self.assertEqual(result.usage, Usage(input_tokens=30, output_tokens=4))
        self.assertEqual(len(state.messages), 4)
        self.assertTrue(state.messages[0].content[0].text.startswith("old question"))
        self.assertIsNotNone(state.compaction)
        assert state.compaction is not None
        self.assertEqual(state.compaction.summary, "manual rolling summary")
        self.assertEqual(state.compaction.first_kept_message_index, 4)
        self.assertEqual(state.usage, result.usage)
        self.assertEqual(provider.requests[0].tools, ())
        self.assertEqual(provider.requests[0].max_output_tokens, 2_048)
        state.validate()

    def test_manual_compaction_without_new_active_history_is_a_noop(self) -> None:
        provider = SequenceProvider(())
        state = self.state()
        state.messages.append(UserMessage.from_text("only message"))
        state.compaction = CompactionCheckpoint(
            summary="existing summary",
            first_kept_message_index=1,
            tokens_before=100,
            created_at=10,
        )

        result = Runtime(provider, (), state_home=self.root).compact(state)

        self.assertFalse(result.compacted)
        self.assertIsNone(result.error_message)
        self.assertEqual(provider.requests, [])
        self.assertIsNotNone(state.compaction)

    def test_manual_compaction_rejects_summary_that_does_not_reduce_context(self) -> None:
        state = self.state()
        state.messages.append(UserMessage.from_text("short message", timestamp=1))
        provider = SequenceProvider(
            (
                text_message(
                    "summary " + "x" * 1_000,
                    usage=Usage(input_tokens=20, output_tokens=10),
                ),
            )
        )

        result = Runtime(provider, (), state_home=self.root).compact(state)

        self.assertFalse(result.compacted)
        self.assertIsNone(result.error_message)
        self.assertIsNotNone(result.tokens_before)
        self.assertIsNotNone(result.tokens_after)
        assert result.tokens_before is not None
        assert result.tokens_after is not None
        self.assertGreaterEqual(result.tokens_after, result.tokens_before)
        self.assertIsNone(state.compaction)
        self.assertEqual(state.usage, Usage(input_tokens=20, output_tokens=10))

    def test_invalid_manual_summary_keeps_prior_checkpoint_and_records_usage(self) -> None:
        state = self.state()
        state.messages.extend(
            (
                UserMessage.from_text("already summarized", timestamp=1),
                text_message("old answer"),
                UserMessage.from_text("recent question", timestamp=2),
                text_message("recent answer"),
            )
        )
        previous = CompactionCheckpoint(
            summary="previous summary",
            first_kept_message_index=1,
            tokens_before=500,
            created_at=10,
        )
        state.compaction = previous
        invalid = tool_message(ToolCall(id="call-1", name="echo"))
        invalid = AssistantMessage(
            content=invalid.content,
            provider=invalid.provider,
            model=invalid.model,
            usage=Usage(input_tokens=25, output_tokens=3),
            stop_reason=invalid.stop_reason,
        )
        runtime = Runtime(SequenceProvider((invalid,)), (), state_home=self.root)

        result = runtime.compact(state)

        self.assertFalse(result.compacted)
        self.assertIn("unexpectedly contained tool calls", result.error_message)
        self.assertIs(state.compaction, previous)
        self.assertEqual(state.usage, Usage(input_tokens=25, output_tokens=3))
        self.assertEqual(result.usage, state.usage)
        state.validate()

    def test_manual_compaction_rejects_running_session(self) -> None:
        state = self.state()
        state.status = SessionStatus.RUNNING

        with self.assertRaisesRegex(ValueError, "while a turn is running"):
            Runtime(SequenceProvider(()), (), state_home=self.root).compact(state)


if __name__ == "__main__":
    unittest.main()
