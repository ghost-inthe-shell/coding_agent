from collections import deque
from pathlib import Path
import tempfile
import unittest

from pydantic import Field

from coding_agent.core.events import TurnFinished
from coding_agent.core.messages import AssistantMessage, TextBlock, ToolCall, ToolResultMessage
from coding_agent.core.results import ToolResult
from coding_agent.core.runtime import Runtime, RuntimeLimits
from coding_agent.core.session import SessionState
from coding_agent.core.types import RunStatus, StopReason, ToolResultStatus
from coding_agent.permissions import PermissionDecision, PermissionRequest
from coding_agent.providers import CompletionRequest, LLMProvider, ProviderError
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
    def __init__(self, messages):
        self.messages = deque(messages)
        self.requests: list[CompletionRequest] = []

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


def text_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text),),
        provider="fake",
        model="fake",
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
        permission_handler = lambda request: PermissionDecision.ALLOW
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

        result = Runtime(FailingProvider(), (), state_home=self.root).run_turn(
            self.state(), "work"
        )

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
            SequenceProvider((partial,)),
            (),
            state_home=self.root,
        ).run_turn(self.state(), "work")

        self.assertEqual(result.status, RunStatus.LIMIT_REACHED)
        self.assertEqual(result.final_text, "partial answer")
        self.assertEqual(result.stop_reason, StopReason.LENGTH)


if __name__ == "__main__":
    unittest.main()
