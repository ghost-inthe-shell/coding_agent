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
from coding_agent.providers import CompletionRequest, LLMProvider, ProviderError
from coding_agent.tools import Tool, ToolContext, ToolInput


class EchoInput(ToolInput):
    text: str = Field(min_length=1)


class EchoTool(Tool[EchoInput]):
    name = "echo"
    description = "Echo text."
    input_model = EchoInput

    def execute(self, arguments: EchoInput, context: ToolContext) -> ToolResult:
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
        runtime = Runtime(provider, (EchoTool(),), event_sink=events.append, state_home=self.root)

        result = runtime.run_turn(state, "work")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.final_text, "done")
        self.assertEqual(provider.requests[0].system_prompt, "Original prompt snapshot.")
        self.assertIsInstance(state.messages[2], ToolResultMessage)
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


if __name__ == "__main__":
    unittest.main()
