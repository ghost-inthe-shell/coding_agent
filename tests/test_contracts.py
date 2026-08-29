import tempfile
import unittest
from pathlib import Path

from pydantic import Field

from coding_agent.core.events import (
    ModelRequested,
    ModelResponded,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
)
from coding_agent.core.messages import AssistantMessage, TextBlock, ToolCall, UserMessage
from coding_agent.core.results import ToolResult
from coding_agent.core.types import StopReason, ToolResultStatus
from coding_agent.providers import CompletionRequest, LLMProvider
from coding_agent.tools import (
    ArtifactStore,
    Tool,
    ToolContext,
    ToolExecutor,
    ToolInput,
    ToolResultProcessor,
)


class EchoInput(ToolInput):
    text: str = Field(min_length=1)


class EchoTool(Tool[EchoInput]):
    name = "echo"
    description = "Echo one string."
    input_model = EchoInput

    def execute(self, arguments: EchoInput, context: ToolContext) -> ToolResult:
        return ToolResult.from_text(arguments.text)


class FakeProvider(LLMProvider):
    def complete(self, request: CompletionRequest) -> AssistantMessage:
        return AssistantMessage(
            content=(TextBlock(f"received {len(request.messages)} message(s)"),),
            provider="fake",
            model="fake-model",
            stop_reason=StopReason.STOP,
        )


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_provider_uses_normalized_request_and_response(self) -> None:
        spec = EchoTool().spec
        request = CompletionRequest(
            system_prompt="Be useful.",
            messages=(UserMessage.from_text("hello"),),
            tools=(spec,),
        )

        response = FakeProvider().complete(request)

        self.assertEqual(response.text, "received 1 message(s)")
        self.assertEqual(response.provider, "fake")
        self.assertEqual(spec.input_schema["additionalProperties"], False)

    def test_executor_strictly_validates_arguments(self) -> None:
        store = ArtifactStore("session-1", state_home=Path(self.temporary.name))
        executor = ToolExecutor((EchoTool(),), ToolResultProcessor(store))
        context = ToolContext("session-1", "/repo", str(store.root), "/repo")

        result = executor.execute(
            ToolCall(id="call-1", name="echo", arguments={"text": "ok", "extra": 1}),
            context,
        )

        self.assertEqual(result.status, ToolResultStatus.ERROR)
        self.assertIn("extra_forbidden", result.content)

    def test_runtime_event_protocol_has_exactly_six_types(self) -> None:
        event_types = {
            TurnStarted,
            ModelRequested,
            ModelResponded,
            ToolStarted,
            ToolFinished,
            TurnFinished,
        }
        self.assertEqual(len(event_types), 6)


if __name__ == "__main__":
    unittest.main()
