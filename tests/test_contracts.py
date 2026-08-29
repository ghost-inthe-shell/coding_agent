import json
import unittest

from coding_agent.core.messages import AssistantMessage, TextBlock, ToolCall, UserMessage
from coding_agent.core.results import ToolResult
from coding_agent.core.types import StopReason, ToolResultStatus
from coding_agent.observability.events import Event, EventType
from coding_agent.providers.base import CompletionRequest, LLMProvider
from coding_agent.tools.base import Tool, ToolContext, ToolSpec


READ_FILE_SPEC = ToolSpec(
    name="read_file",
    description="Read a UTF-8 text file.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    read_only=True,
    concurrency_safe=True,
)


class FakeProvider(LLMProvider):
    def complete(self, request: CompletionRequest) -> AssistantMessage:
        return AssistantMessage(
            content=(TextBlock(f"received {len(request.messages)} message(s)"),),
            provider="fake",
            model="fake-model",
            stop_reason=StopReason.STOP,
        )


class FakeReadTool(Tool):
    spec = READ_FILE_SPEC

    def execute(self, arguments, context: ToolContext) -> ToolResult:
        return ToolResult.from_text(f"{context.workspace_root}/{arguments['path']}")


class ContractTests(unittest.TestCase):
    def test_provider_uses_normalized_request_and_response(self) -> None:
        request = CompletionRequest(
            messages=(UserMessage.from_text("hello"),),
            tools=(READ_FILE_SPEC,),
            max_output_tokens=128,
        )

        response = FakeProvider().complete(request)

        self.assertEqual(response.text, "received 1 message(s)")
        self.assertEqual(response.provider, "fake")

    def test_tool_result_converts_to_a_paired_message(self) -> None:
        tool = FakeReadTool()
        call = ToolCall(id="call-1", name=tool.spec.name, arguments={"path": "README.md"})
        result = tool.execute(
            call.arguments,
            ToolContext(session_id="session-1", workspace_root="/repo", cwd="/repo"),
        )

        message = result.to_message(call, timestamp=123)

        self.assertEqual(message.tool_call_id, call.id)
        self.assertEqual(message.tool_name, call.name)
        self.assertEqual(message.text, "/repo/README.md")
        self.assertEqual(message.status, ToolResultStatus.SUCCESS)

    def test_event_round_trip_is_json_serializable(self) -> None:
        event = Event(
            type=EventType.TOOL_RESULT,
            session_id="session-1",
            sequence=4,
            payload={"tool_name": "read_file", "status": "success"},
            timestamp=100,
        )

        restored = Event.from_dict(json.loads(json.dumps(event.to_dict())))

        self.assertEqual(restored, event)


if __name__ == "__main__":
    unittest.main()

