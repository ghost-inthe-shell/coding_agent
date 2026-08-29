import json
import unittest

from coding_agent.core.messages import (
    AssistantMessage,
    ProtocolError,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    message_from_dict,
    message_to_dict,
    validate_message_sequence,
)
from coding_agent.core.types import StopReason, ToolResultStatus
from coding_agent.core.usage import Usage


class MessageProtocolTests(unittest.TestCase):
    def test_zero_timestamp_is_preserved(self) -> None:
        self.assertEqual(UserMessage.from_text("epoch", timestamp=0).timestamp, 0)

    def test_messages_round_trip_through_json(self) -> None:
        messages = [
            UserMessage.from_text("inspect the repository", timestamp=100),
            AssistantMessage(
                content=(
                    TextBlock("I will inspect it."),
                    ToolCall(
                        id="call-1",
                        name="read_file",
                        arguments={"path": "README.md"},
                        raw_arguments='{"path":"README.md"}',
                    ),
                ),
                provider="fake",
                model="fake-model",
                usage=Usage(input_tokens=10, output_tokens=5),
                stop_reason=StopReason.TOOL_USE,
                response_id="response-1",
                timestamp=101,
            ),
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="read_file",
                content=(TextBlock("1: # Coding Agent"),),
                status=ToolResultStatus.SUCCESS,
                metadata={"characters": 17},
                timestamp=102,
            ),
            AssistantMessage(
                content=(TextBlock("Inspection complete."),),
                provider="fake",
                model="fake-model",
                stop_reason=StopReason.STOP,
                timestamp=103,
            ),
        ]

        encoded = json.loads(json.dumps([message_to_dict(message) for message in messages]))
        decoded = [message_from_dict(message) for message in encoded]

        self.assertEqual(decoded, messages)
        validate_message_sequence(decoded)

    def test_invalid_tool_arguments_are_preserved_for_error_feedback(self) -> None:
        call = ToolCall(
            id="call-1",
            name="read_file",
            raw_arguments="{not-json",
            parse_error="invalid JSON",
        )
        message = AssistantMessage(
            content=(call,),
            provider="fake",
            model="fake-model",
            stop_reason=StopReason.TOOL_USE,
        )

        decoded = message_from_dict(message_to_dict(message))

        self.assertIsInstance(decoded, AssistantMessage)
        assert isinstance(decoded, AssistantMessage)
        self.assertEqual(decoded.tool_calls[0].raw_arguments, "{not-json")
        self.assertEqual(decoded.tool_calls[0].parse_error, "invalid JSON")

    def test_tool_results_must_follow_source_order(self) -> None:
        assistant = AssistantMessage(
            content=(
                ToolCall(id="first", name="read_file"),
                ToolCall(id="second", name="grep_search"),
            ),
            provider="fake",
            model="fake-model",
            stop_reason=StopReason.TOOL_USE,
        )
        wrong_result = ToolResultMessage(
            tool_call_id="second",
            tool_name="grep_search",
            content=(TextBlock("result"),),
        )

        with self.assertRaisesRegex(ProtocolError, "expected result for 'first'"):
            validate_message_sequence([assistant, wrong_result])

    def test_pending_tool_call_is_only_valid_for_recovery_snapshot(self) -> None:
        assistant = AssistantMessage(
            content=(ToolCall(id="call-1", name="read_file"),),
            provider="fake",
            model="fake-model",
            stop_reason=StopReason.TOOL_USE,
        )

        with self.assertRaisesRegex(ProtocolError, "pending tool calls"):
            validate_message_sequence([assistant])

        validate_message_sequence([assistant], allow_pending_tail=True)

    def test_tool_use_stop_reason_requires_a_tool_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires at least one tool call"):
            AssistantMessage(
                content=(TextBlock("no call"),),
                provider="fake",
                model="fake-model",
                stop_reason=StopReason.TOOL_USE,
            )


class UsageTests(unittest.TestCase):
    def test_usage_accumulates_normalized_fields(self) -> None:
        total = Usage(input_tokens=5, cache_read_tokens=3) + Usage(
            input_tokens=7,
            output_tokens=2,
            reasoning_tokens=1,
        )

        self.assertEqual(total.input_tokens, 12)
        self.assertEqual(total.output_tokens, 2)
        self.assertEqual(total.total_tokens, 14)
        self.assertEqual(total.cache_read_tokens, 3)

    def test_usage_rejects_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_tokens"):
            Usage(input_tokens=-1)


if __name__ == "__main__":
    unittest.main()
