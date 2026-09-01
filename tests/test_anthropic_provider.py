import unittest
from types import SimpleNamespace

from coding_agent.core.messages import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.types import StopReason, ToolResultStatus
from coding_agent.providers import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicProvider,
    CompletionRequest,
    CompletionTextDelta,
    CompletionThinkingDelta,
    ProviderError,
    ReasoningLevel,
)
from coding_agent.tools.base import ToolSpec


def namespace(**values):
    return SimpleNamespace(**values)


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.arguments = None

    def create(self, **arguments):
        self.arguments = arguments
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, messages):
        self.messages = messages


def response(*, content=None, stop_reason="end_turn", stop_details=None):
    return namespace(
        id="message-1",
        model="actual-model",
        content=content or (namespace(type="text", text="done"),),
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=namespace(
            input_tokens=20,
            output_tokens=6,
            cache_read_input_tokens=8,
            cache_creation_input_tokens=3,
        ),
    )


def message_start():
    return namespace(
        type="message_start",
        message=namespace(
            id="message-stream-1",
            model="actual-stream-model",
            usage=namespace(
                input_tokens=30,
                output_tokens=1,
                cache_read_input_tokens=7,
                cache_creation_input_tokens=2,
            ),
        ),
    )


def block_start(index, block_type, **fields):
    return namespace(
        type="content_block_start",
        index=index,
        content_block=namespace(type=block_type, **fields),
    )


def block_delta(index, delta_type, **fields):
    return namespace(
        type="content_block_delta",
        index=index,
        delta=namespace(type=delta_type, **fields),
    )


def message_delta(stop_reason="end_turn"):
    return namespace(
        type="message_delta",
        delta=namespace(stop_reason=stop_reason),
        usage=namespace(output_tokens=12),
    )


class AnthropicProviderTests(unittest.TestCase):
    def test_streams_thinking_and_text_then_returns_complete_message(self) -> None:
        events = (
            message_start(),
            block_start(0, "thinking", thinking="", signature=""),
            block_delta(0, "thinking_delta", thinking="inspect "),
            block_delta(0, "thinking_delta", thinking="first"),
            block_delta(0, "signature_delta", signature="opaque"),
            namespace(type="content_block_stop", index=0),
            block_start(1, "text", text=""),
            block_delta(1, "text_delta", text="hel"),
            block_delta(1, "text_delta", text="lo"),
            namespace(type="content_block_stop", index=1),
            message_delta(),
            namespace(type="message_stop"),
        )
        messages = FakeMessages(events)
        provider = AnthropicProvider(
            "requested-model",
            stream=True,
            client=FakeClient(messages),
        )
        deltas = []

        result = provider.complete(
            CompletionRequest(messages=(), system_prompt="System"),
            event_sink=deltas.append,
        )

        self.assertEqual(
            deltas,
            [
                CompletionThinkingDelta("inspect "),
                CompletionThinkingDelta("first"),
                CompletionTextDelta("hel"),
                CompletionTextDelta("lo"),
            ],
        )
        self.assertEqual(result.thinking, "inspect first")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.response_id, "message-stream-1")
        self.assertEqual(result.model, "actual-stream-model")
        self.assertEqual(result.usage.input_tokens, 30)
        self.assertEqual(result.usage.output_tokens, 12)
        self.assertEqual(result.usage.cache_read_tokens, 7)
        self.assertTrue(messages.arguments["stream"])

    def test_stream_assembles_tool_input_without_emitting_partial_call(self) -> None:
        events = (
            message_start(),
            block_start(0, "tool_use", id="call-1", name="read_file", input={}),
            block_delta(0, "input_json_delta", partial_json='{"pa'),
            block_delta(0, "input_json_delta", partial_json='th":"a.py"}'),
            namespace(type="content_block_stop", index=0),
            message_delta("tool_use"),
            namespace(type="message_stop"),
        )
        messages = FakeMessages(events)
        provider = AnthropicProvider("model", stream=True, client=FakeClient(messages))
        deltas = []

        result = provider.complete(
            CompletionRequest(messages=(), system_prompt="System"),
            event_sink=deltas.append,
        )

        self.assertEqual(deltas, [])
        self.assertEqual(result.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(result.tool_calls[0].arguments, {"path": "a.py"})
        self.assertEqual(result.tool_calls[0].raw_arguments, '{"path":"a.py"}')

    def test_stream_preserves_truncated_tool_input(self) -> None:
        events = (
            message_start(),
            block_start(0, "tool_use", id="call-1", name="read_file", input={}),
            block_delta(0, "input_json_delta", partial_json='{"path":'),
            message_delta("max_tokens"),
            namespace(type="message_stop"),
        )
        provider = AnthropicProvider(
            "model",
            stream=True,
            client=FakeClient(FakeMessages(events)),
        )

        result = provider.complete(
            CompletionRequest(messages=(), system_prompt="System"),
            event_sink=lambda event: None,
        )

        self.assertEqual(result.stop_reason, StopReason.LENGTH)
        self.assertEqual(result.tool_calls[0].raw_arguments, '{"path":')
        self.assertIsNotNone(result.tool_calls[0].parse_error)

    def test_stream_error_is_reported_after_prior_display_delta(self) -> None:
        events = (
            message_start(),
            block_start(0, "text", text=""),
            block_delta(0, "text_delta", text="partial"),
            namespace(
                type="error",
                error=namespace(type="overloaded_error", message="Overloaded"),
            ),
        )
        provider = AnthropicProvider("model", stream=True, client=FakeClient(FakeMessages(events)))
        deltas = []

        with self.assertRaisesRegex(ProviderError, "Overloaded"):
            provider.complete(
                CompletionRequest(messages=(), system_prompt="System"),
                event_sink=deltas.append,
            )

        self.assertEqual(deltas, [CompletionTextDelta("partial")])

    def test_stream_configuration_requires_an_event_sink(self) -> None:
        messages = FakeMessages(response())
        provider = AnthropicProvider("model", stream=True, client=FakeClient(messages))

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.text, "done")
        self.assertFalse(messages.arguments["stream"])

    def test_non_stream_response_preserves_thinking_as_non_replayed_content(self) -> None:
        provider = AnthropicProvider(
            "model",
            client=FakeClient(
                FakeMessages(
                    response(
                        content=(
                            namespace(
                                type="thinking",
                                thinking="private",
                                signature="opaque",
                            ),
                            namespace(type="text", text="answer"),
                        )
                    )
                )
            ),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.thinking, "private")
        thinking = result.content[0]
        assert isinstance(thinking, ThinkingBlock)
        self.assertIsNone(thinking.replay_field)

    def test_groups_tool_results_and_converts_request(self) -> None:
        messages = FakeMessages(response())
        provider = AnthropicProvider("requested-model", client=FakeClient(messages))
        assistant = AssistantMessage(
            content=(
                ThinkingBlock(
                    "Foreign provider reasoning.",
                    replay_field="reasoning_content",
                ),
                TextBlock("checking"),
                ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="call-2", name="grep_search", arguments={"pattern": "x"}),
            ),
            provider="anthropic",
            model="requested-model",
            stop_reason=StopReason.TOOL_USE,
        )
        request = CompletionRequest(
            system_prompt="System prompt",
            messages=(
                UserMessage.from_text("Inspect"),
                assistant,
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read_file",
                    content=(TextBlock("contents"),),
                ),
                ToolResultMessage(
                    tool_call_id="call-2",
                    tool_name="grep_search",
                    content=(TextBlock("failed"),),
                    status=ToolResultStatus.ERROR,
                ),
            ),
            tools=(
                ToolSpec(
                    name="read_file",
                    description="Read a file.",
                    input_schema={"type": "object"},
                ),
            ),
        )

        result = provider.complete(request)

        sent = messages.arguments
        self.assertEqual(sent["system"], "System prompt")
        self.assertEqual(len(sent["messages"]), 3)
        self.assertEqual(sent["messages"][1]["content"][0], {"type": "text", "text": "checking"})
        result_blocks = sent["messages"][2]["content"]
        self.assertEqual(
            [block["tool_use_id"] for block in result_blocks],
            ["call-1", "call-2"],
        )
        self.assertFalse(result_blocks[0]["is_error"])
        self.assertTrue(result_blocks[1]["is_error"])
        self.assertEqual(sent["tools"][0]["input_schema"], {"type": "object"})
        self.assertEqual(result.model, "actual-model")
        self.assertEqual(result.usage.input_tokens, 20)
        self.assertEqual(result.usage.output_tokens, 6)
        self.assertEqual(result.usage.cache_read_tokens, 8)
        self.assertEqual(result.usage.cache_write_tokens, 3)
        self.assertEqual(provider.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(sent["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS)

    def test_converts_tool_use_response(self) -> None:
        tool_use = namespace(
            type="tool_use",
            id="call-1",
            name="read_file",
            input={"path": "README.md"},
        )
        provider = AnthropicProvider(
            "model",
            client=FakeClient(FakeMessages(response(content=(tool_use,), stop_reason="tool_use"))),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(result.tool_calls[0].arguments, {"path": "README.md"})

    def test_minimal_reasoning_uses_current_non_thinking_request(self) -> None:
        messages = FakeMessages(response())
        provider = AnthropicProvider("model", client=FakeClient(messages))

        provider.complete(
            CompletionRequest(
                messages=(),
                system_prompt="Summarize",
                reasoning=ReasoningLevel.MINIMAL,
            )
        )

        self.assertNotIn("thinking", messages.arguments)

    def test_rejects_unimplemented_anthropic_reasoning_level(self) -> None:
        provider = AnthropicProvider(
            "model",
            client=FakeClient(FakeMessages(response())),
        )

        with self.assertRaisesRegex(ProviderError, "not implemented"):
            provider.complete(
                CompletionRequest(
                    messages=(),
                    system_prompt="System",
                    reasoning=ReasoningLevel.HIGH,
                )
            )

    def test_preserves_non_object_tool_input_as_parse_error(self) -> None:
        tool_use = namespace(type="tool_use", id="call-1", name="read_file", input="bad")
        provider = AnthropicProvider(
            "model",
            client=FakeClient(FakeMessages(response(content=(tool_use,), stop_reason="tool_use"))),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.tool_calls[0].arguments, {})
        self.assertEqual(result.tool_calls[0].raw_arguments, '"bad"')
        self.assertIsNotNone(result.tool_calls[0].parse_error)

    def test_preserves_max_tokens_tool_use_for_runtime_rejection(self) -> None:
        tool_use = namespace(type="tool_use", id="call-1", name="read_file", input={})
        provider = AnthropicProvider(
            "model",
            client=FakeClient(
                FakeMessages(response(content=(tool_use,), stop_reason="max_tokens"))
            ),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.stop_reason, StopReason.LENGTH)
        self.assertEqual(result.tool_calls[0].id, "call-1")

    def test_request_can_use_a_smaller_output_limit(self) -> None:
        messages = FakeMessages(response(content=(namespace(type="text", text="summary"),)))
        client = FakeClient(messages)
        provider = AnthropicProvider("test-model", client=client)

        provider.complete(
            CompletionRequest(
                system_prompt="Summarize.",
                messages=(),
                max_output_tokens=2_048,
            )
        )

        self.assertEqual(messages.arguments["max_tokens"], 2_048)

    def test_maps_refusal_to_error_message(self) -> None:
        provider = AnthropicProvider(
            "model",
            client=FakeClient(
                FakeMessages(
                    response(
                        stop_reason="refusal",
                        stop_details=namespace(explanation="unsafe request"),
                    )
                )
            ),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.stop_reason, StopReason.ERROR)
        self.assertEqual(result.error_message, "unsafe request")

    def test_wraps_client_failure(self) -> None:
        provider = AnthropicProvider(
            "model", client=FakeClient(FakeMessages(error=RuntimeError("offline")))
        )

        with self.assertRaisesRegex(ProviderError, "offline"):
            provider.complete(CompletionRequest(messages=(), system_prompt="System"))


if __name__ == "__main__":
    unittest.main()
