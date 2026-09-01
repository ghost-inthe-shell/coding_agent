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
    ApiDialect,
    CompletionRequest,
    CompletionTextDelta,
    CompletionThinkingDelta,
    OpenAICompatibleProvider,
    ProviderError,
    ReasoningLevel,
)
from coding_agent.tools.base import ToolSpec


def namespace(**values):
    return SimpleNamespace(**values)


class FakeCompletions:
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
    def __init__(self, completions):
        self.chat = namespace(completions=completions)


def response(
    *,
    content="done",
    tool_calls=None,
    finish_reason="stop",
    **message_fields,
):
    return namespace(
        id="response-1",
        model="actual-model",
        choices=(
            namespace(
                finish_reason=finish_reason,
                message=namespace(
                    content=content,
                    tool_calls=tool_calls or (),
                    **message_fields,
                ),
            ),
        ),
        usage=namespace(
            prompt_tokens=12,
            completion_tokens=5,
            prompt_tokens_details=namespace(cached_tokens=4),
            completion_tokens_details=namespace(reasoning_tokens=2),
        ),
    )


def stream_chunk(
    *,
    content=None,
    tool_calls=(),
    finish_reason=None,
    usage=None,
    **delta_fields,
):
    return namespace(
        id="response-stream-1",
        model="actual-stream-model",
        choices=(
            namespace(
                index=0,
                finish_reason=finish_reason,
                delta=namespace(
                    content=content,
                    tool_calls=tool_calls,
                    **delta_fields,
                ),
            ),
        ),
        usage=usage,
    )


def usage_chunk():
    return namespace(
        id="response-stream-1",
        model="actual-stream-model",
        choices=(),
        usage=namespace(
            prompt_tokens=20,
            completion_tokens=8,
            prompt_tokens_details=namespace(cached_tokens=3),
            completion_tokens_details=namespace(reasoning_tokens=4),
        ),
    )


class FailingStream:
    def __iter__(self):
        yield stream_chunk(content="partial")
        raise RuntimeError("connection dropped")


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_streams_thinking_and_text_then_returns_complete_message(self) -> None:
        completions = FakeCompletions(
            (
                stream_chunk(reasoning_content="inspect "),
                stream_chunk(reasoning_content="first"),
                stream_chunk(content="hel"),
                stream_chunk(content="lo", finish_reason="stop"),
                usage_chunk(),
            )
        )
        provider = OpenAICompatibleProvider(
            "requested-model",
            stream=True,
            client=FakeClient(completions),
        )
        events = []

        result = provider.complete(
            CompletionRequest(messages=(), system_prompt="System"),
            event_sink=events.append,
        )

        self.assertEqual(
            events,
            [
                CompletionThinkingDelta("inspect "),
                CompletionThinkingDelta("first"),
                CompletionTextDelta("hel"),
                CompletionTextDelta("lo"),
            ],
        )
        self.assertEqual(result.thinking, "inspect first")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.model, "actual-stream-model")
        self.assertEqual(result.response_id, "response-stream-1")
        self.assertEqual(result.usage.input_tokens, 20)
        self.assertEqual(result.usage.output_tokens, 8)
        thinking = result.content[0]
        assert isinstance(thinking, ThinkingBlock)
        self.assertEqual(thinking.replay_field, "reasoning_content")
        self.assertTrue(completions.arguments["stream"])

    def test_stream_assembles_interleaved_tool_call_fragments(self) -> None:
        first_calls = (
            namespace(
                index=0,
                id="call-1",
                function=namespace(name="read_", arguments='{"pa'),
            ),
            namespace(
                index=1,
                id="call-2",
                function=namespace(name="grep_", arguments='{"query":"x"'),
            ),
        )
        second_calls = (
            namespace(
                index=1,
                id=None,
                function=namespace(name="search", arguments="}"),
            ),
            namespace(
                index=0,
                id=None,
                function=namespace(name="file", arguments='th":"a.py"}'),
            ),
        )
        completions = FakeCompletions(
            (
                stream_chunk(tool_calls=first_calls),
                stream_chunk(tool_calls=second_calls, finish_reason="tool_calls"),
            )
        )
        provider = OpenAICompatibleProvider("model", stream=True, client=FakeClient(completions))
        events = []

        result = provider.complete(
            CompletionRequest(messages=(), system_prompt="System"),
            event_sink=events.append,
        )

        self.assertEqual(events, [])
        self.assertEqual(result.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(
            [(call.id, call.name, call.arguments) for call in result.tool_calls],
            [
                ("call-1", "read_file", {"path": "a.py"}),
                ("call-2", "grep_search", {"query": "x"}),
            ],
        )

    def test_stream_preserves_truncated_tool_arguments(self) -> None:
        call = namespace(
            index=0,
            id="call-1",
            function=namespace(name="read_file", arguments='{"path":'),
        )
        provider = OpenAICompatibleProvider(
            "model",
            stream=True,
            client=FakeClient(
                FakeCompletions((stream_chunk(tool_calls=(call,), finish_reason="length"),))
            ),
        )

        result = provider.complete(
            CompletionRequest(messages=(), system_prompt="System"),
            event_sink=lambda event: None,
        )

        self.assertEqual(result.stop_reason, StopReason.LENGTH)
        self.assertEqual(result.tool_calls[0].raw_arguments, '{"path":')
        self.assertIsNotNone(result.tool_calls[0].parse_error)

    def test_stream_iteration_failure_keeps_only_displayed_delta(self) -> None:
        completions = FakeCompletions(FailingStream())
        provider = OpenAICompatibleProvider("model", stream=True, client=FakeClient(completions))
        events = []

        with self.assertRaisesRegex(ProviderError, "connection dropped"):
            provider.complete(
                CompletionRequest(messages=(), system_prompt="System"),
                event_sink=events.append,
            )

        self.assertEqual(events, [CompletionTextDelta("partial")])

    def test_stream_configuration_requires_an_event_sink(self) -> None:
        completions = FakeCompletions(response())
        provider = OpenAICompatibleProvider("model", stream=True, client=FakeClient(completions))

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.text, "done")
        self.assertFalse(completions.arguments["stream"])

    def test_converts_request_messages_tools_and_text_response(self) -> None:
        completions = FakeCompletions(response())
        provider = OpenAICompatibleProvider("requested-model", client=FakeClient(completions))
        assistant = AssistantMessage(
            content=(
                ThinkingBlock("Inspect the file.", replay_field="reasoning_content"),
                ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"}),
            ),
            provider="openai-compatible",
            model="requested-model",
            stop_reason=StopReason.TOOL_USE,
        )
        request = CompletionRequest(
            system_prompt="System prompt",
            messages=(
                UserMessage.from_text("Read it"),
                assistant,
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read_file",
                    content=(TextBlock("contents"),),
                    status=ToolResultStatus.SUCCESS,
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

        self.assertEqual(result.text, "done")
        self.assertEqual(result.model, "actual-model")
        self.assertEqual(result.response_id, "response-1")
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(result.usage.output_tokens, 5)
        self.assertEqual(result.usage.cache_read_tokens, 4)
        self.assertEqual(result.usage.reasoning_tokens, 2)
        self.assertEqual(provider.max_output_tokens, DEFAULT_MAX_OUTPUT_TOKENS)
        sent = completions.arguments
        self.assertEqual(sent["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "System prompt"})
        self.assertEqual(sent["messages"][2]["reasoning_content"], "Inspect the file.")
        self.assertEqual(sent["messages"][2]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(sent["messages"][3]["role"], "tool")
        self.assertEqual(sent["tools"][0]["function"]["parameters"], {"type": "object"})

    def test_preserves_invalid_tool_arguments(self) -> None:
        raw_call = namespace(
            id="call-1",
            function=namespace(name="read_file", arguments="{not-json"),
        )
        provider = OpenAICompatibleProvider(
            "model",
            client=FakeClient(
                FakeCompletions(
                    response(content=None, tool_calls=(raw_call,), finish_reason="tool_calls")
                )
            ),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.stop_reason, StopReason.TOOL_USE)
        self.assertEqual(result.tool_calls[0].raw_arguments, "{not-json")
        self.assertIsNotNone(result.tool_calls[0].parse_error)
        self.assertEqual(result.tool_calls[0].arguments, {})

    def test_extracts_first_nonempty_reasoning_field(self) -> None:
        provider = OpenAICompatibleProvider(
            "model",
            client=FakeClient(
                FakeCompletions(
                    response(
                        reasoning_content="primary reasoning",
                        reasoning="duplicate reasoning",
                    )
                )
            ),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.thinking, "primary reasoning")
        self.assertIsInstance(result.content[0], ThinkingBlock)
        thinking = result.content[0]
        assert isinstance(thinking, ThinkingBlock)
        self.assertEqual(thinking.replay_field, "reasoning_content")
        self.assertEqual(result.text, "done")

    def test_replays_each_supported_reasoning_field(self) -> None:
        for field_name in ("reasoning_content", "reasoning", "reasoning_text"):
            with self.subTest(field_name=field_name):
                completions = FakeCompletions(response())
                provider = OpenAICompatibleProvider("model", client=FakeClient(completions))
                assistant = AssistantMessage(
                    content=(ThinkingBlock("reasoning", replay_field=field_name),),
                    provider="openai-compatible",
                    model="model",
                )

                provider.complete(
                    CompletionRequest(
                        messages=(assistant,),
                        system_prompt="System",
                    )
                )

                self.assertEqual(completions.arguments["messages"][1][field_name], "reasoning")

    def test_rejects_non_text_reasoning_content(self) -> None:
        provider = OpenAICompatibleProvider(
            "model",
            client=FakeClient(
                FakeCompletions(response(reasoning_content={"unexpected": "object"}))
            ),
        )

        with self.assertRaisesRegex(ProviderError, "reasoning_content was not text"):
            provider.complete(CompletionRequest(messages=(), system_prompt="System"))

    def test_maps_length_without_tool_calls(self) -> None:
        provider = OpenAICompatibleProvider(
            "model", client=FakeClient(FakeCompletions(response(finish_reason="length")))
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.stop_reason, StopReason.LENGTH)

    def test_request_can_use_a_smaller_output_limit(self) -> None:
        completions = FakeCompletions(response(content="summary"))
        client = FakeClient(completions)
        provider = OpenAICompatibleProvider("test-model", client=client)

        provider.complete(
            CompletionRequest(
                system_prompt="Summarize.",
                messages=(),
                max_output_tokens=2_048,
            )
        )

        self.assertEqual(completions.arguments["max_tokens"], 2_048)

    def test_reasoning_levels_are_mapped_by_explicit_api_dialect(self) -> None:
        cases = (
            (
                ApiDialect.DEEPSEEK,
                ReasoningLevel.OFF,
                {"extra_body": {"thinking": {"type": "disabled"}}},
            ),
            (
                ApiDialect.DEEPSEEK,
                ReasoningLevel.LOW,
                {
                    "reasoning_effort": "low",
                    "extra_body": {"thinking": {"type": "enabled"}},
                },
            ),
            (
                ApiDialect.DASHSCOPE,
                ReasoningLevel.OFF,
                {"extra_body": {"enable_thinking": False}},
            ),
            (
                ApiDialect.DASHSCOPE,
                ReasoningLevel.MEDIUM,
                {"extra_body": {"reasoning_effort": "medium"}},
            ),
            (
                ApiDialect.MOONSHOT,
                ReasoningLevel.OFF,
                {"extra_body": {"thinking": {"type": "disabled"}}},
            ),
        )
        for dialect, reasoning, expected in cases:
            with self.subTest(dialect=dialect, reasoning=reasoning):
                completions = FakeCompletions(response())
                provider = OpenAICompatibleProvider(
                    "model",
                    dialect=dialect,
                    reasoning=reasoning,
                    client=FakeClient(completions),
                )

                provider.complete(CompletionRequest(messages=(), system_prompt="System"))

                for name, value in expected.items():
                    self.assertEqual(completions.arguments[name], value)

    def test_minimal_reasoning_overrides_main_request_configuration(self) -> None:
        cases = (
            (
                ApiDialect.GENERIC,
                ReasoningLevel.DEFAULT,
                {},
            ),
            (
                ApiDialect.DEEPSEEK,
                ReasoningLevel.HIGH,
                {"extra_body": {"thinking": {"type": "disabled"}}},
            ),
            (
                ApiDialect.DASHSCOPE,
                ReasoningLevel.HIGH,
                {"extra_body": {"reasoning_effort": "low"}},
            ),
            (
                ApiDialect.MOONSHOT,
                ReasoningLevel.DEFAULT,
                {"extra_body": {"thinking": {"type": "disabled"}}},
            ),
        )
        for dialect, main_reasoning, expected in cases:
            with self.subTest(dialect=dialect):
                completions = FakeCompletions(response())
                provider = OpenAICompatibleProvider(
                    "model",
                    dialect=dialect,
                    reasoning=main_reasoning,
                    client=FakeClient(completions),
                )

                provider.complete(
                    CompletionRequest(
                        messages=(),
                        system_prompt="Summarize",
                        reasoning=ReasoningLevel.MINIMAL,
                    )
                )

                for name, value in expected.items():
                    self.assertEqual(completions.arguments[name], value)
                if not expected:
                    self.assertNotIn("extra_body", completions.arguments)
                    self.assertNotIn("reasoning_effort", completions.arguments)

    def test_rejects_reasoning_level_unsupported_by_selected_dialect(self) -> None:
        invalid = (
            (ApiDialect.GENERIC, ReasoningLevel.LOW),
            (ApiDialect.DEEPSEEK, ReasoningLevel.MEDIUM),
            (ApiDialect.MOONSHOT, ReasoningLevel.HIGH),
            (ApiDialect.DEEPSEEK, ReasoningLevel.MINIMAL),
        )
        for dialect, reasoning in invalid:
            with self.subTest(dialect=dialect, reasoning=reasoning), self.assertRaises(ValueError):
                OpenAICompatibleProvider(
                    "model",
                    dialect=dialect,
                    reasoning=reasoning,
                    client=FakeClient(FakeCompletions(response())),
                )

    def test_preserves_truncated_tool_calls_for_runtime_rejection(self) -> None:
        raw_call = namespace(
            id="call-1",
            function=namespace(name="read_file", arguments='{"path":"a.py"}'),
        )
        provider = OpenAICompatibleProvider(
            "model",
            client=FakeClient(
                FakeCompletions(response(tool_calls=(raw_call,), finish_reason="length"))
            ),
        )

        result = provider.complete(CompletionRequest(messages=(), system_prompt="System"))

        self.assertEqual(result.stop_reason, StopReason.LENGTH)
        self.assertEqual(result.tool_calls[0].id, "call-1")

    def test_wraps_client_failure(self) -> None:
        provider = OpenAICompatibleProvider(
            "model", client=FakeClient(FakeCompletions(error=RuntimeError("offline")))
        )

        with self.assertRaisesRegex(ProviderError, "offline"):
            provider.complete(CompletionRequest(messages=(), system_prompt="System"))


if __name__ == "__main__":
    unittest.main()
