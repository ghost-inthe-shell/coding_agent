from types import SimpleNamespace
import unittest

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
    CompletionRequest,
    OpenAICompatibleProvider,
    ProviderError,
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


class OpenAICompatibleProviderTests(unittest.TestCase):
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
                provider = OpenAICompatibleProvider(
                    "model", client=FakeClient(completions)
                )
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
