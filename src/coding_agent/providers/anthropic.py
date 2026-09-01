"""Synchronous Anthropic Messages API provider."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from coding_agent.core.messages import (
    AssistantMessage,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.types import StopReason
from coding_agent.core.usage import Usage
from coding_agent.tools.base import ToolSpec

from .base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    CompletionEventSink,
    CompletionRequest,
    CompletionTextDelta,
    CompletionThinkingDelta,
    LLMProvider,
    ProviderError,
)
from .reasoning import ReasoningLevel


@dataclass(slots=True)
class _StreamingContentBlock:
    type: str
    id: str | None = None
    name: str | None = None
    text_parts: list[str] = field(default_factory=list)


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        stream: bool = False,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.max_tokens = max_tokens
        self.stream = stream
        if client is not None:
            self._client = client
            return

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicProvider requires the 'anthropic' extra: "
                "pip install 'coding-agent[anthropic]'"
            ) from exc
        self._client = Anthropic(api_key=api_key)

    @property
    def max_output_tokens(self) -> int:
        return self.max_tokens

    def complete(
        self,
        request: CompletionRequest,
        *,
        event_sink: CompletionEventSink | None = None,
    ) -> AssistantMessage:
        if request.reasoning not in {
            ReasoningLevel.DEFAULT,
            ReasoningLevel.OFF,
            ReasoningLevel.MINIMAL,
        }:
            raise ProviderError("reasoning levels are not implemented for the Anthropic provider")
        use_stream = self.stream and event_sink is not None
        arguments: dict[str, Any] = {
            "model": self.model,
            "max_tokens": min(
                request.max_output_tokens or self.max_tokens,
                self.max_tokens,
            ),
            "messages": _convert_messages(request.messages),
            "stream": use_stream,
        }
        if request.system_prompt:
            arguments["system"] = request.system_prompt
        if request.tools:
            arguments["tools"] = [_convert_tool(spec) for spec in request.tools]

        try:
            response = self._client.messages.create(**arguments)
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        if use_stream:
            try:
                return self._consume_stream(response, event_sink)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"Anthropic stream failed: {exc}") from exc
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        return _convert_response(response, self.model)

    def _consume_stream(
        self,
        events: Any,
        event_sink: CompletionEventSink,
    ) -> AssistantMessage:
        blocks: dict[int, _StreamingContentBlock] = {}
        response_id: str | None = None
        response_model = self.model
        usage = Usage()
        raw_stop_reason: Any = None
        stop_source: Any = None
        saw_message_start = False

        for event in events:
            event_type = _field(event, "type")
            if event_type == "message_start":
                message = _field(event, "message")
                if message is None:
                    raise ProviderError("Anthropic stream contained no starting message")
                saw_message_start = True
                response_id = _optional_text(message, "id", "message id")
                response_model = _optional_text(message, "model", "model") or self.model
                usage = _merge_stream_usage(usage, _field(message, "usage"))
            elif event_type == "content_block_start":
                _start_streaming_block(blocks, event)
            elif event_type == "content_block_delta":
                _apply_streaming_delta(blocks, event, event_sink)
            elif event_type == "message_delta":
                stop_source = _field(event, "delta")
                stop_reason = _field(stop_source, "stop_reason")
                if stop_reason is not None:
                    raw_stop_reason = stop_reason
                usage = _merge_stream_usage(usage, _field(event, "usage"))
            elif event_type == "error":
                error = _field(event, "error")
                message = _field(error, "message")
                raise ProviderError(
                    f"Anthropic stream error: {message or _field(error, 'type') or 'unknown error'}"
                )

        if not saw_message_start:
            raise ProviderError("Anthropic stream contained no starting message")
        content = _finish_streaming_blocks(blocks)
        stop_reason, error_message = _validated_stop_reason(
            content,
            stop_source,
            raw_stop_reason,
        )
        return AssistantMessage(
            content=tuple(content),
            provider="anthropic",
            model=response_model,
            usage=usage,
            stop_reason=stop_reason,
            response_id=response_id,
            raw_stop_reason=raw_stop_reason,
            error_message=error_message,
        )


def _convert_response(response: Any, configured_model: str) -> AssistantMessage:
    content: list[TextBlock | ThinkingBlock | ToolCall] = []
    for block in _field(response, "content", ()):
        block_type = _field(block, "type")
        if block_type == "text":
            text = _field(block, "text")
            if not isinstance(text, str):
                raise ProviderError("Anthropic response text block contained no text")
            content.append(TextBlock(text))
        elif block_type == "thinking":
            thinking = _field(block, "thinking")
            if not isinstance(thinking, str):
                raise ProviderError("Anthropic response thinking block contained no thinking")
            content.append(ThinkingBlock(thinking))
        elif block_type == "tool_use":
            parsed_input, raw_arguments, parse_error = _normalize_tool_input(
                _field(block, "input", {})
            )
            content.append(
                ToolCall(
                    id=_required_text(block, "id", "tool use id"),
                    name=_required_text(block, "name", "tool name"),
                    arguments=parsed_input,
                    raw_arguments=raw_arguments,
                    parse_error=parse_error,
                )
            )

    raw_stop_reason = _field(response, "stop_reason")
    stop_reason, error_message = _validated_stop_reason(
        content,
        response,
        raw_stop_reason,
    )

    return AssistantMessage(
        content=tuple(content),
        provider="anthropic",
        model=_field(response, "model", configured_model) or configured_model,
        usage=_convert_usage(_field(response, "usage")),
        stop_reason=stop_reason,
        response_id=_field(response, "id"),
        raw_stop_reason=raw_stop_reason,
        error_message=error_message,
    )


def _start_streaming_block(
    blocks: dict[int, _StreamingContentBlock],
    event: Any,
) -> None:
    index = _stream_index(event)
    if index in blocks:
        raise ProviderError(f"Anthropic stream started content block index {index} more than once")
    raw_block = _field(event, "content_block")
    block_type = _field(raw_block, "type")
    if not isinstance(block_type, str) or not block_type:
        raise ProviderError("Anthropic stream contained an invalid content block")
    if block_type == "tool_use":
        blocks[index] = _StreamingContentBlock(
            type=block_type,
            id=_required_text(raw_block, "id", "tool use id"),
            name=_required_text(raw_block, "name", "tool name"),
        )
    else:
        blocks[index] = _StreamingContentBlock(type=block_type)


def _apply_streaming_delta(
    blocks: dict[int, _StreamingContentBlock],
    event: Any,
    event_sink: CompletionEventSink,
) -> None:
    index = _stream_index(event)
    block = blocks.get(index)
    if block is None:
        raise ProviderError(f"Anthropic stream updated unknown content block index {index}")
    delta = _field(event, "delta")
    delta_type = _field(delta, "type")
    if delta_type == "text_delta":
        text = _required_delta_text(delta, "text", "text")
        if block.type != "text":
            raise ProviderError("Anthropic text delta targeted a non-text block")
        block.text_parts.append(text)
        event_sink(CompletionTextDelta(text))
    elif delta_type == "thinking_delta":
        thinking = _required_delta_text(delta, "thinking", "thinking")
        if block.type != "thinking":
            raise ProviderError("Anthropic thinking delta targeted a non-thinking block")
        block.text_parts.append(thinking)
        event_sink(CompletionThinkingDelta(thinking))
    elif delta_type == "input_json_delta":
        partial_json = _required_delta_text(delta, "partial_json", "tool input")
        if block.type != "tool_use":
            raise ProviderError("Anthropic tool input delta targeted a non-tool block")
        block.text_parts.append(partial_json)
    # Signature, citation, ping, and future delta types do not affect the
    # normalized content supported by this agent.


def _finish_streaming_blocks(
    blocks: dict[int, _StreamingContentBlock],
) -> list[TextBlock | ThinkingBlock | ToolCall]:
    content: list[TextBlock | ThinkingBlock | ToolCall] = []
    for index in sorted(blocks):
        block = blocks[index]
        joined = "".join(block.text_parts)
        if block.type == "text":
            content.append(TextBlock(joined))
        elif block.type == "thinking":
            content.append(ThinkingBlock(joined))
        elif block.type == "tool_use":
            parsed, raw_arguments, parse_error = _parse_streaming_tool_input(joined)
            content.append(
                ToolCall(
                    id=block.id or "",
                    name=block.name or "",
                    arguments=parsed,
                    raw_arguments=raw_arguments,
                    parse_error=parse_error,
                )
            )
    return content


def _parse_streaming_tool_input(
    raw_arguments: str,
) -> tuple[dict[str, Any], str, str | None]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return {}, raw_arguments, str(exc)
    if not isinstance(parsed, dict):
        return {}, raw_arguments, "tool input must decode to a JSON object"
    return parsed, raw_arguments, None


def _normalize_tool_input(
    raw_input: Any,
) -> tuple[dict[str, Any], str | None, str | None]:
    if isinstance(raw_input, dict):
        return raw_input, None, None
    return {}, json.dumps(raw_input, ensure_ascii=False), "tool input must be a JSON object"


def _validated_stop_reason(
    content: Sequence[TextBlock | ThinkingBlock | ToolCall],
    response: Any,
    raw_stop_reason: Any,
) -> tuple[StopReason, str | None]:
    stop_reason, error_message = _map_stop_reason(response, raw_stop_reason)
    has_tool_calls = any(isinstance(block, ToolCall) for block in content)
    valid_tool_stop_reasons = {StopReason.TOOL_USE, StopReason.LENGTH}
    if has_tool_calls and stop_reason not in valid_tool_stop_reasons:
        raise ProviderError(
            f"Anthropic response returned tool calls with stop_reason={raw_stop_reason!r}"
        )
    if stop_reason is StopReason.TOOL_USE and not has_tool_calls:
        raise ProviderError("Anthropic response stopped for tool use but contained no calls")
    return stop_reason, error_message


def _merge_stream_usage(current: Usage, raw_usage: Any) -> Usage:
    if raw_usage is None:
        return current

    def updated(name: str, current_value: int) -> int:
        value = _field(raw_usage, name)
        return value if isinstance(value, int) and not isinstance(value, bool) else current_value

    return Usage(
        input_tokens=updated("input_tokens", current.input_tokens),
        output_tokens=updated("output_tokens", current.output_tokens),
        cache_read_tokens=updated("cache_read_input_tokens", current.cache_read_tokens),
        cache_write_tokens=updated("cache_creation_input_tokens", current.cache_write_tokens),
        reasoning_tokens=current.reasoning_tokens,
    )


def _stream_index(event: Any) -> int:
    index = _field(event, "index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ProviderError("Anthropic stream contained an invalid content block index")
    return index


def _required_delta_text(value: Any, name: str, label: str) -> str:
    text = _field(value, name)
    if not isinstance(text, str) or not text:
        raise ProviderError(f"Anthropic streaming {label} delta contained no text")
    return text


def _optional_text(value: Any, name: str, label: str) -> str | None:
    text = _field(value, name)
    if text is None:
        return None
    if not isinstance(text, str) or not text:
        raise ProviderError(f"Anthropic stream contained an invalid {label}")
    return text


def _convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, ToolResultMessage):
            results: list[dict[str, Any]] = []
            while index < len(messages) and isinstance(messages[index], ToolResultMessage):
                result = messages[index]
                assert isinstance(result, ToolResultMessage)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": result.text,
                        "is_error": result.is_error,
                    }
                )
                index += 1
            converted.append({"role": "user", "content": results})
            continue
        converted.append(_convert_message(message))
        index += 1
    return converted


def _convert_message(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {
            "role": "user",
            "content": [{"type": "text", "text": block.text} for block in message.content],
        }
    if isinstance(message, ToolResultMessage):
        raise AssertionError(  # noqa: TRY004 - internal invariant, not caller input
            "tool results must be grouped by _convert_messages"
        )

    content: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                }
            )
        elif not isinstance(block, ThinkingBlock):
            raise AssertionError(  # noqa: TRY004 - closed normalized block union
                f"unsupported assistant content block: {type(block)!r}"
            )
    return {"role": "assistant", "content": content}


def _convert_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }


def _convert_usage(raw_usage: Any) -> Usage:
    if raw_usage is None:
        return Usage()
    return Usage(
        input_tokens=_integer_field(raw_usage, "input_tokens"),
        output_tokens=_integer_field(raw_usage, "output_tokens"),
        cache_read_tokens=_integer_field(raw_usage, "cache_read_input_tokens"),
        cache_write_tokens=_integer_field(raw_usage, "cache_creation_input_tokens"),
    )


def _map_stop_reason(response: Any, raw_reason: Any) -> tuple[StopReason, str | None]:
    if raw_reason in {"end_turn", "stop_sequence"}:
        return StopReason.STOP, None
    if raw_reason == "tool_use":
        return StopReason.TOOL_USE, None
    if raw_reason == "max_tokens":
        return StopReason.LENGTH, None
    if raw_reason == "refusal":
        stop_details = _field(response, "stop_details")
        explanation = _field(stop_details, "explanation")
        return StopReason.ERROR, explanation or "Anthropic refused the request"
    return StopReason.ERROR, f"Anthropic stop_reason: {raw_reason}"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer_field(value: Any, name: str) -> int:
    field = _field(value, name, 0)
    return field if isinstance(field, int) and not isinstance(field, bool) else 0


def _required_text(value: Any, name: str, label: str) -> str:
    field = _field(value, name)
    if not isinstance(field, str) or not field:
        raise ProviderError(f"Anthropic response contained an invalid {label}")
    return field
