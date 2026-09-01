"""Synchronous OpenAI Chat Completions compatible provider."""

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
from .reasoning import ApiDialect, ReasoningLevel

_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


@dataclass(slots=True)
class _StreamingToolCall:
    id_parts: list[str] = field(default_factory=list)
    name_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        dialect: ApiDialect = ApiDialect.GENERIC,
        reasoning: ReasoningLevel = ReasoningLevel.DEFAULT,
        stream: bool = False,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not isinstance(dialect, ApiDialect):
            raise TypeError("dialect must be an ApiDialect")
        if not isinstance(reasoning, ReasoningLevel):
            raise TypeError("reasoning must be a ReasoningLevel")
        _validate_configured_reasoning(dialect, reasoning)
        self.model = model
        self.max_tokens = max_tokens
        self.dialect = dialect
        self.reasoning = reasoning
        self.stream = stream
        if client is not None:
            self._client = client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleProvider requires the 'openai' extra: "
                "pip install 'coding-agent[openai]'"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def max_output_tokens(self) -> int:
        return self.max_tokens

    def complete(
        self,
        request: CompletionRequest,
        *,
        event_sink: CompletionEventSink | None = None,
    ) -> AssistantMessage:
        use_stream = self.stream and event_sink is not None
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": _convert_messages(request),
            "max_tokens": min(
                request.max_output_tokens or self.max_tokens,
                self.max_tokens,
            ),
            "stream": use_stream,
        }
        if request.tools:
            arguments["tools"] = [_convert_tool(spec) for spec in request.tools]
        requested_reasoning = (
            self.reasoning if request.reasoning is ReasoningLevel.DEFAULT else request.reasoning
        )
        arguments.update(_reasoning_arguments(self.dialect, requested_reasoning))

        try:
            response = self._client.chat.completions.create(**arguments)
        except Exception as exc:
            raise ProviderError(f"OpenAI-compatible request failed: {exc}") from exc

        if use_stream:
            try:
                return self._consume_stream(response, event_sink)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(f"OpenAI-compatible stream failed: {exc}") from exc
        return _convert_response(response, self.model)

    def _consume_stream(
        self,
        chunks: Any,
        event_sink: CompletionEventSink,
    ) -> AssistantMessage:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_field: str | None = None
        tool_calls: dict[int, _StreamingToolCall] = {}
        response_id: str | None = None
        response_model = self.model
        raw_stop_reason: Any = None
        usage = Usage()
        saw_choice = False

        for chunk in chunks:
            chunk_id = _field(chunk, "id")
            if isinstance(chunk_id, str) and chunk_id:
                response_id = chunk_id
            chunk_model = _field(chunk, "model")
            if isinstance(chunk_model, str) and chunk_model:
                response_model = chunk_model
            raw_usage = _field(chunk, "usage")
            if raw_usage is not None:
                usage = _convert_usage(raw_usage)

            choices = _field(chunk, "choices", ()) or ()
            if not choices:
                continue
            choice = _stream_choice(choices)
            saw_choice = True
            finish_reason = _field(choice, "finish_reason")
            if finish_reason is not None:
                raw_stop_reason = finish_reason
            delta = _field(choice, "delta")
            if delta is None:
                continue

            extracted_thinking = _extract_thinking_value(delta)
            if extracted_thinking is not None:
                field_name, value = extracted_thinking
                if thinking_field is None:
                    thinking_field = field_name
                elif field_name != thinking_field:
                    raise ProviderError(
                        "OpenAI-compatible stream changed reasoning field from "
                        f"{thinking_field!r} to {field_name!r}"
                    )
                thinking_parts.append(value)
                event_sink(CompletionThinkingDelta(value))

            text = _field(delta, "content")
            if text is not None and text != "":
                if not isinstance(text, str):
                    raise ProviderError("OpenAI-compatible response content was not text")
                text_parts.append(text)
                event_sink(CompletionTextDelta(text))

            for raw_call in _field(delta, "tool_calls", ()) or ():
                _accumulate_tool_call(tool_calls, raw_call)

        if not saw_choice:
            raise ProviderError("OpenAI-compatible response contained no choices")

        content: list[TextBlock | ThinkingBlock | ToolCall] = []
        if thinking_parts:
            content.append(
                ThinkingBlock(
                    "".join(thinking_parts),
                    replay_field=thinking_field,
                )
            )
        if text_parts:
            content.append(TextBlock("".join(text_parts)))
        content.extend(_finish_streaming_tool_calls(tool_calls))
        stop_reason, error_message = _validated_stop_reason(content, raw_stop_reason)
        return AssistantMessage(
            content=tuple(content),
            provider="openai-compatible",
            model=response_model,
            usage=usage,
            stop_reason=stop_reason,
            response_id=response_id,
            raw_stop_reason=raw_stop_reason,
            error_message=error_message,
        )


def _convert_response(response: Any, configured_model: str) -> AssistantMessage:
    choices = _field(response, "choices", ())
    if not choices:
        raise ProviderError("OpenAI-compatible response contained no choices")
    choice = choices[0]
    message = _field(choice, "message")
    if message is None:
        raise ProviderError("OpenAI-compatible response contained no message")

    content: list[TextBlock | ThinkingBlock | ToolCall] = []
    thinking = _extract_thinking(message)
    if thinking is not None:
        content.append(thinking)
    text = _field(message, "content")
    if text:
        if not isinstance(text, str):
            raise ProviderError("OpenAI-compatible response content was not text")
        content.append(TextBlock(text))

    for raw_call in _field(message, "tool_calls", ()) or ():
        function = _field(raw_call, "function")
        if function is None:
            raise ProviderError("OpenAI-compatible tool call contained no function")
        raw_arguments = _field(function, "arguments", "")
        if not isinstance(raw_arguments, str):
            raise ProviderError("OpenAI-compatible tool arguments were not a JSON string")
        parsed, parse_error = _parse_arguments(raw_arguments)
        content.append(
            ToolCall(
                id=_required_text(raw_call, "id", "tool call id"),
                name=_required_text(function, "name", "tool name"),
                arguments=parsed,
                raw_arguments=raw_arguments,
                parse_error=parse_error,
            )
        )

    raw_stop_reason = _field(choice, "finish_reason")
    stop_reason, error_message = _validated_stop_reason(content, raw_stop_reason)

    return AssistantMessage(
        content=tuple(content),
        provider="openai-compatible",
        model=_field(response, "model", configured_model) or configured_model,
        usage=_convert_usage(_field(response, "usage")),
        stop_reason=stop_reason,
        response_id=_field(response, "id"),
        raw_stop_reason=raw_stop_reason,
        error_message=error_message,
    )


def _stream_choice(choices: Sequence[Any]) -> Any:
    for choice in choices:
        if _field(choice, "index") == 0:
            return choice
    return choices[0]


def _accumulate_tool_call(
    tool_calls: dict[int, _StreamingToolCall],
    raw_call: Any,
) -> None:
    index = _field(raw_call, "index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ProviderError("OpenAI-compatible streaming tool call contained an invalid index")
    call = tool_calls.setdefault(index, _StreamingToolCall())
    _append_text_part(call.id_parts, _field(raw_call, "id"), "tool call id")
    function = _field(raw_call, "function")
    if function is None:
        return
    _append_text_part(call.name_parts, _field(function, "name"), "tool name")
    _append_text_part(
        call.argument_parts,
        _field(function, "arguments"),
        "tool arguments",
    )


def _append_text_part(parts: list[str], value: Any, label: str) -> None:
    if value is None or value == "":
        return
    if not isinstance(value, str):
        raise ProviderError(f"OpenAI-compatible streaming {label} fragment was not text")
    parts.append(value)


def _finish_streaming_tool_calls(
    tool_calls: dict[int, _StreamingToolCall],
) -> list[ToolCall]:
    result: list[ToolCall] = []
    for index in sorted(tool_calls):
        call = tool_calls[index]
        call_id = "".join(call.id_parts)
        name = "".join(call.name_parts)
        raw_arguments = "".join(call.argument_parts)
        if not call_id:
            raise ProviderError("OpenAI-compatible response contained an invalid tool call id")
        if not name:
            raise ProviderError("OpenAI-compatible response contained an invalid tool name")
        parsed, parse_error = _parse_arguments(raw_arguments)
        result.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=parsed,
                raw_arguments=raw_arguments,
                parse_error=parse_error,
            )
        )
    return result


def _validated_stop_reason(
    content: Sequence[TextBlock | ThinkingBlock | ToolCall],
    raw_stop_reason: Any,
) -> tuple[StopReason, str | None]:
    stop_reason, error_message = _map_stop_reason(raw_stop_reason)
    has_tool_calls = any(isinstance(block, ToolCall) for block in content)
    valid_tool_stop_reasons = {StopReason.TOOL_USE, StopReason.LENGTH}
    if has_tool_calls and stop_reason not in valid_tool_stop_reasons:
        raise ProviderError(
            f"OpenAI-compatible response returned tool calls with finish_reason={raw_stop_reason!r}"
        )
    if stop_reason is StopReason.TOOL_USE and not has_tool_calls:
        raise ProviderError("OpenAI-compatible response ended with tool_calls but had no calls")
    return stop_reason, error_message


def _convert_messages(request: CompletionRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": request.system_prompt}]
    messages.extend(_convert_message(message) for message in request.messages)
    return messages


def _convert_message(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": _text(message.content)}
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.text,
        }

    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.text or None,
    }
    for block in message.content:
        if isinstance(block, ThinkingBlock) and block.replay_field is not None:
            prior = result.get(block.replay_field, "")
            if not isinstance(prior, str):
                raise AssertionError("thinking replay field collided with a non-text value")
            result[block.replay_field] = prior + block.thinking
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    if call.raw_arguments is not None
                    else json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
            for call in message.tool_calls
        ]
    return result


def _convert_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _convert_usage(raw_usage: Any) -> Usage:
    if raw_usage is None:
        return Usage()
    prompt_details = _field(raw_usage, "prompt_tokens_details")
    completion_details = _field(raw_usage, "completion_tokens_details")
    return Usage(
        input_tokens=_integer_field(raw_usage, "prompt_tokens"),
        output_tokens=_integer_field(raw_usage, "completion_tokens"),
        cache_read_tokens=_integer_field(prompt_details, "cached_tokens"),
        reasoning_tokens=_integer_field(completion_details, "reasoning_tokens"),
    )


def _map_stop_reason(raw_reason: Any) -> tuple[StopReason, str | None]:
    if raw_reason == "stop" or raw_reason is None:
        return StopReason.STOP, None
    if raw_reason in {"tool_calls", "function_call"}:
        return StopReason.TOOL_USE, None
    if raw_reason == "length":
        return StopReason.LENGTH, None
    return StopReason.ERROR, f"OpenAI-compatible finish_reason: {raw_reason}"


def _parse_arguments(raw_arguments: str) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return {}, str(exc)
    if not isinstance(parsed, dict):
        return {}, "tool arguments must decode to a JSON object"
    return parsed, None


def _extract_thinking(message: Any) -> ThinkingBlock | None:
    extracted = _extract_thinking_value(message)
    if extracted is None:
        return None
    field_name, value = extracted
    return ThinkingBlock(value, replay_field=field_name)


def _extract_thinking_value(value: Any) -> tuple[str, str] | None:
    for field_name in _REASONING_FIELDS:
        field_value = _field(value, field_name)
        if field_value is None or field_value == "":
            continue
        if not isinstance(field_value, str):
            raise ProviderError(f"OpenAI-compatible response {field_name} was not text")
        return field_name, field_value
    return None


def _text(blocks: Sequence[TextBlock]) -> str:
    return "".join(block.text for block in blocks)


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
        raise ProviderError(f"OpenAI-compatible response contained an invalid {label}")
    return field


def _validate_configured_reasoning(
    dialect: ApiDialect,
    reasoning: ReasoningLevel,
) -> None:
    if reasoning is ReasoningLevel.MINIMAL:
        raise ValueError("minimal reasoning is reserved for internal requests")
    supported = {
        ApiDialect.GENERIC: {ReasoningLevel.DEFAULT},
        ApiDialect.DEEPSEEK: {
            ReasoningLevel.DEFAULT,
            ReasoningLevel.OFF,
            ReasoningLevel.LOW,
            ReasoningLevel.HIGH,
            ReasoningLevel.MAX,
        },
        ApiDialect.DASHSCOPE: {
            ReasoningLevel.DEFAULT,
            ReasoningLevel.OFF,
            ReasoningLevel.LOW,
            ReasoningLevel.MEDIUM,
            ReasoningLevel.HIGH,
            ReasoningLevel.MAX,
        },
        ApiDialect.MOONSHOT: {
            ReasoningLevel.DEFAULT,
            ReasoningLevel.OFF,
        },
    }[dialect]
    if reasoning not in supported:
        raise ValueError(
            f"reasoning level {reasoning.value!r} is not supported by "
            f"the {dialect.value!r} API dialect"
        )


def _reasoning_arguments(
    dialect: ApiDialect,
    reasoning: ReasoningLevel,
) -> dict[str, Any]:
    if reasoning is ReasoningLevel.DEFAULT:
        return {}
    if reasoning is ReasoningLevel.MINIMAL:
        if dialect is ApiDialect.GENERIC:
            return {}
        if dialect is ApiDialect.DASHSCOPE:
            return {"extra_body": {"reasoning_effort": "low"}}
        return {"extra_body": {"thinking": {"type": "disabled"}}}

    try:
        _validate_configured_reasoning(dialect, reasoning)
    except ValueError as exc:
        raise ProviderError(str(exc)) from exc

    if dialect is ApiDialect.DEEPSEEK:
        if reasoning is ReasoningLevel.OFF:
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {
            "reasoning_effort": reasoning.value,
            "extra_body": {"thinking": {"type": "enabled"}},
        }
    if dialect is ApiDialect.DASHSCOPE:
        if reasoning is ReasoningLevel.OFF:
            return {"extra_body": {"enable_thinking": False}}
        return {"extra_body": {"reasoning_effort": reasoning.value}}
    if dialect is ApiDialect.MOONSHOT:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    raise AssertionError("generic dialect accepts only default reasoning")
