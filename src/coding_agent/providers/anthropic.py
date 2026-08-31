"""Synchronous Anthropic Messages API provider."""

from __future__ import annotations

import json
from collections.abc import Sequence
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

from .base import DEFAULT_MAX_OUTPUT_TOKENS, CompletionRequest, LLMProvider, ProviderError
from .reasoning import ReasoningLevel


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.max_tokens = max_tokens
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

    def complete(self, request: CompletionRequest) -> AssistantMessage:
        if request.reasoning not in {
            ReasoningLevel.DEFAULT,
            ReasoningLevel.OFF,
            ReasoningLevel.MINIMAL,
        }:
            raise ProviderError(
                "reasoning levels are not implemented for the Anthropic provider"
            )
        arguments: dict[str, Any] = {
            "model": self.model,
            "max_tokens": min(
                request.max_output_tokens or self.max_tokens,
                self.max_tokens,
            ),
            "messages": _convert_messages(request.messages),
        }
        if request.system_prompt:
            arguments["system"] = request.system_prompt
        if request.tools:
            arguments["tools"] = [_convert_tool(spec) for spec in request.tools]

        try:
            response = self._client.messages.create(**arguments)
        except Exception as exc:
            raise ProviderError(f"Anthropic request failed: {exc}") from exc

        content: list[TextBlock | ToolCall] = []
        for block in _field(response, "content", ()):
            block_type = _field(block, "type")
            if block_type == "text":
                text = _field(block, "text")
                if not isinstance(text, str):
                    raise ProviderError("Anthropic response text block contained no text")
                content.append(TextBlock(text))
            elif block_type == "tool_use":
                raw_input = _field(block, "input", {})
                if isinstance(raw_input, dict):
                    parsed_input = raw_input
                    raw_arguments = None
                    parse_error = None
                else:
                    parsed_input = {}
                    raw_arguments = json.dumps(raw_input, ensure_ascii=False)
                    parse_error = "tool input must be a JSON object"
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
        stop_reason, error_message = _map_stop_reason(response, raw_stop_reason)
        has_tool_calls = any(isinstance(block, ToolCall) for block in content)
        valid_tool_stop_reasons = {StopReason.TOOL_USE, StopReason.LENGTH}
        if has_tool_calls and stop_reason not in valid_tool_stop_reasons:
            raise ProviderError(
                f"Anthropic response returned tool calls with stop_reason={raw_stop_reason!r}"
            )
        if stop_reason is StopReason.TOOL_USE and not has_tool_calls:
            raise ProviderError("Anthropic response stopped for tool use but contained no calls")

        return AssistantMessage(
            content=tuple(content),
            provider="anthropic",
            model=_field(response, "model", self.model) or self.model,
            usage=_convert_usage(_field(response, "usage")),
            stop_reason=stop_reason,
            response_id=_field(response, "id"),
            raw_stop_reason=raw_stop_reason,
            error_message=error_message,
        )


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
            "content": [
                {"type": "text", "text": block.text} for block in message.content
            ],
        }
    if isinstance(message, ToolResultMessage):
        raise AssertionError("tool results must be grouped by _convert_messages")

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
            raise AssertionError(f"unsupported assistant content block: {type(block)!r}")
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
