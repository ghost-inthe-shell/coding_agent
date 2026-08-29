"""Synchronous OpenAI Chat Completions compatible provider."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

from coding_agent.core.messages import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.types import StopReason
from coding_agent.core.usage import Usage
from coding_agent.tools.base import ToolSpec

from .base import CompletionRequest, LLMProvider, ProviderError


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
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
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleProvider requires the 'openai' extra: "
                "pip install 'coding-agent[openai]'"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, request: CompletionRequest) -> AssistantMessage:
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": _convert_messages(request),
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if request.tools:
            arguments["tools"] = [_convert_tool(spec) for spec in request.tools]

        try:
            response = self._client.chat.completions.create(**arguments)
        except Exception as exc:
            raise ProviderError(f"OpenAI-compatible request failed: {exc}") from exc

        choices = _field(response, "choices", ())
        if not choices:
            raise ProviderError("OpenAI-compatible response contained no choices")
        choice = choices[0]
        message = _field(choice, "message")
        if message is None:
            raise ProviderError("OpenAI-compatible response contained no message")

        content: list[TextBlock | ToolCall] = []
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
        stop_reason, error_message = _map_stop_reason(raw_stop_reason)
        has_tool_calls = any(isinstance(block, ToolCall) for block in content)
        valid_tool_stop_reasons = {StopReason.TOOL_USE, StopReason.LENGTH}
        if has_tool_calls and stop_reason not in valid_tool_stop_reasons:
            raise ProviderError(
                f"OpenAI-compatible response returned tool calls with finish_reason={raw_stop_reason!r}"
            )
        if stop_reason is StopReason.TOOL_USE and not has_tool_calls:
            raise ProviderError(
                "OpenAI-compatible response ended with tool_calls but had no calls"
            )

        return AssistantMessage(
            content=tuple(content),
            provider="openai-compatible",
            model=_field(response, "model", self.model) or self.model,
            usage=_convert_usage(_field(response, "usage")),
            stop_reason=stop_reason,
            response_id=_field(response, "id"),
            raw_stop_reason=raw_stop_reason,
            error_message=error_message,
        )


def _convert_messages(request: CompletionRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": request.system_prompt}
    ]
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
