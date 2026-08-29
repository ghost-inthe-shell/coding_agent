"""Validate model arguments, run a registered tool, and process its result."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from coding_agent.core.messages import ToolCall
from coding_agent.core.results import ToolResult

from .base import Tool, ToolContext, ToolSpec
from .result_processor import ToolResultProcessor


class ToolExecutor:
    def __init__(self, tools: Iterable[Tool], result_processor: ToolResultProcessor) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool
        self._result_processor = result_processor

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return self._process(call, ToolResult.error(f"unknown tool: {call.name}"))
        if call.parse_error is not None:
            return self._process(
                call,
                ToolResult.error(f"invalid JSON arguments: {call.parse_error}"),
            )

        try:
            arguments = tool.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            error = exc.errors(include_url=False)
            return self._process(
                call,
                ToolResult.error(
                    f"invalid arguments for {call.name}: {error}",
                    metadata={"validation_errors": error},
                ),
            )

        return self._process(call, tool.execute(arguments, context))

    def _process(self, call: ToolCall, result: ToolResult) -> ToolResult:
        return self._result_processor.process(call, result)
