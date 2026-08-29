"""Tool declaration and execution protocol."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from coding_agent.core.json_types import JsonObject
from coding_agent.core.results import ToolResult


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonObject
    read_only: bool = False
    concurrency_safe: bool = False
    destructive: bool = False
    max_result_chars: int = 24_000

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not self.description:
            raise ValueError("tool description must not be empty")
        if self.max_result_chars <= 0:
            raise ValueError("max_result_chars must be positive")
        if self.read_only and self.destructive:
            raise ValueError("a read-only tool cannot be destructive")
        object.__setattr__(self, "input_schema", dict(self.input_schema))


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    workspace_root: str
    cwd: str


class Tool(ABC):
    spec: ToolSpec

    @abstractmethod
    def execute(self, arguments: JsonObject, context: ToolContext) -> ToolResult:
        """Execute validated arguments and return a structured result."""
