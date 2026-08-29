"""Minimal typed tool declaration and execution protocol."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from coding_agent.core.json_types import JsonObject
from coding_agent.core.results import ToolResult


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: JsonObject

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not self.description:
            raise ValueError("tool description must not be empty")
        object.__setattr__(self, "input_schema", dict(self.input_schema))


@dataclass(frozen=True, slots=True)
class ToolContext:
    session_id: str
    workspace_root: str
    artifact_root: str
    cwd: str


class ToolInput(BaseModel):
    """Base for every model-supplied tool input."""

    model_config = ConfigDict(strict=True, extra="forbid")


InputT = TypeVar("InputT", bound=ToolInput)


class Tool(ABC, Generic[InputT]):
    name: str
    description: str
    input_model: type[InputT]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )

    @abstractmethod
    def execute(self, arguments: InputT, context: ToolContext) -> ToolResult:
        """Execute validated arguments and return a structured result."""
