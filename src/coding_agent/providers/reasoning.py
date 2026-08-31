"""Provider-facing reasoning intent and OpenAI-compatible API dialects."""

from enum import Enum


class ApiDialect(str, Enum):
    GENERIC = "generic"
    DEEPSEEK = "deepseek"
    DASHSCOPE = "dashscope"
    MOONSHOT = "moonshot"


class ReasoningLevel(str, Enum):
    DEFAULT = "default"
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
    MINIMAL = "minimal"
