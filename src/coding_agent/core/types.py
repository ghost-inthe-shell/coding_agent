"""Stable discriminants used across provider, runtime, and storage boundaries."""

from enum import Enum


class StopReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_USE = "tool_use"
    ERROR = "error"
    ABORTED = "aborted"


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    LIMIT_REACHED = "limit_reached"
    INTERRUPTED = "interrupted"
    PROVIDER_ERROR = "provider_error"
    RUNTIME_ERROR = "runtime_error"


class SessionStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    IDLE = "idle"
    INTERRUPTED = "interrupted"
    ERROR = "error"

