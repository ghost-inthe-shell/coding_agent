"""Derive deterministic evaluation metrics from one stable session checkpoint."""

from __future__ import annotations

from dataclasses import dataclass

from coding_agent.core.json_types import JsonObject
from coding_agent.core.messages import AssistantMessage, ToolCall, ToolResultMessage, UserMessage
from coding_agent.core.session import SessionState
from coding_agent.core.types import ToolResultStatus
from coding_agent.core.usage import Usage


@dataclass(frozen=True, slots=True)
class ToolMetrics:
    name: str
    calls: int = 0
    successes: int = 0
    errors: int = 0
    denied: int = 0
    timeouts: int = 0
    cancelled: int = 0

    def to_dict(self) -> JsonObject:
        return {
            "calls": self.calls,
            "successes": self.successes,
            "errors": self.errors,
            "denied": self.denied,
            "timeouts": self.timeouts,
            "cancelled": self.cancelled,
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    case_id: str
    passed: bool
    session_id: str
    workspace_root: str
    providers: tuple[str, ...]
    models: tuple[str, ...]
    user_turns: int
    agent_model_calls: int
    tool_calls: int
    tool_results: int
    tool_successes: int
    tool_errors: int
    tool_denied: int
    tool_timeouts: int
    tool_cancelled: int
    conversation_span_ms: int
    usage: Usage
    compacted: bool
    tools: tuple[ToolMetrics, ...]

    def to_dict(self) -> JsonObject:
        return {
            "case_id": self.case_id,
            "verdict": "PASS" if self.passed else "FAIL",
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "providers": list(self.providers),
            "models": list(self.models),
            "user_turns": self.user_turns,
            "agent_model_calls": self.agent_model_calls,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "tool_successes": self.tool_successes,
            "tool_errors": self.tool_errors,
            "tool_denied": self.tool_denied,
            "tool_timeouts": self.tool_timeouts,
            "tool_cancelled": self.tool_cancelled,
            "conversation_span_ms": self.conversation_span_ms,
            "usage": self.usage.to_dict(),
            "compacted": self.compacted,
            "tools": {tool.name: tool.to_dict() for tool in self.tools},
        }


@dataclass(slots=True)
class _MutableToolMetrics:
    calls: int = 0
    successes: int = 0
    errors: int = 0
    denied: int = 0
    timeouts: int = 0
    cancelled: int = 0


def build_eval_report(case_id: str, state: SessionState, *, passed: bool) -> EvalReport:
    """Build one report without reading or mutating external state."""

    if not case_id:
        raise ValueError("case_id must not be empty")
    state.validate()

    providers: list[str] = []
    models: list[str] = []
    tool_metrics: dict[str, _MutableToolMetrics] = {}
    user_turns = 0
    agent_model_calls = 0
    tool_calls = 0
    tool_results = 0
    tool_successes = 0
    tool_errors = 0
    tool_denied = 0
    tool_timeouts = 0
    tool_cancelled = 0
    first_user_timestamp: int | None = None
    last_message_timestamp: int | None = None

    for message in state.messages:
        last_message_timestamp = message.timestamp
        if isinstance(message, UserMessage):
            user_turns += 1
            if first_user_timestamp is None:
                first_user_timestamp = message.timestamp
        elif isinstance(message, AssistantMessage):
            agent_model_calls += 1
            _append_unique(providers, message.provider)
            _append_unique(models, message.model)
            for block in message.content:
                if isinstance(block, ToolCall):
                    tool_calls += 1
                    tool_metrics.setdefault(block.name, _MutableToolMetrics()).calls += 1
        elif isinstance(message, ToolResultMessage):
            tool_results += 1
            metrics = tool_metrics.setdefault(message.tool_name, _MutableToolMetrics())
            if message.status is ToolResultStatus.SUCCESS:
                tool_successes += 1
                metrics.successes += 1
            elif message.status is ToolResultStatus.ERROR:
                tool_errors += 1
                metrics.errors += 1
            elif message.status is ToolResultStatus.DENIED:
                tool_denied += 1
                metrics.denied += 1
            elif message.status is ToolResultStatus.TIMEOUT:
                tool_timeouts += 1
                metrics.timeouts += 1
            elif message.status is ToolResultStatus.CANCELLED:
                tool_cancelled += 1
                metrics.cancelled += 1

    conversation_span_ms = 0
    if first_user_timestamp is not None and last_message_timestamp is not None:
        conversation_span_ms = max(0, last_message_timestamp - first_user_timestamp)

    return EvalReport(
        case_id=case_id,
        passed=passed,
        session_id=state.session_id,
        workspace_root=state.workspace_root,
        providers=tuple(providers),
        models=tuple(models),
        user_turns=user_turns,
        agent_model_calls=agent_model_calls,
        tool_calls=tool_calls,
        tool_results=tool_results,
        tool_successes=tool_successes,
        tool_errors=tool_errors,
        tool_denied=tool_denied,
        tool_timeouts=tool_timeouts,
        tool_cancelled=tool_cancelled,
        conversation_span_ms=conversation_span_ms,
        usage=state.usage,
        compacted=state.compaction is not None,
        tools=tuple(
            ToolMetrics(
                name=name,
                calls=metrics.calls,
                successes=metrics.successes,
                errors=metrics.errors,
                denied=metrics.denied,
                timeouts=metrics.timeouts,
                cancelled=metrics.cancelled,
            )
            for name, metrics in tool_metrics.items()
        ),
    )


def format_eval_report(report: EvalReport) -> str:
    """Render stable, human-readable report text."""

    usage = report.usage
    lines = [
        f"Case: {report.case_id}",
        f"Verdict: {'PASS' if report.passed else 'FAIL'}",
        f"Session: {report.session_id}",
        f"Workspace: {report.workspace_root}",
        f"Provider: {_format_names(report.providers)}",
        f"Model: {_format_names(report.models)}",
        f"User turns: {report.user_turns}",
        f"Agent model calls: {report.agent_model_calls}",
        f"Tool calls: {report.tool_calls}",
        f"Tool results: {report.tool_results}",
        f"Tool successes: {report.tool_successes}",
        f"Tool errors: {report.tool_errors}",
        f"Tool denied: {report.tool_denied}",
        f"Tool timeouts: {report.tool_timeouts}",
        f"Tool cancelled: {report.tool_cancelled}",
        f"Conversation span: {report.conversation_span_ms / 1000:.1f}s",
        f"Input tokens: {usage.input_tokens}",
        f"Cache-read tokens: {usage.cache_read_tokens}",
        f"Cache-write tokens: {usage.cache_write_tokens}",
        f"Output tokens: {usage.output_tokens}",
        f"Reasoning tokens: {usage.reasoning_tokens}",
        f"Compacted: {'yes' if report.compacted else 'no'}",
        "",
        "Tools:",
    ]
    if not report.tools:
        lines.append("  (none)")
    else:
        width = max(len(tool.name) for tool in report.tools)
        for tool in report.tools:
            lines.append(
                f"  {tool.name:<{width}}  calls={tool.calls} success={tool.successes} "
                f"error={tool.errors} denied={tool.denied} timeout={tool.timeouts} "
                f"cancelled={tool.cancelled}"
            )
    return "\n".join(lines)


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _format_names(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "(none)"
