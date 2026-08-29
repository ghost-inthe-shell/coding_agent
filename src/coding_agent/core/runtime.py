"""Synchronous agent loop; SessionState is its only conversation state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from coding_agent.providers import CompletionRequest, LLMProvider, ProviderError
from coding_agent.tools import ArtifactStore, Tool, ToolContext, ToolExecutor, ToolResultProcessor
from coding_agent.tools.base import ToolSpec

from .events import (
    EventSink,
    ModelRequested,
    ModelResponded,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
    RuntimeEvent,
)
from .messages import AssistantMessage, ToolCall, UserMessage
from .results import RunResult, ToolResult
from .session import SessionState
from .types import RunStatus, SessionStatus, StopReason
from .usage import Usage


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    max_model_calls: int = 8
    max_tool_calls: int = 32

    def __post_init__(self) -> None:
        if self.max_model_calls <= 0 or self.max_tool_calls <= 0:
            raise ValueError("runtime limits must be positive")


class Runtime:
    def __init__(
        self,
        provider: LLMProvider,
        tools: Iterable[Tool],
        *,
        limits: RuntimeLimits | None = None,
        event_sink: EventSink | None = None,
        state_home: Path | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tuple(tools)
        self._limits = limits or RuntimeLimits()
        self._event_sink = event_sink
        self._state_home = state_home

    def run_turn(self, state: SessionState, user_input: str) -> RunResult:
        state.validate()
        state.status = SessionStatus.RUNNING
        state.messages.append(UserMessage.from_text(user_input))
        state.touch()
        self._emit(TurnStarted(state.session_id))

        artifact_store = ArtifactStore(state.session_id, state_home=self._state_home)
        executor = ToolExecutor(self._tools, ToolResultProcessor(artifact_store))
        context = ToolContext(
            session_id=state.session_id,
            workspace_root=state.workspace_root,
            artifact_root=str(artifact_store.root),
            cwd=state.workspace_root,
        )
        turn_usage = Usage()
        model_calls = 0
        tool_calls = 0

        try:
            while model_calls < self._limits.max_model_calls:
                assistant = self._complete(state, executor.specs, model_calls + 1)
                model_calls += 1
                turn_usage = turn_usage + assistant.usage
                state.usage = state.usage + assistant.usage
                state.messages.append(assistant)
                state.touch()
                state.validate(allow_pending_tool_calls=True)

                if assistant.stop_reason is StopReason.ERROR:
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.PROVIDER_ERROR,
                            final_text=assistant.text,
                            usage=turn_usage,
                            model_turns=model_calls,
                            tool_calls=tool_calls,
                            stop_reason=assistant.stop_reason,
                            error_message=assistant.error_message,
                        ),
                    )
                if assistant.stop_reason is StopReason.ABORTED:
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.INTERRUPTED,
                            final_text=assistant.text,
                            usage=turn_usage,
                            model_turns=model_calls,
                            tool_calls=tool_calls,
                            stop_reason=assistant.stop_reason,
                            error_message=assistant.error_message,
                        ),
                    )
                if not assistant.tool_calls:
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.COMPLETED,
                            final_text=assistant.text,
                            usage=turn_usage,
                            model_turns=model_calls,
                            tool_calls=tool_calls,
                            stop_reason=assistant.stop_reason,
                        ),
                    )

                budget_exhausted = False
                for call in assistant.tool_calls:
                    self._emit(ToolStarted(state.session_id, call))
                    if tool_calls < self._limits.max_tool_calls:
                        result = executor.execute(call, context)
                        tool_calls += 1
                    else:
                        budget_exhausted = True
                        result = ToolResult.error(
                            "tool call skipped: turn tool-call limit reached",
                            metadata={"limit_reached": True},
                        )
                    message = result.to_message(call)
                    state.messages.append(message)
                    state.touch()
                    self._emit(ToolFinished(state.session_id, message))

                if budget_exhausted:
                    if model_calls < self._limits.max_model_calls:
                        final = self._complete(state, (), model_calls + 1)
                        model_calls += 1
                        turn_usage = turn_usage + final.usage
                        state.usage = state.usage + final.usage
                        if final.tool_calls:
                            raise RuntimeError(
                                "provider returned tool calls when no tools were offered"
                            )
                        state.messages.append(final)
                        state.touch()
                        final_text = final.text
                        stop_reason = final.stop_reason
                    else:
                        final_text = assistant.text
                        stop_reason = assistant.stop_reason
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.LIMIT_REACHED,
                            final_text=final_text,
                            usage=turn_usage,
                            model_turns=model_calls,
                            tool_calls=tool_calls,
                            stop_reason=stop_reason,
                            error_message="turn tool-call limit reached",
                        ),
                    )

            return self._finish(
                state,
                RunResult(
                    status=RunStatus.LIMIT_REACHED,
                    final_text="",
                    usage=turn_usage,
                    model_turns=model_calls,
                    tool_calls=tool_calls,
                    error_message="turn model-call limit reached",
                ),
            )
        except ProviderError as exc:
            return self._finish(
                state,
                RunResult(
                    status=RunStatus.PROVIDER_ERROR,
                    final_text="",
                    usage=turn_usage,
                    model_turns=model_calls,
                    tool_calls=tool_calls,
                    error_message=str(exc),
                ),
            )
        except KeyboardInterrupt:
            return self._finish(
                state,
                RunResult(
                    status=RunStatus.INTERRUPTED,
                    final_text="",
                    usage=turn_usage,
                    model_turns=model_calls,
                    tool_calls=tool_calls,
                    error_message="interrupted by user",
                ),
            )
        except Exception:
            state.status = SessionStatus.ERROR
            state.touch()
            raise

    def _complete(
        self,
        state: SessionState,
        tools: tuple[ToolSpec, ...],
        model_call: int,
    ) -> AssistantMessage:
        self._emit(ModelRequested(state.session_id, model_call))
        message = self._provider.complete(
            CompletionRequest(
                system_prompt=state.system_prompt,
                messages=tuple(state.messages),
                tools=tools,
            )
        )
        self._emit(ModelResponded(state.session_id, model_call, message))
        return message

    def _finish(self, state: SessionState, result: RunResult) -> RunResult:
        if result.status in {RunStatus.COMPLETED, RunStatus.LIMIT_REACHED}:
            state.status = SessionStatus.IDLE
        elif result.status is RunStatus.INTERRUPTED:
            state.status = SessionStatus.INTERRUPTED
        else:
            state.status = SessionStatus.ERROR
        state.touch()
        state.validate()
        self._emit(TurnFinished(state.session_id, result))
        return result

    def _emit(self, event: RuntimeEvent) -> None:
        if self._event_sink is not None:
            self._event_sink(event)
