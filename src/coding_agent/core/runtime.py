"""Synchronous agent loop; SessionState is its only conversation state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from coding_agent.context import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
    ContextBudget,
    ContextCompactionError,
    active_messages,
    checkpoint_from_summary,
    estimate_request_tokens,
    estimate_text_tokens,
    prepare_compaction,
)
from coding_agent.permissions import PermissionHandler
from coding_agent.prompts import load_compaction_prompt
from coding_agent.providers import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    CompletionRequest,
    LLMProvider,
    ProviderError,
)
from coding_agent.tools import ArtifactStore, Tool, ToolContext, ToolExecutor, ToolResultProcessor
from coding_agent.tools.base import ToolSpec

from .events import (
    EventSink,
    ModelRequested,
    ModelResponded,
    RuntimeEvent,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
)
from .messages import AssistantMessage, UserMessage
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


@dataclass(slots=True)
class _TurnProgress:
    usage: Usage = field(default_factory=Usage)
    model_calls: int = 0
    tool_calls: int = 0

    def record_model(self, message: AssistantMessage) -> None:
        self.model_calls += 1
        self.usage = self.usage + message.usage


class Runtime:
    def __init__(
        self,
        provider: LLMProvider,
        tools: Iterable[Tool],
        *,
        limits: RuntimeLimits | None = None,
        event_sink: EventSink | None = None,
        state_home: Path | None = None,
        permission_handler: PermissionHandler | None = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        self._provider = provider
        self._tools = tuple(tools)
        self._limits = limits or RuntimeLimits()
        self._event_sink = event_sink
        self._state_home = state_home
        self._permission_handler = permission_handler
        provider_output_tokens = provider.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS
        self._context_budget = ContextBudget(
            context_window=context_window,
            max_output_tokens=provider_output_tokens,
        )
        self._summary_budget = ContextBudget(
            context_window=context_window,
            max_output_tokens=min(
                DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
                provider_output_tokens,
            ),
        )
        self._compaction_prompt = load_compaction_prompt()
        self._summary_input_token_limit = (
            self._summary_budget.output_reserve_threshold
            - estimate_text_tokens(self._compaction_prompt)
        )

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
            permission_handler=self._permission_handler,
            read_file_versions=state.read_file_versions,
        )
        progress = _TurnProgress()

        try:
            while progress.model_calls < self._limits.max_model_calls:
                self._auto_compact(state, executor.specs, progress)
                if progress.model_calls >= self._limits.max_model_calls:
                    break
                assistant = self._complete(
                    state,
                    executor.specs,
                    progress.model_calls + 1,
                )
                progress.record_model(assistant)
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
                            usage=progress.usage,
                            model_turns=progress.model_calls,
                            tool_calls=progress.tool_calls,
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
                            usage=progress.usage,
                            model_turns=progress.model_calls,
                            tool_calls=progress.tool_calls,
                            stop_reason=assistant.stop_reason,
                            error_message=assistant.error_message,
                        ),
                    )
                if assistant.stop_reason is StopReason.LENGTH:
                    for call in assistant.tool_calls:
                        self._emit(ToolStarted(state.session_id, call))
                        result = ToolResult.error(
                            "tool call was not executed because the model response reached "
                            "its output-token limit; reissue it with complete arguments",
                            metadata={
                                "not_executed": True,
                                "reason": "truncated_model_response",
                            },
                        )
                        message = result.to_message(call)
                        state.messages.append(message)
                        state.touch()
                        self._emit(ToolFinished(state.session_id, message))

                    if assistant.tool_calls and progress.model_calls < self._limits.max_model_calls:
                        continue
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.LIMIT_REACHED,
                            final_text=assistant.text,
                            usage=progress.usage,
                            model_turns=progress.model_calls,
                            tool_calls=progress.tool_calls,
                            stop_reason=assistant.stop_reason,
                            error_message="model output-token limit reached",
                        ),
                    )
                if not assistant.tool_calls:
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.COMPLETED,
                            final_text=assistant.text,
                            usage=progress.usage,
                            model_turns=progress.model_calls,
                            tool_calls=progress.tool_calls,
                            stop_reason=assistant.stop_reason,
                        ),
                    )

                budget_exhausted = False
                for call in assistant.tool_calls:
                    self._emit(ToolStarted(state.session_id, call))
                    if progress.tool_calls < self._limits.max_tool_calls:
                        result = executor.execute(call, context)
                        progress.tool_calls += 1
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
                    final_text = assistant.text
                    stop_reason = assistant.stop_reason
                    if progress.model_calls < self._limits.max_model_calls:
                        self._auto_compact(state, (), progress)
                    if progress.model_calls < self._limits.max_model_calls:
                        final = self._complete(state, (), progress.model_calls + 1)
                        progress.record_model(final)
                        state.usage = state.usage + final.usage
                        if final.tool_calls:
                            raise RuntimeError(
                                "provider returned tool calls when no tools were offered"
                            )
                        state.messages.append(final)
                        state.touch()
                        final_text = final.text
                        stop_reason = final.stop_reason
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.LIMIT_REACHED,
                            final_text=final_text,
                            usage=progress.usage,
                            model_turns=progress.model_calls,
                            tool_calls=progress.tool_calls,
                            stop_reason=stop_reason,
                            error_message="turn tool-call limit reached",
                        ),
                    )

            return self._finish(
                state,
                RunResult(
                    status=RunStatus.LIMIT_REACHED,
                    final_text="",
                    usage=progress.usage,
                    model_turns=progress.model_calls,
                    tool_calls=progress.tool_calls,
                    error_message="turn model-call limit reached",
                ),
            )
        except (ContextCompactionError, ProviderError) as exc:
            return self._finish(
                state,
                RunResult(
                    status=RunStatus.PROVIDER_ERROR,
                    final_text="",
                    usage=progress.usage,
                    model_turns=progress.model_calls,
                    tool_calls=progress.tool_calls,
                    error_message=str(exc),
                ),
            )
        except KeyboardInterrupt:
            return self._finish(
                state,
                RunResult(
                    status=RunStatus.INTERRUPTED,
                    final_text="",
                    usage=progress.usage,
                    model_turns=progress.model_calls,
                    tool_calls=progress.tool_calls,
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
                messages=active_messages(state),
                tools=tools,
            )
        )
        self._emit(ModelResponded(state.session_id, model_call, message))
        return message

    def _auto_compact(
        self,
        state: SessionState,
        tools: tuple[ToolSpec, ...],
        progress: _TurnProgress,
    ) -> None:
        while progress.model_calls < self._limits.max_model_calls:
            plan = prepare_compaction(
                state,
                self._context_budget,
                tools,
                summary_input_token_limit=self._summary_input_token_limit,
            )
            if plan is None:
                self._ensure_context_fits(state, tools)
                return

            summary_tokens = estimate_request_tokens(
                self._compaction_prompt,
                plan.messages,
                (),
            )
            if summary_tokens >= self._summary_budget.output_reserve_threshold:
                raise ContextCompactionError("history segment is too large to summarize safely")

            model_call = progress.model_calls + 1
            self._emit(ModelRequested(state.session_id, model_call))
            response = self._provider.complete(
                CompletionRequest(
                    system_prompt=self._compaction_prompt,
                    messages=plan.messages,
                    tools=(),
                    max_output_tokens=DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
                )
            )
            self._emit(ModelResponded(state.session_id, model_call, response))
            progress.record_model(response)
            state.usage = state.usage + response.usage
            state.compaction = checkpoint_from_summary(plan, response)
            state.touch()
            state.validate()

    def _ensure_context_fits(
        self,
        state: SessionState,
        tools: tuple[ToolSpec, ...],
    ) -> None:
        estimated = estimate_request_tokens(
            state.system_prompt,
            active_messages(state),
            tools,
        )
        if estimated >= self._context_budget.output_reserve_threshold:
            raise ContextCompactionError(
                "active context is too large and has no safe message group to compact"
            )

    def _finish(self, state: SessionState, result: RunResult) -> RunResult:
        if result.max_output_tokens is None:
            result = replace(
                result,
                max_output_tokens=self._provider.max_output_tokens,
            )
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
