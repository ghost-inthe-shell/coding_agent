"""Synchronous agent loop; SessionState is its only conversation state."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from coding_agent.context import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
    CompactionPlan,
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
    CompletionEvent,
    CompletionEventSink,
    CompletionRequest,
    CompletionTextDelta,
    CompletionThinkingDelta,
    LLMProvider,
    ProviderError,
    ReasoningLevel,
)
from coding_agent.tools import ArtifactStore, Tool, ToolContext, ToolExecutor, ToolResultProcessor
from coding_agent.tools.base import ToolSpec

from .compaction import CompactionCheckpoint
from .events import (
    EventSink,
    ModelRequested,
    ModelResponded,
    ModelTextDelta,
    ModelThinkingDelta,
    RuntimeEvent,
    ToolFinished,
    ToolStarted,
    TurnFinished,
    TurnStarted,
)
from .messages import AssistantMessage, UserMessage
from .results import CompactionResult, RunResult, ToolResult
from .session import SessionState
from .session_store import SessionStore
from .types import RunStatus, SessionStatus, StopReason
from .usage import Usage

DEFAULT_MAX_MODEL_CALLS = 32

_FINAL_MODEL_CALL_INSTRUCTION = """
# Runtime limit

This is the final model call available for the current user turn. No tools are available. Give a
concise final response that states what was completed, what remains incomplete, and any next step
the user needs to take. Do not claim that unverified work succeeded.
""".strip()


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    max_tool_calls: int = 32

    def __post_init__(self) -> None:
        if self.max_model_calls <= 0 or self.max_tool_calls <= 0:
            raise ValueError("runtime limits must be positive")


@dataclass(slots=True)
class _TurnProgress:
    usage: Usage = field(default_factory=Usage)
    model_calls: int = 0
    provider_calls: int = 0
    tool_calls: int = 0

    def record_agent_model(self, message: AssistantMessage) -> None:
        self.model_calls += 1
        self.provider_calls += 1
        self.usage = self.usage + message.usage

    def record_internal_model(self, message: AssistantMessage) -> None:
        self.provider_calls += 1
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

    def compact(self, state: SessionState) -> CompactionResult:
        """Force one rolling compaction at a stable session boundary."""

        if state.status is SessionStatus.RUNNING:
            raise ValueError("cannot compact a session while a turn is running")
        state.validate()
        tools = tuple(tool.spec for tool in self._tools)
        try:
            plan = self._make_compaction_plan(state, tools, force=True)
        except ContextCompactionError as exc:
            return CompactionResult(compacted=False, error_message=str(exc))
        if plan is None:
            return CompactionResult(compacted=False)

        try:
            response = self._request_compaction(state, plan)
        except ProviderError as exc:
            return CompactionResult(compacted=False, error_message=str(exc))
        except KeyboardInterrupt:
            return CompactionResult(
                compacted=False,
                error_message="compaction interrupted by user",
            )

        state.usage = state.usage + response.usage
        state.touch()
        try:
            checkpoint, tokens_after = self._evaluate_compaction(
                state,
                tools,
                plan,
                response,
            )
        except ContextCompactionError as exc:
            return CompactionResult(
                compacted=False,
                usage=response.usage,
                error_message=str(exc),
            )

        if tokens_after >= plan.tokens_before:
            return CompactionResult(
                compacted=False,
                usage=response.usage,
                tokens_before=plan.tokens_before,
                tokens_after=tokens_after,
            )

        state.compaction = checkpoint
        state.validate()
        return CompactionResult(
            compacted=True,
            usage=response.usage,
            summarized_messages=plan.summarized_message_count,
            tokens_before=plan.tokens_before,
            tokens_after=tokens_after,
        )

    def run_turn(self, state: SessionState, user_input: str) -> RunResult:
        state.validate()
        state.status = SessionStatus.RUNNING
        state.messages.append(UserMessage.from_text(user_input))
        state.touch()
        self._emit(TurnStarted(state.session_id))

        session_directory = SessionStore(state_home=self._state_home).path_for_state(state).parent
        artifact_store = ArtifactStore(
            state.session_id,
            session_directory=session_directory,
        )
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
                final_model_call = progress.model_calls + 1 == self._limits.max_model_calls
                available_tools = () if final_model_call else executor.specs
                self._auto_compact(state, available_tools, progress)
                assistant = self._complete(
                    state,
                    available_tools,
                    progress.provider_calls + 1,
                    final_model_call=final_model_call,
                )
                progress.record_agent_model(assistant)
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
                if final_model_call:
                    if assistant.tool_calls:
                        raise RuntimeError(
                            "provider returned tool calls when no tools were offered"
                        )
                    return self._finish(
                        state,
                        RunResult(
                            status=RunStatus.LIMIT_REACHED,
                            final_text=assistant.text,
                            usage=progress.usage,
                            model_turns=progress.model_calls,
                            tool_calls=progress.tool_calls,
                            stop_reason=assistant.stop_reason,
                            error_message="turn model-call limit reached",
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
                        final = self._complete(
                            state,
                            (),
                            progress.provider_calls + 1,
                            final_model_call=True,
                        )
                        progress.record_agent_model(final)
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
        *,
        final_model_call: bool = False,
    ) -> AssistantMessage:
        self._emit(ModelRequested(state.session_id, model_call))
        system_prompt = state.system_prompt
        if final_model_call:
            system_prompt = f"{system_prompt}\n\n{_FINAL_MODEL_CALL_INSTRUCTION}"
        message = self._provider.complete(
            CompletionRequest(
                system_prompt=system_prompt,
                messages=active_messages(state),
                tools=tools,
            ),
            event_sink=self._completion_event_sink(state.session_id, model_call),
        )
        self._emit(ModelResponded(state.session_id, model_call, message))
        return message

    def _completion_event_sink(
        self,
        session_id: str,
        model_call: int,
    ) -> CompletionEventSink | None:
        if self._event_sink is None:
            return None

        def forward(event: CompletionEvent) -> None:
            if isinstance(event, CompletionTextDelta):
                self._emit(ModelTextDelta(session_id, model_call, event.text))
            elif isinstance(event, CompletionThinkingDelta):
                self._emit(
                    ModelThinkingDelta(session_id, model_call, event.thinking)
                )
            else:  # pragma: no cover - closed by the CompletionEvent union
                raise TypeError(f"unsupported completion event: {event!r}")

        return forward

    def _auto_compact(
        self,
        state: SessionState,
        tools: tuple[ToolSpec, ...],
        progress: _TurnProgress,
    ) -> None:
        while True:
            plan = self._make_compaction_plan(state, tools)
            if plan is None:
                self._ensure_context_fits(state, tools)
                return

            model_call = progress.provider_calls + 1
            response = self._request_compaction(state, plan, model_call=model_call)
            progress.record_internal_model(response)
            state.usage = state.usage + response.usage
            state.touch()
            checkpoint, tokens_after = self._evaluate_compaction(
                state,
                tools,
                plan,
                response,
            )
            if tokens_after >= plan.tokens_before:
                raise ContextCompactionError(
                    "compaction did not reduce estimated context: "
                    f"before={plan.tokens_before}, after={tokens_after}"
                )
            state.compaction = checkpoint
            state.validate()

    def _make_compaction_plan(
        self,
        state: SessionState,
        tools: tuple[ToolSpec, ...],
        *,
        force: bool = False,
    ) -> CompactionPlan | None:
        plan = prepare_compaction(
            state,
            self._context_budget,
            tools,
            force=force,
            summary_input_token_limit=self._summary_input_token_limit,
        )
        if plan is None:
            return None
        summary_tokens = estimate_request_tokens(
            self._compaction_prompt,
            plan.messages,
            (),
        )
        if summary_tokens >= self._summary_budget.output_reserve_threshold:
            raise ContextCompactionError("history segment is too large to summarize safely")
        return plan

    def _request_compaction(
        self,
        state: SessionState,
        plan: CompactionPlan,
        *,
        model_call: int | None = None,
    ) -> AssistantMessage:
        if model_call is not None:
            self._emit(ModelRequested(state.session_id, model_call))
        response = self._provider.complete(
            CompletionRequest(
                system_prompt=self._compaction_prompt,
                messages=plan.messages,
                tools=(),
                max_output_tokens=DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
                reasoning=ReasoningLevel.MINIMAL,
            )
        )
        if model_call is not None:
            self._emit(ModelResponded(state.session_id, model_call, response))
        return response

    def _evaluate_compaction(
        self,
        state: SessionState,
        tools: tuple[ToolSpec, ...],
        plan: CompactionPlan,
        response: AssistantMessage,
    ) -> tuple[CompactionCheckpoint, int]:
        checkpoint = checkpoint_from_summary(plan, response)
        candidate = replace(state, compaction=checkpoint)
        tokens_after = estimate_request_tokens(
            candidate.system_prompt,
            active_messages(candidate),
            tools,
        )
        return checkpoint, tokens_after

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
