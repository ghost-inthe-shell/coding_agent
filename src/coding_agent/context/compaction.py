"""Pure planning and validation for rolling conversation compaction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from coding_agent.core.compaction import CompactionCheckpoint
from coding_agent.core.messages import (
    AssistantMessage,
    Message,
    TextBlock,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.session import SessionState
from coding_agent.core.types import StopReason
from coding_agent.tools.base import ToolSpec

from .budget import ContextBudget, estimate_request_tokens

DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS = 2_048
_SUMMARY_PREFIX = "[Previous conversation summary]\n"
_SUMMARY_SUFFIX = "\n[End previous conversation summary]\n\n"
_SUMMARY_REQUEST = "Create the updated rolling checkpoint summary now."


class ContextCompactionError(RuntimeError):
    """Raised when a requested compaction cannot produce a safe checkpoint."""


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    messages: tuple[Message, ...]
    first_kept_message_index: int
    tokens_before: int
    summarized_message_count: int


def active_messages(state: SessionState) -> tuple[Message, ...]:
    """Project full persisted history into the messages visible to the provider."""

    if state.compaction is None:
        return tuple(state.messages)
    recent = tuple(state.messages[state.compaction.first_kept_message_index :])
    return _prepend_summary(state.compaction.summary, recent)


def prepare_compaction(
    state: SessionState,
    budget: ContextBudget,
    tools: Sequence[ToolSpec],
    *,
    force: bool = False,
    summary_input_token_limit: int | None = None,
) -> CompactionPlan | None:
    """Choose a safe forward-only cut and build its rolling summarization input."""

    state.validate()
    visible = active_messages(state)
    tokens_before = estimate_request_tokens(state.system_prompt, visible, tools)
    if not force and not budget.should_compact(tokens_before):
        return None

    start = state.compaction.first_kept_message_index if state.compaction else 0
    cut = find_compaction_cut(
        state.messages,
        start,
        budget.keep_recent_tokens,
    )
    if cut is None:
        return None

    if summary_input_token_limit is not None and summary_input_token_limit <= 0:
        raise ValueError("summary_input_token_limit must be positive")

    candidate_cuts = [
        group_start
        for group_start, _ in _message_groups(state.messages, start)[1:]
        if group_start <= cut
    ]
    summary_input: tuple[Message, ...] | None = None
    for candidate in reversed(candidate_cuts):
        candidate_input = _summary_input(state, start, candidate)
        if (
            summary_input_token_limit is None
            or estimate_request_tokens("", candidate_input, ()) < summary_input_token_limit
        ):
            cut = candidate
            summary_input = candidate_input
            break
    if summary_input is None:
        return None

    return CompactionPlan(
        messages=summary_input,
        first_kept_message_index=cut,
        tokens_before=tokens_before,
        summarized_message_count=cut - start,
    )


def find_compaction_cut(
    messages: Sequence[Message],
    start: int,
    keep_recent_tokens: int,
) -> int | None:
    """Return a group boundary that keeps a recent suffix without splitting tools."""

    if start < 0 or start > len(messages):
        raise ValueError("compaction start is outside the message sequence")
    if keep_recent_tokens <= 0:
        raise ValueError("keep_recent_tokens must be positive")

    groups = _message_groups(messages, start)
    if len(groups) < 2:
        return None

    kept_tokens = 0
    cut = groups[-1][0]
    for group_start, group_end in reversed(groups):
        group_tokens = estimate_request_tokens("", messages[group_start:group_end], ())
        if kept_tokens and kept_tokens + group_tokens > keep_recent_tokens:
            break
        kept_tokens += group_tokens
        cut = group_start
        if kept_tokens >= keep_recent_tokens:
            break

    if cut == start:
        cut = groups[1][0]
    return cut


def checkpoint_from_summary(
    plan: CompactionPlan,
    response: AssistantMessage,
) -> CompactionCheckpoint:
    """Accept only a complete, text-only summary response."""

    if response.tool_calls:
        raise ContextCompactionError("compaction response unexpectedly contained tool calls")
    if response.stop_reason is StopReason.LENGTH:
        raise ContextCompactionError("compaction summary reached its output-token limit")
    if response.stop_reason is not StopReason.STOP:
        detail = response.error_message or response.stop_reason.value
        raise ContextCompactionError(f"compaction response failed: {detail}")
    summary = response.text.strip()
    if not summary:
        raise ContextCompactionError("compaction response contained no summary text")
    return CompactionCheckpoint(
        summary=summary,
        first_kept_message_index=plan.first_kept_message_index,
        tokens_before=plan.tokens_before,
        created_at=response.timestamp,
    )


def _message_groups(
    messages: Sequence[Message],
    start: int,
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    index = start
    while index < len(messages):
        end = index + 1
        message = messages[index]
        if isinstance(message, AssistantMessage) and message.tool_calls:
            end += len(message.tool_calls)
            if end > len(messages) or not all(
                isinstance(result, ToolResultMessage) for result in messages[index + 1 : end]
            ):
                raise ValueError("assistant tool calls do not have a complete result group")
        elif isinstance(message, ToolResultMessage):
            raise ValueError("tool result appears outside its assistant message group")
        groups.append((index, end))
        index = end
    return groups


def _prepend_summary(summary: str, messages: tuple[Message, ...]) -> tuple[Message, ...]:
    summary_block = TextBlock(_SUMMARY_PREFIX + summary + _SUMMARY_SUFFIX)
    if messages and isinstance(messages[0], UserMessage):
        first = messages[0]
        merged = UserMessage(
            content=(summary_block, *first.content),
            timestamp=first.timestamp,
        )
        return (merged, *messages[1:])
    return (UserMessage(content=(summary_block,), timestamp=0), *messages)


def _append_summary_request(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    request_block = TextBlock("\n\n" + _SUMMARY_REQUEST)
    if messages and isinstance(messages[-1], UserMessage):
        last = messages[-1]
        merged = UserMessage(
            content=(*last.content, request_block),
            timestamp=last.timestamp,
        )
        return (*messages[:-1], merged)
    return (*messages, UserMessage(content=(request_block,), timestamp=0))


def _summary_input(
    state: SessionState,
    start: int,
    cut: int,
) -> tuple[Message, ...]:
    newly_evicted = tuple(state.messages[start:cut])
    if state.compaction is None:
        messages = newly_evicted
    else:
        messages = _prepend_summary(state.compaction.summary, newly_evicted)
    return _append_summary_request(messages)
