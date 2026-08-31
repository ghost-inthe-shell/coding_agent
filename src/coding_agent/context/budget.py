"""Conservative token estimates and context-window thresholds."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from coding_agent.core.messages import Message, message_to_dict
from coding_agent.tools.base import ToolSpec

DEFAULT_CONTEXT_WINDOW = 128_000


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_window: int = DEFAULT_CONTEXT_WINDOW
    max_output_tokens: int = 16_384

    def __post_init__(self) -> None:
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.max_output_tokens + self.safety_margin >= self.context_window:
            raise ValueError(
                "max_output_tokens plus context safety margin must be smaller than context_window"
            )

    @property
    def safety_margin(self) -> int:
        return max(1_024, ceil(self.context_window * 0.02))

    @property
    def compact_threshold(self) -> int:
        ratio_threshold = int(self.context_window * 0.8)
        return min(ratio_threshold, self.output_reserve_threshold)

    @property
    def output_reserve_threshold(self) -> int:
        return self.context_window - self.max_output_tokens - self.safety_margin

    @property
    def keep_recent_tokens(self) -> int:
        return min(20_000, max(1, int(self.context_window * 0.25)))

    def should_compact(self, estimated_tokens: int) -> bool:
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be non-negative")
        return estimated_tokens >= self.compact_threshold


def estimate_request_tokens(
    system_prompt: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
) -> int:
    """Estimate all provider-visible request content without provider tokenizers."""

    message_text = json.dumps(
        [message_to_dict(message) for message in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    tool_text = json.dumps(
        [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in tools
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sum(estimate_text_tokens(text) for text in (system_prompt, message_text, tool_text))


def estimate_text_tokens(text: str) -> int:
    """Estimate ASCII near four chars/token and non-ASCII near three bytes/token."""

    ascii_characters = 0
    non_ascii_bytes = 0
    for character in text:
        if character.isascii():
            ascii_characters += 1
        else:
            non_ascii_bytes += len(character.encode("utf-8"))
    return ceil(ascii_characters / 4) + ceil(non_ascii_bytes / 3)
