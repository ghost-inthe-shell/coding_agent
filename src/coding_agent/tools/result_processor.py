"""Apply one context limit and one raw-output limit to every tool result."""

from __future__ import annotations

from dataclasses import dataclass

from coding_agent.core.messages import ToolCall
from coding_agent.core.results import ToolResult

from .artifacts import ArtifactStore, DEFAULT_MAX_ARTIFACT_BYTES


DEFAULT_MAX_RESULT_CHARS = 50_000


@dataclass(slots=True)
class ToolResultProcessor:
    artifact_store: ArtifactStore
    max_result_chars: int = DEFAULT_MAX_RESULT_CHARS
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES

    def __post_init__(self) -> None:
        if self.max_result_chars <= 0:
            raise ValueError("max_result_chars must be positive")
        if self.max_artifact_bytes <= 0:
            raise ValueError("max_artifact_bytes must be positive")

    def process(self, call: ToolCall, result: ToolResult) -> ToolResult:
        if len(result.content) <= self.max_result_chars:
            return result

        metadata = dict(result.metadata)
        metadata.update(
            {
                "truncated": True,
                "original_chars": len(result.content),
                "original_bytes": len(result.content.encode("utf-8")),
                "preview_strategy": "head_tail",
            }
        )

        try:
            artifact = self.artifact_store.write(
                call.id,
                result.content,
                max_bytes=self.max_artifact_bytes,
            )
        except OSError as exc:
            metadata["artifact_error"] = str(exc)
            notice = "[Tool output truncated; saving the full output failed.]"
        else:
            metadata.update(
                {
                    "artifact_path": str(artifact.path),
                    "artifact_bytes": artifact.stored_bytes,
                    "artifact_incomplete": artifact.incomplete
                    or bool(metadata.get("artifact_incomplete", False)),
                }
            )
            completeness = " (incomplete: raw output limit reached)" if metadata[
                "artifact_incomplete"
            ] else ""
            notice = (
                "[Tool output truncated for model context.]\n"
                f"Full output: {artifact.path}{completeness}"
            )

        preview = _head_tail_preview(result.content, notice, self.max_result_chars)
        return ToolResult(content=preview, status=result.status, metadata=metadata)


def _head_tail_preview(content: str, notice: str, limit: int) -> str:
    separator = "\n\n--- omitted middle ---\n\n"
    notice_budget = max(0, limit // 2)
    if len(notice) > notice_budget:
        notice = notice[: max(0, notice_budget - 3)] + "..."
    fixed = len(notice) + len(separator) + 2
    available = max(0, limit - fixed)
    head_chars = available // 2
    tail_chars = available - head_chars
    head = content[:head_chars]
    tail = content[-tail_chars:] if tail_chars else ""
    return f"{notice}\n\n{head}{separator}{tail}"[:limit]
