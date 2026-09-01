"""Small ANSI renderer for the synchronous terminal client."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum
from typing import TextIO

from coding_agent.core.results import CompactionResult, RunResult
from coding_agent.core.types import RunStatus


class ColorMode(str, Enum):
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_YELLOW = "\x1b[33m"
_CYAN = "\x1b[36m"


class TerminalRenderer:
    """Render stable CLI output with optional ANSI styling."""

    def __init__(
        self,
        output_stream: TextIO,
        *,
        color: ColorMode = ColorMode.AUTO,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.output_stream = output_stream
        current_environment = os.environ if environment is None else environment
        self.color_enabled = _color_enabled(color, output_stream, current_environment)

    def write(self, text: str) -> None:
        self.output_stream.write(text)
        self.output_stream.flush()

    def banner(self, session_id: str, workspace_root: str) -> None:
        title = self._style("Coding Agent", _BOLD, _CYAN)
        self.write(
            f"{title}\n"
            f"Session: {session_id}\n"
            f"Workspace: {workspace_root}\n"
            "Type /help for commands.\n"
        )

    def input_prompt(self, *, continuation: bool, readline: bool) -> str:
        prompt = "... " if continuation else "> "
        return self._style(prompt, _BOLD, _CYAN, readline=readline)

    def assistant(self, text: str) -> None:
        if text:
            self.write(f"{self._style('assistant>', _BOLD, _GREEN)} {text}\n")

    def run_result(self, result: RunResult) -> None:
        self.assistant(result.final_text)
        if result.status is RunStatus.COMPLETED:
            return

        detail = result.error_message or result.status.value
        diagnostics = [
            f"model_turns={result.model_turns}",
            f"tool_calls={result.tool_calls}",
            f"output_tokens={result.usage.output_tokens}",
            f"reasoning_tokens={result.usage.reasoning_tokens}",
        ]
        if result.max_output_tokens is not None:
            diagnostics.append(f"max_output_tokens_per_call={result.max_output_tokens}")
        status = f"[{result.status.value}]"
        self.write(f"{self._style(status, _BOLD, _RED)} {detail} ({', '.join(diagnostics)})\n")

    def compaction_result(self, result: CompactionResult) -> None:
        if result.compacted:
            label = self._style("[compacted]", _CYAN)
            self.write(
                f"{label} "
                f"summarized_messages={result.summarized_messages}, "
                f"tokens_before={result.tokens_before}, "
                f"tokens_after={result.tokens_after}\n"
            )
        elif result.error_message is not None:
            label = self._style("[compact_error]", _BOLD, _RED)
            self.write(f"{label} {result.error_message}\n")
        elif result.tokens_before is not None and result.tokens_after is not None:
            label = self._style("[compact]", _CYAN)
            self.write(
                f"{label} not applied: "
                f"tokens_before={result.tokens_before}, "
                f"tokens_after={result.tokens_after}\n"
            )
        else:
            label = self._style("[compact]", _CYAN)
            self.write(f"{label} nothing to compact\n")

    def help(self, text: str) -> None:
        self.write(text)

    def unknown_command(self, command: str) -> None:
        label = self._style("Unknown command:", _RED)
        self.write(f"{label} {command}\n")

    def permission_request(self, question: str, label: str, target: str) -> None:
        self.write(
            f"{self._style(question, _YELLOW)}\n  {self._style(label + ':', _DIM)} {target}\n"
        )

    def permission_decision(self, *, approved: bool) -> None:
        if approved:
            self.write(f"{self._style('Approved.', _GREEN)}\n")
        else:
            self.write(f"{self._style('Denied.', _RED)}\n")

    def session_error(self, message: str) -> None:
        label = self._style("[session_error]", _BOLD, _RED)
        self.write(f"{label} checkpoint failed; REPL terminated: {message}\n")

    def _style(self, text: str, *codes: str, readline: bool = False) -> str:
        if not self.color_enabled:
            return text
        prefix = "".join(codes)
        if readline:
            return f"\001{prefix}\002{text}\001{_RESET}\002"
        return f"{prefix}{text}{_RESET}"


def _color_enabled(
    mode: ColorMode,
    output_stream: TextIO,
    environment: Mapping[str, str],
) -> bool:
    if mode is ColorMode.ALWAYS:
        return True
    if mode is ColorMode.NEVER or "NO_COLOR" in environment:
        return False
    return bool(getattr(output_stream, "isatty", lambda: False)())
