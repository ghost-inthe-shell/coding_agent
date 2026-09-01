"""Minimal synchronous command-line REPL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TextIO
from uuid import uuid4

try:
    import readline as _readline
except ImportError:  # pragma: no cover - readline is expected on Linux
    _readline = None

from coding_agent.context import DEFAULT_CONTEXT_WINDOW, ContextBudget
from coding_agent.core.results import CompactionResult, RunResult
from coding_agent.core.runtime import DEFAULT_MAX_MODEL_CALLS, Runtime, RuntimeLimits
from coding_agent.core.session import SessionState
from coding_agent.core.session_store import SessionStore, SessionStoreError
from coding_agent.core.usage import Usage
from coding_agent.permissions import (
    PermissionDecision,
    PermissionOperation,
    PermissionRequest,
)
from coding_agent.providers import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicProvider,
    ApiDialect,
    LLMProvider,
    OpenAICompatibleProvider,
    ReasoningLevel,
)
from coding_agent.tools import (
    EditFileTool,
    GlobFilesTool,
    GrepSearchTool,
    ReadFileTool,
    RunShellTool,
    WriteFileTool,
)
from coding_agent.ui import ColorMode, TerminalRenderer

HELP_TEXT = """Commands:
  /compact  Summarize older conversation history now.
  /help     Show this help.
  /exit     End the session.

Input:
  End a line with \\ and press Enter to insert a newline and continue at the ... prompt.
"""


class _ReplRuntime(Protocol):
    def run_turn(self, state: SessionState, user_input: str) -> RunResult: ...

    def compact(self, state: SessionState) -> CompactionResult: ...


class _SessionSaver(Protocol):
    def save(self, state: SessionState) -> Path: ...


class InteractivePermissionHandler:
    """Ask once on the REPL streams; approvals are never remembered."""

    def __init__(
        self,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
        renderer: TerminalRenderer | None = None,
    ) -> None:
        self._input_stream = input_stream
        self._output_stream = output_stream
        self._renderer = renderer or TerminalRenderer(output_stream)

    def __call__(self, request: PermissionRequest) -> PermissionDecision:
        if request.operation is PermissionOperation.READ:
            question = "Allow read outside the workspace?"
            label = "Target"
        elif request.operation is PermissionOperation.WRITE:
            question = "Allow write in the workspace?"
            label = "Target"
        elif request.operation is PermissionOperation.EXECUTE:
            question = "Run this shell command in the workspace?"
            label = "Command"
        else:
            raise ValueError(f"unsupported permission operation: {request.operation!r}")
        displayed_target = json.dumps(request.target, ensure_ascii=False)
        self._renderer.permission_request(question, label, displayed_target)
        try:
            answer = _read_line(
                "Approve once? [y/N] ",
                self._input_stream,
                self._output_stream,
            )
        except KeyboardInterrupt:
            self._renderer.write("\n")
            self._renderer.permission_decision(approved=False)
            return PermissionDecision.DENY
        if answer is None:
            self._renderer.write("\n")
            self._renderer.permission_decision(approved=False)
            return PermissionDecision.DENY
        if answer.strip().lower() in {"y", "yes"}:
            self._renderer.permission_decision(approved=True)
            return PermissionDecision.ALLOW
        self._renderer.permission_decision(approved=False)
        return PermissionDecision.DENY


def run_repl(
    runtime: _ReplRuntime,
    state: SessionState,
    *,
    session_store: _SessionSaver | None = None,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    renderer: TerminalRenderer | None = None,
) -> int:
    """Run a synchronous multi-turn REPL over one SessionState."""

    renderer = renderer or TerminalRenderer(output_stream)
    _configure_terminal_input(input_stream, output_stream)
    renderer.banner(state.session_id, state.workspace_root)
    while True:
        try:
            prompt = _read_prompt(input_stream, output_stream, renderer)
        except KeyboardInterrupt:
            renderer.write("\n")
            return 130

        if prompt is None:
            renderer.write("\n")
            return 0

        command = prompt.strip()
        if not command:
            continue
        if command == "/exit":
            return 0
        if command == "/help":
            renderer.help(HELP_TEXT)
            continue
        if command == "/compact":
            compaction = runtime.compact(state)
            renderer.compaction_result(compaction)
            if (
                session_store is not None
                and (compaction.compacted or compaction.usage != Usage())
                and not _save_checkpoint(session_store, state, renderer)
            ):
                return 1
            continue
        if command.startswith("/"):
            renderer.unknown_command(command)
            continue

        result = runtime.run_turn(state, prompt)
        renderer.run_result(result)
        if session_store is not None and not _save_checkpoint(session_store, state, renderer):
            return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synchronous coding-agent REPL.")
    parser.add_argument(
        "--provider",
        choices=("openai-compatible", "anthropic"),
        default="openai-compatible",
        help="model API provider (default: openai-compatible)",
    )
    parser.add_argument("--model", required=True, help="provider model identifier")
    session_source = parser.add_mutually_exclusive_group()
    session_source.add_argument(
        "--workspace",
        help="workspace directory for a new session (default: current directory)",
    )
    session_source.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume the explicitly identified session and its saved workspace",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL; API keys come from provider environment variables",
    )
    parser.add_argument(
        "--api-dialect",
        choices=tuple(dialect.value for dialect in ApiDialect),
        default=ApiDialect.GENERIC.value,
        help="OpenAI-compatible vendor extensions (default: generic)",
    )
    parser.add_argument(
        "--reasoning",
        choices=tuple(
            level.value for level in ReasoningLevel if level is not ReasoningLevel.MINIMAL
        ),
        default=ReasoningLevel.DEFAULT.value,
        help="main-request reasoning level (default: provider default)",
    )
    parser.add_argument(
        "--max-turns",
        type=_positive_int,
        default=DEFAULT_MAX_MODEL_CALLS,
        help=(f"maximum agent model calls per user turn (default: {DEFAULT_MAX_MODEL_CALLS})"),
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(f"maximum output tokens per model call (default: {DEFAULT_MAX_OUTPUT_TOKENS})"),
    )
    parser.add_argument(
        "--context-window",
        type=_positive_int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"model context window in tokens (default: {DEFAULT_CONTEXT_WINDOW})",
    )
    parser.add_argument(
        "--color",
        choices=tuple(mode.value for mode in ColorMode),
        default=ColorMode.AUTO.value,
        help="terminal color mode (default: auto)",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stream model text and thinking as they arrive (default: enabled)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.provider == "anthropic" and arguments.base_url is not None:
        parser.error("--base-url is only valid with the openai-compatible provider")
    if arguments.provider == "anthropic" and arguments.api_dialect != "generic":
        parser.error("--api-dialect is only valid with the openai-compatible provider")
    if arguments.provider == "anthropic" and arguments.reasoning != "default":
        parser.error("--reasoning is not implemented for the anthropic provider")
    try:
        ContextBudget(
            context_window=arguments.context_window,
            max_output_tokens=arguments.max_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))

    session_store = SessionStore()
    if arguments.resume is not None:
        try:
            state = session_store.load(arguments.resume)
        except SessionStoreError as exc:
            _write(sys.stderr, f"coding-agent: session error: {exc}\n")
            return 1
        workspace = Path(state.workspace_root)
        if not workspace.is_dir():
            parser.error(f"saved workspace is not a directory: {state.workspace_root}")
    else:
        workspace_argument = arguments.workspace if arguments.workspace is not None else "."
        workspace = Path(workspace_argument).expanduser().resolve()
        if not workspace.is_dir():
            parser.error(f"workspace is not a directory: {workspace_argument}")
        state = SessionState.create(uuid4().hex, workspace)

    try:
        provider = _create_provider(
            arguments.provider,
            arguments.model,
            base_url=arguments.base_url,
            max_tokens=arguments.max_tokens,
            api_dialect=ApiDialect(arguments.api_dialect),
            reasoning=ReasoningLevel(arguments.reasoning),
            stream=arguments.stream,
        )
    except (ImportError, ValueError) as exc:
        parser.error(str(exc))

    renderer = TerminalRenderer(sys.stdout, color=ColorMode(arguments.color))
    runtime = Runtime(
        provider,
        (
            ReadFileTool(),
            GlobFilesTool(),
            GrepSearchTool(),
            WriteFileTool(),
            EditFileTool(),
            RunShellTool(),
        ),
        permission_handler=InteractivePermissionHandler(renderer=renderer),
        event_sink=renderer,
        context_window=arguments.context_window,
        limits=RuntimeLimits(max_model_calls=arguments.max_turns),
    )
    if arguments.resume is None:
        try:
            session_store.save(state)
        except SessionStoreError as exc:
            _write(sys.stderr, f"coding-agent: session error: {exc}\n")
            return 1
    return run_repl(runtime, state, session_store=session_store, renderer=renderer)


def _create_provider(
    provider_name: str,
    model: str,
    *,
    base_url: str | None,
    max_tokens: int,
    api_dialect: ApiDialect,
    reasoning: ReasoningLevel,
    stream: bool,
) -> LLMProvider:
    if provider_name == "anthropic":
        if api_dialect is not ApiDialect.GENERIC:
            raise ValueError("api_dialect is only supported by openai-compatible")
        if reasoning is not ReasoningLevel.DEFAULT:
            raise ValueError("reasoning is not implemented for anthropic")
        return AnthropicProvider(model, max_tokens=max_tokens, stream=stream)
    return OpenAICompatibleProvider(
        model,
        base_url=base_url,
        max_tokens=max_tokens,
        dialect=api_dialect,
        reasoning=reasoning,
        stream=stream,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _save_checkpoint(
    session_store: _SessionSaver,
    state: SessionState,
    renderer: TerminalRenderer,
) -> bool:
    try:
        session_store.save(state)
    except SessionStoreError as exc:
        renderer.session_error(str(exc))
        return False
    return True


def _write(stream: TextIO, text: str) -> None:
    stream.write(text)
    stream.flush()


def _configure_terminal_input(input_stream: TextIO, output_stream: TextIO) -> None:
    if _uses_terminal_editor(input_stream, output_stream):
        assert _readline is not None
        _readline.parse_and_bind("set enable-bracketed-paste on")


def _read_prompt(
    input_stream: TextIO,
    output_stream: TextIO,
    renderer: TerminalRenderer,
) -> str | None:
    """Read one prompt; an odd trailing backslash inserts a newline."""

    lines: list[str] = []
    uses_terminal_editor = _uses_terminal_editor(input_stream, output_stream)
    prompt = renderer.input_prompt(
        continuation=False,
        readline=uses_terminal_editor,
    )
    while True:
        line = _read_line(prompt, input_stream, output_stream)
        if line is None:
            return None
        line = line.replace("\r\n", "\n").replace("\r", "\n")
        trailing_backslashes = len(line) - len(line.rstrip("\\"))
        if trailing_backslashes % 2 == 1:
            lines.append(line[:-1])
            prompt = renderer.input_prompt(
                continuation=True,
                readline=uses_terminal_editor,
            )
            continue
        lines.append(line)
        return "\n".join(lines)


def _read_line(
    prompt: str,
    input_stream: TextIO,
    output_stream: TextIO,
) -> str | None:
    """Read one line, using GNU readline only for the real interactive terminal."""

    if _uses_terminal_editor(input_stream, output_stream):
        try:
            return input(prompt)
        except EOFError:
            return None

    _write(output_stream, prompt)
    line = input_stream.readline()
    if line == "":
        return None
    return line.removesuffix("\n").removesuffix("\r")


def _uses_terminal_editor(input_stream: TextIO, output_stream: TextIO) -> bool:
    return (
        _readline is not None
        and input_stream is sys.stdin
        and output_stream is sys.stdout
        and input_stream.isatty()
    )


if __name__ == "__main__":
    raise SystemExit(main())
