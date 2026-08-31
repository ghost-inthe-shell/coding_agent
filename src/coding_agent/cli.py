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
from coding_agent.core.results import RunResult
from coding_agent.core.runtime import Runtime
from coding_agent.core.session import SessionState
from coding_agent.core.session_store import SessionStore, SessionStoreError
from coding_agent.core.types import RunStatus
from coding_agent.permissions import (
    PermissionDecision,
    PermissionOperation,
    PermissionRequest,
)
from coding_agent.providers import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnthropicProvider,
    LLMProvider,
    OpenAICompatibleProvider,
)
from coding_agent.tools import (
    EditFileTool,
    GlobFilesTool,
    GrepSearchTool,
    ReadFileTool,
    RunShellTool,
    WriteFileTool,
)

HELP_TEXT = """Commands:
  /help  Show this help.
  /exit  End the session.
"""


class _TurnRunner(Protocol):
    def run_turn(self, state: SessionState, user_input: str) -> RunResult:
        ...


class _SessionSaver(Protocol):
    def save(self, state: SessionState) -> Path:
        ...


class InteractivePermissionHandler:
    """Ask once on the REPL streams; approvals are never remembered."""

    def __init__(
        self,
        *,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        self._input_stream = input_stream
        self._output_stream = output_stream

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
        _write(
            self._output_stream,
            f"{question}\n"
            f"  {label}: {displayed_target}\n",
        )
        try:
            answer = _read_line(
                "Approve once? [y/N] ",
                self._input_stream,
                self._output_stream,
            )
        except KeyboardInterrupt:
            _write(self._output_stream, "\nDenied.\n")
            return PermissionDecision.DENY
        if answer == "":
            _write(self._output_stream, "\nDenied.\n")
            return PermissionDecision.DENY
        if answer.strip().lower() in {"y", "yes"}:
            _write(self._output_stream, "Approved.\n")
            return PermissionDecision.ALLOW
        _write(self._output_stream, "Denied.\n")
        return PermissionDecision.DENY


def run_repl(
    runtime: _TurnRunner,
    state: SessionState,
    *,
    session_store: _SessionSaver | None = None,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    """Run a single-line, multi-turn REPL over one SessionState."""

    _write(
        output_stream,
        f"Coding Agent\n"
        f"Session: {state.session_id}\n"
        f"Workspace: {state.workspace_root}\n"
        "Type /help for commands.\n",
    )
    while True:
        try:
            line = _read_line("> ", input_stream, output_stream)
        except KeyboardInterrupt:
            _write(output_stream, "\n")
            return 130

        if line == "":
            _write(output_stream, "\n")
            return 0

        user_input = line.strip()
        if not user_input:
            continue
        if user_input == "/exit":
            return 0
        if user_input == "/help":
            _write(output_stream, HELP_TEXT)
            continue
        if user_input.startswith("/"):
            _write(output_stream, f"Unknown command: {user_input}\n")
            continue

        result = runtime.run_turn(state, user_input)
        _print_result(result, output_stream)
        if session_store is not None:
            try:
                session_store.save(state)
            except SessionStoreError as exc:
                _write(
                    output_stream,
                    f"[session_error] checkpoint failed; REPL terminated: {exc}\n",
                )
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
        "--max-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(
            "maximum output tokens per model call "
            f"(default: {DEFAULT_MAX_OUTPUT_TOKENS})"
        ),
    )
    parser.add_argument(
        "--context-window",
        type=_positive_int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=f"model context window in tokens (default: {DEFAULT_CONTEXT_WINDOW})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.provider == "anthropic" and arguments.base_url is not None:
        parser.error("--base-url is only valid with the openai-compatible provider")
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
        )
    except (ImportError, ValueError) as exc:
        parser.error(str(exc))

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
        permission_handler=InteractivePermissionHandler(),
        context_window=arguments.context_window,
    )
    if arguments.resume is None:
        try:
            session_store.save(state)
        except SessionStoreError as exc:
            _write(sys.stderr, f"coding-agent: session error: {exc}\n")
            return 1
    return run_repl(runtime, state, session_store=session_store)


def _create_provider(
    provider_name: str,
    model: str,
    *,
    base_url: str | None,
    max_tokens: int,
) -> LLMProvider:
    if provider_name == "anthropic":
        return AnthropicProvider(model, max_tokens=max_tokens)
    return OpenAICompatibleProvider(
        model,
        base_url=base_url,
        max_tokens=max_tokens,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _print_result(result: RunResult, output_stream: TextIO) -> None:
    if result.final_text:
        _write(output_stream, f"assistant> {result.final_text}\n")
    if result.status is not RunStatus.COMPLETED:
        detail = result.error_message or result.status.value
        diagnostics = [
            f"model_turns={result.model_turns}",
            f"tool_calls={result.tool_calls}",
            f"output_tokens={result.usage.output_tokens}",
            f"reasoning_tokens={result.usage.reasoning_tokens}",
        ]
        if result.max_output_tokens is not None:
            diagnostics.append(
                f"max_output_tokens_per_call={result.max_output_tokens}"
            )
        _write(
            output_stream,
            f"[{result.status.value}] {detail} ({', '.join(diagnostics)})\n",
        )


def _write(stream: TextIO, text: str) -> None:
    stream.write(text)
    stream.flush()


def _read_line(prompt: str, input_stream: TextIO, output_stream: TextIO) -> str:
    """Read one line, using GNU readline only for the real interactive terminal."""

    if (
        _readline is not None
        and input_stream is sys.stdin
        and output_stream is sys.stdout
        and input_stream.isatty()
    ):
        try:
            return input(prompt)
        except EOFError:
            return ""

    _write(output_stream, prompt)
    return input_stream.readline()


if __name__ == "__main__":
    raise SystemExit(main())
