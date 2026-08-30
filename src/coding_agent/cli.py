"""Minimal synchronous command-line REPL."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Protocol, TextIO
from uuid import uuid4

from coding_agent.core.results import RunResult
from coding_agent.core.runtime import Runtime
from coding_agent.core.session import SessionState
from coding_agent.core.types import RunStatus
from coding_agent.permissions import PermissionDecision, PermissionRequest
from coding_agent.providers import AnthropicProvider, LLMProvider, OpenAICompatibleProvider
from coding_agent.tools import GlobFilesTool, GrepSearchTool, ReadFileTool


HELP_TEXT = """Commands:
  /help  Show this help.
  /exit  End the session.
"""


class _TurnRunner(Protocol):
    def run_turn(self, state: SessionState, user_input: str) -> RunResult:
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
        _write(
            self._output_stream,
            f"Allow {request.operation.value} outside the workspace?\n"
            f"  {request.target}\n"
            "Approve once? [y/N] ",
        )
        try:
            answer = self._input_stream.readline()
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
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> int:
    """Run a single-line, multi-turn REPL over one SessionState."""

    _write(output_stream, f"Coding Agent ({state.workspace_root})\nType /help for commands.\n")
    while True:
        try:
            _write(output_stream, "> ")
            line = input_stream.readline()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synchronous coding-agent REPL.")
    parser.add_argument(
        "--provider",
        choices=("openai-compatible", "anthropic"),
        default="openai-compatible",
        help="model API provider (default: openai-compatible)",
    )
    parser.add_argument("--model", required=True, help="provider model identifier")
    parser.add_argument(
        "--workspace",
        default=".",
        help="workspace directory (default: current directory)",
    )
    parser.add_argument(
        "--base-url",
        help="OpenAI-compatible API base URL; API keys come from provider environment variables",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=4096,
        help="maximum output tokens per model call (default: 4096)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    workspace = Path(arguments.workspace).expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"workspace is not a directory: {arguments.workspace}")
    if arguments.provider == "anthropic" and arguments.base_url is not None:
        parser.error("--base-url is only valid with the openai-compatible provider")

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
        (ReadFileTool(), GlobFilesTool(), GrepSearchTool()),
        permission_handler=InteractivePermissionHandler(),
    )
    state = SessionState.create(uuid4().hex, workspace)
    return run_repl(runtime, state)


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
        _write(output_stream, f"[{result.status.value}] {detail}\n")


def _write(stream: TextIO, text: str) -> None:
    stream.write(text)
    stream.flush()


if __name__ == "__main__":
    raise SystemExit(main())
