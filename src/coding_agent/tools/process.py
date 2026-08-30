"""Linux-first subprocess capture with a global raw-output ceiling."""

from __future__ import annotations

import math
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .artifacts import DEFAULT_MAX_ARTIFACT_BYTES


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    stdout: str
    stderr: str
    returncode: int
    incomplete: bool
    timed_out: bool
    duration_ms: int


def run_limited_process(
    command: list[str],
    *,
    cwd: Path,
    max_output_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    timeout_seconds: float | None = None,
    env: Mapping[str, str] | None = None,
) -> ProcessOutput:
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    if (
        timeout_seconds is not None
        and (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        )
    ):
        raise ValueError("timeout_seconds must be a positive finite number")

    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stderr is not None

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    incomplete = False
    timed_out = False
    deadline = started_at + timeout_seconds if timeout_seconds is not None else None

    try:
        while selector.get_map():
            select_timeout = None
            if deadline is not None:
                select_timeout = max(0.0, deadline - time.monotonic())
                if select_timeout == 0:
                    timed_out = True
                    break
            events = selector.select(select_timeout)
            if not events and deadline is not None:
                timed_out = True
                break

            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue

                remaining = max_output_bytes - total
                if remaining <= 0:
                    incomplete = True
                    break
                captured[key.data].extend(chunk[:remaining])
                total += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    incomplete = True
                    break
            if incomplete:
                break
    except BaseException:
        _terminate_process_group(process)
        process.stdout.close()
        process.stderr.close()
        raise
    finally:
        selector.close()

    if incomplete or timed_out:
        _terminate_process_group(process)
    else:
        process.wait()
    process.stdout.close()
    process.stderr.close()

    return ProcessOutput(
        stdout=bytes(captured["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(captured["stderr"]).decode("utf-8", errors="replace"),
        returncode=process.returncode,
        incomplete=incomplete,
        timed_out=timed_out,
        duration_ms=max(0, round((time.monotonic() - started_at) * 1000)),
    )


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 0.2,
) -> None:
    """Terminate the Linux process group, then reap the direct child."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    grace_deadline = time.monotonic() + grace_seconds
    if process.poll() is None:
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            pass
    remaining_grace = grace_deadline - time.monotonic()
    if remaining_grace > 0:
        time.sleep(remaining_grace)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()
