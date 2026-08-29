"""Linux-first subprocess capture with a global raw-output ceiling."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import subprocess

from .artifacts import DEFAULT_MAX_ARTIFACT_BYTES


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    stdout: str
    stderr: str
    returncode: int
    incomplete: bool


def run_limited_process(
    command: list[str],
    *,
    cwd: Path,
    max_output_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> ProcessOutput:
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")

    process = subprocess.Popen(
        command,
        cwd=cwd,
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

    try:
        while selector.get_map():
            for key, _ in selector.select():
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
    finally:
        selector.close()

    if incomplete:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()
    process.stdout.close()
    process.stderr.close()

    return ProcessOutput(
        stdout=bytes(captured["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(captured["stderr"]).decode("utf-8", errors="replace"),
        returncode=process.returncode,
        incomplete=incomplete,
    )
