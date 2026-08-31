"""Compile the candidate and compare its answers with a deterministic oracle."""

from __future__ import annotations

import bisect
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def expected_output(values: list[int], queries: list[int]) -> str:
    answers: list[str] = []
    for query in queries:
        start = bisect.bisect_left(values, query)
        if start == len(values) or values[start] != query:
            answers.append("-1 -1")
            continue
        terminal = bisect.bisect_right(values, query) - 1
        answers.append(f"{start} {terminal}")
    return "\n".join(answers) + "\n"


def run_case(binary: Path, values: list[int], queries: list[int]) -> None:
    input_text = (
        f"{len(values)} {len(queries)}\n"
        + " ".join(str(value) for value in values)
        + "\n"
        + "\n".join(str(query) for query in queries)
        + "\n"
    )
    completed = subprocess.run(
        [str(binary)],
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=3,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"program exited with {completed.returncode}: {completed.stderr.strip()}"
        )
    expected = expected_output(values, queries)
    if completed.stdout != expected:
        raise AssertionError(
            "wrong answer\n"
            f"values={values[:30]}{'...' if len(values) > 30 else ''}\n"
            f"queries={queries}\n"
            f"expected={expected!r}\n"
            f"actual={completed.stdout!r}"
        )


def verify(workspace: Path) -> None:
    source = workspace / "1.cpp"
    if not source.is_file():
        raise AssertionError("missing 1.cpp")

    with tempfile.TemporaryDirectory(prefix="coding-agent-cpp-eval-") as temporary:
        binary = Path(temporary) / "solution"
        compiled = subprocess.run(
            ["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        if compiled.returncode != 0:
            raise AssertionError(f"compilation failed:\n{compiled.stderr}")

        run_case(binary, [1, 2, 2, 3, 3, 4], [3, 4, 5])
        run_case(binary, [7], [1, 7, 10_000])
        run_case(binary, [5, 5, 5, 5], [4, 5, 6])
        run_case(binary, [1, 1, 2, 4, 4, 10_000], [1, 2, 3, 4, 9_999, 10_000])

        random_generator = random.Random(20260831)
        for size in (2, 3, 8, 31, 100, 997):
            values = sorted(random_generator.randint(1, 250) for _ in range(size))
            queries = [1, 250]
            queries.extend(random_generator.randint(1, 250) for _ in range(40))
            run_case(binary, values, queries)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify.py WORKSPACE", file=sys.stderr)
        return 2
    try:
        verify(Path(sys.argv[1]).resolve())
    except (AssertionError, OSError, subprocess.SubprocessError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print("all C++ boundary-search tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
