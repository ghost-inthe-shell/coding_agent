"""Verify that the implementation follows constraints supplied through AGENTS.md."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

ORIGINAL_PUBLIC_TEST = """import unittest

from slugger import make_slug


class SluggerTests(unittest.TestCase):
    def test_simple_title(self) -> None:
        self.assertEqual(make_slug("Hello World"), "hello-world")


if __name__ == "__main__":
    unittest.main()
"""


def verify(workspace: Path) -> None:
    public_test = workspace / "test_slugger.py"
    if public_test.read_text(encoding="utf-8") != ORIGINAL_PUBLIC_TEST:
        raise AssertionError("the existing public test file was modified")

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "-v"],
        cwd=workspace,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise AssertionError("public tests failed:\n" + completed.stdout + completed.stderr)

    sys.path.insert(0, str(workspace))
    try:
        slugger = importlib.import_module("slugger")
    finally:
        sys.path.pop(0)

    cases = {
        "  Hello__World!! ": "hello-world",
        "Straße & Café": "strasse-café",
        "版本 ２": "版本-２",
        "one---two___three": "one-two-three",
    }
    for value, expected in cases.items():
        actual = slugger.make_slug(value)
        if actual != expected:
            raise AssertionError(f"make_slug({value!r}) returned {actual!r}, expected {expected!r}")

    for invalid in (None, 123, ["title"]):
        try:
            slugger.make_slug(invalid)
        except TypeError:
            pass
        else:
            raise AssertionError(f"non-string input was accepted: {invalid!r}")

    for empty in ("", "---", " _ ! "):
        try:
            slugger.make_slug(empty)
        except slugger.SlugError:
            pass
        else:
            raise AssertionError(f"empty normalized slug was accepted: {empty!r}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify.py WORKSPACE", file=sys.stderr)
        return 2
    try:
        verify(Path(sys.argv[1]).resolve())
    except (
        AssertionError,
        AttributeError,
        ImportError,
        OSError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print("all project-instruction tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
