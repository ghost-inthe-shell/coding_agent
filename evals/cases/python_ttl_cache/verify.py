"""Run public tests and additional deterministic TTL cache checks."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType


class FakeClock:
    def __init__(self, now: float = 10.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def load_candidate(workspace: Path) -> ModuleType:
    source = workspace / "ttl_cache.py"
    if not source.is_file():
        raise AssertionError("missing ttl_cache.py")
    spec = importlib.util.spec_from_file_location("eval_ttl_cache", source)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import ttl_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(workspace: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    public_tests = subprocess.run(
        [sys.executable, "-m", "unittest", "-v"],
        cwd=workspace,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if public_tests.returncode != 0:
        raise AssertionError("public tests failed:\n" + public_tests.stdout + public_tests.stderr)

    module = load_candidate(workspace)
    cache_class = getattr(module, "TTLCache", None)
    if cache_class is None:
        raise AssertionError("missing TTLCache")

    clock = FakeClock()
    cache = cache_class(clock)
    for invalid_ttl in (0, -0.1):
        try:
            cache.put("invalid", "value", invalid_ttl)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive TTL must raise ValueError")

    cache.put("key", "old", 2)
    clock.now = 11.5
    cache.put("key", "new", 5)
    clock.now = 12.0
    if cache.get("key") != "new":
        raise AssertionError("overwriting a key must replace its value and expiry")
    clock.now = 16.5
    try:
        cache.get("key")
    except KeyError:
        pass
    else:
        raise AssertionError("value remained available at its exact deadline")
    if len(cache) != 0:
        raise AssertionError("expired entries must not contribute to len(cache)")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify.py WORKSPACE", file=sys.stderr)
        return 2
    try:
        verify(Path(sys.argv[1]).resolve())
    except (AssertionError, ImportError, OSError, subprocess.SubprocessError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print("all TTL cache tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
