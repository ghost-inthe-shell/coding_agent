"""Verify the priority model, compatibility, persistence, CLI, and ordering."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(command: list[str], workspace: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(workspace)
    return subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )


def verify(workspace: Path) -> None:
    public_tests = run_command(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        workspace,
    )
    if public_tests.returncode != 0:
        raise AssertionError(
            "public tests failed:\n" + public_tests.stdout + public_tests.stderr
        )

    sys.path.insert(0, str(workspace))
    try:
        model = importlib.import_module("tasker.model")
        store = importlib.import_module("tasker.store")
    finally:
        sys.path.pop(0)

    task_class = model.Task
    if task_class("default").priority != "normal":
        raise AssertionError("Task priority must default to normal")
    for invalid_priority in ("urgent", "", "HIGH"):
        try:
            task_class("invalid", priority=invalid_priority)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid priority was accepted: {invalid_priority!r}")

    with tempfile.TemporaryDirectory(prefix="coding-agent-tasker-eval-") as temporary:
        database = Path(temporary) / "tasks.json"
        database.write_text(
            json.dumps([{"title": "legacy", "completed": True}]) + "\n",
            encoding="utf-8",
        )
        legacy = store.load_tasks(database)
        if len(legacy) != 1 or legacy[0].priority != "normal":
            raise AssertionError("legacy JSON without priority must load as normal")

        base_command = [
            sys.executable,
            "-m",
            "tasker.cli",
            "--db",
            str(database),
        ]
        additions = [
            ("low task", "low"),
            ("high first", "high"),
            ("default task", None),
            ("high second", "high"),
        ]
        for title, priority in additions:
            command = [*base_command, "add", title]
            if priority is not None:
                command.extend(["--priority", priority])
            completed = run_command(command, workspace)
            if completed.returncode != 0:
                raise AssertionError(
                    f"add command failed for {title!r}: {completed.stderr.strip()}"
                )

        raw_tasks = json.loads(database.read_text(encoding="utf-8"))
        priorities = [raw_task.get("priority") for raw_task in raw_tasks]
        if priorities != ["normal", "low", "high", "normal", "high"]:
            raise AssertionError(f"priorities were not persisted: {priorities!r}")

        listed = run_command([*base_command, "list"], workspace)
        expected = (
            "[HIGH] [ ] high first\n"
            "[HIGH] [ ] high second\n"
            "[NORMAL] [x] legacy\n"
            "[NORMAL] [ ] default task\n"
            "[LOW] [ ] low task\n"
        )
        if listed.returncode != 0:
            raise AssertionError(f"list command failed: {listed.stderr.strip()}")
        if listed.stdout != expected:
            raise AssertionError(
                f"list output mismatch\nexpected={expected!r}\nactual={listed.stdout!r}"
            )


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
    print("all task-priority tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
