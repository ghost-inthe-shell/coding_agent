"""Command-line interface for Tasker."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import Task
from .store import load_tasks, save_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tasker")
    parser.add_argument("--db", type=Path, default=Path("tasks.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("title")
    subparsers.add_parser("list")
    return parser


def format_tasks(tasks: list[Task]) -> str:
    return "\n".join(
        f"[{'x' if task.completed else ' '}] {task.title}" for task in tasks
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = load_tasks(args.db)
    if args.command == "add":
        tasks.append(Task(args.title))
        save_tasks(args.db, tasks)
        print(f"Added: {args.title}")
        return 0
    if args.command == "list":
        output = format_tasks(tasks)
        if output:
            print(output)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
