"""JSON persistence for tasks."""

from __future__ import annotations

import json
from pathlib import Path

from .model import Task


def load_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    raw_tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_tasks, list):
        raise TypeError("task database must contain a JSON array")
    return [Task.from_dict(raw_task) for raw_task in raw_tasks]


def save_tasks(path: Path, tasks: list[Task]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([task.to_dict() for task in tasks], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
