import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tasker.model import Task
from tasker.store import load_tasks, save_tasks


class TaskerTests(unittest.TestCase):
    def test_model_round_trip(self) -> None:
        task = Task("write tests", completed=True)

        self.assertEqual(Task.from_dict(task.to_dict()), task)

    def test_store_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "tasks.json"

            save_tasks(database, [Task("one"), Task("two", completed=True)])

            self.assertEqual(
                load_tasks(database),
                [Task("one"), Task("two", completed=True)],
            )
            self.assertIsInstance(json.loads(database.read_text()), list)

    def test_cli_add_then_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "tasks.json"
            base_command = [sys.executable, "-m", "tasker.cli", "--db", str(database)]

            added = subprocess.run(
                [*base_command, "add", "write docs"],
                text=True,
                capture_output=True,
                check=False,
            )
            listed = subprocess.run(
                [*base_command, "list"],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(added.returncode, 0, added.stderr)
            self.assertEqual(listed.returncode, 0, listed.stderr)
            self.assertIn("write docs", listed.stdout)


if __name__ == "__main__":
    unittest.main()
