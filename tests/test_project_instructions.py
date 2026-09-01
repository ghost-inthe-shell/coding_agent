import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.prompts import (
    MAX_PROJECT_INSTRUCTIONS_CHARS,
    ProjectInstructionsError,
    compose_session_system_prompt,
    load_project_instructions,
    load_system_prompt,
)


class ProjectInstructionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "workspace"
        self.workspace.mkdir()
        self.path = self.workspace / "AGENTS.md"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_missing_or_blank_file_is_ignored(self) -> None:
        self.assertIsNone(load_project_instructions(self.workspace))
        self.path.write_text("  \n\t\n", encoding="utf-8")

        self.assertIsNone(load_project_instructions(self.workspace))
        self.assertEqual(compose_session_system_prompt(self.workspace), load_system_prompt())

    def test_utf8_content_is_preserved_and_composed_once(self) -> None:
        content = "# 项目约束\n\n    保留这段缩进。\n"
        self.path.write_text(content, encoding="utf-8")

        loaded = load_project_instructions(self.workspace)
        prompt = compose_session_system_prompt(self.workspace)

        self.assertEqual(loaded, content)
        self.assertEqual(prompt.count("# Project instructions"), 1)
        self.assertEqual(prompt.count("# 项目约束"), 1)
        self.assertIn("    保留这段缩进。", prompt)
        self.assertTrue(prompt.startswith(load_system_prompt()))

    def test_binary_invalid_utf8_and_oversized_content_are_rejected(self) -> None:
        for raw, error in (
            (b"text\x00data", "binary data"),
            (b"\xff", "not valid UTF-8"),
            (b"x" * (MAX_PROJECT_INSTRUCTIONS_CHARS + 1), "character limit"),
        ):
            with self.subTest(error=error):
                self.path.write_bytes(raw)
                with self.assertRaisesRegex(ProjectInstructionsError, error):
                    load_project_instructions(self.workspace)

    def test_non_file_and_broken_symlink_are_rejected(self) -> None:
        self.path.mkdir()
        with self.assertRaisesRegex(ProjectInstructionsError, "regular file"):
            load_project_instructions(self.workspace)

        self.path.rmdir()
        os.symlink(self.workspace / "missing.md", self.path)
        with self.assertRaisesRegex(ProjectInstructionsError, "could not resolve"):
            load_project_instructions(self.workspace)

    def test_symlink_inside_workspace_is_allowed_but_escape_is_rejected(self) -> None:
        source = self.workspace / "instructions.md"
        source.write_text("Inside instructions.\n", encoding="utf-8")
        os.symlink(source, self.path)

        self.assertEqual(load_project_instructions(self.workspace), "Inside instructions.\n")

        self.path.unlink()
        outside = Path(self.temporary.name) / "outside.md"
        outside.write_text("Private instructions.\n", encoding="utf-8")
        os.symlink(outside, self.path)
        with self.assertRaisesRegex(ProjectInstructionsError, "outside the workspace"):
            load_project_instructions(self.workspace)

    def test_read_errors_are_not_silently_ignored(self) -> None:
        self.path.write_text("Instructions.\n", encoding="utf-8")

        with (
            patch.object(Path, "open", side_effect=PermissionError("denied")),
            self.assertRaisesRegex(ProjectInstructionsError, "could not read AGENTS.md"),
        ):
            load_project_instructions(self.workspace)

    def test_invalid_workspace_is_rejected(self) -> None:
        missing = self.workspace / "missing"
        with self.assertRaisesRegex(ProjectInstructionsError, "could not resolve workspace"):
            load_project_instructions(missing)


if __name__ == "__main__":
    unittest.main()
