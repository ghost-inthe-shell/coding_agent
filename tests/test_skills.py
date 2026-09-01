import os
import tempfile
import unittest
from pathlib import Path

from coding_agent.prompts import (
    MAX_PROJECT_SKILLS,
    MAX_SKILL_CATALOG_CHARS,
    MAX_SKILL_CHARS,
    ProjectSkillsError,
    discover_project_skills,
    format_project_skills_for_prompt,
    load_project_skill,
)


def skill_document(
    name: str,
    description: str = "Use this skill for focused work.",
    instructions: str = "# Steps\n\nDo the work.\n",
) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n"
        f"{instructions}"
    )


class ProjectSkillsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "workspace"
        self.workspace.mkdir()
        self.skills_root = self.workspace / ".agents" / "skills"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_skill(self, name: str, content: str | None = None) -> Path:
        directory = self.skills_root / name
        directory.mkdir(parents=True)
        path = directory / "SKILL.md"
        path.write_text(content or skill_document(name), encoding="utf-8")
        return path

    def test_missing_empty_and_non_skill_entries_are_ignored(self) -> None:
        self.assertEqual(discover_project_skills(self.workspace), ())
        self.skills_root.mkdir(parents=True)
        (self.skills_root / "README.md").write_text("notes", encoding="utf-8")
        (self.skills_root / "group").mkdir()

        self.assertEqual(discover_project_skills(self.workspace), ())

    def test_discovers_sorted_direct_children_and_preserves_instructions(self) -> None:
        self.write_skill("zebra", skill_document("zebra", instructions="Zebra steps.\n"))
        self.write_skill(
            "alpha",
            skill_document(
                "alpha",
                description="|-\n  Alpha workflow for\n  source files.",
                instructions="# Alpha\n\n    Preserve indentation.\n",
            ),
        )
        nested = self.skills_root / "group" / "nested"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text(skill_document("nested"), encoding="utf-8")

        skills = discover_project_skills(self.workspace)

        self.assertEqual([skill.name for skill in skills], ["alpha", "zebra"])
        self.assertEqual(skills[0].description, "Alpha workflow for\nsource files.")
        self.assertEqual(skills[0].path, ".agents/skills/alpha/SKILL.md")
        self.assertEqual(skills[0].directory, ".agents/skills/alpha")
        self.assertEqual(skills[0].instructions, "# Alpha\n\n    Preserve indentation.\n")
        self.assertEqual(load_project_skill(self.workspace, "zebra"), skills[1])

    def test_frontmatter_is_strict_and_name_must_match_directory(self) -> None:
        cases = (
            ("wrong", skill_document("other"), "must match directory"),
            (
                "extra",
                "---\nname: extra\ndescription: Valid.\nallowed-tools: read_file\n---\nDo it.\n",
                "extra_forbidden",
            ),
            ("Bad_Name", skill_document("Bad_Name"), "invalid frontmatter"),
            ("blank", "---\nname: blank\ndescription: ''\n---\nDo it.\n", "invalid frontmatter"),
            ("missing", "# No frontmatter\n", "must start"),
            ("open", "---\nname: open\ndescription: Missing close\n", "unterminated"),
            ("yaml", "---\nname: [\n---\nDo it.\n", "invalid YAML"),
            ("body", "---\nname: body\ndescription: Valid.\n---\n", "no instructions"),
        )
        for directory_name, content, error in cases:
            with self.subTest(directory_name=directory_name):
                self.skills_root.mkdir(parents=True, exist_ok=True)
                directory = self.skills_root / directory_name
                directory.mkdir()
                (directory / "SKILL.md").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ProjectSkillsError, error):
                    discover_project_skills(self.workspace)
                for child in directory.iterdir():
                    child.unlink()
                directory.rmdir()

    def test_binary_invalid_utf8_and_oversized_skill_are_rejected(self) -> None:
        for raw, error in (
            (b"---\nname: binary\ndescription: x\n---\na\x00b", "binary data"),
            (b"\xff", "not valid UTF-8"),
            (b"x" * (MAX_SKILL_CHARS + 1), "character limit"),
            (b"x" * (MAX_SKILL_CHARS * 4 + 1), "character limit"),
        ):
            with self.subTest(error=error):
                path = self.skills_root / "binary" / "SKILL.md"
                path.parent.mkdir(parents=True)
                path.write_bytes(raw)
                with self.assertRaisesRegex(ProjectSkillsError, error):
                    discover_project_skills(self.workspace)
                path.unlink()
                path.parent.rmdir()

    def test_skill_paths_cannot_escape_workspace(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (outside / "SKILL.md").write_text(skill_document("escape"), encoding="utf-8")
        self.skills_root.mkdir(parents=True)
        os.symlink(outside, self.skills_root / "escape")

        with self.assertRaisesRegex(ProjectSkillsError, "outside the workspace"):
            discover_project_skills(self.workspace)

    def test_internal_skill_symlink_is_allowed(self) -> None:
        source = self.workspace / "shared-skill"
        source.mkdir()
        (source / "SKILL.md").write_text(skill_document("linked"), encoding="utf-8")
        self.skills_root.mkdir(parents=True)
        os.symlink(source, self.skills_root / "linked")

        skill = discover_project_skills(self.workspace)[0]

        self.assertEqual(skill.name, "linked")
        self.assertEqual(skill.path, ".agents/skills/linked/SKILL.md")

    def test_skill_count_and_explicit_name_are_bounded(self) -> None:
        for index in range(MAX_PROJECT_SKILLS + 1):
            name = f"skill-{index}"
            self.write_skill(name)

        with self.assertRaisesRegex(ProjectSkillsError, "more than"):
            discover_project_skills(self.workspace)

        with self.assertRaisesRegex(ProjectSkillsError, "invalid skill name"):
            load_project_skill(self.workspace, "../escape")
        with self.assertRaisesRegex(ProjectSkillsError, "not found"):
            load_project_skill(self.workspace, "not-found")

    def test_prompt_catalog_contains_only_escaped_metadata(self) -> None:
        self.write_skill(
            "review",
            skill_document(
                "review",
                description="Review <code> & tests.",
                instructions="SECRET BODY MUST BE LOADED LATER.\n",
            ),
        )

        catalog = format_project_skills_for_prompt(
            discover_project_skills(self.workspace)
        )

        self.assertIn("<name>review</name>", catalog)
        self.assertIn("Review &lt;code&gt; &amp; tests.", catalog)
        self.assertIn(".agents/skills/review/SKILL.md", catalog)
        self.assertNotIn("SECRET BODY", catalog)
        self.assertLessEqual(len(catalog), MAX_SKILL_CATALOG_CHARS)
        self.assertEqual(format_project_skills_for_prompt(()), "")


if __name__ == "__main__":
    unittest.main()
