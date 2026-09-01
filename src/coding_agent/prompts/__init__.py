"""Stable prompt resources for the coding agent."""

"""Stable prompt resources."""

from .loader import load_compaction_prompt, load_system_prompt
from .project import (
    MAX_PROJECT_INSTRUCTIONS_CHARS,
    ProjectInstructionsError,
    compose_session_system_prompt,
    load_project_instructions,
)
from .skills import (
    MAX_PROJECT_SKILLS,
    MAX_SKILL_CHARS,
    ProjectSkill,
    ProjectSkillsError,
    discover_project_skills,
    load_project_skill,
)

__all__ = [
    "MAX_PROJECT_INSTRUCTIONS_CHARS",
    "MAX_PROJECT_SKILLS",
    "MAX_SKILL_CHARS",
    "ProjectSkill",
    "ProjectSkillsError",
    "ProjectInstructionsError",
    "compose_session_system_prompt",
    "discover_project_skills",
    "load_compaction_prompt",
    "load_project_instructions",
    "load_project_skill",
    "load_system_prompt",
]
