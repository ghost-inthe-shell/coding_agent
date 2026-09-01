"""Stable prompt resources for the coding agent."""

"""Stable prompt resources."""

from .loader import load_compaction_prompt, load_system_prompt
from .project import (
    MAX_PROJECT_INSTRUCTIONS_CHARS,
    ProjectInstructionsError,
    compose_session_system_prompt,
    load_project_instructions,
)

__all__ = [
    "MAX_PROJECT_INSTRUCTIONS_CHARS",
    "ProjectInstructionsError",
    "compose_session_system_prompt",
    "load_compaction_prompt",
    "load_project_instructions",
    "load_system_prompt",
]
