"""Load stable prompt resources packaged with the agent."""

from importlib.resources import files


def load_system_prompt() -> str:
    return _load_prompt("system.md")


def load_compaction_prompt() -> str:
    return _load_prompt("compact.md")


def _load_prompt(name: str) -> str:
    prompt = files("coding_agent.prompts").joinpath(name).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"prompt resource must not be empty: {name}")
    return prompt
