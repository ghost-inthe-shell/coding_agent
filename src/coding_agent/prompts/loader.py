"""Load stable prompt resources packaged with the agent."""

from importlib.resources import files


def load_system_prompt() -> str:
    prompt = files("coding_agent.prompts").joinpath("system.md").read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("system prompt must not be empty")
    return prompt
