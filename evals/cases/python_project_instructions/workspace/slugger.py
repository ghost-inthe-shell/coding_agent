class SlugError(ValueError):
    """Raised when a value cannot produce a non-empty slug."""


def make_slug(value: str) -> str:
    """Normalize a title according to this project's slug rules."""

    raise NotImplementedError
