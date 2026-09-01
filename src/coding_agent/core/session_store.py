"""Durable, atomic persistence for stable session checkpoints."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path

from .session import SessionState
from .types import SessionStatus


class SessionStoreError(Exception):
    """Base class for expected session persistence failures."""


class SessionNotFoundError(SessionStoreError):
    """Raised when an explicitly requested session does not exist."""


class InvalidSessionError(SessionStoreError):
    """Raised when persisted session data violates the storage contract."""


class SessionSaveError(SessionStoreError):
    """Raised when a valid checkpoint cannot be written durably."""


class SessionStore:
    """Store one strict JSON checkpoint per session ID."""

    def __init__(self, *, state_home: Path | None = None) -> None:
        base = state_home if state_home is not None else _default_state_home()
        self.root = base.expanduser() / "coding-agent" / "sessions"

    def path_for(self, session_id: str, *, created_at: int | None = None) -> Path:
        _validate_session_id(session_id)
        existing = _find_session_paths(self.root, session_id)
        if len(existing) > 1:
            raise ValueError(f"multiple checkpoints found for session {session_id!r}")
        if existing:
            return existing[0]
        if created_at is not None:
            year, month, day = _session_date_parts(created_at)
            return self.root / year / month / day / session_id / "session.json"
        return self.root / session_id / "session.json"

    def path_for_state(self, state: SessionState) -> Path:
        """Resolve the stable checkpoint path for an existing or new state."""

        return self.path_for(state.session_id, created_at=state.created_at)

    def save(self, state: SessionState) -> Path:
        """Atomically replace the checkpoint for a stable SessionState."""

        try:
            path = self.path_for_state(state)
            _validate_stable_state(state)
            encoded = (
                json.dumps(
                    state.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        except (TypeError, ValueError) as exc:
            raise InvalidSessionError(f"session cannot be saved: {exc}") from exc

        temporary_path: Path | None = None
        try:
            _make_private_directory(self.root.parent)
            _make_private_directory(self.root)
            _make_private_tree(self.root, path.parent)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".session-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.chmod(temporary_path, 0o600)
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())

            os.replace(temporary_path, path)
            temporary_path = None
            os.chmod(path, 0o600)
            _fsync_directory(path.parent)
        except OSError as exc:
            raise SessionSaveError(f"failed to save session {state.session_id!r}: {exc}") from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return path

    def load(self, session_id: str) -> SessionState:
        """Load and strictly validate one stable checkpoint."""

        try:
            path = self.path_for(session_id)
        except ValueError as exc:
            raise InvalidSessionError(str(exc)) from exc

        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SessionNotFoundError(f"session not found: {session_id}") from exc
        except OSError as exc:
            raise SessionStoreError(f"failed to read session {session_id!r}: {exc}") from exc

        try:
            data = json.loads(raw)
            if not isinstance(data, Mapping):
                raise TypeError("session document must be a JSON object")
            state = SessionState.from_dict(data)
            if state.session_id != session_id:
                raise ValueError(
                    f"stored session_id {state.session_id!r} does not match requested ID"
                )
            _validate_stable_state(state)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise InvalidSessionError(f"invalid session {session_id!r}: {exc}") from exc
        return state


def _default_state_home() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    if configured:
        return Path(configured)
    return Path.home() / ".local" / "state"


def _validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id
    ):
        raise ValueError("session_id contains unsafe path characters")
    if session_id in {".", ".."}:
        raise ValueError("session_id contains unsafe path characters")


def _validate_stable_state(state: SessionState) -> None:
    if state.status is SessionStatus.RUNNING:
        raise ValueError("running sessions are not stable checkpoints")
    state.validate()


def _session_date_parts(created_at: int) -> tuple[str, str, str]:
    if not isinstance(created_at, int) or isinstance(created_at, bool):
        raise TypeError("created_at must be an integer timestamp")
    try:
        created = datetime.fromtimestamp(created_at / 1000, timezone.utc).astimezone()
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("created_at is outside the supported timestamp range") from exc
    return f"{created.year:04d}", f"{created.month:02d}", f"{created.day:02d}"


def _find_session_paths(root: Path, session_id: str) -> list[Path]:
    candidates: list[Path] = []
    legacy = root / session_id / "session.json"
    if legacy.is_file():
        candidates.append(legacy)
    if not root.is_dir():
        return candidates
    for candidate in root.glob(f"*/*/*/{session_id}/session.json"):
        try:
            year, month, day, found_id, filename = candidate.relative_to(root).parts
            date(int(year), int(month), int(day))
        except (TypeError, ValueError):
            continue
        if (
            not re.fullmatch(r"\d{4}", year)
            or not re.fullmatch(r"\d{2}", month)
            or not re.fullmatch(r"\d{2}", day)
            or found_id != session_id
            or filename != "session.json"
            or not candidate.is_file()
        ):
            continue
        candidates.append(candidate)
    return sorted(candidates)


def _make_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _make_private_tree(root: Path, leaf: Path) -> None:
    try:
        relative = leaf.relative_to(root)
    except ValueError as exc:  # pragma: no cover - constructed internally
        raise OSError(f"session directory escaped storage root: {leaf}") from exc
    current = root
    for part in relative.parts:
        current /= part
        _make_private_directory(current)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
