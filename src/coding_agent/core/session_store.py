"""Durable, atomic persistence for stable session checkpoints."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
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

    def path_for(self, session_id: str) -> Path:
        _validate_session_id(session_id)
        return self.root / session_id / "session.json"

    def save(self, state: SessionState) -> Path:
        """Atomically replace the checkpoint for a stable SessionState."""

        try:
            path = self.path_for(state.session_id)
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
            _make_private_directory(path.parent)
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


def _make_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
