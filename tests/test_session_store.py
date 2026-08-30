import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.core.messages import AssistantMessage, ToolCall
from coding_agent.core.session import SessionState
from coding_agent.core.session_store import (
    InvalidSessionError,
    SessionNotFoundError,
    SessionSaveError,
    SessionStore,
)
from coding_agent.core.types import SessionStatus, StopReason


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.state_home = Path(self.temporary.name) / "state"
        self.store = SessionStore(state_home=self.state_home)
        self.state = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="Be useful.",
            status=SessionStatus.IDLE,
            created_at=1,
            updated_at=2,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_save_and_load_round_trip_with_private_permissions(self) -> None:
        path = self.store.save(self.state)

        self.assertEqual(
            path,
            self.state_home / "coding-agent" / "sessions" / "session-1" / "session.json",
        )
        self.assertEqual(self.store.load("session-1"), self.state)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_atomic_failure_preserves_previous_checkpoint_and_removes_temp_file(self) -> None:
        path = self.store.save(self.state)
        original = path.read_text(encoding="utf-8")
        self.state.updated_at = 3

        with (
            patch(
                "coding_agent.core.session_store.os.replace",
                side_effect=OSError("replace failed"),
            ),
            self.assertRaisesRegex(SessionSaveError, "replace failed"),
        ):
            self.store.save(self.state)

        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertEqual(list(path.parent.glob(".session-*.tmp")), [])

    def test_save_rejects_running_or_protocol_incomplete_state(self) -> None:
        self.state.status = SessionStatus.RUNNING
        with self.assertRaisesRegex(InvalidSessionError, "stable checkpoint"):
            self.store.save(self.state)

        self.state.status = SessionStatus.IDLE
        self.state.messages.append(
            AssistantMessage(
                content=(ToolCall(id="call-1", name="read_file"),),
                provider="fake",
                model="fake",
                stop_reason=StopReason.TOOL_USE,
            )
        )
        with self.assertRaisesRegex(InvalidSessionError, "pending tool calls"):
            self.store.save(self.state)

    def test_load_rejects_corrupt_mismatched_or_unstable_checkpoint(self) -> None:
        path = self.store.path_for("session-1")
        path.parent.mkdir(parents=True)

        path.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(InvalidSessionError, "invalid session"):
            self.store.load("session-1")

        document = self.state.to_dict()
        document["session_id"] = "another-session"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(InvalidSessionError, "does not match"):
            self.store.load("session-1")

        document["session_id"] = "session-1"
        document["status"] = "running"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(InvalidSessionError, "stable checkpoint"):
            self.store.load("session-1")

    def test_load_missing_session_and_unsafe_ids_have_explicit_errors(self) -> None:
        with self.assertRaisesRegex(SessionNotFoundError, "session not found"):
            self.store.load("missing")

        for session_id in ("../escape", "/absolute", "with/slash", "", "a" * 129):
            with (
                self.subTest(session_id=session_id),
                self.assertRaisesRegex(InvalidSessionError, "unsafe"),
            ):
                self.store.load(session_id)

    def test_default_root_uses_xdg_state_home(self) -> None:
        configured = Path(self.temporary.name) / "custom"
        with patch.dict(os.environ, {"XDG_STATE_HOME": str(configured)}):
            store = SessionStore()

        self.assertEqual(store.root, configured / "coding-agent" / "sessions")


if __name__ == "__main__":
    unittest.main()
