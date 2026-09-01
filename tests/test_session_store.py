import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
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
            created_at=int(datetime(2024, 2, 3, 4, 0, tzinfo=timezone.utc).timestamp() * 1000),
            updated_at=2,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_save_and_load_round_trip_with_private_permissions(self) -> None:
        path = self.store.save(self.state)

        self.assertEqual(
            path,
            self.state_home
            / "coding-agent"
            / "sessions"
            / "2024"
            / "02"
            / "03"
            / "session-1"
            / "session.json",
        )
        self.assertEqual(self.store.load("session-1"), self.state)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
        for directory in (
            path.parent.parent,
            path.parent.parent.parent,
            path.parent.parent.parent.parent,
        ):
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)

    def test_existing_legacy_layout_is_loaded_and_saved_in_place(self) -> None:
        legacy_path = self.store.path_for("session-1")
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(
            json.dumps(self.state.to_dict()),
            encoding="utf-8",
        )

        loaded = self.store.load("session-1")
        saved_path = self.store.save(loaded)

        self.assertEqual(loaded, self.state)
        self.assertEqual(saved_path, legacy_path)
        self.assertFalse((self.store.root / "2024").exists())

    def test_resume_keeps_the_project_instruction_snapshot(self) -> None:
        workspace = Path(self.temporary.name) / "workspace"
        workspace.mkdir()
        instructions = workspace / "AGENTS.md"
        instructions.write_text("Original project rule.\n", encoding="utf-8")
        state = SessionState.create("snapshot-session", workspace)
        self.store.save(state)

        instructions.write_bytes(b"\xff")
        restored = self.store.load("snapshot-session")

        self.assertEqual(restored.system_prompt, state.system_prompt)
        self.assertIn("Original project rule.", restored.system_prompt)

    def test_duplicate_legacy_and_dated_checkpoints_are_rejected(self) -> None:
        dated_path = self.store.save(self.state)
        legacy_path = self.store.root / "session-1" / "session.json"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(dated_path.read_text(encoding="utf-8"), encoding="utf-8")

        with self.assertRaisesRegex(InvalidSessionError, "multiple checkpoints"):
            self.store.load("session-1")

    def test_save_rejects_created_at_outside_datetime_range(self) -> None:
        self.state.created_at = 10**30

        with self.assertRaisesRegex(InvalidSessionError, "supported timestamp range"):
            self.store.save(self.state)

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
