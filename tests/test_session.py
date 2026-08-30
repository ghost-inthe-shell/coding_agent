import json
import unittest

from coding_agent.core.file_state import FileVersion
from coding_agent.core.messages import (
    AssistantMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.session import SessionState
from coding_agent.core.types import SessionStatus, StopReason
from coding_agent.core.usage import Usage


class SessionStateTests(unittest.TestCase):
    def test_session_round_trip_is_provider_independent(self) -> None:
        state = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="Be useful.",
            messages=[
                UserMessage.from_text("read the file", timestamp=10),
                AssistantMessage(
                    content=(ToolCall(id="call-1", name="read_file"),),
                    provider="fake",
                    model="fake-model",
                    stop_reason=StopReason.TOOL_USE,
                    timestamp=11,
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read_file",
                    content=(TextBlock("contents"),),
                    timestamp=12,
                ),
            ],
            read_file_versions={
                "src/example.py": FileVersion(
                    mtime_ns=123,
                    size=7,
                    sha256="a" * 64,
                )
            },
            usage=Usage(input_tokens=20, output_tokens=4),
            status=SessionStatus.IDLE,
            created_at=1,
            updated_at=2,
        )

        encoded = json.loads(json.dumps(state.to_dict()))
        restored = SessionState.from_dict(encoded)

        self.assertEqual(restored, state)
        restored.validate()

    def test_schema_version_one_is_migrated_with_an_empty_version_table(self) -> None:
        state = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="Be useful.",
        )
        legacy = state.to_dict()
        legacy["schema_version"] = 1
        del legacy["read_file_versions"]

        restored = SessionState.from_dict(legacy)

        self.assertEqual(restored.schema_version, 2)
        self.assertEqual(restored.read_file_versions, {})

        current_without_versions = state.to_dict()
        del current_without_versions["read_file_versions"]
        with self.assertRaisesRegex(ValueError, "required"):
            SessionState.from_dict(current_without_versions)

    def test_file_versions_require_canonical_relative_paths_and_valid_digests(self) -> None:
        version = FileVersion(mtime_ns=-1, size=2, sha256="A" * 64)

        self.assertEqual(version.mtime_ns, -1)
        self.assertEqual(version.sha256, "a" * 64)
        for path in ("", ".", "/absolute.py", "src/../example.py", "src//example.py"):
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "workspace-relative"
            ):
                SessionState(
                    session_id="session-1",
                    workspace_root="/tmp/workspace",
                    system_prompt="Be useful.",
                    read_file_versions={path: version},
                )

        with self.assertRaisesRegex(ValueError, "sha256"):
            FileVersion(mtime_ns=1, size=2, sha256="not-a-digest")

    def test_new_session_loads_a_system_prompt_snapshot(self) -> None:
        state = SessionState.create("session-1", "/tmp")

        self.assertTrue(state.system_prompt)
        self.assertEqual(state.workspace_root, "/tmp")


if __name__ == "__main__":
    unittest.main()
