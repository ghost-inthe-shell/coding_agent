import json
import unittest

from coding_agent.core.compaction import CompactionCheckpoint
from coding_agent.core.file_state import FileVersion
from coding_agent.core.messages import (
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
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
                    content=(
                        ThinkingBlock(
                            "The file is relevant.",
                            replay_field="reasoning_content",
                        ),
                        ToolCall(id="call-1", name="read_file"),
                    ),
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
            compaction=CompactionCheckpoint(
                summary="The user asked to read the file.",
                first_kept_message_index=1,
                tokens_before=123,
                created_at=13,
            ),
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

    def test_legacy_schema_versions_are_migrated(self) -> None:
        state = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="Be useful.",
        )
        legacy = state.to_dict()
        legacy["schema_version"] = 1
        del legacy["read_file_versions"]
        del legacy["compaction"]

        restored = SessionState.from_dict(legacy)

        self.assertEqual(restored.schema_version, 4)
        self.assertEqual(restored.read_file_versions, {})

        for schema_version in (2, 3):
            with self.subTest(schema_version=schema_version):
                legacy_version = state.to_dict()
                legacy_version["schema_version"] = schema_version
                legacy_version.pop("compaction", None)
                restored_version = SessionState.from_dict(legacy_version)
                self.assertEqual(restored_version.schema_version, 4)
                self.assertIsNone(restored_version.compaction)

        current_without_versions = state.to_dict()
        del current_without_versions["read_file_versions"]
        with self.assertRaisesRegex(ValueError, "required"):
            SessionState.from_dict(current_without_versions)

        current_without_compaction = state.to_dict()
        del current_without_compaction["compaction"]
        with self.assertRaisesRegex(ValueError, "compaction is required"):
            SessionState.from_dict(current_without_compaction)

    def test_compaction_checkpoint_requires_a_valid_active_message_boundary(self) -> None:
        call = ToolCall(id="call-1", name="read_file")
        messages = [
            UserMessage.from_text("read", timestamp=1),
            AssistantMessage(
                content=(call,),
                provider="fake",
                model="fake",
                stop_reason=StopReason.TOOL_USE,
                timestamp=2,
            ),
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="read_file",
                content=(TextBlock("contents"),),
                timestamp=3,
            ),
            AssistantMessage(
                content=(TextBlock("done"),),
                provider="fake",
                model="fake",
                timestamp=4,
            ),
        ]

        valid = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="Be useful.",
            messages=messages,
            compaction=CompactionCheckpoint("summary", 1, 100, created_at=5),
        )
        valid.validate()

        cuts_tool_results = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="Be useful.",
            messages=messages,
            compaction=CompactionCheckpoint("summary", 2, 100, created_at=5),
        )
        with self.assertRaisesRegex(ValueError, "no pending tool call"):
            cuts_tool_results.validate()

        beyond_history = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="Be useful.",
            messages=messages,
            compaction=CompactionCheckpoint("summary", 5, 100, created_at=5),
        )
        with self.assertRaisesRegex(ValueError, "exceeds message count"):
            beyond_history.validate()

    def test_compaction_checkpoint_fields_are_strict(self) -> None:
        for arguments in (
            {"summary": "", "first_kept_message_index": 1, "tokens_before": 1},
            {"summary": "ok", "first_kept_message_index": 0, "tokens_before": 1},
            {"summary": "ok", "first_kept_message_index": 1, "tokens_before": -1},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                CompactionCheckpoint(**arguments)

    def test_file_versions_require_canonical_relative_paths_and_valid_digests(self) -> None:
        version = FileVersion(mtime_ns=-1, size=2, sha256="A" * 64)

        self.assertEqual(version.mtime_ns, -1)
        self.assertEqual(version.sha256, "a" * 64)
        for path in ("", ".", "/absolute.py", "src/../example.py", "src//example.py"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "workspace-relative"):
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
