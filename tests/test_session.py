import json
import unittest

from coding_agent.core.messages import (
    AssistantMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.session import FileVersion, SessionState
from coding_agent.core.types import SessionStatus, StopReason
from coding_agent.core.usage import Usage


class SessionStateTests(unittest.TestCase):
    def test_session_round_trip_is_provider_independent(self) -> None:
        state = SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
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
            usage=Usage(input_tokens=20, output_tokens=4),
            read_file_versions={
                "README.md": FileVersion(
                    mtime_ns=123,
                    size=42,
                    sha256="a" * 64,
                )
            },
            modified_files=["src/example.py"],
            status=SessionStatus.IDLE,
            created_at=1,
            updated_at=2,
        )

        encoded = json.loads(json.dumps(state.to_dict()))
        restored = SessionState.from_dict(encoded)

        self.assertEqual(restored, state)
        restored.validate()

    def test_file_version_requires_a_sha256_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "sha256"):
            FileVersion(mtime_ns=1, size=1, sha256="not-a-digest")


if __name__ == "__main__":
    unittest.main()

