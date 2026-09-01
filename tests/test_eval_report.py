import json
import unittest

from coding_agent.core.compaction import CompactionCheckpoint
from coding_agent.core.messages import (
    AssistantMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.session import SessionState
from coding_agent.core.types import SessionStatus, StopReason, ToolResultStatus
from coding_agent.core.usage import Usage
from evals.report import build_eval_report, format_eval_report


class EvalReportTests(unittest.TestCase):
    def test_builds_metrics_from_a_stable_session(self) -> None:
        state = SessionState(
            session_id="session-1",
            workspace_root="/workspace",
            system_prompt="System.",
            messages=[
                UserMessage.from_text("work", timestamp=100),
                AssistantMessage(
                    content=(ToolCall(id="call-1", name="read_file"),),
                    provider="provider-a",
                    model="model-a",
                    stop_reason=StopReason.TOOL_USE,
                    timestamp=110,
                ),
                ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read_file",
                    content=(TextBlock("contents"),),
                    timestamp=120,
                ),
                AssistantMessage(
                    content=(
                        ToolCall(id="call-2", name="write_file"),
                        ToolCall(id="call-3", name="run_shell"),
                        ToolCall(id="call-4", name="slow_shell"),
                        ToolCall(id="call-5", name="cancelled_tool"),
                    ),
                    provider="provider-a",
                    model="model-b",
                    stop_reason=StopReason.TOOL_USE,
                    timestamp=130,
                ),
                ToolResultMessage(
                    tool_call_id="call-2",
                    tool_name="write_file",
                    content=(TextBlock("exists"),),
                    status=ToolResultStatus.ERROR,
                    timestamp=140,
                ),
                ToolResultMessage(
                    tool_call_id="call-3",
                    tool_name="run_shell",
                    content=(TextBlock("declined"),),
                    status=ToolResultStatus.DENIED,
                    timestamp=150,
                ),
                ToolResultMessage(
                    tool_call_id="call-4",
                    tool_name="slow_shell",
                    content=(TextBlock("timed out"),),
                    status=ToolResultStatus.TIMEOUT,
                    timestamp=155,
                ),
                ToolResultMessage(
                    tool_call_id="call-5",
                    tool_name="cancelled_tool",
                    content=(TextBlock("cancelled"),),
                    status=ToolResultStatus.CANCELLED,
                    timestamp=158,
                ),
                AssistantMessage(
                    content=(TextBlock("done"),),
                    provider="provider-a",
                    model="model-b",
                    timestamp=160,
                ),
            ],
            compaction=CompactionCheckpoint("summary", 1, 100, created_at=90),
            usage=Usage(
                input_tokens=1000,
                output_tokens=200,
                reasoning_tokens=30,
                cache_read_tokens=400,
                cache_write_tokens=50,
            ),
            status=SessionStatus.IDLE,
        )

        report = build_eval_report("sample_case", state, passed=False)

        self.assertEqual(report.providers, ("provider-a",))
        self.assertEqual(report.models, ("model-a", "model-b"))
        self.assertEqual(report.user_turns, 1)
        self.assertEqual(report.agent_model_calls, 3)
        self.assertEqual(report.tool_calls, 5)
        self.assertEqual(report.tool_results, 5)
        self.assertEqual(report.tool_successes, 1)
        self.assertEqual(report.tool_errors, 1)
        self.assertEqual(report.tool_denied, 1)
        self.assertEqual(report.tool_timeouts, 1)
        self.assertEqual(report.tool_cancelled, 1)
        self.assertEqual(report.conversation_span_ms, 60)
        self.assertTrue(report.compacted)
        self.assertEqual(
            [
                (
                    tool.name,
                    tool.calls,
                    tool.successes,
                    tool.errors,
                    tool.denied,
                    tool.timeouts,
                    tool.cancelled,
                )
                for tool in report.tools
            ],
            [
                ("read_file", 1, 1, 0, 0, 0, 0),
                ("write_file", 1, 0, 1, 0, 0, 0),
                ("run_shell", 1, 0, 0, 1, 0, 0),
                ("slow_shell", 1, 0, 0, 0, 1, 0),
                ("cancelled_tool", 1, 0, 0, 0, 0, 1),
            ],
        )

        document = report.to_dict()
        self.assertEqual(document["verdict"], "FAIL")
        self.assertEqual(document["usage"]["cache_read_tokens"], 400)
        self.assertEqual(document["tools"]["write_file"]["errors"], 1)
        json.dumps(document)

        rendered = format_eval_report(report)
        self.assertIn("Verdict: FAIL", rendered)
        self.assertIn("Model: model-a, model-b", rendered)
        self.assertIn("Tool errors: 1", rendered)
        self.assertIn(
            "write_file      calls=1 success=0 error=1 denied=0 timeout=0 cancelled=0",
            rendered,
        )

    def test_empty_session_has_zero_metrics(self) -> None:
        state = SessionState(
            session_id="empty-session",
            workspace_root="/workspace",
            system_prompt="System.",
        )

        report = build_eval_report("sample_case", state, passed=True)

        self.assertEqual(report.user_turns, 0)
        self.assertEqual(report.agent_model_calls, 0)
        self.assertEqual(report.conversation_span_ms, 0)
        self.assertEqual(report.tools, ())
        self.assertIn("Provider: (none)", format_eval_report(report))
        self.assertIn("  (none)", format_eval_report(report))

    def test_rejects_empty_case_id_or_incomplete_session(self) -> None:
        state = SessionState(
            session_id="session-1",
            workspace_root="/workspace",
            system_prompt="System.",
        )
        with self.assertRaisesRegex(ValueError, "case_id"):
            build_eval_report("", state, passed=True)

        state.messages.append(
            AssistantMessage(
                content=(ToolCall(id="pending", name="read_file"),),
                provider="provider",
                model="model",
                stop_reason=StopReason.TOOL_USE,
            )
        )
        with self.assertRaisesRegex(ValueError, "pending tool calls"):
            build_eval_report("sample_case", state, passed=True)


if __name__ == "__main__":
    unittest.main()
