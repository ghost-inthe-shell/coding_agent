import unittest

from coding_agent.context import (
    CompactionPlan,
    ContextBudget,
    ContextCompactionError,
    active_messages,
    checkpoint_from_summary,
    find_compaction_cut,
    prepare_compaction,
)
from coding_agent.core.compaction import CompactionCheckpoint
from coding_agent.core.messages import (
    AssistantMessage,
    TextBlock,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from coding_agent.core.session import SessionState
from coding_agent.core.types import StopReason


def assistant_text(text: str, *, stop_reason: StopReason = StopReason.STOP) -> AssistantMessage:
    return AssistantMessage(
        content=(TextBlock(text),),
        provider="fake",
        model="fake",
        stop_reason=stop_reason,
        timestamp=20,
    )


class CompactionTests(unittest.TestCase):
    def state(self, messages) -> SessionState:
        return SessionState(
            session_id="session-1",
            workspace_root="/tmp/workspace",
            system_prompt="System prompt.",
            messages=list(messages),
        )

    def test_active_view_includes_summary_without_mutating_full_history(self) -> None:
        state = self.state(
            (
                UserMessage.from_text("old question", timestamp=1),
                assistant_text("old answer"),
                UserMessage.from_text("recent question", timestamp=3),
            )
        )
        state.compaction = CompactionCheckpoint("old work summary", 2, 100, created_at=4)

        visible = active_messages(state)

        self.assertEqual(len(state.messages), 3)
        self.assertEqual(len(visible), 1)
        self.assertIsInstance(visible[0], UserMessage)
        rendered = "".join(block.text for block in visible[0].content)
        self.assertIn("old work summary", rendered)
        self.assertIn("recent question", rendered)

    def test_cut_never_splits_assistant_tool_calls_from_results(self) -> None:
        call = ToolCall(id="call-1", name="read_file")
        messages = (
            UserMessage.from_text("inspect", timestamp=1),
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
            assistant_text("done"),
        )

        cut = find_compaction_cut(messages, 0, keep_recent_tokens=1)

        self.assertEqual(cut, 3)

    def test_rolling_plan_uses_previous_summary_and_only_newly_evicted_messages(self) -> None:
        state = self.state(
            (
                UserMessage.from_text("already summarized raw text", timestamp=1),
                assistant_text("already summarized answer"),
                UserMessage.from_text("new segment", timestamp=3),
                assistant_text("recent answer"),
            )
        )
        state.compaction = CompactionCheckpoint("previous summary", 2, 100, created_at=4)

        plan = prepare_compaction(state, ContextBudget(), (), force=True)

        self.assertIsNotNone(plan)
        assert plan is not None
        rendered = "\n".join(
            block.text
            for message in plan.messages
            if isinstance(message, UserMessage)
            for block in message.content
        )
        self.assertIn("previous summary", rendered)
        self.assertIn("new segment", rendered)
        self.assertNotIn("already summarized raw text", rendered)
        self.assertEqual(plan.first_kept_message_index, 4)
        self.assertEqual(plan.summarized_message_count, 2)

    def test_summary_response_must_be_complete_text_without_tools(self) -> None:
        plan = CompactionPlan((), 2, 500, 2)

        checkpoint = checkpoint_from_summary(plan, assistant_text("new summary"))
        self.assertEqual(checkpoint.summary, "new summary")
        self.assertEqual(checkpoint.first_kept_message_index, 2)

        invalid_responses = (
            assistant_text("partial", stop_reason=StopReason.LENGTH),
            AssistantMessage(
                content=(ToolCall(id="call-1", name="read_file"),),
                provider="fake",
                model="fake",
                stop_reason=StopReason.TOOL_USE,
            ),
            assistant_text("   "),
        )
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(ContextCompactionError):
                checkpoint_from_summary(plan, response)


if __name__ == "__main__":
    unittest.main()
