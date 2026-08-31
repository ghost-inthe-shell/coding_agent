import unittest

from coding_agent.context import ContextBudget, estimate_request_tokens, estimate_text_tokens
from coding_agent.core.messages import UserMessage
from coding_agent.tools.base import ToolSpec


class ContextBudgetTests(unittest.TestCase):
    def test_threshold_reserves_ratio_output_and_safety_space(self) -> None:
        large = ContextBudget(context_window=128_000, max_output_tokens=16_384)
        small = ContextBudget(context_window=32_000, max_output_tokens=16_384)

        self.assertEqual(large.safety_margin, 2_560)
        self.assertEqual(large.compact_threshold, 102_400)
        self.assertEqual(large.keep_recent_tokens, 20_000)
        self.assertEqual(small.safety_margin, 1_024)
        self.assertEqual(small.compact_threshold, 14_592)
        self.assertEqual(small.keep_recent_tokens, 8_000)
        self.assertFalse(small.should_compact(14_591))
        self.assertTrue(small.should_compact(14_592))

    def test_budget_rejects_an_impossible_output_reserve(self) -> None:
        with self.assertRaisesRegex(ValueError, "safety margin"):
            ContextBudget(context_window=16_000, max_output_tokens=15_000)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            ContextBudget().should_compact(-1)

    def test_estimator_counts_non_ascii_more_conservatively_than_ascii(self) -> None:
        self.assertEqual(estimate_text_tokens("abcdefgh"), 2)
        self.assertEqual(estimate_text_tokens("中文测试"), 4)
        self.assertEqual(estimate_text_tokens(""), 0)

    def test_request_estimate_includes_prompt_messages_and_tool_schema(self) -> None:
        message = UserMessage.from_text("inspect src/main.py", timestamp=1)
        tool = ToolSpec(
            name="read_file",
            description="Read one file.",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        baseline = estimate_request_tokens("system", (), ())
        with_message = estimate_request_tokens("system", (message,), ())
        with_tool = estimate_request_tokens("system", (message,), (tool,))

        self.assertGreater(with_message, baseline)
        self.assertGreater(with_tool, with_message)


if __name__ == "__main__":
    unittest.main()
