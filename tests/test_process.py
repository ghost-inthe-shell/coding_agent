import signal
import sys
import tempfile
import unittest
from pathlib import Path

from coding_agent.tools.process import run_limited_process


class ProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cwd = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_python(self, source: str, **kwargs):
        return run_limited_process(
            [sys.executable, "-c", source],
            cwd=self.cwd,
            **kwargs,
        )

    def test_captures_both_streams_exit_code_and_duration(self) -> None:
        output = self.run_python(
            "import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(3)"
        )

        self.assertEqual(output.stdout, "stdout\n")
        self.assertEqual(output.stderr, "stderr\n")
        self.assertEqual(output.returncode, 3)
        self.assertFalse(output.incomplete)
        self.assertFalse(output.timed_out)
        self.assertGreaterEqual(output.duration_ms, 0)

    def test_timeout_kills_the_process_tree_even_when_sigterm_is_ignored(self) -> None:
        child_source = (
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        )
        output = self.run_python(
            "import signal, subprocess, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"child = subprocess.Popen([sys.executable, '-c', {child_source!r}]); "
            "print(child.pid, flush=True); time.sleep(30)",
            timeout_seconds=0.05,
        )

        child_pid = int(output.stdout.strip())
        self.assertTrue(output.timed_out)
        self.assertFalse(output.incomplete)
        self.assertEqual(output.returncode, -signal.SIGKILL)
        self.assertLess(output.duration_ms, 2000)
        self.assertFalse(_process_is_running(child_pid))

    def test_output_limit_stops_capture_and_marks_it_incomplete(self) -> None:
        output = self.run_python(
            "import sys; sys.stdout.write('x' * 1000); sys.stdout.flush()",
            max_output_bytes=100,
        )

        self.assertEqual(len(output.stdout.encode("utf-8")), 100)
        self.assertEqual(output.stderr, "")
        self.assertTrue(output.incomplete)
        self.assertFalse(output.timed_out)

    def test_explicit_environment_is_used_and_stdin_is_closed(self) -> None:
        output = self.run_python(
            "import os, sys; print(os.environ.get('VISIBLE')); print(repr(sys.stdin.read()))",
            env={"VISIBLE": "yes"},
        )

        self.assertEqual(output.stdout, "yes\n''\n")
        self.assertEqual(output.returncode, 0)

    def test_invalid_timeout_is_rejected_before_spawning(self) -> None:
        for timeout in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(
                ValueError, "positive finite"
            ):
                self.run_python("pass", timeout_seconds=timeout)


def _process_is_running(pid: int) -> bool:
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


if __name__ == "__main__":
    unittest.main()
