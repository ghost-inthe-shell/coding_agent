import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from coding_agent.core.session import SessionState
from coding_agent.core.session_store import SessionNotFoundError
from evals import run as eval_run
from evals.run import (
    EvalConfigurationError,
    evaluate_case,
    load_case,
    prepare_case,
    report_case,
    verify_case,
)


class EvalRunnerTests(unittest.TestCase):
    def test_documented_script_entrypoint_can_load_report_command(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [sys.executable, str(project_root / "evals" / "run.py"), "report", "--help"],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("case_id", completed.stdout)
        self.assertIn("session_id", completed.stdout)
        self.assertIn("--json", completed.stdout)

    def test_load_prepare_and_verify_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_path = self._write_case(root)
            case = load_case(case_path)
            destination = root / "prepared"

            prepared = prepare_case(case, destination)

            self.assertEqual(case.id, "sample_case")
            self.assertEqual(case.instruction, "Fix the sample.")
            self.assertEqual((prepared / "value.txt").read_text(), "broken\n")
            self.assertEqual(verify_case(case, prepared), 0)

    def test_prepare_ignores_python_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case = load_case(self._write_case(root))
            cache = case.workspace_template / "__pycache__"
            cache.mkdir()
            (cache / "module.cpython-310.pyc").write_bytes(b"cache")
            (case.workspace_template / "stray.pyc").write_bytes(b"cache")
            (case.workspace_template / "stray.pyo").write_bytes(b"cache")

            prepared = prepare_case(case, root / "prepared")

            self.assertFalse((prepared / "__pycache__").exists())
            self.assertFalse((prepared / "stray.pyc").exists())
            self.assertFalse((prepared / "stray.pyo").exists())

    def test_prepare_refuses_to_overwrite_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case = load_case(self._write_case(root))
            destination = root / "prepared"
            destination.mkdir()

            with self.assertRaises(FileExistsError):
                prepare_case(case, destination)

    def test_case_id_must_match_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case_path = self._write_case(root)
            manifest_path = case_path / "case.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["id"] = "another_id"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaises(EvalConfigurationError):
                load_case(case_path)

    def test_report_command_loads_session_and_emits_only_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case = load_case(self._write_case(root))
            state = SessionState(
                session_id="session-1",
                workspace_root=str(case.workspace_template),
                system_prompt="System.",
            )
            store = Mock()
            store.load.return_value = state
            output = StringIO()
            errors = StringIO()

            with (
                patch.object(eval_run, "load_cases", return_value={case.id: case}),
                patch.object(eval_run, "SessionStore", return_value=store),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                exit_code = eval_run.main(["report", "sample_case", "session-1", "--json"])

            self.assertEqual(exit_code, 0)
            store.load.assert_called_once_with("session-1")
            document = json.loads(output.getvalue())
            self.assertEqual(document["verdict"], "PASS")
            self.assertEqual(document["session_id"], "session-1")
            self.assertEqual(errors.getvalue(), "")

    def test_failed_report_keeps_metrics_on_stdout_and_diagnostics_on_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case = load_case(self._write_case(root))
            failed_workspace = root / "failed-workspace"
            failed_workspace.mkdir()
            state = SessionState(
                session_id="failed-session",
                workspace_root=str(failed_workspace),
                system_prompt="System.",
            )
            output = StringIO()
            errors = StringIO()

            with redirect_stdout(output), redirect_stderr(errors):
                exit_code = report_case(case, state)

            self.assertEqual(exit_code, 1)
            self.assertIn("Verdict: FAIL", output.getvalue())
            self.assertIn("Verifier: verifier exited with 1", errors.getvalue())

    def test_verifier_timeout_is_a_structured_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            case = load_case(self._write_case(Path(temporary_directory)))
            with patch.object(
                eval_run.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["verify"],
                    timeout=case.timeout_seconds,
                    output=b"partial stdout",
                    stderr=b"partial stderr",
                ),
            ):
                outcome = evaluate_case(case, case.workspace_template)

            self.assertFalse(outcome.passed)
            self.assertTrue(outcome.timed_out)
            self.assertEqual(
                outcome.summary,
                f"verifier timed out after {case.timeout_seconds}s",
            )
            self.assertEqual(outcome.stdout, "partial stdout")
            self.assertEqual(outcome.stderr, "partial stderr")

    def test_report_command_reports_missing_session_as_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            case = load_case(self._write_case(Path(temporary_directory)))
            store = Mock()
            store.load.side_effect = SessionNotFoundError("session not found: missing")
            errors = StringIO()

            with (
                patch.object(eval_run, "load_cases", return_value={case.id: case}),
                patch.object(eval_run, "SessionStore", return_value=store),
                redirect_stderr(errors),
            ):
                exit_code = eval_run.main(["report", "sample_case", "missing"])

            self.assertEqual(exit_code, 2)
            self.assertIn("session not found: missing", errors.getvalue())

    def _write_case(self, root: Path) -> Path:
        case_path = root / "sample_case"
        workspace = case_path / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "value.txt").write_text("broken\n")
        (case_path / "instruction.md").write_text("Fix the sample.\n")
        (case_path / "case.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": "sample_case",
                    "title": "Sample",
                    "category": "bugfix",
                    "language": "text",
                    "timeout_seconds": 10,
                }
            )
        )
        (case_path / "verify.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "raise SystemExit(0 if (Path(sys.argv[1]) / 'value.txt').is_file() else 1)\n"
        )
        return case_path


if __name__ == "__main__":
    unittest.main()
