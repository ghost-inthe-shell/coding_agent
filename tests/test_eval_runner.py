import json
import tempfile
import unittest
from pathlib import Path

from evals.run import EvalConfigurationError, load_case, prepare_case, verify_case


class EvalRunnerTests(unittest.TestCase):
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
