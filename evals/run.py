"""Prepare and verify the repository's deterministic acceptance cases."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from coding_agent.core.session import SessionState
from coding_agent.core.session_store import SessionStore, SessionStoreError

if __package__:
    from .report import build_eval_report, format_eval_report
else:  # Support the documented ``python evals/run.py`` entry point.
    from report import build_eval_report, format_eval_report

CASES_DIR = Path(__file__).resolve().parent / "cases"
CASE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


class EvalConfigurationError(ValueError):
    """Raised when an evaluation case is incomplete or malformed."""


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    title: str
    category: str
    language: str
    timeout_seconds: int
    path: Path

    @property
    def instruction(self) -> str:
        return (self.path / "instruction.md").read_text(encoding="utf-8").strip()

    @property
    def workspace_template(self) -> Path:
        return self.path / "workspace"

    @property
    def verifier(self) -> Path:
        return self.path / "verify.py"


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    passed: bool
    returncode: int | None
    timed_out: bool
    timeout_seconds: int | None
    stdout: str
    stderr: str

    @property
    def summary(self) -> str:
        if self.timed_out:
            return f"verifier timed out after {self.timeout_seconds}s"
        return f"verifier exited with {self.returncode}"


def load_cases(cases_dir: Path = CASES_DIR) -> dict[str, EvalCase]:
    if not cases_dir.is_dir():
        return {}

    cases: dict[str, EvalCase] = {}
    for case_path in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        case = load_case(case_path)
        if case.id in cases:
            raise EvalConfigurationError(f"duplicate case id: {case.id}")
        cases[case.id] = case
    return cases


def load_case(case_path: Path) -> EvalCase:
    manifest_path = case_path / "case.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalConfigurationError(f"missing manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise EvalConfigurationError(f"invalid JSON in {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise EvalConfigurationError(f"manifest must be an object: {manifest_path}")
    if manifest.get("schema_version") != 1:
        raise EvalConfigurationError(f"unsupported schema_version in {manifest_path}")

    case_id = _required_text(manifest, "id", manifest_path)
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise EvalConfigurationError(f"invalid case id: {case_id!r}")
    if case_id != case_path.name:
        raise EvalConfigurationError(f"case id {case_id!r} must match directory {case_path.name!r}")

    timeout_seconds = manifest.get("timeout_seconds")
    if not isinstance(timeout_seconds, int) or isinstance(timeout_seconds, bool):
        raise EvalConfigurationError(f"timeout_seconds must be an integer: {manifest_path}")
    if timeout_seconds <= 0:
        raise EvalConfigurationError(f"timeout_seconds must be positive: {manifest_path}")

    case = EvalCase(
        id=case_id,
        title=_required_text(manifest, "title", manifest_path),
        category=_required_text(manifest, "category", manifest_path),
        language=_required_text(manifest, "language", manifest_path),
        timeout_seconds=timeout_seconds,
        path=case_path.resolve(),
    )
    _validate_case_files(case)
    return case


def prepare_case(case: EvalCase, output: Path) -> Path:
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        case.workspace_template,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return destination


def verify_case(case: EvalCase, workspace: Path) -> int:
    outcome = evaluate_case(case, workspace)
    _forward_verifier_output(outcome)
    if outcome.passed:
        print(f"PASS {case.id}")
        return 0
    print(f"FAIL {case.id}: {outcome.summary}", file=sys.stderr)
    return 1


def evaluate_case(case: EvalCase, workspace: Path) -> VerificationOutcome:
    """Run the external verifier while capturing output for verify or report."""

    workspace = workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace does not exist: {workspace}")

    try:
        completed = subprocess.run(
            [sys.executable, str(case.verifier), str(workspace)],
            check=False,
            timeout=case.timeout_seconds,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired as exc:
        return VerificationOutcome(
            passed=False,
            returncode=None,
            timed_out=True,
            timeout_seconds=case.timeout_seconds,
            stdout=_timeout_output(exc.stdout),
            stderr=_timeout_output(exc.stderr),
        )
    return VerificationOutcome(
        passed=completed.returncode == 0,
        returncode=completed.returncode,
        timed_out=False,
        timeout_seconds=None,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def report_case(
    case: EvalCase,
    state: SessionState,
    *,
    json_output: bool = False,
) -> int:
    """Verify the saved workspace and print metrics without mutating it or the session."""

    outcome = evaluate_case(case, Path(state.workspace_root))
    report = build_eval_report(case.id, state, passed=outcome.passed)
    if json_output:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_eval_report(report))
    if outcome.passed:
        return 0
    print(f"Verifier: {outcome.summary}", file=sys.stderr)
    _forward_verifier_output(outcome, stdout_stream=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list available cases")

    show = subparsers.add_parser("show", help="print one case instruction")
    show.add_argument("case_id")

    prepare = subparsers.add_parser("prepare", help="copy a clean case workspace")
    prepare.add_argument("case_id")
    prepare.add_argument("output", type=Path)

    verify = subparsers.add_parser("verify", help="run the hidden verifier")
    verify.add_argument("case_id")
    verify.add_argument("workspace", type=Path)

    report = subparsers.add_parser(
        "report",
        help="verify a saved session workspace and report deterministic metrics",
    )
    report.add_argument("case_id")
    report.add_argument("session_id")
    report.add_argument("--json", action="store_true", dest="json_output")

    subparsers.add_parser("check", help="validate all case manifests and files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_cases()
        if args.command == "check":
            print(f"OK: {len(cases)} case(s)")
            return 0
        if args.command == "list":
            for case in cases.values():
                print(f"{case.id}\t{case.category}\t{case.language}\t{case.title}")
            return 0

        case = cases.get(args.case_id)
        if case is None:
            available = ", ".join(cases) or "none"
            raise EvalConfigurationError(f"unknown case {args.case_id!r}; available: {available}")
        if args.command == "show":
            print(case.instruction)
            return 0
        if args.command == "prepare":
            destination = prepare_case(case, args.output)
            print(f"Workspace: {destination}")
            print("\nInstruction:\n")
            print(case.instruction)
            return 0
        if args.command == "verify":
            return verify_case(case, args.workspace)
        if args.command == "report":
            state = SessionStore().load(args.session_id)
            return report_case(case, state, json_output=args.json_output)
    except (
        EvalConfigurationError,
        FileExistsError,
        FileNotFoundError,
        SessionStoreError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    raise AssertionError(f"unhandled command: {args.command}")


def _required_text(manifest: dict[str, object], key: str, path: Path) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalConfigurationError(f"{key} must be non-empty text: {path}")
    return value.strip()


def _validate_case_files(case: EvalCase) -> None:
    if not case.instruction:
        raise EvalConfigurationError(f"instruction must not be empty: {case.path}")
    if not case.workspace_template.is_dir():
        raise EvalConfigurationError(f"missing workspace directory: {case.path}")
    if not case.verifier.is_file():
        raise EvalConfigurationError(f"missing verifier: {case.verifier}")


def _forward_verifier_output(
    outcome: VerificationOutcome,
    *,
    stdout_stream: TextIO | None = None,
) -> None:
    stdout_stream = stdout_stream or sys.stdout
    if outcome.stdout:
        print(outcome.stdout, end="" if outcome.stdout.endswith("\n") else "\n", file=stdout_stream)
    if outcome.stderr:
        print(outcome.stderr, end="" if outcome.stderr.endswith("\n") else "\n", file=sys.stderr)


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
