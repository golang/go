# pyright: strict
"""Require every frozen Node-backed Wasm test to run and pass under the CI env."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import NoReturn, cast

DIRECTORY = "logos-doc/validation/upstream-host"
WRAPPER = ".github/scripts/logos-test-env.sh"
PACKAGE = "cmd/compile/internal/logos"
MANIFEST_SHA256 = "9d6314cef3d9f2156c8ebd1dc0b413afd0405b01517c0b8a13a04773ffe545ea"
NAMES = frozenset(
    (
        "TestWasmProgramRunsUnderNode",
        "TestWasmProgramRunsUnderNode/resume_points_everywhere",
        "TestWasmProgramRunsUnderNode/selective_resume_points",
        "TestWasmWrongMayYieldTraps",
        "TestWasmYieldFreeSourcesRunUnderNode",
        "TestWasmYieldFreeSourcesRunUnderNode/plain_go",
        "TestWasmYieldFreeSourcesRunUnderNode/logos",
    )
)


def reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-JSON numeric constant {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return cast(dict[str, object], value)


def json_object(text: str) -> dict[str, object]:
    return mapping(
        json.loads(
            text, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
    )


def string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected JSON string")
    return value


def accept(stdout: str, stderr: bytes, exit_code: int) -> None:
    if exit_code != 0 or stderr:
        raise ValueError("Wasm gate requires exit zero and empty stderr")
    runs: Counter[str] = Counter()
    passes: Counter[str] = Counter()
    active: set[str] = set()
    output: list[str] = []
    started, finished = False, False
    for line in stdout.splitlines():
        event = json_object(line)
        if finished:
            raise ValueError("event after package completion")
        action = string(event.get("Action"))
        if action in ("fail", "skip", "build-fail") or "FailedBuild" in event:
            raise ValueError("failed, skipped, or unbuilt Wasm test")
        if event.get("OutputType") in ("error", "error-continue"):
            raise ValueError("test error output in Wasm evidence")
        if "Output" in event:
            output.append(string(event["Output"]))
            if action not in ("output", "build-output"):
                raise ValueError("unexpected output field on a non-output action")
        if action == "build-output":
            string(event.get("ImportPath"))
            string(event.get("Output"))
            if "Test" in event:
                raise ValueError("build output has a Test field")
            continue
        if event.get("Package") != PACKAGE:
            raise ValueError("unexpected package in Wasm events")
        if "Test" in event:
            name = string(event["Test"])
            if name not in NAMES or not started or finished:
                raise ValueError("unexpected test identity or package lifetime")
            if action == "run":
                parent = name.rpartition("/")[0]
                if runs[name] or (parent in NAMES and parent not in active):
                    raise ValueError("duplicate run or child outside its parent")
                runs[name] += 1
                active.add(name)
            elif action == "pass":
                if name not in active or any(
                    child.startswith(name + "/") for child in active
                ):
                    raise ValueError("pass without a completed active test")
                active.remove(name)
                passes[name] += 1
            elif action == "output":
                string(event.get("Output"))
                if name not in active:
                    raise ValueError("test output outside its active lifetime")
            else:
                raise ValueError(f"unexpected test action {action!r}")
        elif action == "start":
            if started or finished:
                raise ValueError("duplicate package start")
            started = True
        elif action == "pass":
            if not started or finished or active:
                raise ValueError("package passed outside its completed lifetime")
            finished = True
        elif action == "output":
            string(event.get("Output"))
            if not started:
                raise ValueError("package output before its start")
        else:
            raise ValueError(f"unexpected package action {action!r}")
    if (
        not started
        or not finished
        or runs != Counter(NAMES)
        or passes != Counter(NAMES)
    ):
        raise ValueError("missing exact seven named runs/passes and package completion")
    if re.search(
        r"\b(?:panic:|fatal error:|signal:|SIG[A-Z0-9]+:|internal compiler error:|test timed out)",
        "".join(output),
    ):
        raise ValueError("crash or timeout diagnostic in Wasm output")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rooted(repo: Path, relative: str) -> Path:
    path = (repo / relative).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(repo):
        raise ValueError(f"input outside repository: {relative}")
    return path


def verify_frozen(repo: Path) -> tuple[str, ...]:
    manifest = rooted(repo, f"{DIRECTORY}/frozen.json")
    if digest(manifest) != MANIFEST_SHA256:
        raise ValueError("frozen manifest digest mismatch")
    data = json_object(manifest.read_text(encoding="utf-8"))
    entries = data["files"]
    if not isinstance(entries, list):
        raise TypeError("frozen files must be a list")
    paths = [f"{DIRECTORY}/frozen.json"]
    for value in cast(list[object], entries):
        entry = mapping(value)
        path = string(entry["path"])
        if digest(rooted(repo, path)) != entry["sha256"]:
            raise ValueError(f"frozen source digest mismatch: {path}")
        paths.append(path)
    return tuple(paths)


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repo), *args), text=True, timeout=30
    ).strip()


def identity(repo: Path, paths: tuple[str, ...], node: Path) -> dict[str, object]:
    return {
        "head": git_value(repo, "rev-parse", "HEAD"),
        "tree": git_value(repo, "rev-parse", "HEAD^{tree}"),
        "source_tree": git_value(repo, "rev-parse", "HEAD:src"),
        "tracked_status": git_value(
            repo, "status", "--porcelain", "--untracked-files=no"
        ),
        "node_path": str(node),
        "node_sha256": digest(node),
        "sha256": {
            path: digest(rooted(repo, path))
            for path in (
                *paths,
                WRAPPER,
                f"{DIRECTORY}/check_wasm.py",
                "bin/go",
                "bin/logos",
                "pkg/tool/linux_amd64/compile",
                "pkg/tool/linux_amd64/link",
            )
        },
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def compact_environment(repo: Path) -> dict[str, str]:
    raw = subprocess.check_output(
        (str(repo / WRAPPER), "/usr/bin/env", "-0"), timeout=10
    )
    result: dict[str, str] = {}
    for entry in raw.decode("utf-8").split("\0"):
        if not entry:
            continue
        key, separator, value = entry.partition("=")
        if not separator or not key or key in result:
            raise ValueError("malformed compact environment receipt")
        result[key] = value
    for key in (
        "HOME",
        "USER",
        "PATH",
        "TMPDIR",
        "GOEXPERIMENT",
        "GO_BUILDER_NAME",
        "UPROOT1272",
        "UPROOT1273",
        "UPROOT1274",
    ):
        if (key in result) != (key in os.environ) or result.get(key) != os.environ.get(
            key
        ):
            raise ValueError(f"compact environment changed {key} value or presence")
    return result


def capture(
    command: tuple[str, ...], repo: Path, out: Path, timeout: float = 900
) -> dict[str, object]:
    stdout, stderr = out / "wasm.jsonl", out / "wasm.stderr.txt"
    result: dict[str, object] = {
        "command": command,
        "cwd": str(repo / "src"),
        "accepted": False,
        "exit_code": None,
        "timed_out": False,
    }
    started = time.monotonic()
    try:
        with (
            stdout.open("xb") as output,
            stderr.open("xb") as errors,
            subprocess.Popen(
                command,
                cwd=repo / "src",
                stdout=output,
                stderr=errors,
                start_new_session=True,
            ) as process,
        ):
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
        result["exit_code"] = exit_code
        if result["timed_out"] or any(
            path.stat().st_size > 16 * 1024 * 1024 for path in (stdout, stderr)
        ):
            raise ValueError("Wasm command exceeded its time or event-log limit")
        accept(stdout.read_text(encoding="utf-8"), stderr.read_bytes(), exit_code)
        result["accepted"] = True
    except (OSError, TypeError, ValueError) as error:
        result["rejection"] = str(error)
    result.update(
        elapsed_seconds=time.monotonic() - started,
        stdout_sha256=digest(stdout),
        stderr_sha256=digest(stderr),
    )
    write_json(out / "wasm.status.json", result)
    return result


def run_gate(repo: Path, out: Path) -> bool:
    report: dict[str, object] = {
        "accepted": False,
        "expected_named_tests": sorted(NAMES),
        "expected_leaf_cases": 5,
    }
    before: dict[str, object] | None = None
    paths: tuple[str, ...] = ()
    node: Path | None = None
    result: dict[str, object] = {"accepted": False}
    try:
        paths = verify_frozen(repo)
        if git_value(repo, "rev-parse", "--show-toplevel") != str(repo):
            raise ValueError("--repo must be the repository root")
        env = compact_environment(repo)
        if env.get("GOROOT") not in (None, "", str(repo)):
            raise ValueError("GOROOT does not identify the requested installed source")
        node_name = shutil.which("node", path=env.get("PATH", ""))
        if node_name is None:
            raise ValueError("Node is required; a missing host is not a passing skip")
        node = Path(node_name).resolve(strict=True)
        before = identity(repo, paths, node)
        write_json(out / "before.json", before)
        if before["tracked_status"]:
            raise ValueError("tracked checkout files are modified")
        write_json(
            out / "environment.json",
            {
                "names": sorted(env),
                "environment_only_staging_bytes": sum(
                    (len(f"{key}={value}".encode()) + 8) // 8 * 8 + 8
                    for key, value in env.items()
                ),
                "wasm_argv_env_capacity_bytes": 8192,
                "controls": {
                    key: {"present": key in env, "value": env.get(key)}
                    for key in (
                        "GOEXPERIMENT",
                        "GO_BUILDER_NAME",
                        "GO_TEST_TIMEOUT_SCALE",
                    )
                },
            },
        )
        roots = sorted(name for name in NAMES if "/" not in name)
        command = (
            str(repo / WRAPPER),
            str(repo / "bin/go"),
            "test",
            "-json",
            "-count=1",
            "-short",
            "-timeout=10m",
            "-run",
            "^(" + "|".join(roots) + ")$",
            PACKAGE,
        )
        result = capture(command, repo, out)
        report["test_result"] = result
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        report["rejection"] = str(error)
    finally:
        if before is not None and node is not None:
            try:
                after = identity(repo, paths, node)
                write_json(out / "after.json", after)
                stable = before == after
                report["source_and_tools_unchanged"] = stable
                report["accepted"] = (
                    stable and result["accepted"] is True and "rejection" not in report
                )
            except (
                OSError,
                TypeError,
                ValueError,
                subprocess.SubprocessError,
            ) as error:
                report["after_rejection"] = str(error)
        write_json(out / "report.json", report)
    return report["accepted"] is True


class Arguments(argparse.Namespace):
    repo: Path
    evidence: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(namespace=Arguments())
    repo, out = args.repo.resolve(strict=True), args.evidence.resolve()
    if out.is_relative_to(repo) or repo.is_relative_to(out):
        raise ValueError("evidence must be a new directory outside the repository")
    if Path(__file__).resolve() != rooted(repo, f"{DIRECTORY}/check_wasm.py"):
        raise ValueError("run the checker from the requested source checkout")
    out.mkdir(parents=True, exist_ok=False)
    return 0 if run_gate(repo, out) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        print(f"Wasm evidence rejected: {error}", file=sys.stderr)
        raise SystemExit(1) from error
