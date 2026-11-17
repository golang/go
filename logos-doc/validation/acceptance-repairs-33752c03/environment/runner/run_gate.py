"""Record a bounded acceptance-repair gate without changing its input files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import cast

ROOT = Path("/workspace/scratch/9949907ca71e")
sys.path.insert(0, str(ROOT / "recovery-evidence/parity-timeout-investigation"))
from record_isolated import binding, digest, stop_descendants, write_json


@dataclass(frozen=True)
class Gate:
    command: tuple[str, ...]
    cwd: Path
    output: Path
    environment: dict[str, str]
    files: tuple[Path, ...]
    repositories: tuple[Path, ...]
    expected_exit: int
    required_markers: tuple[str, ...]
    forbidden_markers: tuple[str, ...]
    timeout_seconds: int
    boundary: str


def strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected a list of strings")
    return tuple(cast(list[str], value))


def load_gate(path: Path) -> Gate:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a gate object")
    data = cast(dict[str, object], value)
    environment = data["environment"]
    if not isinstance(environment, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()):
        raise ValueError("Expected string environment overrides")
    for key in ("cwd", "output", "boundary"):
        if not isinstance(data[key], str):
            raise ValueError(f"Expected string {key}")
    for key in ("expected_exit", "timeout_seconds"):
        if type(data[key]) is not int:
            raise ValueError(f"Expected integer {key}")
    return Gate(
        command=strings(data["command"]), cwd=Path(cast(str, data["cwd"])),
        output=Path(cast(str, data["output"])),
        environment=cast(dict[str, str], environment),
        files=tuple(Path(p) for p in strings(data["files"])),
        repositories=tuple(Path(p) for p in strings(data["repositories"])),
        expected_exit=cast(int, data["expected_exit"]),
        required_markers=strings(data["required_markers"]),
        forbidden_markers=strings(data["forbidden_markers"]),
        timeout_seconds=cast(int, data["timeout_seconds"]),
        boundary=cast(str, data["boundary"]),
    )


def repository_state(path: Path) -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=path, text=True)
    return {
        "head": git("rev-parse", "HEAD").strip(),
        "source_tree": git("rev-parse", "HEAD:src").strip(),
        "status": git("status", "--porcelain=v1"),
        "tracked_diff": git("diff", "HEAD", "--"),
    }


def main() -> int:
    spec = Path(sys.argv[1]).resolve()
    gate = load_gate(spec)
    before = binding()
    frozen = {str(path): digest(path) for path in gate.files}
    states = {str(path): repository_state(path) for path in gate.repositories}
    gate.output.mkdir()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GO", "CGO_", "BOOT_GO_", "LOGOS_"))}
    environment.update(gate.environment)
    write_json(gate.output / "inputs.json", {
        "binding": before, "files_sha256": frozen, "repositories": states,
        "command": gate.command, "cwd": str(gate.cwd),
        "environment_overrides": gate.environment, "boundary": gate.boundary,
        "expected_exit": gate.expected_exit, "required_markers": gate.required_markers,
        "forbidden_markers": gate.forbidden_markers,
        "outer_timeout_seconds": gate.timeout_seconds,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": digest(spec), "runner_sha256": digest(Path(__file__)),
    })
    started = time.monotonic()
    interrupted = False
    with (gate.output / "stdout.log").open("xb") as stdout, (gate.output / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(gate.command, cwd=gate.cwd, env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while True:
                remaining = gate.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(gate.command, gate.timeout_seconds)
                try:
                    code = process.wait(timeout=min(25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(json.dumps({"gate": gate.output.name, "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            interrupted = True
            stop_descendants(process)
            assert process.returncode is not None
            code = process.returncode
    output = (gate.output / "stdout.log").read_text() + (gate.output / "stderr.log").read_text()
    markers = {marker: marker in output for marker in gate.required_markers}
    forbidden = {marker: marker in output for marker in gate.forbidden_markers}
    unchanged = all(digest(Path(path)) == sha for path, sha in frozen.items())
    states_after = {str(path): repository_state(path) for path in gate.repositories}
    matched = (code == gate.expected_exit and not interrupted and unchanged and
               states == states_after and all(markers.values()) and not any(forbidden.values()))
    result = {
        "exit_code": code, "interrupted_or_outer_timeout": interrupted,
        "expected_outcome_matched": matched, "required_markers": markers,
        "forbidden_markers": forbidden, "frozen_files_unchanged": unchanged,
        "repository_states_unchanged": states == states_after,
        "duration_seconds": round(time.monotonic() - started, 3),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_sha256": digest(gate.output / "stdout.log"),
        "stderr_sha256": digest(gate.output / "stderr.log"), "binding_after": binding(),
    }
    write_json(gate.output / "result.json", result)
    print(json.dumps({"gate": gate.output.name, **result}), flush=True)
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
