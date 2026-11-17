"""Record an installed gate against the frozen combined acceptance repair."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import cast

ROOT = Path("/workspace/scratch/9949907ca71e")
REPO = ROOT / "logos-acceptance-fixes"
EVIDENCE = ROOT / "recovery-evidence"
EXPECTED_SOURCE = "33752c03819ed5afcb37f87633a3cac668e181a8"
sys.path.insert(0, str(EVIDENCE / "acceptance-failure-repairs"))
from run_gate import load_gate, repository_state
sys.path.insert(0, str(EVIDENCE / "parity-timeout-investigation"))
from record_isolated import stop_descendants


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast(dict[str, object], value)


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def binding() -> dict[str, object]:
    build_path = EVIDENCE / "fresh-make-repair-33752c03-002/result.json"
    build = read_object(build_path)
    if build["make_exit_code"] != 0 or build["source_before"] != build["source_after"]:
        raise ValueError("The fresh combined build did not complete on unchanged source")
    built_source = build["source_after"]
    if not isinstance(built_source, dict) or built_source.get("src_tree") != EXPECTED_SOURCE:
        raise ValueError("The build receipt does not identify the expected source tree")
    if build["postbuild_cache_still_empty"] is not True:
        raise ValueError("The postbuild cache was used before the fresh build completed")
    tree = git("write-tree")
    source = git("rev-parse", tree + ":src")
    if source != EXPECTED_SOURCE or git("diff", "--name-only", "--", "src"):
        raise ValueError("Installed candidate source differs from the frozen index")
    installed = build["installed_binary_sha256"]
    if not isinstance(installed, dict):
        raise ValueError("Invalid installed tool manifest")
    hashes: dict[str, str] = {}
    for name, expected in installed.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("Invalid installed tool entry")
        actual = digest(REPO / name)
        if actual != expected:
            raise ValueError(f"Installed tool changed: {name}")
        hashes[name] = actual
    integration_path = EVIDENCE / "acceptance-failure-repairs/integration-source-001.json"
    integration = read_object(integration_path)
    groups = integration["groups"]
    if not isinstance(groups, dict):
        raise ValueError("Invalid integration groups")
    for group in groups.values():
        if not isinstance(group, dict) or not isinstance(group.get("files_sha256"), dict):
            raise ValueError("Invalid integration file group")
        for name, expected in group["files_sha256"].items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise ValueError("Invalid integration file entry")
            actual = digest(REPO / name)
            if actual != expected:
                raise ValueError(f"Frozen repair file changed: {name}")
            hashes[name] = actual
    return {
        "head": git("rev-parse", "HEAD"), "index_tree": tree,
        "source_tree": source, "status_porcelain": git("status", "--porcelain=v1"),
        "build_result_sha256": digest(build_path),
        "integration_manifest_sha256": digest(integration_path), "sha256": hashes,
        "boundary": "Fresh installed combined source snapshot; no clean published-tip claim",
    }


def resources() -> dict[str, int]:
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.split(":")[0] in ("MemAvailable", "Shmem")}
    disk = os.statvfs(ROOT)
    oom = next(int(line.split()[1]) for line in Path("/proc/vmstat").read_text().splitlines()
               if line.startswith("oom_kill "))
    return {**memory, "workspace_available_bytes": disk.f_bavail * disk.f_frsize,
            "global_oom_kill_counter": oom}


def main() -> int:
    spec = Path(sys.argv[1]).resolve()
    gate = load_gate(spec)
    spec_record = read_object(spec)
    output_names = spec_record.get("produced_files", [])
    if not isinstance(output_names, list) or not all(isinstance(name, str) for name in output_names):
        raise ValueError("Expected a list of produced-file paths")
    produced_files = tuple(Path(name) for name in cast(list[str], output_names))
    if any(path.exists() for path in produced_files):
        raise ValueError("A recorded build output already exists")
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
        "expected_produced_files": [str(path) for path in produced_files],
        "dependencies_sha256": {
            "run_gate.py": digest(EVIDENCE / "acceptance-failure-repairs/run_gate.py"),
            "record_isolated.py": digest(EVIDENCE / "parity-timeout-investigation/record_isolated.py"),
        },
        "resources_before": resources(),
    })
    started = time.monotonic()
    interrupted = False
    with (gate.output / "stdout.log").open("xb") as stdout, \
            (gate.output / "stderr.log").open("xb") as stderr, \
            (gate.output / "resources.jsonl").open("x") as samples:
        process = subprocess.Popen(gate.command, cwd=gate.cwd, env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while True:
                elapsed = time.monotonic() - started
                sample = {"elapsed_seconds": round(elapsed, 3), **resources()}
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                remaining = gate.timeout_seconds - elapsed
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(gate.command, gate.timeout_seconds)
                try:
                    code = process.wait(timeout=min(25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(json.dumps({"gate": gate.output.name, **sample}), flush=True)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            interrupted = True
            stop_descendants(process)
            assert process.returncode is not None
            code = process.returncode
    output = (gate.output / "stdout.log").read_bytes() + (gate.output / "stderr.log").read_bytes()
    markers = {marker: marker.encode() in output for marker in gate.required_markers}
    forbidden = {marker: marker.encode() in output for marker in gate.forbidden_markers}
    unchanged = all(digest(Path(path)) == sha for path, sha in frozen.items())
    states_after = {str(path): repository_state(path) for path in gate.repositories}
    after = binding()
    produced = {str(path): digest(path) for path in produced_files if path.is_file()}
    matched = (code == gate.expected_exit and not interrupted and unchanged and
               states == states_after and before == after and
               len(produced) == len(produced_files) and
               all(markers.values()) and not any(forbidden.values()))
    result = {
        "exit_code": code, "interrupted_or_outer_timeout": interrupted,
        "expected_outcome_matched": matched, "required_markers": markers,
        "forbidden_markers": forbidden, "frozen_files_unchanged": unchanged,
        "repository_states_unchanged": states == states_after,
        "installed_binding_unchanged": before == after,
        "duration_seconds": round(time.monotonic() - started, 3),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_sha256": digest(gate.output / "stdout.log"),
        "stderr_sha256": digest(gate.output / "stderr.log"),
        "produced_files_sha256": produced,
        "resources_after": resources(), "binding_after": after,
    }
    write_json(gate.output / "result.json", result)
    print(json.dumps({"gate": gate.output.name, "exit_code": code,
                      "matched": matched, "duration_seconds": result["duration_seconds"],
                      "missing_markers": [key for key, present in markers.items() if not present],
                      "forbidden_markers_found": [key for key, present in forbidden.items() if present]}), flush=True)
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
