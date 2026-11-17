"""Record a controlled at08 fan-out comparison and its resource envelope."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import cast


ROOT = Path("/workspace/scratch/9949907ca71e")
HERE = Path(__file__).resolve().parent
REPO = ROOT / "logos-final-acceptance"
OUT = HERE / "fanout-001"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return cast(dict[str, object], value)


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def binding() -> dict[str, object]:
    expected = {
        "HEAD": "e96a570da271c43a80e23bb1a188aac054acff25",
        "HEAD:src": "e237c399aae8f8065b724e8a0e27585686c7da18",
    }
    for query, value in expected.items():
        actual = subprocess.check_output(["git", "rev-parse", query], cwd=REPO, text=True).strip()
        if actual != value:
            raise ValueError(f"Source identity changed: {query}")
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=REPO, text=True)
    if status:
        raise ValueError(f"Installed worktree changed: {status}")
    manifest = read_object(HERE / "fanout-identity.json")
    hashes: dict[str, str] = {}
    for key, base in (("inputs", REPO), ("prepared", HERE)):
        entries = manifest[key]
        if not isinstance(entries, list):
            raise ValueError(f"Invalid input list: {key}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid input record")
            path, expected_hash = entry.get("path"), entry.get("sha256")
            if not isinstance(path, str) or not isinstance(expected_hash, str):
                raise ValueError("Invalid input path or hash")
            actual_hash = digest(base / path)
            if actual_hash != expected_hash:
                raise ValueError(f"Frozen input changed: {path}")
            hashes[str(base / path)] = actual_hash
    installed = read_object(ROOT / "recovery-evidence/final-acceptance-tools/identity-e96a570d.json")
    binaries = installed["installed_binary_sha256"]
    if not isinstance(binaries, dict):
        raise ValueError("Invalid installed binary identity")
    for path, expected_hash in binaries.items():
        if not isinstance(path, str) or not isinstance(expected_hash, str):
            raise ValueError("Invalid installed binary record")
        if digest(REPO / path) != expected_hash:
            raise ValueError(f"Installed binary changed: {path}")
        hashes[str(REPO / path)] = expected_hash
    return {"source": expected, "clean": True, "sha256": hashes}


def descendants(pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
            parents[int(path.parent.name)] = int(fields[1])
        except FileNotFoundError:
            continue
    owned = {pid}
    while extra := {pid for pid, parent in parents.items() if parent in owned} - owned:
        owned.update(extra)
    return owned


def resources(pid: int | None = None) -> dict[str, object]:
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.split(":")[0] in ("MemAvailable", "Shmem")}
    vmstat = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    rss = 0
    process_count = 0
    if pid is not None:
        for child in descendants(pid):
            try:
                status = Path(f"/proc/{child}/status").read_text()
            except FileNotFoundError:
                continue
            process_count += 1
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1]) * 1024
                    break
    pressure = Path("/proc/pressure/memory")
    return {**memory, "oom_kill": int(vmstat["oom_kill"]),
            "observed_owned_processes": process_count,
            "observed_aggregate_rss_bytes": rss,
            "memory_pressure": pressure.read_text() if pressure.exists() else None,
            "observed_utc": datetime.now(timezone.utc).isoformat()}


def stop_descendants(process: subprocess.Popen[bytes]) -> None:
    owned = descendants(process.pid)
    for pid in sorted(owned, reverse=True):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    process.wait(timeout=10)


def main() -> int:
    before = binding()
    before_resources = resources()
    available = before_resources["MemAvailable"]
    if not isinstance(available, int) or available < 5 * 1024**3:
        raise RuntimeError(f"Insufficient memory reserve: {before_resources}")
    OUT.mkdir()
    physical_temp = Path("/home/agent/logos-parity-isolated-e96-002")
    if not physical_temp.is_dir() or physical_temp.is_symlink():
        raise ValueError("Expected the existing real diagnostic temporary directory")
    for ancestor in (physical_temp, *physical_temp.parents):
        if (ancestor / ".git").exists():
            raise ValueError(f"Temporary source directory has a VCS ancestor: {ancestor}")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GO", "CGO_", "BOOT_GO_", "LOGOS_"))}
    selected = {
        "GOROOT": str(REPO), "GOCACHE": str(ROOT / "final-postbuild-cache"),
        "GOTOOLCHAIN": "local", "GOENV": "off", "GOWORK": "off",
        "GOPROXY": "off", "GOSUMDB": "off", "GOMAXPROCS": "2",
        "GOGC": "50", "GOMEMLIMIT": "2GiB", "GOFLAGS": "-p=2",
        "TMPDIR": str(physical_temp), "GOTMPDIR": str(physical_temp),
        "UPROOT1274": str(ROOT / "toolchains/go1.27.4-ci"),
        "PATH": str(REPO / "bin") + os.pathsep + environment["PATH"],
    }
    environment.update(selected)
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if len(allowed_cpus) < 2:
        raise RuntimeError(f"Two CPU IDs are required: {allowed_cpus}")
    selected_cpus = allowed_cpus[:2]
    command = ["/usr/bin/taskset", "--cpu-list", ",".join(map(str, selected_cpus)),
               str(REPO / "bin/go"), "test", "-count=1", "-v", "-timeout=10m",
               "-overlay=" + str(HERE / "fanout-overlay.json"), "cmd/internal/goparity",
               "-run=^TestParityAt08FanoutProbe$"]
    write_json(OUT / "inputs.json", {
        "binding": before, "command": command, "cwd": str(REPO / "src"),
        "environment_overrides": selected, "outer_timeout_seconds": 660,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "recorder_sha256": digest(Path(__file__)),
        "boundary": "Controlled resource diagnostic only; not a replay of the complete release composition or a release stamp",
        "allowed_cpu_ids": allowed_cpus, "selected_cpu_ids": selected_cpus,
        "resources_before": before_resources,
        "resource_abort_below_available_bytes": 1024**3,
    })
    started = time.monotonic()
    timed_out = False
    interrupted = False
    resource_aborted = False
    samples: list[dict[str, object]] = []
    with (OUT / "stdout.log").open("xb") as stdout, (OUT / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=REPO / "src", env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while True:
                remaining = 660 - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, 660)
                try:
                    code = process.wait(timeout=min(5, remaining))
                    break
                except subprocess.TimeoutExpired:
                    sample = resources(process.pid)
                    samples.append(sample)
                    print(json.dumps({"elapsed_seconds": round(time.monotonic() - started, 3), "resources": sample}), flush=True)
                    available = sample["MemAvailable"]
                    if not isinstance(available, int) or available < 1024**3:
                        resource_aborted = True
                        stop_descendants(process)
                        assert process.returncode is not None
                        code = process.returncode
                        break
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            interrupted = isinstance(error, KeyboardInterrupt)
            stop_descendants(process)
            assert process.returncode is not None
            code = process.returncode
    after_resources = resources()
    result = {
        "exit_code": code, "timed_out": timed_out, "interrupted": interrupted,
        "resource_aborted": resource_aborted, "resource_samples": samples,
        "resources_after": after_resources,
        "oom_kill_counter_unchanged": before_resources["oom_kill"] == after_resources["oom_kill"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_sha256": digest(OUT / "stdout.log"),
        "stderr_sha256": digest(OUT / "stderr.log"), "binding_after": binding(),
    }
    write_json(OUT / "result.json", result)
    print(json.dumps({key: result[key] for key in (
        "exit_code", "timed_out", "interrupted", "resource_aborted",
        "oom_kill_counter_unchanged", "duration_seconds", "stdout_sha256", "stderr_sha256",
    )}), flush=True)
    return (125 if resource_aborted or not result["oom_kill_counter_unchanged"]
            else 124 if timed_out else 130 if interrupted else code)


if __name__ == "__main__":
    raise SystemExit(main())
