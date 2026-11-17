from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
from typing import cast


ROOT = Path("/workspace/scratch/9949907ca71e")
REPO = ROOT / "logos-final-acceptance"
OUT = ROOT / "recovery-evidence/json-installed-e96a570d"
IDENTITY = ROOT / "recovery-evidence/final-acceptance-tools/identity-e96a570d.json"
IDENTITY_SHA = "a52fd73da636ff7b668f8652af4f2e4de672b0e3b79b9b9014d0bccae95ef262"
COMPOSITION = ROOT / "recovery-evidence/final-acceptance-tools/expected-json-composition.json"
COMPOSITION_SHA = "3b23a93368366946439ebf453e96c1569c833210e8d0060d09f20ad821a57869"
COMMIT = "e96a570da271c43a80e23bb1a188aac054acff25"
CGO_FIXTURE = "src/cmd/compile/internal/logos/json_cgo_quoted_test.go"
CGO_SHA = "bc1903980a8fa11ad0724c8eada997d0b5bdbfb5713efbe46d27cffdacac3084"
OUTER_BOUND = 1260


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, object], value)


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected an object field")
    return cast(dict[str, object], value)


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def resources() -> dict[str, int]:
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.split(":")[0] in ("MemAvailable", "Shmem")}
    disk = os.statvfs(ROOT)
    home = os.statvfs("/home/agent")
    return {**memory, "workspace_free_bytes": disk.f_bavail * disk.f_frsize,
            "home_free_bytes": home.f_bavail * home.f_frsize}


def binding() -> dict[str, object]:
    if digest(IDENTITY) != IDENTITY_SHA or digest(COMPOSITION) != COMPOSITION_SHA:
        raise ValueError("Frozen identity or JSON composition changed")
    identity = read_object(IDENTITY)
    actual = {key: subprocess.check_output(
        ["git", "--no-optional-locks", "rev-parse", query], cwd=REPO, text=True).strip()
        for key, query in (("commit", "HEAD"), ("tree", "HEAD^{tree}"), ("src_tree", "HEAD:src"))}
    if actual["commit"] != COMMIT or any(actual[key] != identity.get(key) for key in actual):
        raise ValueError(f"Source identity differs: {actual}")
    status = subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--porcelain=v1"], cwd=REPO, text=True)
    if status:
        raise ValueError(f"Source worktree is not clean: {status}")
    binaries = object_value(identity["installed_binary_sha256"])
    for relative, expected in binaries.items():
        if digest(REPO / relative) != expected:
            raise ValueError(f"Installed tool changed: {relative}")
    files = object_value(read_object(COMPOSITION)["expected_files"])
    source_hashes: dict[str, str] = {}
    for relative, record in files.items():
        observed = digest(REPO / relative)
        if observed != object_value(record)["sha256"]:
            raise ValueError(f"JSON composition differs: {relative}")
        source_hashes[relative] = observed
    if digest(REPO / CGO_FIXTURE) != CGO_SHA:
        raise ValueError("Quoted-C frozen fixture changed")
    source_hashes[CGO_FIXTURE] = CGO_SHA
    return {**actual, "clean": True, "installed_binary_sha256": binaries,
            "source_sha256": source_hashes, "identity_sha256": IDENTITY_SHA,
            "composition_sha256": COMPOSITION_SHA}


def execute(label: str, command: list[str], environment: dict[str, str]) -> dict[str, object]:
    directory = OUT / label
    directory.mkdir()
    selected = {key: environment[key] for key in (
        "GOROOT", "GOCACHE", "GOPATH", "GOTMPDIR", "TMPDIR", "GOMAXPROCS", "GOGC",
        "GOMEMLIMIT", "GOFLAGS", "GOEXPERIMENT", "GOTOOLCHAIN", "GOENV", "GOWORK",
        "GOPROXY", "GOSUMDB", "CGO_ENABLED", "CC", "PATH", "PWD")}
    before = resources()
    if before["MemAvailable"] < 5 * 1024**3 // 2 or before["workspace_free_bytes"] < 2 * 1024**3:
        raise RuntimeError(f"Insufficient resource reserve before {label}: {before}")
    write_json(directory / "inputs.json", {
        "command": command, "cwd": str(REPO / "src"), "environment": selected,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "outer_timeout_seconds": OUTER_BOUND, "resources_before": before,
        "runner_sha256": digest(Path(__file__)), "overlay": None,
        "absent_environment": ["GO_GCFLAGS", "BOOT_GO_GCFLAGS"],
    })
    began = time.monotonic()
    timed_out = False
    interrupted = False
    with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=REPO / "src", env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while True:
                remaining = OUTER_BOUND - (time.monotonic() - began)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, OUTER_BOUND)
                try:
                    status = process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    print(json.dumps({"stage": label, "elapsed_seconds": round(time.monotonic() - began, 3),
                                      "resources": resources()}), flush=True)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            interrupted = isinstance(error, KeyboardInterrupt)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                status = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                status = process.wait(timeout=10)
    result: dict[str, object] = {
        "exit_code": status, "timed_out": timed_out, "interrupted": interrupted,
        "duration_seconds": round(time.monotonic() - began, 3),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_sha256": digest(directory / "stdout.log"),
        "stderr_sha256": digest(directory / "stderr.log"), "resources_after": resources(),
    }
    write_json(directory / "result.json", result)
    print(json.dumps({"stage": label, **result}), flush=True)
    if status != 0 or timed_out or interrupted:
        raise RuntimeError(f"Stage did not complete successfully: {label}")
    return result


def require_tests(label: str, profile: str, quoted: bool) -> None:
    text = (OUT / label / "stdout.log").read_text(errors="replace")
    passed = re.findall(r"^\s*--- PASS: (\S+) \(", text, re.MULTILINE)
    failed = re.findall(r"^\s*--- FAIL: (\S+) \(", text, re.MULTILINE)
    skipped = re.findall(r"^\s*--- SKIP: (\S+) \(", text, re.MULTILINE)
    started = re.findall(r"^=== RUN\s+(\S+)$", text, re.MULTILINE)
    expected_skips = [] if quoted or profile == "jsonv2" else ["TestWalkEnterFrameDepthBound"]
    required = ["TestEjectQuotedCgoTypes"] if quoted else ["TestWalkEnterFrameBalances"]
    if not quoted and profile == "jsonv2":
        required += ["TestWalkEnterFrameDepthBound", "TestQuotedTextAppendCannotCorruptToken",
                     "TestQuotedTextPermissiveUnquoteMatchesEncodingJSON"]
    if failed or skipped != expected_skips or any(name not in started or name not in passed for name in required):
        raise ValueError(f"Unexpected tests/skips in {label}: {failed}, {skipped}, {required}")
    if quoted and (passed != required or started != required):
        raise ValueError("Quoted-C selector did not execute exactly the requested test")
    if expected_skips and "encoding/json v1 has no depth bound" not in text:
        raise ValueError("Classic skip did not report its known profile reason")
    write_json(OUT / label / "test-check.json", {
        "required_passed": required, "passed_records": len(passed),
        "top_level_started": [name for name in started if "/" not in name],
        "top_level_passed": [name for name in passed if "/" not in name],
        "skipped": skipped, "failed": failed,
        "scope": "exact quoted-C test executed" if quoted else "full runtime package, run pattern .",
    })


def main() -> int:
    OUT.mkdir(exist_ok=False)
    before = binding()
    write_json(OUT / "binding-before.json", before)
    source_temp = Path(tempfile.mkdtemp(prefix="logos-json-final-e96-", dir="/home/agent"))
    if source_temp.resolve() != source_temp:
        raise ValueError("Source temp is not a real directory")
    markers = [str(parent / name) for parent in (source_temp, *source_temp.parents)
               for name in (".git", ".hg", ".svn") if (parent / name).exists()]
    if markers:
        raise ValueError(f"Source temp has VCS ancestry: {markers}")
    build_temp = ROOT / "final-json-build-tmp"
    build_temp.mkdir(exist_ok=True)
    binaries = OUT / "binaries"
    binaries.mkdir()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GO", "BOOT_GO_", "CGO_", "LOGOS_"))}
    environment.update({
        "GOROOT": str(REPO), "GOCACHE": str(ROOT / "final-postbuild-cache"),
        "GOPATH": str(ROOT / "acceptance-gopath"), "GOTMPDIR": str(build_temp),
        "TMPDIR": str(source_temp), "GOMAXPROCS": "1", "GOGC": "20", "GOMEMLIMIT": "512MiB",
        "GOFLAGS": "-p=1", "GOTOOLCHAIN": "local", "GOENV": "off", "GOWORK": "off",
        "GOPROXY": "off", "GOSUMDB": "off", "CGO_ENABLED": "1", "CC": "/usr/bin/cc",
        "PATH": str(REPO / "bin") + os.pathsep + environment.get("PATH", ""),
        "PWD": str(REPO / "src"),
    })
    write_json(OUT / "preparation.json", {
        "source_temp": str(source_temp), "source_temp_is_real": True, "vcs_markers": markers,
        "build_temp": str(build_temp), "source_temp_reason": "testing.TempDir uses GOTMPDIR; executed binaries use real-home GOTMPDIR",
        "cc_resolved": str(Path("/usr/bin/cc").resolve()), "cc_sha256": digest(Path("/usr/bin/cc")),
        "resources": resources(), "source_edits": False, "overlays": False, "new_mutations": False,
    })
    passed = False
    error_text: str | None = None
    try:
        for profile in ("jsonv2", "nojsonv2"):
            environment["GOEXPERIMENT"] = profile
            for kind, package, selector in (("runtime", "logos/rt/json", "."),
                                            ("quoted-cgo", "cmd/compile/internal/logos", "^TestEjectQuotedCgoTypes$")):
                binary = binaries / f"{kind}-{profile}.test"
                execute(f"{profile}-{kind}-compile", [str(REPO / "bin/go"), "test", "-p=1", "-race", "-c",
                        "-o", str(binary), package], environment)
                binary_sha = digest(binary)
                write_json(OUT / f"{profile}-{kind}-compile" / "binary.json", {
                    "path": str(binary), "sha256": binary_sha, "bytes": binary.stat().st_size,
                    "profile": profile, "race": True, "source_commit": COMMIT,
                })
                runtime_environment = {**environment, "GOTMPDIR": str(source_temp)}
                label = f"{profile}-{kind}-run"
                execute(label, [str(binary), "-test.v", "-test.count=1", "-test.parallel=1", "-test.timeout=20m",
                                "-test.run=" + selector], runtime_environment)
                require_tests(label, profile, kind == "quoted-cgo")
                if digest(binary) != binary_sha:
                    raise ValueError(f"Test binary changed while executing: {binary}")
            execute(f"{profile}-vet", [str(REPO / "bin/go"), "vet", "-p=1", "logos/rt/json"], environment)
        passed = True
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        error_text = f"{type(error).__name__}: {error}"
        print(error_text, flush=True)
    finally:
        after = binding()
        write_json(OUT / "binding-after.json", after)
        write_json(OUT / "summary.json", {
            "passed": passed, "error": error_text, "identities_unchanged": before == after,
            "source_commit": COMMIT, "profiles": ["jsonv2", "nojsonv2"],
            "scope": "full JSON runtime race suites and normal runtime vet; explicit quoted-C race test in both profiles",
            "overlays": False, "new_mutations": False,
            "remaining_scope": "broader hosted frontend/fixture/script profile workflow and full release gates are separate",
        })
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
