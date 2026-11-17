"""Replay the two main-workflow failures with the frozen installed source."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/workspace/scratch/9949907ca71e")
sys.path.insert(0, str(ROOT / "recovery-evidence/parity-timeout-investigation"))
from record_isolated import binding, digest, stop_descendants, write_json

REPO = ROOT / "logos-final-acceptance"
OUT = Path(__file__).resolve().parent / "baseline-001"
RACE = ROOT / "recovery-evidence/array-index-acceptance/runs/e96-final-001/binaries/cmdgo-race.test"


def main() -> int:
    before = binding()
    if digest(RACE) != "9645693e7edb1d9b4a1a02c0a74b158283db73158189b62c0c4feb45b3103f0a":
        raise ValueError("Installed command race test binary changed")
    frozen = {path: digest(REPO / path) for path in (
        "src/go/types/generate_test.go", "src/go/types/builtins.go",
        "src/go/types/typestring.go", "src/cmd/compile/internal/types2/builtins.go",
        "src/cmd/compile/internal/types2/typestring.go",
        "src/cmd/go/testdata/script/logos_alias.txt",
    )}
    OUT.mkdir()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GO", "CGO_", "BOOT_GO_", "LOGOS_"))}
    selected = {
        "GOROOT": str(REPO), "GOCACHE": str(ROOT / "final-postbuild-cache"),
        "GOTOOLCHAIN": "local", "GOENV": "off", "GOWORK": "off",
        "GOPROXY": "off", "GOSUMDB": "off", "GOMAXPROCS": "2",
        "GOGC": "50", "GOMEMLIMIT": "2GiB", "GOFLAGS": "-p=2",
        "TMPDIR": "/home/agent/logos-parity-isolated-e96-002",
        "GOTMPDIR": "/home/agent/logos-parity-isolated-e96-002",
        "UPROOT1274": str(ROOT / "toolchains/go1.27.4-ci"),
        "PATH": str(REPO / "bin") + os.pathsep + environment["PATH"],
    }
    environment.update(selected)
    stages = (
        ("go-types-generate", REPO / "src", [str(REPO / "bin/go"), "test", "-count=1",
          "-short", "-v", "-timeout=5m", "go/types", "-run=^TestGenerate$"],
         ("builtins.go (on disk", "typestring.go (on disk", "--- FAIL: TestGenerate")),
        ("logos-alias", REPO / "src/cmd/go", [str(RACE), "-test.v", "-test.count=1",
          "-test.timeout=5m", "-test.run=^TestScript$/^logos_alias$"],
         ("correctness::duplicate_stem_filename", "example.com/m/dup", "--- FAIL: TestScript/logos_alias")),
    )
    for name, cwd, command, markers in stages:
        directory = OUT / name
        directory.mkdir()
        write_json(directory / "inputs.json", {
            "binding": before, "frozen_inputs": frozen, "command": command,
            "cwd": str(cwd), "environment_overrides": selected,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "outer_timeout_seconds": 330, "runner_sha256": digest(Path(__file__)),
        })
        started = time.monotonic()
        interrupted = False
        with (directory / "stdout.log").open("xb") as stdout, (directory / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=cwd, env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=330)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                interrupted = True
                stop_descendants(process)
                assert process.returncode is not None
                code = process.returncode
        output = (directory / "stdout.log").read_text() + (directory / "stderr.log").read_text()
        red = code == 1 and not interrupted and all(marker in output for marker in markers)
        result = {
            "exit_code": code, "interrupted_or_outer_timeout": interrupted,
            "expected_semantic_red": red, "required_markers": markers,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout_sha256": digest(directory / "stdout.log"),
            "stderr_sha256": digest(directory / "stderr.log"), "binding_after": binding(),
        }
        if any(digest(REPO / path) != sha for path, sha in frozen.items()):
            raise ValueError("Frozen failure inputs changed")
        write_json(directory / "result.json", result)
        print(f"{name}: exit={code} semantic_red={red} seconds={result['duration_seconds']}", flush=True)
        if not red:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
