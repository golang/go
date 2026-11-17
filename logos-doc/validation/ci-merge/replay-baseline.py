from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main() -> None:
    task: Path = Path("/workspace/scratch/9949907ca71e")
    root: Path = task / "logos-json-integration"
    evidence: Path = task / "language-evidence/ci-merge"
    replay: Path = evidence / "fresh-baseline-replay"
    before: str = "e82c916ff4d8c2befc6f1891decb0f623401a94b"
    release: str = "9a628f5b87f3c69a392ba539d11c7c9a37b98566"

    def read_git(revision: str, path: str) -> str:
        return subprocess.run(
            ["git", "show", f"{revision}:{path}"], cwd=root,
            check=True, capture_output=True, text=True,
        ).stdout

    paths: list[str] = [line.split("  ", 1)[1] for line in (evidence / "recovered-inputs.sha256").read_text().splitlines()]
    for path in paths:
        target: Path = replay / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / path).read_bytes())
    for name in ("compiler-stability", "deferred-callees", "gc-stability", "generic-type-arguments", "json-profile-gates"):
        path = f".github/workflows/{name}.yml"
        (replay / path).write_text(read_git(before, path))
    config: str = ".github/actionlint.yaml"
    (replay / config).write_text(read_git(release, config))
    simd: str = ".github/workflows/simd-host-gates.yml"
    old: str = '          export GOROOT_BOOTSTRAP="$(go env GOROOT)"\n          GOEXPERIMENT= ./make.bash\n'
    fixed: str = '          GOROOT_BOOTSTRAP=$(go env GOROOT)\n          export GOROOT_BOOTSTRAP\n          GOEXPERIMENT="" ./make.bash\n'
    original_simd: str = read_git(before, simd)
    assert original_simd.count(old) == 1
    assert read_git(release, simd).count(fixed) == 1
    (replay / simd).write_text(original_simd.replace(old, fixed, 1))
    (replay / "inputs.sha256").write_text("".join(
        f"{hashlib.sha256((replay / path).read_bytes()).hexdigest()}  {path}\n" for path in paths
    ))
    command: list[str] = [str(task / "language-ci-tools/bin/actionlint"), "-color", "-config-file", str(replay / config)]
    command.extend(str(path) for path in sorted((replay / ".github/workflows").glob("*.yml")))
    environment: dict[str, str] = dict(os.environ)
    environment["PATH"] = f"{task / 'language-ci-tools/bin'}:{environment['PATH']}"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    started: float = time.monotonic()
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, check=False)
    elapsed: float = round(time.monotonic() - started, 3)
    (replay / "actionlint.stdout").write_bytes(result.stdout)
    (replay / "actionlint.stderr").write_bytes(result.stderr)
    normalized: str = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout.decode())
    (replay / "actionlint.normalized.txt").write_text(normalized)
    counts: dict[str, int] = {code: normalized.count(f"{code}:") for code in ("SC2155", "SC1007", "SC2086")}
    counts["runner-label"] = normalized.count("[runner-label]")
    record = {
        "command": command, "cwd": str(root), "status": result.returncode,
        "seconds": elapsed, "stdoutSha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderrSha256": hashlib.sha256(result.stderr).hexdigest(),
        "provenance": "fresh diagnostic replay, not recovered historical raw output",
        "preMergeParent": before, "releaseParent": release, "diagnosticCounts": counts,
        "construction": "Five workflows from e82; config from 9a; SIMD from e82 with the exact 9a bootstrap block; hash-verified merged logos.yml and both installers retained",
    }
    (replay / "actionlint.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(normalized)
    assert result.returncode == 1
    assert counts == {"SC2155": 5, "SC1007": 3, "SC2086": 2, "runner-label": 1}
    assert len(re.findall(r"^.*:\d+:\d+: .* \[[^]]+\]$", normalized, re.MULTILINE)) == 11


if __name__ == "__main__":
    main()
