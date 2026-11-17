from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    output = Path(sys.argv[1])
    environment_file = Path(sys.argv[2])
    timeout_seconds = int(sys.argv[3])
    cwd = Path(sys.argv[4]).resolve(strict=True)
    command = sys.argv[5:]
    assert command and 0 < timeout_seconds <= 900
    overrides: dict[str, str] = json.loads(environment_file.read_text())
    assert all(isinstance(key, str) and isinstance(value, str) for key, value in overrides.items())
    environment = dict(os.environ)
    removed = (
        "GO_GCFLAGS", "BOOT_GO_GCFLAGS", "GOOS", "GOARCH", "GOAMD64",
        "GOARM64", "GODEBUG", "GOEXPERIMENT", "GOROOT", "GOFLAGS",
    )
    for key in removed:
        environment.pop(key, None)
    environment.update(overrides)
    environment["PWD"] = str(cwd)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {
        "started_utc": utc_now(),
        "command": command,
        "cwd": str(cwd),
        "timeout_seconds": timeout_seconds,
        "environment_overrides": overrides,
        "removed_before_overrides": removed,
        "runner_sha256": digest(Path(__file__)),
    }
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    started = time.monotonic()
    timed_out = False
    stdout_path = output / "stdout.log"
    stderr_path = output / "stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait(timeout=10)
    result = {
        "completed_utc": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "exit_code": return_code,
        "timed_out": timed_out,
        "stdout_sha256": digest(stdout_path),
        "stderr_sha256": digest(stderr_path),
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"record": str(output), **result}), flush=True)
    raise SystemExit(124 if timed_out else return_code)


if __name__ == "__main__":
    main()
