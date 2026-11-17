#!/usr/bin/env bash
# Run only in an isolated Linux/amd64 GitHub Actions checkout after make.bash.
# Usage: bash logos-doc/validation/capability-inline-context/check_ci.sh EVIDENCE
set -euo pipefail

if [[ $# != 1 || ${GITHUB_ACTIONS:-} != true ]]; then
  printf 'usage: %s EVIDENCE (isolated GitHub Actions runner required)\n' "$0" >&2
  exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
mkdir -p -- "$(dirname -- "$1")"
mkdir -- "$1"
evidence=$(cd -- "$1" && pwd -P)
case "$evidence/" in
  "$repo/"*) printf 'evidence must be outside the source checkout\n' >&2; exit 2 ;;
esac
cd "$repo"

phase=verify
replaced=0
link="$repo/pkg/tool/linux_amd64/link"
work=

install_link() {
  cp -p -- "$1" "$link.cap-inline-$$" || return 1
  mv -f -- "$link.cap-inline-$$" "$link" || return 1
}

restore_link() {
  if (( replaced )); then
    install_link "$work/candidate-link" || return 1
    cmp -- "$work/candidate-link" "$link" || return 1
    replaced=0
  fi
  if [[ -f "$evidence/tools-before.sha256" ]]; then
    sha256sum --check --strict "$evidence/tools-before.sha256" >> "$evidence/restoration.log" || return 1
  fi
}

finish() {
  local status=$?
  trap - EXIT INT TERM HUP
  if ! restore_link; then
    printf 'failed to restore or verify the installed tools\n' >&2
    status=1
    phase=restoration-failed
  fi
  printf '%s\n' "$status" > "$evidence/runner.status.txt"
  printf 'exit=%s\nphase=%s\nlink_replaced=%s\n' "$status" "$phase" "$replaced" > "$evidence/summary.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

check() {
  uv run --no-project python - "$1" "$repo" "$evidence" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NoReturn, cast

FROZEN_SHA256 = "0a2ed87e4b0e9af5b05ffd6fbac0c7cf4bbdf78a094e160ec6cd76725531b6ee"
SCRIPT = "TestScript/logos_cap_inline_context"
EXPECTED = Counter({"TestScript": 1, SCRIPT: 1})


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], value)


def string_value(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a JSON string")
    return value


def array_value(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected a JSON array")
    return cast(list[object], value)


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value: str) -> NoReturn:
    raise ValueError(f"non-JSON numeric constant: {value}")


def parse_object(data: str) -> dict[str, object]:
    value: object = json.loads(
        data, object_pairs_hook=unique_object, parse_constant=invalid_constant
    )
    return object_value(value)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(repo: Path, evidence: Path) -> dict[str, object]:
    manifest_path = repo / "logos-doc/validation/capability-inline-context/frozen.json"
    if digest(manifest_path) != FROZEN_SHA256:
        raise ValueError("frozen.json SHA-256 mismatch")
    manifest = parse_object(manifest_path.read_text())
    checked: dict[str, str] = {}
    for value in array_value(manifest["new_tests"]) + array_value(
        manifest["preserved_validation"]
    ):
        item = object_value(value)
        path = string_value(item["path"])
        actual = digest(repo / path)
        if actual != item["sha256"]:
            raise ValueError(f"frozen validation SHA-256 mismatch: {path}")
        checked[path] = actual
    original = object_value(manifest["original_production"])
    archive = repo / string_value(original["archive_path"])
    if digest(archive) != original["sha256"]:
        raise ValueError("archived auditor SHA-256 mismatch")
    checked[str(archive.relative_to(repo))] = digest(archive)
    overlay = {"Replace": {str(repo / string_value(original["path"])): str(archive)}}
    (evidence / "old-link-overlay.json").write_text(
        json.dumps(overlay, indent=2) + "\n"
    )
    return {"frozen_sha256": FROZEN_SHA256, "verified": checked}


def events(evidence: Path, phase: str) -> dict[str, object]:
    expected_exit = 1 if phase == "negative" else 0
    status = int((evidence / f"{phase}.status.txt").read_text().strip())
    stdout = evidence / f"{phase}.json"
    stderr = evidence / f"{phase}.stderr.log"
    if status != expected_exit:
        raise ValueError(f"{phase}: exit {status}, expected {expected_exit}")
    if stderr.read_bytes():
        raise ValueError(f"{phase}: unexpected nonempty stderr")
    counts: dict[str, Counter[str]] = {
        action: Counter() for action in ("run", "pass", "fail")
    }
    package: Counter[str] = Counter()
    output: list[str] = []
    child_output: list[str] = []
    for line in stdout.read_text().splitlines():
        event = parse_object(line)
        action = string_value(event.get("Action"))
        if "FailedBuild" in event:
            raise ValueError(f"{phase}: failed-build metadata")
        if action == "build-output":
            string_value(event.get("ImportPath"))
            output.append(string_value(event.get("Output")))
            if "Test" in event:
                raise ValueError("build output unexpectedly names a test")
            continue
        if action not in ("start", "run", "pass", "fail", "output", "pause", "cont"):
            raise ValueError(f"{phase}: forbidden or unknown action {action!r}")
        if event.get("Package") != "cmd/go":
            raise ValueError(f"{phase}: unexpected package")
        name = event.get("Test")
        if "Test" not in event:
            if action not in ("start", "pass", "fail", "output"):
                raise ValueError(f"{phase}: {action} requires a test name")
            if action != "output":
                package[action] += 1
        elif not isinstance(name, str) or name not in EXPECTED or action == "start":
            raise ValueError(f"{phase}: unexpected test identity")
        elif action in counts:
            counts[action][name] += 1
        if action == "output":
            text = string_value(event.get("Output"))
            output.append(text)
            if name == SCRIPT:
                child_output.append(text)
    result = "fail" if phase == "negative" else "pass"
    other = "pass" if result == "fail" else "fail"
    if counts["run"] != EXPECTED or counts[result] != EXPECTED or counts[other]:
        raise ValueError(
            f"{phase}: missing, duplicate, or unexpected named test results"
        )
    if package != Counter({"start": 1, result: 1}):
        raise ValueError(f"{phase}: unexpected package results")
    transcript = "".join(output)
    if re.search(
        r"panic:|fatal error:|internal compiler error|error obtaining buildID|"
        r"exec format error|signal:|\bSIG[A-Z0-9]+(?:\s|:)|test timed out",
        transcript,
    ):
        raise ValueError(f"{phase}: compiler, runtime, or tool failure")
    if phase == "negative" and not re.search(
        r"ERROR \[capabilities::unauthorized_net\]: Transitive dependency "
        r"'example\.com/inlinecapdep/safe' reaches "
        r"'(?:net\.sysSocket|syscall\.(?:Socket|Connect))'",
        "".join(child_output),
    ):
        raise ValueError(
            "negative: missing the safe dependency's network-capability witness"
        )
    return {
        "exit": status,
        "stdout_sha256": digest(stdout),
        "stderr_sha256": digest(stderr),
        "named_results": dict(counts[result]),
    }


def run(mode: str, repo: Path, evidence: Path) -> bool:
    report: dict[str, object] = {"mode": mode, "accepted": False}
    try:
        if mode in ("verify", "verify-after"):
            report.update(verify(repo, evidence))
        elif mode in ("negative", "candidate"):
            report.update(events(evidence, mode))
        else:
            raise ValueError(f"unknown checker mode: {mode}")
        report["accepted"] = True
    except (OSError, TypeError, ValueError) as error:
        report["error"] = str(error)
        print(error, file=sys.stderr)
    (evidence / f"{mode}.check.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report["accepted"] is True


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("checker requires MODE REPO EVIDENCE")
    raise SystemExit(0 if run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])) else 1)
PY
}

check verify > "$evidence/verification.log" 2>&1
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]]
git diff --exit-code HEAD -- src > "$evidence/source.diff.txt"
git rev-parse HEAD HEAD:src > "$evidence/source-before.txt"
sha256sum "$repo/bin/go" "$repo/bin/logos" "$repo/pkg/tool/linux_amd64/compile" "$link" > "$evidence/tools-before.sha256"

work=$(mktemp -d "${RUNNER_TEMP:?}/logos-inline-context.XXXXXX")
export GOROOT="$repo" GOTOOLCHAIN=local GOENV=off GOWORK=off GOFLAGS=-p=1
export PATH="$repo/bin:$PATH"
export GOMAXPROCS="${GOMAXPROCS:-2}" GOMEMLIMIT="${GOMEMLIMIT:-2GiB}" GOGC="${GOGC:-50}"
printf 'GOMAXPROCS=%s\nGOMEMLIMIT=%s\nGOGC=%s\nGOFLAGS=%s\nGOEXPERIMENT_present=%s\nGOEXPERIMENT=%s\n' \
  "$GOMAXPROCS" "$GOMEMLIMIT" "$GOGC" "$GOFLAGS" "${GOEXPERIMENT+x}" "${GOEXPERIMENT-}" > "$evidence/profile.txt"

phase=build-old-linker
build_status=0
timeout --kill-after=30s 10m "$repo/bin/go" build -p=1 -overlay="$evidence/old-link-overlay.json" -o "$work/old-link" cmd/link \
  > "$evidence/old-link-build.stdout.log" 2> "$evidence/old-link-build.stderr.log" || build_status=$?
printf '%s\n' "$build_status" > "$evidence/old-link-build.status.txt"
[[ $build_status == 0 && -x "$work/old-link" ]]
sha256sum "$work/old-link" > "$evidence/old-link.sha256"
if cmp -s -- "$link" "$work/old-link"; then
  printf 'archived old linker is identical to the installed candidate\n' >&2
  exit 1
fi
cp -p -- "$link" "$work/candidate-link"
cmp -- "$link" "$work/candidate-link"

phase=negative
replaced=1
install_link "$work/old-link"
cmp -- "$work/old-link" "$link"
sha256sum "$link" > "$evidence/link-during-negative.sha256"
negative_status=0
timeout --kill-after=30s 25m "$repo/bin/go" test -v -json -count=1 -race -parallel=1 -timeout=20m cmd/go \
  -run '^TestScript$/^logos_cap_inline_context$' > "$evidence/negative.json" 2> "$evidence/negative.stderr.log" || negative_status=$?
printf '%s\n' "$negative_status" > "$evidence/negative.status.txt"
restore_link
sha256sum "$repo/bin/go" "$repo/bin/logos" "$repo/pkg/tool/linux_amd64/compile" "$link" > "$evidence/tools-restored.sha256"
cmp -- "$evidence/tools-before.sha256" "$evidence/tools-restored.sha256"
check negative > "$evidence/negative-validation.log" 2>&1

phase=candidate
candidate_status=0
timeout --kill-after=30s 25m "$repo/bin/go" test -v -json -count=1 -race -parallel=1 -timeout=20m cmd/go \
  -run '^TestScript$/^logos_cap_inline_context$' > "$evidence/candidate.json" 2> "$evidence/candidate.stderr.log" || candidate_status=$?
printf '%s\n' "$candidate_status" > "$evidence/candidate.status.txt"
check candidate > "$evidence/candidate-validation.log" 2>&1
check verify-after > "$evidence/verification-after.log" 2>&1
git diff --exit-code HEAD -- src > "$evidence/source-after.diff.txt"
git rev-parse HEAD HEAD:src > "$evidence/source-after.txt"
cmp -- "$evidence/source-before.txt" "$evidence/source-after.txt"
restore_link
phase=accepted
