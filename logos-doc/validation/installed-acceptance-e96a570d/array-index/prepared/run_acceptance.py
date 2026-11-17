from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import FrameType
from typing import cast


ROOT = Path(__file__).resolve().parent
LINT = "cmd/compile/internal/logos/lint"
RULE = "src/" + LINT + "/rule_array_index_out_of_bounds.go"
PROBE_SELECTOR = "^TestArrayIndexOutOfBounds$/^guardedReceiver$/^x\\.(go|lgo)$"
PROBE_LEAVES = (
    "TestArrayIndexOutOfBounds/guardedReceiver/x.go",
    "TestArrayIndexOutOfBounds/guardedReceiver/x.lgo",
)


@dataclass(frozen=True)
class Binding:
    worktree: Path
    commit: str
    driver_sha256: str
    compiler_sha256: str
    cc: str
    cache: Path
    source_temp: Path
    output: Path
    phase: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(worktree: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "--no-optional-locks", *arguments], cwd=worktree, text=True
    ).strip()


def verify(binding: Binding) -> dict[str, object]:
    if git(binding.worktree, "rev-parse", "HEAD") != binding.commit:
        raise ValueError("Final worktree commit does not match supplied binding")
    status = git(binding.worktree, "status", "--porcelain=v1", "--untracked-files=normal")
    if status:
        raise ValueError(f"Final worktree is not clean:\n{status}")
    hashes: dict[str, str] = {}
    for line in (ROOT / "frozen-inputs.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        observed = digest(binding.worktree / relative)
        if observed != expected or digest(ROOT / "frozen" / relative) != expected:
            raise ValueError(f"Frozen input mismatch: {relative}")
        hashes[relative] = observed
    for relative, expected in (
        ("bin/go", binding.driver_sha256),
        ("bin/logos", binding.driver_sha256),
        ("pkg/tool/linux_amd64/compile", binding.compiler_sha256),
    ):
        observed = digest(binding.worktree / relative)
        if observed != expected:
            raise ValueError(f"Installed tool mismatch: {relative}")
        hashes[relative] = observed
    original = (ROOT / "frozen" / RULE).read_bytes()
    mutant = (ROOT / "mutation/rule_array_index_out_of_bounds.go").read_bytes()
    anchor = b"proof.block(fn.Body.List) == arrayIndexReached"
    if original.count(anchor) != 1 or mutant != original.replace(
        anchor, b"proof.block(fn.Body.List) != arrayIndexReached", 1
    ):
        raise ValueError("External mutation differs from the sole frozen condition")
    return {"commit": binding.commit,
            "tree": git(binding.worktree, "rev-parse", "HEAD^{tree}"),
            "src_tree": git(binding.worktree, "rev-parse", "HEAD:src"),
            "clean": True, "sha256": hashes,
            "mutant_sha256": digest(ROOT / "mutation/rule_array_index_out_of_bounds.go")}


def execute(binding: Binding, name: str, command: list[str]) -> int:
    stage = binding.output / name
    stage.mkdir()
    environment = dict(os.environ)
    removed = ("GO_GCFLAGS", "BOOT_GO_GCFLAGS", "GOOS", "GOARCH", "GOAMD64",
               "GOARM64", "GODEBUG", "GOEXPERIMENT", "GOROOT", "GOFLAGS")
    for key in removed:
        environment.pop(key, None)
    overrides = {
        "GOROOT": str(binding.worktree), "GOCACHE": str(binding.cache),
        "GOTMPDIR": str(binding.source_temp), "TMPDIR": str(binding.source_temp),
        "GOMAXPROCS": "1", "GOGC": "20", "GOMEMLIMIT": "512MiB",
        "GOTOOLCHAIN": "local", "GOENV": "off", "GOWORK": "off",
        "GOPROXY": "off", "GOSUMDB": "off",
        "CGO_ENABLED": "1", "CC": binding.cc,
        "PATH": str(binding.worktree / "bin") + os.pathsep + environment.get("PATH", ""),
        "PWD": str(binding.worktree / "src"),
    }
    environment.update(overrides)
    write_json(stage / "inputs.json", {
        "command": command, "cwd": str(binding.worktree / "src"),
        "environment_overrides": overrides, "removed_environment_keys": removed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "outer_timeout_seconds": 660, "runner_sha256": digest(Path(__file__)),
    })
    started = time.monotonic()
    timed_out = False
    interrupted = False
    with (stage / "stdout.log").open("wb") as stdout, (stage / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=binding.worktree / "src", env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            code = process.wait(timeout=660)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            interrupted = isinstance(exc, KeyboardInterrupt)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=10)
    result: dict[str, object] = {
        "exit_code": code, "timed_out": timed_out, "interrupted": interrupted,
        "duration_seconds": round(time.monotonic() - started, 3),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_sha256": digest(stage / "stdout.log"),
        "stderr_sha256": digest(stage / "stderr.log"),
    }
    write_json(stage / "result.json", result)
    print(json.dumps({"stage": name, **result}), flush=True)
    if interrupted:
        raise KeyboardInterrupt("Stopped the active process group; no further Go stage launched")
    return 124 if timed_out else code


def require_leaves(binding: Binding, stage: str, expected: tuple[str, ...], action: str) -> None:
    outcomes: dict[str, str] = {}
    output: dict[str, str] = {}
    skipped: list[str] = []
    for line in (binding.output / stage / "stdout.log").read_text().splitlines():
        raw: object = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("go test JSON event is not an object")
        event = cast(dict[str, object], raw)
        test = event.get("Test")
        event_action = event.get("Action")
        if isinstance(test, str) and isinstance(event_action, str):
            if event_action in ("pass", "fail", "skip"):
                outcomes[test] = event_action
            if event_action == "skip":
                skipped.append(test)
            text = event.get("Output")
            if isinstance(text, str):
                output[test] = output.get(test, "") + text
    wrong = {name: outcomes.get(name, "missing") for name in expected
             if outcomes.get(name) != action}
    if action == "fail":
        for name in expected:
            text = output.get(name, "")
            if "correctness::array_index_out_of_bounds deny" not in text or "want []" not in text:
                wrong[name] = "missing predicted unwanted-diagnostic assertion"
    write_json(binding.output / stage / "leaf-check.json", {
        "expected": list(expected), "expected_action": action,
        "matched": len(expected) - len(wrong), "wrong": wrong,
        "all_observed_skips": skipped,
    })
    if wrong:
        raise ValueError(f"{stage}: required leaf outcomes differ: {wrong}")


def tests(binding: Binding, package: str, *flags: str) -> list[str]:
    return [str(binding.worktree / "bin/go"), "test", "-json", "-p=1", "-count=1",
            "-parallel=1", "-timeout=10m", *flags, package]


def passed(binding: Binding, stage: str, command: list[str], leaves: tuple[str, ...]) -> None:
    if execute(binding, stage, command) != 0:
        raise ValueError(f"{stage}: command did not pass")
    if leaves:
        require_leaves(binding, stage, leaves, "pass")


def gates(binding: Binding) -> None:
    leaves = tuple((ROOT / "expected-unit-leaves.txt").read_text().splitlines())
    if len(leaves) != 96 or len(set(leaves)) != 96:
        raise ValueError("Expected exactly 96 unique frozen unit leaves")
    passed(binding, "lint-full", tests(binding, LINT, "-short"), leaves)
    passed(binding, "fixture-completeness", tests(binding, "cmd/compile/internal/logos",
           "-run=^TestEveryLintRuleHasFixture$"), ("TestEveryLintRuleHasFixture",))
    passed(binding, "installed-fixtures", tests(binding, "cmd/internal/testdir",
           "-run=^Test$/^logos$/^lint$/^(array_index_out_of_bounds\\.lgo|array_index_out_of_bounds_go\\.go)$"),
           ("Test/logos/lint/array_index_out_of_bounds.lgo",
            "Test/logos/lint/array_index_out_of_bounds_go.go"))
    binary_directory = binding.output / "binaries"
    binary_directory.mkdir()
    race_binary = binary_directory / "cmdgo-race.test"
    passed(binding, "guarded-enum-race", tests(binding, "cmd/go", "-race",
           "-o=" + str(race_binary),
           "-run=^TestScript$/^logos_guarded_enum_index$"),
           ("TestScript/logos_guarded_enum_index",))
    write_json(binding.output / "guarded-enum-race/retained-binary.json", {
        "path": str(race_binary), "sha256": digest(race_binary),
        "bytes": race_binary.stat().st_size, "source_commit": binding.commit,
        "driver_sha256": binding.driver_sha256,
        "compiler_sha256": binding.compiler_sha256,
        "scope": "race-enabled cmd/go package test binary; retained for queued checks, never committed",
    })
    passed(binding, "vet", [str(binding.worktree / "bin/go"), "vet", "-p=1", LINT], ())
    paths = [line.split("  ", 1)[1]
             for line in (ROOT / "frozen-inputs.sha256").read_text().splitlines()
             if line.endswith(".go")]
    passed(binding, "gofmt", [str(binding.worktree / "bin/gofmt"), "-l",
           *(str(binding.worktree / path) for path in paths)], ())
    if (binding.output / "gofmt/stdout.log").read_bytes():
        raise ValueError("gofmt lists a frozen Go input")


def probe(binding: Binding) -> None:
    overlay = binding.output / "external-overlay.json"
    write_json(overlay, {"Replace": {str(binding.worktree / RULE):
                                   str(ROOT / "mutation/rule_array_index_out_of_bounds.go")}})
    code = execute(binding, "mutation", tests(binding, LINT, "-short",
                   "-overlay=" + str(overlay), "-run=" + PROBE_SELECTOR))
    if code == 124:
        raise ValueError("Mutation timed out; no restoration test launched after the bound")
    try:
        if code == 0:
            raise ValueError("Mutation was not caught by a completed failing test")
        require_leaves(binding, "mutation", PROBE_LEAVES, "fail")
    finally:
        passed(binding, "restored", tests(binding, LINT, "-short",
               "-run=" + PROBE_SELECTOR), PROBE_LEAVES)


def interrupted(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def main() -> None:
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description="Bounded, identity-bound array-index acceptance")
    for name in ("worktree", "cache", "source-temp", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("commit", "driver-sha256", "compiler-sha256", "cc"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--phase", choices=("all", "gates", "probe"), default="all")
    args = parser.parse_args()
    binding = Binding(args.worktree.resolve(strict=True), args.commit, args.driver_sha256,
                      args.compiler_sha256, args.cc, args.cache.resolve(), args.source_temp.resolve(),
                      args.output.resolve(), args.phase)
    if not binding.source_temp.is_relative_to(Path("/home/agent").resolve()):
        raise ValueError("Use a real /home/agent source/temp directory to avoid workspace VCS ancestry")
    if binding.output.is_relative_to(binding.worktree) or ROOT.is_relative_to(binding.worktree):
        raise ValueError("Runner, mutation, and evidence must remain outside the final worktree")
    binding.output.mkdir(parents=True, exist_ok=False)
    binding.cache.mkdir(parents=True, exist_ok=True)
    binding.source_temp.mkdir(parents=True, exist_ok=True)
    before = verify(binding)
    write_json(binding.output / "binding-before.json", before)
    error: str | None = None
    completed = False
    try:
        if binding.phase in ("all", "gates"):
            gates(binding)
        if binding.phase in ("all", "probe"):
            probe(binding)
        completed = True
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        error = str(exc)
        raise
    finally:
        after = verify(binding)
        write_json(binding.output / "binding-after.json", after)
        write_json(binding.output / "summary.json", {
            "phase": binding.phase, "passed": completed and before == after,
            "error": error if error is not None or completed else "execution interrupted before completion",
            "identities_unchanged": before == after,
            "mutation_scope": "external overlay recompiles the lint package; installed compiler is unchanged",
            "release_scope": "focused acceptance only; full language and Go regression remain separate",
        })
        if before != after:
            raise ValueError("Final source or installed tools changed during acceptance")


if __name__ == "__main__":
    main()
