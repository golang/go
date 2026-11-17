# pyright: strict
"""Verify the frozen tagged-fact regression gate and retain its raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

PACKAGE = "cmd/compile/internal/logos"
EVIDENCE_DIR = "logos-doc/validation/tagged-package-facts"
SOURCE_DIR = "src/cmd/compile/internal/logos"
NEGATIVE_ROOT = "TestFactScopeTaggedFunctionAssertions"
NEGATIVE_TEST = f"{NEGATIVE_ROOT}/Project@simd128/mask=0"
NEGATIVE_NAMES = frozenset((NEGATIVE_ROOT, NEGATIVE_TEST))
NEGATIVE_DIAGNOSTIC = "assertion metadata known=false"
FROZEN_PINS = (
    (
        f"{SOURCE_DIR}/optfacts_tagged_test.go",
        "cecb8d8738e9c80777700da61c4cbec395cbe9e034d550679101d510f10b6ead",
    ),
    (
        f"{SOURCE_DIR}/compile_tagged_facts_test.go",
        "f3b308710d0f57ed15a2ef94d53750944c14ae9b10bafc788d8a1b26361ecbd8",
    ),
    (
        f"{SOURCE_DIR}/optfacts_scope_test.go",
        "0899e0c6da711b269e4a586ce96e2f9daaa90ccf211541664c30e9977add3317",
    ),
    (
        f"{SOURCE_DIR}/compile_local_type_import_test.go",
        "05daf14fc22048833996164c3ca56231cc857d67c8a7456bdb3b24d1cd9183af",
    ),
    (
        f"{EVIDENCE_DIR}/old-optfacts.go.txt",
        "5ff9035b2998b75772223068366f1fdca121b3b738f1e35ba8bc7cc4da00be9d",
    ),
    (
        f"{EVIDENCE_DIR}/frozen.json",
        "fe38455d4fab1e2928336f7d9845c61893ce97d38986edd0154bedab63cc4b4a",
    ),
)
OPERATIONS = (
    "functions",
    "readonly",
    "empty functions",
    "empty readonly",
    "declaration clone",
    "structural clone",
    "readonly symbol",
)


def expected_names() -> frozenset[str]:
    groups: dict[str, tuple[str, ...]] = {
        "TestFactScopeLeavesCompilerSymbolsLazy": tuple(
            f"{symbol}/{operation}"
            for symbol in ("local·1", "hidden·23", "Box.Method", "Box[go.shape.int]")
            for operation in OPERATIONS
        ),
        "TestCompileImportedLocalType": tuple(
            f"{case}/{mode}"
            for case in (
                "generic method",
                "generic function",
                "nested two parameters",
                "nongeneric local control",
                "package generic control",
            )
            for mode in ("go", "logos go", "logos interop", "logos overlay")
        ),
        NEGATIVE_ROOT: tuple(
            f"{name}/mask={mask}"
            for name in (
                "Project",
                "Project@simd0",
                "Project@simd128",
                "Project@simd256",
                "Project@simd512",
                "Ω@simd128",
                "_value2@simd128",
            )
            for mask in ("0", "1")
        ),
        "TestFactScopeTaggedFunctionRejectsInvalidMasks": (
            "outside function arity",
            "receiver bit on function",
            "missing receiver mask",
        ),
        "TestFactScopeTaggedReadonlyType": ("MRead", "FView.MRead"),
        "TestFactScopeTaggedReachableMethod": (),
        "TestFactScopeTaggedClones": ("declaration", "structural"),
        "TestFactScopeRejectsMalformedTaggedSymbols": tuple(
            f"{symbol}/{operation}"
            for symbol in (
                "Project@",
                "Project@1",
                "Project@simd128!",
                "Project@simd128@again",
                "@simd128",
                "for@simd128",
                "Project[go.shape.int]@simd128",
                "local·1@simd128",
                "Box.Method@simd128",
                "Project@simd128.Read",
            )
            for operation in OPERATIONS
        ),
        "TestCompileSIMDTaggedFuncFacts": (
            "Float32sFromArch@simd0",
            "Float32sFromArch@simd128",
        ),
    }
    # Each frozen t.Run receives one complete name, even when it contains '/'.
    names = set(groups)
    names.update(
        f"{root}/{child.replace(' ', '_')}"
        for root, children in groups.items()
        for child in children
    )
    return frozenset(names)


CANDIDATE_NAMES = expected_names()


@dataclass(frozen=True)
class Events:
    runs: tuple[str, ...]
    passes: tuple[str, ...]
    failures: tuple[str, ...]
    skips: tuple[str, ...]
    package_starts: int
    package_results: tuple[str, ...]
    build_failures: tuple[str, ...]
    negative_output: str


def reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-JSON numeric constant {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def json_object(text: str) -> dict[str, object]:
    value: object = json.loads(
        text, object_pairs_hook=unique_object, parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise TypeError("JSON event is not an object")
    return cast(dict[str, object], value)


def string_field(event: dict[str, object], key: str, *, empty: bool = False) -> str:
    value = event.get(key)
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"event has invalid {key}: {value!r}")
    return value


def parse_events(text: str) -> Events:
    outcomes: dict[str, list[str]] = {
        action: [] for action in ("run", "pass", "fail", "skip")
    }
    starts = 0
    package_results: list[str] = []
    build_failures: list[str] = []
    negative_output: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ValueError(f"empty JSON event at line {number}")
        event = json_object(line)
        action = string_field(event, "Action")
        if action in ("build-output", "build-fail"):
            if "Test" in event:
                raise ValueError("build event contains a Test field")
            import_path = string_field(event, "ImportPath", empty=True)
            if action == "build-fail":
                build_failures.append(import_path)
            else:
                string_field(event, "Output", empty=True)
            continue
        if string_field(event, "Package") != PACKAGE:
            raise ValueError("event belongs to an unexpected package")
        if "Test" in event:
            name = string_field(event, "Test")
            if action in outcomes:
                outcomes[action].append(name)
            elif action == "output":
                output = string_field(event, "Output", empty=True)
                if name == NEGATIVE_TEST:
                    negative_output.append(output)
            elif action not in ("pause", "cont"):
                raise ValueError(f"unknown test action {action!r}")
        elif action == "start":
            starts += 1
        elif action in ("pass", "fail", "skip"):
            package_results.append(action)
        elif action == "output":
            string_field(event, "Output", empty=True)
        else:
            raise ValueError(f"unknown package action {action!r}")
    return Events(
        tuple(outcomes["run"]),
        tuple(outcomes["pass"]),
        tuple(outcomes["fail"]),
        tuple(outcomes["skip"]),
        starts,
        tuple(package_results),
        tuple(build_failures),
        "".join(negative_output),
    )


def require_exact(
    actual: tuple[str, ...], expected: frozenset[str], label: str
) -> None:
    got, want = Counter(actual), Counter(expected)
    if got != want:
        raise ValueError(
            f"{label}: missing={dict(want - got)}, extra={dict(got - want)}"
        )


def accept(
    events: Events, exit_code: int, *, negative: bool, stderr: bytes = b""
) -> None:
    if stderr:
        raise ValueError("unexpected nonempty stderr; inspect the retained stderr log")
    if events.build_failures or events.skips:
        raise ValueError(
            f"build failures or skipped tests: {events.build_failures}, {events.skips}"
        )
    expected = NEGATIVE_NAMES if negative else CANDIDATE_NAMES
    require_exact(events.runs, expected, "test runs")
    if events.package_starts != 1:
        raise ValueError(f"package start count is {events.package_starts}, want 1")
    if negative:
        require_exact(events.failures, expected, "negative-control failures")
        if exit_code != 1 or events.passes or events.package_results != ("fail",):
            raise ValueError(
                "negative control lacks the exact failing exit/package outcome"
            )
        if NEGATIVE_DIAGNOSTIC not in events.negative_output:
            raise ValueError(
                "negative control lacks the named missing-metadata diagnostic"
            )
    else:
        require_exact(events.passes, expected, "candidate passes")
        if exit_code != 0 or events.failures or events.package_results != ("pass",):
            raise ValueError("candidate has a failing exit/package outcome")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_frozen(root: Path) -> dict[str, str]:
    actual = {path: digest(root / path) for path, _ in FROZEN_PINS}
    for path, expected in FROZEN_PINS:
        if actual[path] != expected:
            raise ValueError(
                f"frozen source digest mismatch: {path}: {actual[path]} != {expected}"
            )
    return actual


def git_value(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *arguments), text=True
    ).strip()


def source_identity(root: Path, expected_head: str) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise ValueError("expected head must be a full Git SHA-1")
    head = git_value(root, "rev-parse", "HEAD")
    if head != expected_head or git_value(
        root, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("checkout does not match the clean requested source revision")
    return {
        "head": head,
        "tree": git_value(root, "rev-parse", "HEAD^{tree}"),
        "source_tree": git_value(root, "rev-parse", "HEAD:src"),
        "production_sha256": digest(root / SOURCE_DIR / "optfacts.go"),
        "frozen_sha256": verify_frozen(root),
        "helper_sha256": digest(Path(__file__)),
        "workflow_sha256": digest(root / ".github/workflows/tagged-facts.yml"),
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


def tool_identity(root: Path) -> dict[str, str]:
    return {
        path: digest(root / path)
        for path in (
            "bin/go",
            "bin/logos",
            "pkg/tool/linux_amd64/compile",
            "pkg/tool/linux_amd64/link",
        )
    }


def capture(
    command: tuple[str, ...], root: Path, out: Path, phase: str, timeout: int
) -> dict[str, object]:
    started = time.monotonic()
    timed_out = False
    stdout, stderr = out / f"{phase}.jsonl", out / f"{phase}.stderr.txt"
    with (
        stdout.open("xb") as output,
        stderr.open("xb") as errors,
        subprocess.Popen(
            command,
            cwd=root / "src",
            stdout=output,
            stderr=errors,
            start_new_session=True,
        ) as process,
    ):
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            exit_code = process.wait()
    result: dict[str, object] = {
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_sha256": digest(stdout),
        "stderr_sha256": digest(stderr),
        "accepted": False,
    }
    try:
        if timed_out:
            raise ValueError(f"{phase} exceeded {timeout} seconds")
        accept(
            parse_events(stdout.read_text(encoding="utf-8")),
            exit_code,
            negative=phase == "negative",
            stderr=stderr.read_bytes(),
        )
        result["accepted"] = True
    except (TypeError, ValueError) as error:
        result["rejection"] = str(error)
    write_json(out / f"{phase}.status.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def run_gate(root: Path, out: Path, expected_head: str) -> bool:
    before = source_identity(root, expected_head)
    tools_before = tool_identity(root)
    write_json(out / "before.json", {"source": before, "tools": tools_before})
    overlay = out / "old-production-overlay.json"
    write_json(
        overlay,
        {
            "Replace": {
                str(root / SOURCE_DIR / "optfacts.go"): str(
                    root / EVIDENCE_DIR / "old-optfacts.go.txt"
                )
            }
        },
    )
    go = str(root / "bin/go")
    common = (go, "test", "-json", "-count=1", "-p=1", "-parallel=1")
    negative = capture(
        (
            *common,
            f"-overlay={overlay}",
            "-timeout=10m",
            "-run",
            f"^{NEGATIVE_ROOT}$/^Project@simd128$/^mask=0$",
            PACKAGE,
        ),
        root,
        out,
        "negative",
        900,
    )
    if negative["accepted"] is not True:
        write_json(
            out / "acceptance.json",
            {"complete": False, "accepted": False, "blocked_phase": "negative"},
        )
        return False
    roots = sorted(name for name in CANDIDATE_NAMES if "/" not in name)
    candidate = capture(
        (*common, "-timeout=30m", "-run", "^(" + "|".join(roots) + ")$", PACKAGE),
        root,
        out,
        "candidate",
        2100,
    )
    after = source_identity(root, expected_head)
    tools_after = tool_identity(root)
    write_json(out / "after.json", {"source": after, "tools": tools_after})
    stable = before == after and tools_before == tools_after
    accepted = stable and candidate["accepted"] is True
    write_json(
        out / "acceptance.json",
        {
            "complete": True,
            "accepted": accepted,
            "source_and_tools_unchanged": stable,
            "candidate_named_events": len(CANDIDATE_NAMES),
            "candidate_leaf_cases": 142,
            "negative_named_events": len(NEGATIVE_NAMES),
        },
    )
    return accepted


class Arguments(argparse.Namespace):
    operation: str
    root: Path
    out: Path
    expected_head: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("verify", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args(namespace=Arguments())
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.operation == "verify":
        write_json(
            out / "verified-source.json",
            {
                "verified_at": datetime.now(UTC).isoformat(),
                "source": source_identity(root, args.expected_head),
                "python": sys.version,
                "resources": {
                    key: os.environ.get(key)
                    for key in (
                        "GOMAXPROCS",
                        "GOMEMLIMIT",
                        "GOGC",
                        "GOFLAGS",
                        "GOTOOLCHAIN",
                        "GOWORK",
                        "GOENV",
                        "GOCACHE",
                        "TMPDIR",
                    )
                },
                "github": {
                    key: os.environ.get(key)
                    for key in (
                        "GITHUB_RUN_ID",
                        "GITHUB_RUN_ATTEMPT",
                        "GITHUB_SHA",
                        "GITHUB_REF",
                    )
                },
            },
        )
        print("Frozen sources and clean checkout verified.")
        return 0
    return 0 if run_gate(root, out, args.expected_head) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"tagged-fact evidence rejected: {error}", file=sys.stderr)
        raise SystemExit(1) from error
