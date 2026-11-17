# pyright: strict
"""Run the frozen tagged-method controls with an installed Linux/amd64 toolchain."""

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
from pathlib import Path
from typing import NoReturn, cast

DIRECTORY = "logos-doc/validation/tagged-method-facts"
SOURCE = "src/cmd/compile/internal/logos"
LOGOS = "cmd/compile/internal/logos"
FACTS = f"{LOGOS}/facts"
MANIFEST_SHA256 = "679b1704bbd94ee292a8312cbacf315e55bbe41ac876b920bb3e549b5d17d9d3"
README_SHA256 = "a7d0930f18a490c611c2af1d4e4755bf80e4bf46e6a2eb3e7bb96c4ede198590"
CONSTRUCTOR = "TestTaggedMethodObjectConstructors"
GRAMMAR = "TestTaggedMethodCodecGrammar"
INTERFACE = "TestTaggedInterfaceMethodFactLookup"
GRAMMAR_CASES = (
    "Box.Read",
    "(*Box).Read",
    "Box@simd128",
    "(*Box)@simd128",
    "(Box).Read@simd128",
    "Box.Read@simd128",
    "(*Box).Read@simd128",
    "Box@simd128.Read",
    "(*Box@simd128).Read",
    "Box@simd0.Read",
    "Ω@simd128.Read",
    "Box@.Read",
    "Box@1.Read",
    "Box@simd128!.Read",
    "Box@@simd128.Read",
    "Box@simd128@again.Read",
    "Box@simd128.Read@again",
    "(*Box@simd128).Read@again",
    "(*Box@).Read",
    "(*Box@simd128.Read",
    "Box@simd128).Read",
    "Box@simd128..Read",
    "Box@simd128.Read.More",
    "local·1@simd128.Read",
    "Box[go.shape.int]@simd128.Read",
    "Box.Read@",
    "Box.Read@1",
    "Box.Read@simd128!",
    "Box.Read@simd128@again",
    "Box..Read@simd128",
    "Box.Read.More@simd128",
    "local·1.Read@simd128",
    "Box[go.shape.int].Read@simd128",
)


def facts_names() -> frozenset[str]:
    names = {CONSTRUCTOR, GRAMMAR, "TestTaggedInterfaceReadonlyFactWriter"}
    # These four intermediate names are actual t.Run calls, not slash prefixes.
    for receiver in ("Box", "Box@simd0", "Box@simd128", "Ω@simd128"):
        parent = f"{CONSTRUCTOR}/{receiver}"
        names.add(parent)
        names.update(
            f"{parent}/{variant}"
            for variant in (
                "value",
                "pointer",
                "alias_value",
                "alias_pointer",
                "const_value",
                "const_pointer",
                "pointer_to_const",
                "const_alias_pointer",
            )
        )
    names.update(f"{GRAMMAR}/{case}" for case in GRAMMAR_CASES)
    return frozenset(names)


def logos_names() -> frozenset[str]:
    groups = {
        "TestTaggedMethodFactLookup": ("value", "pointer"),
        "TestTaggedMethodFactRejectsInvalidMask": ("value", "pointer"),
        INTERFACE: (),
        "TestTaggedMethodStructuralFactLookup": ("value", "pointer"),
        "TestTaggedMethodOptionFactLookup": ("value", "pointer"),
        "TestTaggedMethodIdentityRemainsDistinct": (),
        "TestCompileSIMDTaggedMethodFacts": ("fork-go", "logos"),
    }
    return frozenset(groups) | frozenset(
        f"{root}/{child}" for root, children in groups.items() for child in children
    )


@dataclass(frozen=True)
class Check:
    label: str
    package: str
    names: frozenset[str]
    negative: tuple[str, str] | None = None
    overlay: tuple[str, str] | None = None

    def pattern(self) -> str:
        if self.negative is not None:
            return "/".join(
                f"^{re.escape(part)}$" for part in self.negative[0].split("/")
            )
        return (
            "^("
            + "|".join(sorted(name for name in self.names if "/" not in name))
            + ")$"
        )


CHECKS = (
    Check(
        "old-codec",
        FACTS,
        frozenset((GRAMMAR, f"{GRAMMAR}/Box@simd128.Read")),
        (f"{GRAMMAR}/Box@simd128.Read", "ValidObject = false, want true"),
        (f"{SOURCE}/facts/facts.go", f"{DIRECTORY}/old-facts.go.txt"),
    ),
    Check(
        "old-package-guard",
        LOGOS,
        frozenset((INTERFACE,)),
        (INTERFACE, "resolved an interface method symbol through the package scope"),
        (f"{SOURCE}/optfacts.go", f"{DIRECTORY}/old-optfacts.go.txt"),
    ),
    Check("facts", FACTS, facts_names()),
    Check("logos", LOGOS, logos_names()),
)


def reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-JSON numeric constant {value}")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return cast(dict[str, object], value)


def json_object(text: str) -> dict[str, object]:
    return mapping(
        json.loads(
            text, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
    )


def string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {value!r}")
    return value


def field(record: dict[str, object], name: str) -> str:
    result = string(record.get(name))
    if not result:
        raise ValueError(f"empty {name}")
    return result


@dataclass(frozen=True)
class Events:
    runs: tuple[str, ...]
    passes: tuple[str, ...]
    failures: tuple[str, ...]
    skips: tuple[str, ...]
    starts: int
    results: tuple[str, ...]
    build_failures: tuple[str, ...]
    output: tuple[tuple[str, str], ...]


def parse_events(text: str, check: Check) -> Events:
    outcomes: dict[str, list[str]] = {
        action: [] for action in ("run", "pass", "fail", "skip")
    }
    starts = 0
    active: set[str] = set()
    results: list[str] = []
    build_failures: list[str] = []
    output: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise ValueError(f"empty event at line {number}")
        event = json_object(line)
        action = field(event, "Action")
        if "FailedBuild" in event:
            build_failures.append(field(event, "FailedBuild"))
        if action in ("build-output", "build-fail"):
            if "Test" in event:
                raise ValueError("build event contains a Test field")
            path = string(event.get("ImportPath"))
            if action == "build-fail":
                build_failures.append(path)
            else:
                output.append(("", string(event.get("Output"))))
            continue
        if field(event, "Package") != check.package:
            raise ValueError("event belongs to an unexpected package")
        if "Test" in event:
            name = field(event, "Test")
            if name not in check.names:
                raise ValueError(f"unexpected test name {name!r}")
            if action in outcomes:
                if starts != 1 or results:
                    raise ValueError("test event outside its package lifetime")
                if action == "run":
                    if name in outcomes["run"]:
                        raise ValueError("duplicate test run")
                    parent = name.rpartition("/")[0]
                    if parent in check.names and parent not in active:
                        raise ValueError("child ran outside its parent test")
                    active.add(name)
                elif name not in active:
                    raise ValueError("test result without an active run")
                elif any(child.startswith(name + "/") for child in active):
                    raise ValueError("test completed before its child")
                else:
                    active.remove(name)
                outcomes[action].append(name)
            elif action == "output":
                output.append((name, string(event.get("Output"))))
            else:
                raise ValueError(f"unknown test action {action!r}")
        elif action == "start":
            starts += 1
        elif action in ("pass", "fail", "skip"):
            if starts != 1 or active or results:
                raise ValueError("package result outside its completed test lifetime")
            results.append(action)
        elif action == "output":
            output.append(("", string(event.get("Output"))))
        else:
            raise ValueError(f"unknown package action {action!r}")
    return Events(
        tuple(outcomes["run"]),
        tuple(outcomes["pass"]),
        tuple(outcomes["fail"]),
        tuple(outcomes["skip"]),
        starts,
        tuple(results),
        tuple(build_failures),
        tuple(output),
    )


def require_exact(
    actual: tuple[str, ...], expected: frozenset[str], label: str
) -> None:
    got, want = Counter(actual), Counter(expected)
    if got != want:
        raise ValueError(
            f"{label}: missing={dict(want - got)}, extra={dict(got - want)}"
        )


def accept(events: Events, check: Check, exit_code: int, stderr: bytes = b"") -> None:
    if stderr:
        raise ValueError("nonempty stderr; inspect the retained raw log")
    if events.build_failures or events.skips:
        raise ValueError("build failure or skipped test")
    output = "".join(text for _, text in events.output)
    if re.search(
        r"(?m)^(?:panic:|fatal error:|signal:|SIG[A-Z])|test timed out", output
    ):
        raise ValueError("crash or timeout diagnostic")
    require_exact(events.runs, check.names, "test runs")
    if events.starts != 1:
        raise ValueError("expected exactly one package start")
    if check.negative is None:
        require_exact(events.passes, check.names, "test passes")
        if exit_code != 0 or events.failures or events.results != ("pass",):
            raise ValueError("positive check lacks the exact exit/package outcome")
        return
    require_exact(events.failures, check.names, "negative failures")
    if exit_code != 1 or events.passes or events.results != ("fail",):
        raise ValueError("negative check lacks the exact exit/package outcome")
    name, diagnostic = check.negative
    named_output = "".join(text for owner, text in events.output if owner == name)
    if (
        re.search(
            r"(?m)^\s+\S+\.go:[1-9][0-9]*: " + re.escape(diagnostic) + r"$",
            named_output,
        )
        is None
    ):
        raise ValueError("missing exact diagnostic on the named negative test")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rooted(repo: Path, relative: str) -> Path:
    path = (repo / relative).resolve(strict=True)
    if not path.is_relative_to(repo) or not path.is_file():
        raise ValueError(f"input is not a file inside the repository: {relative}")
    return path


def verify_frozen(repo: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in (
        (f"{DIRECTORY}/frozen.json", MANIFEST_SHA256),
        (f"{DIRECTORY}/README.md", README_SHA256),
    ):
        actual[relative] = digest(rooted(repo, relative))
        if actual[relative] != expected:
            raise ValueError(f"frozen source digest mismatch: {relative}")
    manifest = json_object(
        (repo / DIRECTORY / "frozen.json").read_text(encoding="utf-8")
    )
    records: list[dict[str, object]] = [mapping(manifest["real_source"])]
    for group in ("files", "existing_package_controls", "negative_controls"):
        entries = manifest[group]
        if not isinstance(entries, list):
            raise TypeError(f"manifest {group} must be a list")
        records.extend(mapping(item) for item in cast(list[object], entries))
    for record in records:
        relative = field(record, "path")
        actual[relative] = digest(rooted(repo, relative))
        if actual[relative] != field(record, "sha256"):
            raise ValueError(f"frozen source digest mismatch: {relative}")
    return actual


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repo), *args), text=True, timeout=30
    ).strip()


def identity(repo: Path, frozen_paths: tuple[str, ...]) -> dict[str, object]:
    if git_value(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise ValueError("--repo must be the repository root")
    return {
        "head": git_value(repo, "rev-parse", "HEAD"),
        "tree": git_value(repo, "rev-parse", "HEAD^{tree}"),
        "source_tree": git_value(repo, "rev-parse", "HEAD:src"),
        "tracked_status": git_value(
            repo, "status", "--porcelain", "--untracked-files=no"
        ),
        "sha256": {
            **{path: digest(rooted(repo, path)) for path in frozen_paths},
            **{
                path: digest(rooted(repo, path))
                for path in (
                    f"{SOURCE}/facts/facts.go",
                    f"{SOURCE}/optfacts.go",
                    f"{DIRECTORY}/check_ci.py",
                    "bin/go",
                    "bin/logos",
                    "pkg/tool/linux_amd64/compile",
                    "pkg/tool/linux_amd64/link",
                )
            },
        },
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")


def environment(repo: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PATH": f"{repo / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
        "GOROOT": str(repo),
        "GOTOOLCHAIN": "local",
        "GOENV": "off",
        "GOWORK": "off",
        "GOFLAGS": "-mod=readonly -buildvcs=false",
        "GOEXPERIMENT": "none",
        "GODEBUG": "",
        "GOOS": "linux",
        "GOARCH": "amd64",
        "GOAMD64": "v1",
        "CGO_ENABLED": "0",
        "GOPROXY": "off",
        "GOSUMDB": "off",
        "GOTELEMETRY": "off",
    }


def capture(
    command: tuple[str, ...],
    repo: Path,
    out: Path,
    check: Check,
    env: dict[str, str],
    timeout: float,
) -> dict[str, object]:
    started = time.monotonic()
    stdout, stderr = out / f"{check.label}.jsonl", out / f"{check.label}.stderr.txt"
    result: dict[str, object] = {
        "command": command,
        "cwd": str(repo / "src"),
        "accepted": False,
        "exit_code": None,
        "timed_out": False,
    }
    try:
        with (
            stdout.open("xb") as output,
            stderr.open("xb") as errors,
            subprocess.Popen(
                command,
                cwd=repo / "src",
                env=env,
                stdout=output,
                stderr=errors,
                start_new_session=True,
            ) as process,
        ):
            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
        result.update(exit_code=exit_code, timed_out=timed_out)
        if timed_out:
            raise ValueError(f"command exceeded {timeout} seconds")
        accept(
            parse_events(stdout.read_text(encoding="utf-8"), check),
            check,
            exit_code,
            stderr.read_bytes(),
        )
        result["accepted"] = True
    except (OSError, TypeError, ValueError) as error:
        result["rejection"] = str(error)
    result.update(
        elapsed_seconds=time.monotonic() - started,
        stdout_sha256=digest(stdout),
        stderr_sha256=digest(stderr),
    )
    write_json(out / f"{check.label}.status.json", result)
    print(json.dumps({"phase": check.label, **result}, sort_keys=True), flush=True)
    return result


def run_gate(repo: Path, out: Path) -> bool:
    report: dict[str, object] = {"accepted": False, "complete": False, "phases": []}
    before: dict[str, object] | None = None
    frozen_paths: tuple[str, ...] = ()
    phases: list[dict[str, object]] = []
    try:
        frozen_paths = tuple(verify_frozen(repo))
        before = identity(repo, frozen_paths)
        env = environment(repo)
        write_json(out / "before.json", before)
        if before["tracked_status"]:
            raise ValueError("tracked checkout files are modified")
        write_json(
            out / "environment.json",
            {
                key: value
                for key, value in env.items()
                if key
                in (
                    "PATH",
                    "TMPDIR",
                    "GOROOT",
                    "GOTOOLCHAIN",
                    "GOENV",
                    "GOWORK",
                    "GOFLAGS",
                    "GOEXPERIMENT",
                    "GODEBUG",
                    "GOOS",
                    "GOARCH",
                    "GOAMD64",
                    "CGO_ENABLED",
                    "GOPROXY",
                    "GOSUMDB",
                    "GOTELEMETRY",
                    "GOCACHE",
                    "GOMODCACHE",
                    "GOMAXPROCS",
                    "GOMEMLIMIT",
                    "GOGC",
                    "GITHUB_RUN_ID",
                    "GITHUB_RUN_ATTEMPT",
                    "GITHUB_SHA",
                    "GITHUB_REF",
                )
            },
        )
        for check in CHECKS:
            overlay_flags: tuple[str, ...] = ()
            if check.overlay is not None:
                source, old = check.overlay
                overlay = out / f"{check.label}.overlay.json"
                write_json(
                    overlay,
                    {"Replace": {str(rooted(repo, source)): str(rooted(repo, old))}},
                )
                overlay_flags = (f"-overlay={overlay}",)
            long = check.label == "logos"
            command = (
                str(repo / "bin/go"),
                "test",
                "-json",
                "-count=1",
                "-p=1",
                "-parallel=1",
                "-timeout=30m" if long else "-timeout=10m",
                *overlay_flags,
                "-run",
                check.pattern(),
                check.package,
            )
            result = capture(command, repo, out, check, env, 2100 if long else 900)
            phases.append({"phase": check.label, **result})
            if result["accepted"] is not True:
                report["blocked_phase"] = check.label
                break
        report["complete"] = len(phases) == len(CHECKS)
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
        report["rejection"] = str(error)
    finally:
        report["phases"] = phases
        try:
            after = identity(repo, frozen_paths)
            write_json(out / "after.json", after)
            stable = before is not None and before == after
            report["source_and_tools_unchanged"] = stable
            report["accepted"] = (
                stable
                and len(phases) == len(CHECKS)
                and all(phase["accepted"] is True for phase in phases)
                and "rejection" not in report
            )
        except (OSError, TypeError, ValueError, subprocess.SubprocessError) as error:
            report["after_rejection"] = str(error)
        report["expected_positive_named_tests"] = sum(
            len(check.names) for check in CHECKS if check.negative is None
        )
        report["expected_positive_leaf_cases"] = 78
        write_json(out / "report.json", report)
    return report["accepted"] is True


class Arguments(argparse.Namespace):
    repo: Path
    evidence: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(namespace=Arguments())
    repo, out = args.repo.resolve(strict=True), args.evidence.resolve()
    if out.is_relative_to(repo) or repo.is_relative_to(out):
        raise ValueError("evidence must be a new directory outside the repository")
    if Path(__file__).resolve() != rooted(repo, f"{DIRECTORY}/check_ci.py"):
        raise ValueError("run the checker stored in the requested repository")
    out.mkdir(parents=True, exist_ok=False)
    return 0 if run_gate(repo, out) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        print(f"tagged-method evidence rejected: {error}", file=sys.stderr)
        raise SystemExit(1) from error
