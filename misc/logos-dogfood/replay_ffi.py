#!/usr/bin/env python3
# pyright: strict
"""Record eight explicit purego FFI tests without replacing its full suite."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from run import (
    Assessment,
    Case,
    assess_build,
    copy_source,
    decode,
    environment,
    execute,
    file_hash,
    git,
    obj,
    save_report,
    string,
    toolchain_identity,
    verify_source,
)

ROOT = Path(__file__).resolve().parent
TESTS = (
    "TestSimpleDlsym",
    "TestNestedDlopenCall",
    "TestSyscallN",
    "TestCallGoFromSharedLib",
    "TestNewCallbackFloat64AndIntMix",
    "TestNewCallbackInt64ResultWithStackArgs",
    "TestRegisterFunc_StackCallback",
    "TestRegisterFunc_StackAlignment",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "go-oracle",
        "go-candidate",
        "logos",
        "source-root",
        "module-cache",
        "cache-root",
        "tmpdir",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "report.json"
    if target.exists():
        raise ValueError("report already exists; use a fresh output directory")
    manifest = ROOT / "corpus.json"
    data = decode(manifest.read_text())
    if not isinstance(data, list):
        raise TypeError("manifest must be an array")
    cases = [Case.parse(value) for value in cast(list[object], data)]
    selected = [case for case in cases if case.id == "purego-ffi"]
    if len(selected) != 1:
        raise ValueError("manifest must contain exactly one purego-ffi case")
    case = replace(selected[0], id="purego-ffi-focused", patterns=(".",))
    source = verify_source(
        case, args.source_root.resolve(), args.module_cache.resolve(), ROOT
    )
    tools: dict[str, Path] = {
        "oracle": args.go_oracle.resolve(),
        "go": args.go_candidate.resolve(),
        "logos": args.logos.resolve(),
    }
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "complete": False,
        "scope": "eight selected upstream FFI tests; full suite remains separate",
        "case": asdict(case),
        "source_case_id": selected[0].id,
        "source_digest": source.digest,
        "expected_top_level_tests": TESTS,
        "manifest_sha256": file_hash(manifest),
        "runner_sha256": file_hash(ROOT / "run.py"),
        "replay_sha256": file_hash(Path(__file__)),
        "source_revision": git(ROOT.parents[1], "rev-parse", "HEAD"),
        "rows": rows,
    }
    save_report(target, report)
    baseline = None
    for name, binary in tools.items():
        env = environment(
            case,
            binary,
            args.module_cache.resolve(),
            args.cache_root.resolve() / name,
            1,
        )
        env.update(
            GOMEMLIMIT="384MiB",
            TMPDIR=str(args.tmpdir.resolve()),
            CC="/usr/bin/cc",
            CXX="/usr/bin/c++",
            PUREGO_TEST_PREBUILT_LIBDIR="",
        )
        identity = toolchain_identity(binary, env, ROOT, out / "tools" / name)
        work = out / "work"
        copy_source(source, work)
        run = execute(
            (
                str(binary),
                "test",
                "-json",
                "-count=1",
                "-short",
                "-timeout=120s",
                "-run",
                "^(" + "|".join(TESTS) + ")$",
                ".",
            ),
            work,
            env,
            out / name,
            600,
        )
        assessment = assess_build(run, name, baseline)
        events = [
            obj(decode(line)) for line in Path(run.stdout).read_text().splitlines()
        ]
        test_events = [event for event in events if "Test" in event]
        started = sorted(
            string(event["Test"])
            for event in test_events
            if event["Action"] == "run" and "/" not in string(event["Test"])
        )
        passed = sorted(
            string(event["Test"])
            for event in test_events
            if event["Action"] == "pass" and "/" not in string(event["Test"])
        )
        incomplete = [
            event for event in test_events if event["Action"] in ("fail", "skip")
        ]
        exact = started == sorted(TESTS) and passed == sorted(TESTS) and not incomplete
        if assessment.status == "pass" and not exact:
            assessment = Assessment(
                "upstream-failed" if name == "oracle" else "candidate-failed",
                "selected tests did not all execute and pass without skips",
            )
        if name == "oracle":
            baseline = assessment
        row: dict[str, object] = {
            "stage": name,
            "tool_sha256": file_hash(binary),
            "tool_identity": identity,
            "run": asdict(run),
            "assessment": asdict(assessment),
            "started": started,
            "passed": passed,
            "failed_or_skipped": incomplete,
            "exact_execution": exact,
            "status": "pass" if assessment.status == "pass" and exact else "failed",
        }
        rows.append(row)
        save_report(target, report)
        print(f"{name}: {row['status']}", flush=True)
        shutil.rmtree(work)
    report["complete"] = True
    save_report(target, report)
    return int(any(row["status"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
