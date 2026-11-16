"""Run the pinned modernc wake test with a separately reviewed capability policy."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from replay import (
    RUNNER,
    Case,
    assess_build,
    copy_source,
    decode,
    environment,
    execute,
    file_hash,
    obj,
    save_report,
    test_events,
    toolchain_identity,
    verify_source,
)
from run import Assessment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logos", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--module-cache", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--tmpdir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "report.json"
    if target.exists():
        raise ValueError("report already exists; use a fresh output directory")
    policy: Path = args.policy.resolve()
    if file_hash(policy) != args.policy_sha256:
        raise ValueError("reviewed policy checksum mismatch")
    baseline = obj(decode(args.baseline.read_text()))
    raw_rows = baseline["rows"]
    if not isinstance(raw_rows, list) or not baseline["complete"]:
        raise ValueError("baseline report must be complete")
    matches = [
        obj(row) for row in raw_rows if obj(row)["label"] == "modernc-sqlite-oracle"
    ]
    if len(matches) != 1 or matches[0]["status"] != "pass":
        raise ValueError("the same invocation must contain a passing modernc oracle")
    case = Case.parse(matches[0]["case"])
    source = verify_source(
        case, args.source_root.resolve(), args.module_cache.resolve(), RUNNER
    )
    if source.digest != matches[0]["source_digest"]:
        raise ValueError("source differs from the passing oracle")
    binary: Path = args.logos.resolve()
    old_tools = obj(baseline["tools"])
    if file_hash(binary) != obj(old_tools["logos"])["sha256"]:
        raise ValueError("logos driver differs from the unconfigured replay")
    env = environment(
        case, binary, args.module_cache.resolve(), args.cache.resolve(), 1
    )
    env.update(
        GOMEMLIMIT="512MiB",
        TMPDIR=str(args.tmpdir.resolve()),
        CC="/usr/bin/cc",
        CXX="/usr/bin/c++",
    )
    report: dict[str, object] = {
        "complete": False,
        "case": asdict(case),
        "source_digest": source.digest,
        "baseline_sha256": file_hash(args.baseline),
        "policy_sha256": file_hash(policy),
        "script_sha256": file_hash(Path(__file__)),
        "runner_sha256": file_hash(RUNNER / "run.py"),
        "driver_sha256": file_hash(binary),
        "toolchain": toolchain_identity(binary, env, RUNNER, out / "tools" / "logos"),
    }
    save_report(target, report)
    work = out / "work"
    copy_source(source, work)
    try:
        destination = work / "logos.toml"
        if destination.exists():
            raise ValueError("refuse to replace source-provided logos.toml")
        shutil.copyfile(policy, destination)
        shutil.copyfile(policy, out / "policy.toml")
        run = execute(
            (
                str(binary),
                "test",
                "-json",
                "-count=1",
                "-short",
                "-timeout=60s",
                "-run",
                "^TestMutexWakeOnce$",
                ".",
            ),
            work,
            env,
            out / "modernc-sqlite-logos",
            600,
        )
        assessment = assess_build(run, "logos", Assessment("pass"))
        events = test_events(run, ("TestMutexWakeOnce",))
        report.update(
            run=asdict(run), assessment=asdict(assessment), events=events, complete=True
        )
        report["status"] = (
            "pass"
            if assessment.status == "pass" and events["exact_match"]
            else assessment.status
            if assessment.status != "pass"
            else "failed"
        )
        save_report(target, report)
        print(json.dumps({key: report[key] for key in ("status", "events")}))
        return int(report["status"] != "pass")
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    raise SystemExit(main())
