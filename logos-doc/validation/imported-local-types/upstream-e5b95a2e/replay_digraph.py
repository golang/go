"""Run the pinned digraph tests under three toolchains in fresh private copies."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "misc" / "logos-dogfood"
sys.path.insert(0, str(RUNNER))

from run import (
    Case,
    Run,
    Source,
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
    tree_hash,
    verify_source,
    within,
)

EXPECTED_TESTS = tuple(
    sorted(
        (
            "TestDigraph",
            "TestAllpaths",
            "TestSomepath",
            "TestSplit",
            "TestQuotedLength",
            "TestFocus",
            "TestToDot",
        )
    )
)


def test_events(run: Run) -> dict[str, object]:
    records = [obj(decode(line)) for line in Path(run.stdout).read_text().splitlines()]
    events = [record for record in records if "Test" in record]
    started = sorted(
        string(event["Test"]) for event in events if event["Action"] == "run"
    )
    passed = sorted(
        string(event["Test"]) for event in events if event["Action"] == "pass"
    )
    failed = [event for event in events if event["Action"] in ("fail", "skip")]
    top_started = tuple(name for name in started if "/" not in name)
    top_passed = tuple(name for name in passed if "/" not in name)
    package_passed = any(
        "Test" not in record
        and record.get("Action") == "pass"
        and record.get("Package") == "golang.org/x/tools/cmd/digraph"
        for record in records
    )
    return {
        "expected_top_level": EXPECTED_TESTS,
        "top_level_started": top_started,
        "top_level_passed": top_passed,
        "started": started,
        "passed": passed,
        "failed_or_skipped": failed,
        "exact_match": run.returncode == 0
        and not run.timed_out
        and not run.error
        and package_passed
        and top_started == EXPECTED_TESTS
        and top_passed == EXPECTED_TESTS
        and started == passed
        and len(set(started)) == len(started)
        and not failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "oracle",
        "candidate",
        "logos",
        "module-cache",
        "cache-root",
        "tmpdir",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    modules: Path = args.module_cache.resolve()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = out / "report.json"
    tools = {
        "oracle": args.oracle.resolve(),
        "go": args.candidate.resolve(),
        "logos": args.logos.resolve(),
    }
    case = Case(
        "tools-digraph",
        "module",
        "golang.org/x/tools@v0.52.0",
        "golang.org/x/tools",
        "https://go.googlesource.com/tools",
        "16517d951552611e7bd358c48ee1051fd0e57b0c",
        "v0.52.0",
        "h1:ryxg4U63oaUYk7uOLTF37n43f4qgtGvqV4Ioit1BaEs=",
        "reference",
        "go-only",
        0,
        (),
        ("./cmd/digraph",),
        "test",
    )
    upstream = verify_source(case, modules, modules, RUNNER)
    manifest = Path(__file__).with_name("upstream-input.sha256")
    inputs = tuple(line.split("  ", 1) for line in manifest.read_text().splitlines())
    for digest, name in inputs:
        if file_hash(within(upstream.path, name)) != digest:
            raise ValueError(f"focused input differs from pinned source: {name}")
    files = tuple(name for _, name in inputs)
    subset = Source(upstream.path, files, tree_hash(upstream.path, files))
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "complete": False,
        "all_passed": False,
        "case": asdict(case),
        "source_digest": upstream.digest,
        "subset_digest": subset.digest,
        "input_manifest_sha256": file_hash(manifest),
        "runner_sha256": file_hash(RUNNER / "run.py"),
        "replay_sha256": file_hash(Path(__file__)),
        "checkout_revision": git(ROOT, "rev-parse", "HEAD"),
        "rows": rows,
    }
    save_report(target, report)
    for name, binary in tools.items():
        work = out / name / "work"
        copy_source(subset, work)
        env = environment(case, binary, modules, args.cache_root.resolve() / name, 1)
        env.update(GOMEMLIMIT="384MiB", GOGC="30", TMPDIR=str(args.tmpdir.resolve()))
        identity = toolchain_identity(binary, env, work, out / name / "identity")
        run = execute(
            (
                str(binary),
                "test",
                "-p=1",
                "-count=1",
                "-json",
                "-timeout=120s",
                "./cmd/digraph",
            ),
            work,
            env,
            out / name / "tests",
            180,
        )
        events = test_events(run)
        unchanged = tree_hash(work, files) == subset.digest
        rows.append(
            {
                "driver": name,
                "driver_sha256": file_hash(binary),
                "identity": identity,
                "run": asdict(run),
                "tests": events,
                "source_unchanged": unchanged,
                "passed": events["exact_match"] is True and unchanged,
            }
        )
        save_report(target, report)
        print(
            f"{name}: exit={run.returncode}, tests={events['exact_match']}, source_unchanged={unchanged}",
            flush=True,
        )
    report["complete"] = True
    report["all_passed"] = all(row["passed"] is True for row in rows)
    save_report(target, report)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
