"""Replay the focused runtime checks using the pinned corpus runner primitives."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RUNNER))

from run import (
    Case,
    Run,
    Source,
    all_files,
    assess_build,
    assess_lint,
    copy_source,
    decode,
    environment,
    execute,
    file_hash,
    git,
    logos_records,
    obj,
    save_report,
    string,
    toolchain_identity,
    tree_hash,
    verify_source,
)


def local_case(name: str) -> tuple[Case, Source]:
    root = RUNNER / "fixtures" / name
    files = all_files(root)
    digest = tree_hash(root, files)
    return (
        Case(
            name,
            "local",
            f"fixtures/{name}",
            f"example.com/logos-dogfood/{name}",
            "",
            "",
            "",
            digest,
            "stress",
            "go-only",
            0,
            (),
            (".",),
            "test",
        ),
        Source(root, files, digest),
    )


def expected_lint(source: Path) -> list[dict[str, object]]:
    lines = source.read_bytes().splitlines(keepends=True)

    def diagnostic(
        rule: str, line: int, col: int, token: str, message: str
    ) -> dict[str, object]:
        offset = sum(map(len, lines[: line - 1])) + col - 1
        assert lines[line - 1][col - 1 : col - 1 + len(token)] == token.encode()
        return {
            "schema_version": "1.0",
            "severity": "deny",
            "rule_id": rule,
            "file": "cases.go",
            "span": {
                "start_line": line,
                "start_col": col,
                "end_line": line,
                "end_col": col + len(token),
                "byte_offset": offset,
                "byte_len": len(token),
            },
            "message": message,
            "provenance": {"origin_expr": "", "origin_line": 0, "inferred_type": ""},
            "autofix_candidates": [],
        }

    leak = "r holds an io.Closer returned by acquire(): defer r.Close() or hand it to another owner"
    return [
        diagnostic("correctness::unclosed_io_closer", 34, 2, "r", leak),
        diagnostic("correctness::unclosed_io_closer", 39, 2, "r", leak),
        diagnostic(
            "correctness::infinite_recursion",
            54,
            6,
            "RecurseAfterShadowedPanic",
            "function RecurseAfterShadowedPanic calls itself with its own arguments on every path",
        ),
    ]


def test_events(run: Run, names: tuple[str, ...]) -> dict[str, object]:
    records = [obj(decode(line)) for line in Path(run.stdout).read_text().splitlines()]
    events = [record for record in records if "Test" in record]
    started = sorted(
        string(event["Test"]) for event in events if event["Action"] == "run"
    )
    passed = sorted(
        string(event["Test"]) for event in events if event["Action"] == "pass"
    )
    failed = [event for event in events if event["Action"] in ("fail", "skip")]
    return {
        "expected": names,
        "started": started,
        "passed": passed,
        "failed_or_skipped": failed,
        "exact_match": run.returncode == 0
        and started == sorted(names)
        and passed == sorted(names)
        and not failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--logos", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--module-cache", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tmpdir", type=Path, required=True)
    parser.add_argument("--part", choices=("language", "lint", "sqlite"), required=True)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "report.json"
    if target.exists():
        raise ValueError("report already exists; use a fresh output directory")
    tools = {
        "oracle": args.oracle.resolve(),
        "go": args.candidate.resolve(),
        "logos": args.logos.resolve(),
    }
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "complete": False,
        "part": args.part,
        "rows": rows,
        "runner_sha256": file_hash(RUNNER / "run.py"),
        "replay_sha256": file_hash(Path(__file__)),
        "source_revision": git(RUNNER.parents[1], "rev-parse", "HEAD"),
        "tools": {},
    }
    save_report(target, report)

    def env(case: Case, name: str) -> dict[str, str]:
        values = environment(
            case,
            tools[name],
            args.module_cache.resolve(),
            args.cache_root.resolve() / name,
            1,
        )
        values.update(
            GOMEMLIMIT="512MiB",
            TMPDIR=str(args.tmpdir.resolve()),
            CC="/usr/bin/cc",
            CXX="/usr/bin/c++",
        )
        return values

    identity_case, _ = local_case("language")
    report["tools"] = {
        name: {
            "sha256": file_hash(binary),
            "identity": toolchain_identity(
                binary, env(identity_case, name), RUNNER, out / "tools" / name
            ),
        }
        for name, binary in tools.items()
    }
    save_report(target, report)

    def record(label: str, run: Run, extra: dict[str, object]) -> None:
        rows.append({"label": label, "run": asdict(run), **extra})
        save_report(target, report)
        print(f"{label}: exit={run.returncode} {extra.get('status', '')}", flush=True)

    if args.part == "language":
        case, source = local_case("language")
        report["source_digest"] = source.digest
        work = out / "work"
        copy_source(source, work)
        native = execute(
            (str(tools["logos"]), "run", "."),
            work,
            env(case, "logos"),
            out / "native",
            180,
        )
        expected = b"language semantics ok\n"
        record(
            "native",
            native,
            {
                "status": "pass"
                if native.returncode == 0
                and Path(native.stdout).read_bytes() == expected
                else "failed"
            },
        )
        shutil.rmtree(work)
        copy_source(source, work)
        ejected = out / "ejected"
        eject = execute(
            (str(tools["logos"]), "eject", "-o", str(ejected)),
            work,
            env(case, "logos"),
            out / "eject",
            180,
        )
        record(
            "eject", eject, {"status": "pass" if eject.returncode == 0 else "failed"}
        )
        shutil.rmtree(work)
        if eject.returncode == 0:
            report["ejected_digest"] = tree_hash(ejected, all_files(ejected))
            oracle = execute(
                (str(tools["oracle"]), "run", "."),
                ejected,
                env(case, "oracle"),
                out / "ejected-oracle",
                180,
            )
            record(
                "ejected-oracle",
                oracle,
                {
                    "status": "pass"
                    if oracle.returncode == 0
                    and Path(oracle.stdout).read_bytes() == expected
                    and Path(oracle.stdout).read_bytes()
                    == Path(native.stdout).read_bytes()
                    else "failed"
                },
            )
    elif args.part == "lint":
        case, source = local_case("lint")
        report["source_digest"] = source.digest
        expected_records = expected_lint(source.path / "cases.go")
        (out / "expected.json").write_text(
            json.dumps(expected_records, indent=2) + "\n"
        )
        work = out / "work"
        for label in ("lint", "lint-repeat"):
            copy_source(source, work)
            run = execute(
                (str(tools["logos"]), "lint", "--json-diagnostics", "."),
                work,
                env(case, "logos"),
                out / label,
                180,
            )
            assessment = assess_lint(run, "logos-lint")
            actual = logos_records(Path(run.stderr).read_text())
            for diagnostic in actual:
                diagnostic["file"] = Path(string(diagnostic["file"])).name
            exact = sorted(json.dumps(row, sort_keys=True) for row in actual) == sorted(
                json.dumps(row, sort_keys=True) for row in expected_records
            )
            record(
                label,
                run,
                {
                    "assessment": asdict(assessment),
                    "exact_match": exact,
                    "status": "pass"
                    if exact and assessment.status == "findings"
                    else "failed",
                },
            )
            shutil.rmtree(work)
        names = ("TestExecutedCloseForms", "TestShadowedPanicReturnsTypedNil")
        for name, binary in tools.items():
            copy_source(source, work)
            run = execute(
                (str(binary), "test", "-json", "-count=1", "-timeout=60s", "."),
                work,
                env(case, name),
                out / f"runtime-{name}",
                180,
            )
            events = test_events(run, names)
            record(
                f"runtime-{name}",
                run,
                {
                    "events": events,
                    "status": "pass" if events["exact_match"] else "failed",
                },
            )
            shutil.rmtree(work)
    else:
        raw = decode((RUNNER / "corpus.json").read_text())
        if not isinstance(raw, list):
            raise TypeError("manifest must be an array")
        selected = {
            "modernc-sqlite": ("TestMutexWakeOnce",),
            "sqlite3-cgo": ("TestOpen", "TestClose", "TestInsert"),
        }
        for item in raw:
            case = Case.parse(item)
            if case.id not in selected:
                continue
            source = verify_source(
                case, args.source_root.resolve(), args.module_cache.resolve(), RUNNER
            )
            names = selected[case.id]
            oracle_assessment = None
            work = out / "work"
            for name, binary in tools.items():
                copy_source(source, work)
                run = execute(
                    (
                        str(binary),
                        "test",
                        "-json",
                        "-count=1",
                        "-short",
                        "-timeout=60s",
                        "-run",
                        "^(" + "|".join(names) + ")$",
                        ".",
                    ),
                    work,
                    env(case, name),
                    out / f"{case.id}-{name}",
                    600,
                )
                assessment = assess_build(run, name, oracle_assessment)
                if name == "oracle":
                    oracle_assessment = assessment
                events = test_events(run, names)
                record(
                    f"{case.id}-{name}",
                    run,
                    {
                        "case": asdict(case),
                        "source_digest": source.digest,
                        "assessment": asdict(assessment),
                        "events": events,
                        "status": "pass"
                        if assessment.status == "pass" and events["exact_match"]
                        else assessment.status
                        if assessment.status != "pass"
                        else "failed",
                    },
                )
                shutil.rmtree(work)
    report["complete"] = True
    save_report(target, report)
    return int(any(row["status"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
