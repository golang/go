# pyright: strict
"""Accept a complete corpus replay, with one explicit upstream load-failure control."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import cast

from run import (
    STAGES,
    SUCCESS,
    Assessment,
    Case,
    Run,
    assess_build,
    assess_lint,
    compare_repeats,
    decode,
    file_hash,
    logos_records,
    obj,
    save_report,
    string,
    strings,
    verify_policy,
)

MODES = ("basic", "standard", "strict")
LOAD_PATH = "golang.org/x/tools@v0.20.0/internal/tokeninternal/tokeninternal.go"
LOAD_MESSAGE = "invalid array length -delta * delta (constant -256 of type int64)"
CELLS = tuple(
    (stage, mode)
    for stage in STAGES
    for mode in (MODES if stage == "gglint" else ("",))
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def objects(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise TypeError("expected an array of objects")
    return tuple(obj(item) for item in cast(list[object], value))


def text_map(value: object) -> dict[str, str]:
    return {key: string(item) for key, item in obj(value).items()}


def integer(value: object) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


def sha256(value: object) -> str:
    digest = string(value)
    require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "invalid SHA-256")
    return digest


def content_digest(value: object) -> str:
    digest = string(value)
    require(
        re.fullmatch(r"h1:[A-Za-z0-9+/]{43}=", digest) is not None,
        "invalid source content digest",
    )
    return digest


def checked_log(value: object, digest: object, root: Path) -> Path:
    path = Path(string(value)).resolve(strict=True)
    require(path.is_relative_to(root.resolve()), "log escapes the evidence directory")
    require(path.is_file() and file_hash(path) == sha256(digest), "log hash differs")
    return path


def read_run(value: object, root: Path) -> Run:
    data = obj(value)
    require(data.get("timed_out") is False, "command timed out or lacks timeout state")
    require(data.get("error") == "", "command could not run")
    code = integer(data.get("returncode"))
    require(code in (0, 1), "command exited abnormally")
    elapsed = data.get("seconds")
    require(type(elapsed) in (int, float), "invalid elapsed time")
    seconds = float(cast(int | float, elapsed))
    require(math.isfinite(seconds) and seconds >= 0, "invalid elapsed time")
    out = checked_log(data.get("stdout"), data.get("stdout_sha256"), root)
    err = checked_log(data.get("stderr"), data.get("stderr_sha256"), root)
    command = strings(data.get("command"))
    require(bool(command), "command is empty")
    return Run(
        command,
        string(data.get("cwd")),
        text_map(data.get("environment")),
        code,
        False,
        "",
        seconds,
        str(out),
        str(err),
        file_hash(out),
        file_hash(err),
    )


def known_load_failure(run: Run, stage: str) -> None:
    """Require the pinned tokeninternal error; unrelated failures never qualify."""
    require(run.returncode == 1, "load-failure control must exit one")
    stdout = Path(run.stdout).read_text()
    stderr = Path(run.stderr).read_text()
    if stage == "gglint":
        require(
            all(
                line.startswith("gglint: config ")
                for line in stderr.splitlines()
                if line.strip()
            ),
            "load-failure control has unexplained gglint stderr",
        )
        errors = tuple(
            record
            for record in objects(obj(decode(stdout)).get("findings"))
            if record.get("rule") == "typecheck"
        )
        require(bool(errors), "missing expected typecheck finding")
        for record in errors:
            require(
                string(record.get("path")).endswith("/" + LOAD_PATH)
                and record.get("line") == 64
                and record.get("column") == 9
                and record.get("message") == LOAD_MESSAGE,
                "unexpected typecheck finding",
            )
        return
    require(not stdout.strip(), "load-failure control has unexpected stdout")
    if stage == "logos-lint":
        errors = tuple(
            record for record in logos_records(stderr) if not record["rule_id"]
        )
        require(bool(errors), "missing expected compiler-error record")
        for record in errors:
            span = obj(record["span"])
            require(
                string(record["file"]).endswith("/" + LOAD_PATH)
                and span["start_line"] == 64
                and span["start_col"] == 9
                and record["message"] == LOAD_MESSAGE,
                "unexpected compiler-error record",
            )
        return
    errors_seen = 0
    for line in stderr.splitlines():
        if not line.strip() or line == "# golang.org/x/tools/internal/tokeninternal":
            continue
        require(
            re.fullmatch(r".+/" + re.escape(LOAD_PATH + ":64:9: " + LOAD_MESSAGE), line)
            is not None,
            "unexpected compiler failure",
        )
        errors_seen += 1
    require(errors_seen > 0, "missing expected compiler failure")


def check_inputs(value: object, case: Case, stage: str, source_digest: str) -> None:
    inputs = obj(value)
    effective = content_digest(inputs.get("effective_source_digest"))
    if case.logos_policy and stage in ("logos", "logos-lint"):
        require(
            inputs.get("policy_sha256") == case.logos_policy_sha256
            and effective != source_digest,
            "reviewed policy was not applied to the native input",
        )
    else:
        require(
            inputs == {"effective_source_digest": source_digest},
            "pristine input changed or received a policy",
        )


def check_invocation(
    command: Run,
    case: Case,
    stage: str,
    mode: str,
    tools: dict[str, Path],
    jobs: int,
    root: Path,
) -> None:
    """Bind coverage labels to the actual driver, patterns, tags, and cgo mode."""
    require(
        command.cwd == str(root / "work" / case.id),
        "command did not use the verified private source directory",
    )
    tool = "logos" if stage == "logos-lint" else stage
    binary = tools[tool]
    flags = (f"-tags={','.join(case.tags)}",) if case.tags else ()
    if stage == "gglint":
        config = root / "config" / f"{case.id}.toml"
        require(
            config.read_text()
            == f"[run]\ntests = true\nbuild_tags = {json.dumps(case.tags)}\n",
            "gglint configuration differs from the case",
        )
        expected = (
            str(binary),
            "run",
            "-config",
            str(config),
            "-mode",
            mode,
            "-format",
            "json",
            *case.patterns,
        )
    elif stage == "logos-lint":
        expected = (str(binary), "lint", "--json-diagnostics", *flags, *case.patterns)
    else:
        expected = (str(binary), "build", *flags, *case.patterns)
    require(command.command == expected, "recorded command differs from the case")
    required = {
        "CGO_ENABLED": str(case.cgo),
        "GOFLAGS": f"-mod=readonly -buildvcs=false -p={jobs}",
        "GOMAXPROCS": str(jobs),
        "GOTOOLCHAIN": "local",
        "GOWORK": "off",
        "GOENV": "off",
        "GOPROXY": "off",
        "GOSUMDB": "off",
        "GOTELEMETRY": "off",
    }
    require(
        all(command.environment.get(key) == value for key, value in required.items()),
        "recorded build environment differs from the case",
    )
    driver = tools["oracle"] if stage == "gglint" else binary
    require(
        command.environment.get("PATH", "").split(os.pathsep)[0] == str(driver.parent),
        "toolchain PATH differs from the requested driver",
    )
    require("GOEXPERIMENT" not in command.environment, "unexpected experiment override")


def check_case(
    row: dict[str, object], case: Case, root: Path, tools: dict[str, Path], jobs: int
) -> dict[str, int]:
    require(Case.parse(row.get("case")) == case, "case differs from the manifest")
    require(row.get("status") == "finished", "case did not finish")
    digest = content_digest(row.get("source_digest"))
    if case.source_kind != "git":
        require(digest == case.sum, "source checksum differs from the pin")
    if case.logos_policy:
        policy = obj(row.get("logos_policy"))
        require(policy.get("sha256") == case.logos_policy_sha256, "policy pin differs")
        checked_log(policy.get("saved_path"), case.logos_policy_sha256, root)
    entries = objects(row.get("stages"))
    require(
        tuple((entry.get("stage"), entry.get("mode")) for entry in entries) == CELLS,
        "missing, duplicated, or reordered corpus cells",
    )
    counts: dict[str, int] = {}
    oracle: Assessment | None = None
    for entry, (stage, mode) in zip(entries, CELLS, strict=True):
        check_inputs(entry.get("inputs"), case, stage, digest)
        command = read_run(entry.get("run"), root)
        check_invocation(command, case, stage, mode, tools, jobs, root)
        lint = stage in ("logos-lint", "gglint")
        assessment = (
            assess_lint(command, stage, mode)
            if lint
            else assess_build(command, stage, oracle)
        )
        if stage == "oracle":
            oracle = assessment
        if case.id == "decorder":
            expected = (
                "load-failed"
                if lint
                else ("upstream-failed" if stage == "oracle" else "uncompared")
            )
            require(
                assessment.status == expected, "load-failure control changed status"
            )
            require("repeat" not in entry, "unexpected repeat of a failed load")
            known_load_failure(command, stage)
        else:
            require(
                assessment.status in SUCCESS, f"{case.id}/{stage}/{mode}: {assessment}"
            )
            if lint:
                check_inputs(entry.get("repeat_inputs"), case, stage, digest)
                require(
                    entry["repeat_inputs"] == entry["inputs"], "repeat input changed"
                )
                repeat = read_run(entry.get("repeat"), root)
                check_invocation(repeat, case, stage, mode, tools, jobs, root)
                require(
                    (repeat.command, repeat.cwd, repeat.environment)
                    == (command.command, command.cwd, command.environment),
                    "repeat command or environment changed",
                )
                repeated = assess_lint(repeat, stage, mode)
                require(
                    entry.get("repeat_assessment") == asdict(repeated),
                    "repeat assessment differs",
                )
                assessment = compare_repeats(assessment, repeated, command, repeat)
                require(assessment.status in SUCCESS, "lint report changed on repeat")
        require(
            entry.get("assessment") == asdict(assessment), "stored assessment differs"
        )
        counts[assessment.status] = counts.get(assessment.status, 0) + 1
    return counts


def accept(
    report: dict[str, object], cases: tuple[Case, ...], root: Path
) -> dict[str, object]:
    """Return acceptance only after checking every log, cell, and lint repeat."""
    require(
        report.get("schema_version") == 1 and report.get("complete") is True,
        "incomplete report",
    )
    require(strings(report.get("stages")) == STAGES, "stage selection differs")
    require(strings(report.get("modes")) == MODES, "mode selection differs")
    require(
        report.get("operation_override") == "build",
        "expected compile-only corpus sweep",
    )
    require(
        bool(cases) and len({case.id for case in cases}) == len(cases),
        "empty or duplicate manifest",
    )
    require(
        sum(case.id == "decorder" for case in cases) == 1,
        "missing explicit failed-load control",
    )
    require(report.get("case_count") == len(cases), "case count differs")
    jobs = integer(report.get("jobs"))
    require(jobs > 0, "invalid worker count")
    tools = {
        name: Path(string(obj(value).get("path"))).resolve()
        for name, value in obj(report.get("tools")).items()
    }
    require(
        set(tools) == {"oracle", "go", "logos", "gglint"}, "incomplete tool identities"
    )
    rows = objects(report.get("cases"))
    require(len(rows) == len(cases), "incomplete case coverage")
    counts: dict[str, int] = {}
    for row, case in zip(rows, cases, strict=True):
        for status, count in check_case(row, case, root, tools, jobs).items():
            counts[status] = counts.get(status, 0) + count
    require(report.get("counts") == counts, "summary counts differ from raw evidence")
    return {
        "accepted": True,
        "case_count": len(cases),
        "primary_cells": len(cases) * len(CELLS),
        "accepted_positive_cases": len(cases) - 1,
        "incomplete_analysis_cases": ["decorder"],
        "known_failed_load_cells": len(CELLS),
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest: Path = args.manifest.resolve()
    report_path: Path = args.report.resolve()
    output = report_path.with_name("acceptance.json")
    require(not output.exists(), "acceptance evidence already exists")
    cases = tuple(Case.parse(value) for value in objects(decode(manifest.read_text())))
    baseline = tuple(
        Case.parse(value)
        for value in objects(decode(manifest.with_name("corpus.json").read_text()))
    )
    require(
        tuple(
            {
                key: value
                for key, value in asdict(case).items()
                if not key.startswith("logos_policy")
            }
            for case in cases
        )
        == tuple(
            {
                key: value
                for key, value in asdict(case).items()
                if not key.startswith("logos_policy")
            }
            for case in baseline
        ),
        "CI manifest changed corpus selection or source pins",
    )
    for case in cases:
        verify_policy(case, manifest.parent)
    report = obj(decode(report_path.read_text()))
    require(
        report.get("manifest_sha256") == file_hash(manifest), "manifest hash differs"
    )
    require(
        report.get("runner_sha256") == file_hash(Path(__file__).with_name("run.py")),
        "runner hash differs",
    )
    tools = obj(report.get("tools"))
    require(
        set(tools) == {"oracle", "go", "logos", "gglint"}, "tool identity is incomplete"
    )
    for identity in tools.values():
        tool = obj(identity)
        require(
            file_hash(Path(string(tool.get("path")))) == sha256(tool.get("sha256")),
            "tool binary changed",
        )
    result = accept(report, cases, report_path.parent)
    save_report(
        output,
        {
            **result,
            "report_sha256": file_hash(report_path),
            "checker_sha256": file_hash(Path(__file__)),
        },
    )
    print(
        f"Accepted {result['primary_cells']} cells; decorder remains an incomplete analysis."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
