# pyright: strict
"""Negative controls for corpus completeness and exact failed-load acceptance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from accept_corpus import CELLS, LOAD_MESSAGE, LOAD_PATH, accept, objects
from hypothesis import given
from hypothesis import strategies as st
from run import Assessment, Case, Run, decode, file_hash, obj, strings


def manifest_cases() -> tuple[Case, ...]:
    rows = objects(decode(Path(__file__).with_name("corpus.json").read_text()))
    selected = {str(row["id"]): Case.parse(row) for row in rows}
    return selected["fixture-semantics"], replace(
        selected["decorder"], logos_policy="", logos_policy_sha256=""
    )


def run_record(root: Path, label: str, stdout: str, stderr: str, code: int) -> Run:
    out = root / f"{label}.stdout"
    err = root / f"{label}.stderr"
    out.write_text(stdout)
    err.write_text(stderr)
    return Run(
        ("/tool", "run"),
        str(root),
        {"GOMAXPROCS": "2"},
        code,
        False,
        "",
        0.01,
        str(out),
        str(err),
        file_hash(out),
        file_hash(err),
    )


def complete_report(root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    tools = {
        name: f"/tools/{name}/bin/tool" for name in ("oracle", "go", "logos", "gglint")
    }
    config_dir = root / "config"
    config_dir.mkdir(exist_ok=True)
    for case in manifest_cases():
        entries: list[dict[str, object]] = []
        for stage, mode in CELLS:
            stdout = stderr = ""
            code = 0
            lint = stage in ("gglint", "logos-lint")
            if case.id == "decorder":
                code = 1
                status = (
                    "load-failed"
                    if lint
                    else ("upstream-failed" if stage == "oracle" else "uncompared")
                )
                reason = (
                    "typecheck findings prevent complete analysis"
                    if stage == "gglint"
                    else "ordinary compiler errors prevent complete analysis"
                    if lint
                    else "oracle was not run or did not pass; inspect candidate exit status"
                    if stage != "oracle"
                    else ""
                )
                assessment = Assessment(status, reason, diagnostics=int(lint))
                if stage == "gglint":
                    stdout = json.dumps(
                        {
                            "mode": mode,
                            "findings": [
                                {
                                    "rule": "typecheck",
                                    "path": "/modules/" + LOAD_PATH,
                                    "line": 64,
                                    "column": 9,
                                    "message": LOAD_MESSAGE,
                                }
                            ],
                        }
                    )
                    stderr = "gglint: config config.toml\n"
                elif lint:
                    stderr = (
                        Path(__file__)
                        .with_name("testdata")
                        .joinpath("compiler-error.jsonl")
                        .read_text()
                    )
                else:
                    stderr = (
                        "# golang.org/x/tools/internal/tokeninternal\n/modules/"
                        + LOAD_PATH
                        + ":64:9: "
                        + LOAD_MESSAGE
                        + "\n"
                    )
            elif lint:
                assessment = Assessment(
                    "clean", normalized_sha256=hashlib.sha256(b"").hexdigest()
                )
                if stage == "gglint":
                    stdout = json.dumps({"mode": mode, "findings": []})
            else:
                assessment = Assessment("pass")
            command = run_record(
                root, f"{case.id}-{stage}-{mode}", stdout, stderr, code
            )
            tool = "logos" if stage == "logos-lint" else stage
            driver = "oracle" if stage == "gglint" else tool
            config = config_dir / f"{case.id}.toml"
            config.write_text("[run]\ntests = true\nbuild_tags = []\n")
            argv = (
                (
                    tools[tool],
                    "run",
                    "-config",
                    str(config),
                    "-mode",
                    mode,
                    "-format",
                    "json",
                    *case.patterns,
                )
                if stage == "gglint"
                else (tools[tool], "lint", "--json-diagnostics", *case.patterns)
                if stage == "logos-lint"
                else (tools[tool], "build", *case.patterns)
            )
            env = {
                "PATH": f"/tools/{driver}/bin:/usr/bin",
                "CGO_ENABLED": str(case.cgo),
                "GOFLAGS": "-mod=readonly -buildvcs=false -p=2",
                "GOMAXPROCS": "2",
                "GOTOOLCHAIN": "local",
                "GOWORK": "off",
                "GOENV": "off",
                "GOPROXY": "off",
                "GOSUMDB": "off",
                "GOTELEMETRY": "off",
            }
            command = replace(
                command, command=argv, environment=env, cwd=str(root / "work" / case.id)
            )
            inputs: dict[str, object] = {"effective_source_digest": case.sum}
            entry: dict[str, object] = {
                "stage": stage,
                "mode": mode,
                "inputs": inputs,
                "run": asdict(command),
                "assessment": asdict(assessment),
            }
            if case.id != "decorder" and lint:
                repeated = run_record(
                    root, f"{case.id}-{stage}-{mode}-repeat", stdout, stderr, code
                )
                repeated = replace(
                    repeated,
                    command=argv,
                    environment=env,
                    cwd=str(root / "work" / case.id),
                )
                entry.update(
                    repeat=asdict(repeated),
                    repeat_assessment=asdict(assessment),
                    repeat_inputs=inputs.copy(),
                )
            entries.append(entry)
            counts[assessment.status] = counts.get(assessment.status, 0) + 1
        rows.append(
            {
                "case": asdict(case),
                "status": "finished",
                "source_digest": case.sum,
                "stages": entries,
            }
        )
    # JSON serialization mirrors the public evidence format, including tuple arrays.
    return obj(
        decode(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": True,
                    "stages": ["oracle", "go", "logos", "logos-lint", "gglint"],
                    "modes": ["basic", "standard", "strict"],
                    "operation_override": "build",
                    "case_count": 2,
                    "jobs": 2,
                    "tools": {
                        name: {"path": path, "sha256": "0" * 64}
                        for name, path in tools.items()
                    },
                    "cases": rows,
                    "counts": counts,
                }
            )
        )
    )


def cell(
    report: dict[str, object], case_id: str, stage: str, mode: str = ""
) -> dict[str, object]:
    for row in objects(report["cases"]):
        if obj(row["case"])["id"] == case_id:
            return next(
                entry
                for entry in objects(row["stages"])
                if (entry["stage"], entry["mode"]) == (stage, mode)
            )
    raise AssertionError("fixture cell missing")


def rewrite_log(command: dict[str, object], name: str, text: str) -> None:
    path = Path(str(command[name]))
    path.write_text(text)
    command[name + "_sha256"] = file_hash(path)


def test_complete_evidence_keeps_failed_load_separate(tmp_path: Path) -> None:
    result = accept(complete_report(tmp_path), manifest_cases(), tmp_path)
    assert result["accepted"] is True
    assert result["primary_cells"] == 14
    assert result["accepted_positive_cases"] == 1
    assert result["incomplete_analysis_cases"] == ["decorder"]
    assert result["known_failed_load_cells"] == 7


@pytest.mark.parametrize(
    "defect",
    [
        "incomplete",
        "missing-case",
        "missing-strict",
        "duplicate-basic",
        "forged-counts",
        "changed-pin",
    ],
)
def test_incomplete_or_changed_campaign_is_rejected(
    tmp_path: Path, defect: str
) -> None:
    report = complete_report(tmp_path)
    rows = objects(report["cases"])
    if defect == "incomplete":
        report["complete"] = False
    elif defect == "missing-case":
        report["cases"] = [rows[0]]
    elif defect == "forged-counts":
        report["counts"] = {"pass": 14}
    elif defect == "changed-pin":
        obj(rows[1]["case"])["version"] = "v99.0.0"
    else:
        entries = list(objects(rows[0]["stages"]))
        if defect == "missing-strict":
            entries.pop()
        else:
            entries[-1] = entries[-3]
        rows[0]["stages"] = entries
    with pytest.raises(ValueError):
        accept(report, manifest_cases(), tmp_path)


@pytest.mark.parametrize(
    "defect",
    [
        "timeout",
        "signal",
        "error",
        "tamper",
        "outside-log",
        "empty-error",
        "wrong-error",
        "extra-error",
        "stderr-error",
        "missing-repeat",
        "changed-repeat",
        "changed-input",
        "policy-on-oracle",
    ],
)
def test_raw_evidence_defects_are_rejected(tmp_path: Path, defect: str) -> None:
    report = complete_report(tmp_path)
    control = cell(report, "decorder", "gglint", "strict")
    command = obj(control["run"])
    if defect == "timeout":
        command["timed_out"] = True
    elif defect == "signal":
        command["returncode"] = -9
    elif defect == "error":
        command["error"] = "executable missing"
    elif defect == "tamper":
        Path(str(command["stdout"])).write_text("replaced after measurement")
    elif defect == "outside-log":
        command["stdout"] = __file__
        command["stdout_sha256"] = file_hash(Path(__file__))
    elif defect in ("empty-error", "wrong-error", "extra-error"):
        document = obj(decode(Path(str(command["stdout"])).read_text()))
        findings = list(objects(document["findings"]))
        if defect == "empty-error":
            findings.clear()
        elif defect == "wrong-error":
            findings[0]["message"] = "cannot load unrelated package"
        else:
            findings.append({**findings[0], "message": "internal compiler error"})
        document["findings"] = findings
        rewrite_log(command, "stdout", json.dumps(document))
    elif defect == "stderr-error":
        rewrite_log(
            command,
            "stderr",
            "gglint: config config.toml\nvet failed for another reason\n",
        )
    elif defect in ("missing-repeat", "changed-repeat", "changed-input"):
        positive = cell(report, "fixture-semantics", "gglint", "standard")
        if defect == "missing-repeat":
            del positive["repeat"]
        elif defect == "changed-repeat":
            obj(positive["repeat"])["environment"] = {"GOMAXPROCS": "1"}
        else:
            obj(positive["repeat_inputs"])["effective_source_digest"] = (
                "h1:" + "A" * 43 + "="
            )
    else:
        obj(cell(report, "fixture-semantics", "oracle")["inputs"])["policy_sha256"] = (
            "0" * 64
        )
    with pytest.raises((ValueError, TypeError)):
        accept(report, manifest_cases(), tmp_path)


@given(
    extra=st.text(min_size=1).filter(
        lambda value: any(
            line.strip() and not line.startswith("gglint: config ")
            for line in value.splitlines()
        )
    )
)
def test_arbitrary_tool_stderr_cannot_hide_behind_known_type_error(extra: str) -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        report = complete_report(root)
        command = obj(cell(report, "decorder", "gglint", "basic")["run"])
        rewrite_log(command, "stderr", "gglint: config config.toml\n" + extra + "\n")
        with pytest.raises(ValueError):
            accept(report, manifest_cases(), root)


@pytest.mark.parametrize("stage", ["oracle", "go", "logos", "logos-lint"])
def test_unrelated_native_error_cannot_join_expected_failure(
    tmp_path: Path, stage: str
) -> None:
    report = complete_report(tmp_path)
    command = obj(cell(report, "decorder", stage)["run"])
    original = Path(str(command["stderr"])).read_text()
    rewrite_log(
        command, "stderr", original + "other.go:8:2: unexpected compiler failure\n"
    )
    with pytest.raises(ValueError):
        accept(report, manifest_cases(), tmp_path)


@pytest.mark.parametrize(
    "defect",
    ["driver", "pattern", "cgo", "tags", "mode", "tests", "path", "experiments", "cwd"],
)
def test_coverage_labels_cannot_replace_actual_invocations(
    tmp_path: Path, defect: str
) -> None:
    report = complete_report(tmp_path)
    stage, mode = (
        ("gglint", "basic")
        if defect in ("mode", "tags", "tests", "path")
        else ("oracle", "")
    )
    entry = cell(report, "fixture-semantics", stage, mode)
    command = obj(entry["run"])
    if defect == "driver":
        command["command"] = ["/different/compiler", "build", "."]
    elif defect == "pattern":
        command["command"] = ["/tools/oracle/bin/tool", "build", "fmt"]
    elif defect == "cgo":
        obj(command["environment"])["CGO_ENABLED"] = "1"
    elif defect in ("tags", "tests"):
        config = tmp_path / "config" / "fixture-semantics.toml"
        config.write_text(
            "[run]\ntests = false\nbuild_tags = []\n"
            if defect == "tests"
            else '[run]\ntests = true\nbuild_tags = ["purego"]\n'
        )
    elif defect == "mode":
        command["command"] = [
            item.replace("basic", "strict") for item in strings(command["command"])
        ]
    elif defect == "path":
        obj(command["environment"])["PATH"] = "/different/go/bin:/usr/bin"
    elif defect == "experiments":
        obj(command["environment"])["GOEXPERIMENT"] = "none"
    else:
        command["cwd"] = str(tmp_path / "unverified-source")
    if "repeat" in entry:
        obj(entry["repeat"])["command"] = command["command"]
        obj(entry["repeat"])["environment"] = command["environment"]
        obj(entry["repeat"])["cwd"] = command["cwd"]
    with pytest.raises(ValueError):
        accept(report, manifest_cases(), tmp_path)
