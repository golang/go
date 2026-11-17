# pyright: strict
"""The unlimited probe keeps strict inputs and cannot become build acceptance."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import probe_tools as probe
import pytest
from run import (
    Case,
    Policy,
    Run,
    all_files,
    environment,
    file_hash,
    obj,
    tree_hash,
)

ROOT = Path(__file__).resolve().parent


def test_original_profile_and_explicit_unlimited_argv() -> None:
    case, policy = probe.load_profile(ROOT)
    assert case.source == "golang.org/x/tools@v0.52.0"
    assert case.logos_policy == "policies/tools.toml"
    assert policy.sha256 == (
        "005ba0707611025ef364004190601f0f103ab28ab97da9a6acd81e034d7f4d3f"
    )
    assert probe.command(Path("/native/bin/logos")) == (
        "/native/bin/logos",
        "build",
        "-ldflags=-e",
        "./...",
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("logos_policy", "policies/tools-build.toml"),
        ("logos_policy_sha256", "0" * 64),
        ("source", "golang.org/x/tools@v0.51.0"),
        ("version", "v0.51.0"),
        ("revision", "0" * 40),
        ("sum", "h1:" + "A" * 43 + "="),
        ("cgo", 1),
        ("tags", ["purego"]),
        ("patterns", ["./cmd/deadcode"]),
        ("patterns", ["-ldflags=-e"]),
    ),
)
def test_strict_profile_cannot_drift(tmp_path: Path, key: str, value: object) -> None:
    case, policy = probe.load_profile(ROOT)
    changed = asdict(case)
    changed[key] = value
    (tmp_path / "corpus-permissions.json").write_text(json.dumps([changed]))
    (tmp_path / "policies").mkdir()
    (tmp_path / "policies/tools.toml").write_bytes(policy.content)
    with pytest.raises(ValueError):
        probe.load_profile(tmp_path)


@dataclass(frozen=True)
class Fixture:
    options: probe.Options
    case: Case
    policy: Policy
    source: Path
    env: dict[str, str]
    native: dict[str, object]
    effective: str


def result(
    command: tuple[str, ...],
    cwd: Path,
    env: dict[str, str],
    log: Path,
    *,
    code: int = 0,
    stderr: str = "",
    timeout: bool = False,
) -> Run:
    log.parent.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = log.with_suffix(".stdout"), log.with_suffix(".stderr")
    stdout_path.write_text("")
    stderr_path.write_text(stderr)
    return Run(
        command,
        str(cwd),
        env,
        code,
        timeout,
        "",
        0.1,
        str(stdout_path),
        str(stderr_path),
        file_hash(stdout_path),
        file_hash(stderr_path),
    )


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    original, policy = probe.load_profile(ROOT)
    saved = tmp_path / "strict-policy.toml"
    saved.write_bytes(policy.content)
    policy = replace(policy, path=saved)
    source = tmp_path / "modules" / original.source
    source.mkdir(parents=True)
    (source / "go.mod").write_text("module golang.org/x/tools\n\ngo 1.27\n")
    (source / "tools.go").write_text("package tools\n")
    case = replace(original, sum=tree_hash(source, all_files(source), original.source))
    info = tmp_path / "modules/cache/download/golang.org/x/tools/@v/v0.52.0.info"
    info.parent.mkdir(parents=True)
    info.write_text(
        json.dumps({"Version": case.version, "Origin": {"Hash": case.revision}})
    )
    binary = tmp_path / "native/bin/logos"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"inert logos identity; never executed")
    go = binary.with_name("go")
    go.write_bytes(b"inert go identity; never executed")
    oracle = tmp_path / "stock/bin/go"
    oracle.parent.mkdir(parents=True)
    oracle.write_bytes(b"distinct inert stock oracle; never executed")
    options = probe.Options(
        logos=binary,
        source_root=tmp_path / "sources",
        module_cache=tmp_path / "modules",
        cache_root=tmp_path / "cache",
        baseline=tmp_path / "baseline/report.json",
        out=tmp_path / "evidence",
        jobs=2,
        timeout=600,
    )
    env = environment(
        case, binary, options.module_cache, options.cache_root / "logos", 2
    )
    toolchain: dict[str, object] = {
        "goroot": str(binary.parent.parent),
        "source_revision": "a" * 40,
        "source_status": "",
        "compiler_source_h1": "h1:" + "A" * 43 + "=",
        "tools": {
            name: {"path": f"/native/{name}", "sha256": "1" * 64}
            for name in ("compile", "link", "asm", "cgo")
        },
        "run": {"seconds": 0.1},
    }
    native: dict[str, object] = {
        "drivers": {
            "go": {"path": str(go), "sha256": file_hash(go)},
            "logos": {"path": str(binary), "sha256": file_hash(binary)},
        },
        "source_tree": "b" * 40,
        "toolchain": toolchain,
    }
    effective_root = tmp_path / "effective"
    effective_root.mkdir()
    for name in all_files(source):
        (effective_root / name).write_bytes((source / name).read_bytes())
    (effective_root / "logos.toml").write_bytes(policy.content)
    effective = tree_hash(effective_root, all_files(effective_root), case.source)
    base = options.baseline.parent
    stages: list[dict[str, object]] = []
    for stage in ("oracle", "go", "logos"):
        stage_binary = {"oracle": oracle, "go": go, "logos": binary}[stage]
        stage_env = environment(
            case, stage_binary, options.module_cache, options.cache_root / stage, 2
        )
        invocation = result(
            (str(stage_binary), "build", "./..."),
            base / "work/tools",
            stage_env,
            base / f"logs/{stage}",
            code=int(stage == "logos"),
        )
        stages.append(
            {
                "stage": stage,
                "mode": "",
                "run": asdict(invocation),
                "assessment": {
                    "status": "candidate-failed" if stage == "logos" else "pass"
                },
                "inputs": {
                    "effective_source_digest": effective
                    if stage == "logos"
                    else case.sum,
                    **({"policy_sha256": policy.sha256} if stage == "logos" else {}),
                },
            }
        )
    saved_policy = base / "inputs/tools/logos.toml"
    saved_policy.parent.mkdir(parents=True)
    saved_policy.write_bytes(policy.content)
    oracle_toolchain = copy.deepcopy(toolchain)
    oracle_toolchain["goroot"] = str(oracle.parent.parent)
    oracle_toolchain.pop("source_revision")
    oracle_toolchain.pop("source_status")
    oracle_toolchain["compiler_source_h1"] = "h1:" + "B" * 43 + "="
    oracle_toolchain["tools"] = {
        name: {"path": f"/stock/{name}", "sha256": "2" * 64}
        for name in ("compile", "link", "asm", "cgo")
    }
    baseline: dict[str, object] = {
        "complete": True,
        "case_count": 1,
        "operation_override": "build",
        "runner_sha256": file_hash(ROOT / "run.py"),
        "cases": [
            {
                "case": asdict(case),
                "status": "finished",
                "source_digest": case.sum,
                "logos_policy": {
                    "saved_path": str(saved_policy),
                    "sha256": policy.sha256,
                },
                "stages": stages,
            }
        ],
        "tools": {
            **obj(native["drivers"]),
            "oracle": {"path": str(oracle), "sha256": file_hash(oracle)},
        },
        "versions": {
            **{
                name: {"toolchain": copy.deepcopy(toolchain)}
                for name in ("go", "logos")
            },
            "oracle": {"toolchain": oracle_toolchain},
        },
    }
    options.baseline.write_text(json.dumps(baseline))

    def load_profile(_root: Path) -> tuple[Case, Policy]:
        return case, policy

    def native_identity(
        _binary: Path, _env: dict[str, str], _log: Path
    ) -> dict[str, object]:
        return copy.deepcopy(native)

    monkeypatch.setattr(probe, "load_profile", load_profile)
    monkeypatch.setattr(probe, "native_identity", native_identity)
    return Fixture(options, case, policy, source, env, native, effective)


def read_baseline(fixture: Fixture) -> dict[str, object]:
    return obj(cast(object, json.loads(fixture.options.baseline.read_text())))


def test_successful_stock_and_fork_baseline_is_bound(fixture: Fixture) -> None:
    assert probe.verify_baseline(
        fixture.options.baseline,
        fixture.case,
        fixture.env,
        fixture.native,
        fixture.effective,
    ) == file_hash(fixture.options.baseline)


@pytest.mark.parametrize("stage", (0, 1, 2), ids=("oracle", "go", "logos"))
@pytest.mark.parametrize("status", (False, True, 0.0, 1.0))
def test_baseline_rejects_noninteger_exit_status(
    fixture: Fixture, stage: int, status: object
) -> None:
    baseline = read_baseline(fixture)
    row = obj(cast(list[object], baseline["cases"])[0])
    stages = cast(list[object], row["stages"])
    obj(obj(stages[stage])["run"])["returncode"] = status
    fixture.options.baseline.write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="baseline build/profile did not match"):
        probe.verify_baseline(
            fixture.options.baseline,
            fixture.case,
            fixture.env,
            fixture.native,
            fixture.effective,
        )


@pytest.mark.parametrize(
    "fault",
    (
        "incomplete",
        "missing-oracle",
        "failed-oracle",
        "failed-go",
        "source",
        "policy",
        "environment",
        "driver",
        "compiler",
        "log",
        "truncated-stock",
        "oracle-command",
        "oracle-pattern",
        "go-command",
        "logos-command",
        "cwd",
        "oracle-alias",
        "oracle-missing",
        "oracle-hash",
        "oracle-environment",
    ),
)
def test_baseline_rejects_unmatched_evidence(fixture: Fixture, fault: str) -> None:
    baseline = read_baseline(fixture)
    row = obj(cast(list[object], baseline["cases"])[0])
    stages = [obj(item) for item in cast(list[object], row["stages"])]
    if fault == "incomplete":
        baseline["complete"] = False
    elif fault == "missing-oracle":
        row["stages"] = stages[1:]
    elif fault in ("failed-oracle", "failed-go"):
        obj(stages[int(fault == "failed-go")]["run"])["returncode"] = 1
    elif fault == "source":
        row["source_digest"] = "h1:" + "B" * 43 + "="
    elif fault == "policy":
        obj(row["logos_policy"])["sha256"] = "0" * 64
    elif fault == "environment":
        obj(obj(stages[2]["run"])["environment"])["GOMAXPROCS"] = "9"
    elif fault == "driver":
        obj(obj(baseline["tools"])["logos"])["sha256"] = "0" * 64
    elif fault == "compiler":
        toolchain = obj(obj(obj(baseline["versions"])["logos"])["toolchain"])
        obj(obj(toolchain["tools"])["compile"])["sha256"] = "0" * 64
    elif fault == "log":
        Path(cast(str, obj(stages[0]["run"])["stdout"])).write_text("changed")
    elif fault == "truncated-stock":
        obj(stages[0]["run"])["timed_out"] = True
    elif fault in ("oracle-command", "oracle-pattern", "go-command", "logos-command"):
        stage = {
            "oracle-command": 0,
            "oracle-pattern": 0,
            "go-command": 1,
            "logos-command": 2,
        }[fault]
        invocation = obj(stages[stage]["run"])
        executable = cast(list[object], invocation["command"])[0]
        suffix = {
            "oracle-command": ["version"],
            "oracle-pattern": ["build", "./cmd/deadcode"],
            "go-command": ["env"],
            "logos-command": ["build", "-ldflags=-e", "./..."],
        }[fault]
        invocation["command"] = [executable, *suffix]
    elif fault == "cwd":
        obj(stages[0]["run"])["cwd"] = str(
            fixture.options.baseline.parent / "work/other"
        )
    elif fault == "oracle-alias":
        tools = obj(baseline["tools"])
        tools["oracle"] = copy.deepcopy(tools["go"])
        versions = obj(baseline["versions"])
        versions["oracle"] = copy.deepcopy(versions["go"])
        obj(stages[0]["run"])["command"] = [obj(tools["go"])["path"], "build", "./..."]
        obj(stages[0]["run"])["environment"] = {
            **fixture.env,
            "GOCACHE": str(fixture.options.cache_root / "oracle"),
        }
    elif fault == "oracle-missing":
        obj(baseline["tools"]).pop("oracle")
    elif fault == "oracle-hash":
        obj(obj(baseline["tools"])["oracle"])["sha256"] = "0" * 64
    else:
        obj(obj(stages[0]["run"])["environment"])["GOFLAGS"] = "-mod=mod -p=9"
    fixture.options.baseline.write_text(json.dumps(baseline))
    with pytest.raises(ValueError):
        probe.verify_baseline(
            fixture.options.baseline,
            fixture.case,
            fixture.env,
            fixture.native,
            fixture.effective,
        )


@pytest.mark.parametrize(
    "field",
    (
        "drivers",
        "source_tree",
        "compiler_source_h1",
        "source_revision",
        "source_status",
        "tools",
    ),
)
def test_native_stability_keeps_all_execution_identities(
    fixture: Fixture, field: str
) -> None:
    changed = copy.deepcopy(fixture.native)
    if field in ("drivers", "source_tree"):
        changed[field] = "changed"
    else:
        obj(changed["toolchain"])[field] = "changed"
    assert probe.identity_key(changed) != probe.identity_key(fixture.native)
    unchanged = copy.deepcopy(fixture.native)
    obj(unchanged["toolchain"])["run"] = {"seconds": 22, "stdout": "/new/log"}
    assert probe.identity_key(unchanged) == probe.identity_key(fixture.native)


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "build-pass",
        "timeout",
        "source-write",
        "work-write",
        "policy-write",
        "tool-write",
        "truncation",
        "compile-error",
        "spawn-error",
        "crash",
    ),
)
def test_capture_is_private_and_retains_failures(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    calls: list[tuple[str, ...]] = []

    def execute(
        command: tuple[str, ...],
        cwd: Path,
        env: dict[str, str],
        log: Path,
        timeout: float,
    ) -> Run:
        calls.append(command)
        assert command == (str(fixture.options.logos), "build", "-ldflags=-e", "./...")
        assert cwd != fixture.source and cwd.is_relative_to(fixture.options.out)
        assert (cwd / "logos.toml").read_bytes() == fixture.policy.content
        assert env == fixture.env and timeout == 600
        if fault == "source-write":
            (fixture.source / "tools.go").write_text("package changed\n")
        if fault == "work-write":
            (cwd / "unexpected.txt").write_text("unaccounted output")
        if fault == "policy-write":
            fixture.policy.path.write_text("changed")
        if fault == "tool-write":
            obj(obj(fixture.native["drivers"])["logos"])["sha256"] = "0" * 64
        text = "ERROR [capabilities::unauthorized_net]: dependency reaches syscall.Socket\n"
        if fault == "truncation":
            text += "link: too many errors\n"
        if fault == "compile-error":
            text = "./tools.go:1: undefined: broken\n"
        if fault == "crash":
            text += "panic: simulated compiler crash\n"
        invocation = result(
            command,
            cwd,
            env,
            log,
            code=int(fault != "build-pass"),
            stderr="" if fault == "build-pass" else text,
            timeout=fault == "timeout",
        )
        if fault == "spawn-error":
            return replace(invocation, returncode=None, error="simulated missing tool")
        return invocation

    monkeypatch.setattr(probe, "execute", execute)
    code = probe.capture(fixture.options)
    report = obj(
        cast(object, json.loads((fixture.options.out / "report.json").read_text()))
    )
    assert len(calls) == 1
    assert not (fixture.options.out / "work").exists()
    invocation = obj(report["run"])
    assert Path(cast(str, invocation["stderr"])).is_file()
    assert report["build_succeeded"] is (fault == "build-pass")
    assert report["purpose"] == "diagnostic-only; does not accept the corpus"
    if fault in ("none", "compile-error", "build-pass"):
        assert code == 0 and report["complete"] is True
        assert invocation["returncode"] == int(fault != "build-pass")
    else:
        assert code == 2 and report["complete"] is False
        assert report["error"]


def test_capture_checks_baseline_before_build(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = read_baseline(fixture)
    baseline["complete"] = False
    fixture.options.baseline.write_text(json.dumps(baseline))

    def forbidden(
        _command: tuple[str, ...],
        _cwd: Path,
        _env: dict[str, str],
        _log: Path,
        _timeout: float,
    ) -> Run:
        pytest.fail("build ran without a completed matching baseline")

    monkeypatch.setattr(probe, "execute", forbidden)
    assert probe.capture(fixture.options) == 2
    report = obj(
        cast(object, json.loads((fixture.options.out / "report.json").read_text()))
    )
    assert report["complete"] is False and report["error"]
    assert "run" not in report
    assert not (fixture.options.out / "work").exists()


@pytest.mark.parametrize("timeout", (-1.0, 0.0, float("nan"), float("inf")))
def test_timeout_must_be_positive_and_finite(fixture: Fixture, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        replace(fixture.options, timeout=timeout)


@pytest.mark.parametrize("jobs", (-1, 0))
def test_jobs_must_be_positive(fixture: Fixture, jobs: int) -> None:
    with pytest.raises(ValueError, match="jobs"):
        replace(fixture.options, jobs=jobs)


def test_resource_boundary_accepts_positive_values(fixture: Fixture) -> None:
    assert replace(fixture.options, jobs=1, timeout=1).jobs == 1


@pytest.mark.parametrize(
    "location", ("source", "native", "cache", "cache-inside-output")
)
def test_output_cannot_overlap_inputs_or_caches(
    fixture: Fixture, location: str
) -> None:
    options = fixture.options
    if location == "source":
        options = replace(options, out=fixture.source / "evidence")
    elif location == "native":
        options = replace(options, out=options.logos.parent.parent / "evidence")
    elif location == "cache":
        options = replace(options, out=options.cache_root / "evidence")
    else:
        options = replace(options, cache_root=options.out / "cache")
    with pytest.raises(ValueError, match="overlap"):
        probe.capture(options)
    assert not options.out.exists()


def test_fresh_output_is_required(fixture: Fixture) -> None:
    fixture.options.out.mkdir()
    sentinel = fixture.options.out / "keep.txt"
    sentinel.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        probe.capture(fixture.options)
    assert sentinel.read_text() == "prior evidence"
