#!/usr/bin/env python3
# pyright: strict
"""Compare pinned xxhash and SQLite runtime tests with a stock Go baseline."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from run import (
    Assessment,
    Case,
    Policy,
    Run,
    Source,
    assess_build,
    decode,
    environment,
    execute,
    file_hash,
    git,
    obj,
    prepare_source,
    save_report,
    string,
    toolchain_identity,
    tree_hash,
    verify_policy,
    verify_source,
)

ROOT = Path(__file__).resolve().parent
DRIVERS = ("oracle", "go", "logos")
XXHASH_IDS = ("xxhash-asm-core", "xxhash-purego-core", "xxhash-plugin-cgo")
PLUGIN_CONFIG = ROOT / "profiles" / "xxhash-plugin-arena-off.toml"
PLUGIN_CONFIG_SHA256 = (
    "2ae10a1fe10a8e72ac145fc13e7692747bd06c626ec91a403e9c19d4a4a30e21"
)
MODERNC_POLICY = "policies/modernc-sql-runtime.toml"
MODERNC_POLICY_SHA256 = (
    "5666c90bc1f25d5f0d7d0e5b53e80c2fa8e7ebe341286f7d7b0b1c5651513096"
)
SQLITE_TESTS = ("TestOpen", "TestClose", "TestInsert")
MODERNC_TESTS = (
    "TestNamedParameters",
    "TestMemDB",
    "TestBinding",
    "TestBeginMode",
    "TestMultiStmtQueryStringRoundtrip",
    "TestExecReturningMultiRow",
    "TestMutexWakeOnce",
)
MODERNC_EVENTS = (
    *MODERNC_TESTS,
    "TestBinding/DB",
    "TestBinding/Prepare",
    "TestBeginMode/delete/deferred",
    "TestBeginMode/wal/deferred",
    "TestBeginMode/delete/immediate",
    "TestBeginMode/wal/immediate",
    "TestBeginMode/delete/exclusive",
    "TestBeginMode/wal/exclusive",
)
Pair = tuple[str, str]


@dataclass(frozen=True)
class Replay:
    case: Case
    tests: tuple[str, ...] = ()
    expected: tuple[str, ...] = ()
    required: tuple[Pair, ...] = ()


def read_cases(path: Path) -> dict[str, Case]:
    raw = decode(path.read_text())
    if not isinstance(raw, list):
        raise TypeError(f"manifest must be an array: {path}")
    cases: dict[str, Case] = {}
    for index, value in enumerate(cast(list[object], raw)):
        try:
            case = Case.parse(value)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid case {index} in {path}: {exc}") from exc
        if case.id in cases:
            raise ValueError(f"duplicate case {case.id} in {path}")
        cases[case.id] = case
    return cases


def load_profiles() -> tuple[Replay, ...]:
    runtime = read_cases(ROOT / "corpus-runtime.json")
    if set(runtime) != set(XXHASH_IDS):
        raise ValueError(
            "corpus-runtime.json must contain the three pinned xxhash cases"
        )
    corpus = read_cases(ROOT / "corpus.json")
    result: list[Replay] = []
    for name in XXHASH_IDS:
        case = runtime[name]
        required = ((case.module, "TestAll"),)
        if name == "xxhash-plugin-cgo":
            required += ((case.module + "/dynamic", "TestDynamic"),)
        result.append(Replay(case, required=required))
    for name, tests, expected in (
        ("sqlite3-cgo", SQLITE_TESTS, SQLITE_TESTS),
        ("modernc-sqlite", MODERNC_TESTS, MODERNC_EVENTS),
    ):
        if name not in corpus:
            raise ValueError(f"missing pinned runtime case: {name}")
        result.append(Replay(replace(corpus[name], patterns=(".",)), tests, expected))
    return tuple(result)


@dataclass(frozen=True)
class Events:
    started: tuple[Pair, ...]
    passed: tuple[Pair, ...]
    failures: tuple[Pair, ...]
    skips: tuple[Pair, ...]


def parse_events(text: str) -> Events:
    started: list[Pair] = []
    passed: list[Pair] = []
    failures: list[Pair] = []
    skips: list[Pair] = []
    actions = {"run", "pause", "cont", "pass", "bench", "fail", "output", "skip"}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = obj(decode(line))
            action = string(event.get("Action"))
            if action in ("build-output", "build-fail"):
                package = string(event.get("ImportPath"))
                if "Test" in event:
                    raise ValueError("build events must not carry a test name")
                if action == "build-fail":
                    failures.append((package, ""))
                else:
                    string(event.get("Output"))
                continue
            package = string(event.get("Package"))
            if not package:
                raise ValueError("test events require a nonempty package")
            if "Test" not in event:
                if action not in {"start", "output", "pass", "fail", "skip"}:
                    raise ValueError("unknown action or missing test name")
                if action == "fail":
                    failures.append((package, ""))
                continue
            name = string(event["Test"])
            if not name or action not in actions:
                raise ValueError(
                    "test events require package, test name and a known action"
                )
            pair = (package, name)
            if action == "run":
                started.append(pair)
            elif action == "pass":
                passed.append(pair)
            elif action == "fail":
                failures.append(pair)
            elif action == "skip":
                skips.append(pair)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid test event at line {line_number}: {exc}"
            ) from exc
    return Events(
        tuple(sorted(started)),
        tuple(sorted(passed)),
        tuple(sorted(failures)),
        tuple(sorted(skips)),
    )


@dataclass(frozen=True)
class RuntimeResult:
    profile: Replay
    assessment: Assessment
    events: Events | None
    event_error: str = ""

    @property
    def accepted(self) -> bool:
        return self.assessment.status == "pass"


def execution_error(profile: Replay, events: Events) -> str:
    if not events.started:
        return "no tests executed"
    if events.started != events.passed:
        return "test run/pass event multisets differ"
    if events.failures or events.skips:
        return "test failure or test-level skip recorded"
    expected = tuple(sorted((profile.case.module, name) for name in profile.expected))
    if expected and events.started != expected:
        return "executed test events differ from the exact named selection"
    missing = [pair for pair in profile.required if pair not in events.started]
    if missing:
        return f"required tests did not execute and pass: {missing}"
    return ""


def assess_runtime(
    run: Run, stage: str, profile: Replay, baseline: RuntimeResult | None
) -> RuntimeResult:
    oracle = (
        baseline.assessment
        if baseline is not None and baseline.profile == profile
        else None
    )
    assessment = assess_build(run, stage, oracle)
    try:
        events = parse_events(Path(run.stdout).read_text())
    except (UnicodeError, TypeError, ValueError) as exc:
        if assessment.status == "pass":
            assessment = Assessment("invalid-report", str(exc))
        return RuntimeResult(profile, assessment, None, str(exc))
    if assessment.status != "pass":
        return RuntimeResult(profile, assessment, events)
    error = execution_error(profile, events)
    if not error and stage != "oracle":
        assert baseline is not None and baseline.events is not None
        if (
            events.started != baseline.events.started
            or events.passed != baseline.events.passed
        ):
            error = "candidate test event multisets differ from the successful stock baseline"
    if error:
        assessment = Assessment(
            "upstream-failed" if stage == "oracle" else "candidate-failed", error
        )
    return RuntimeResult(profile, assessment, events)


def make_command(profile: Replay, stage: str, binary: Path) -> tuple[str, ...]:
    flags = (
        ("-O=1",) if stage == "logos" and profile.case.id == "xxhash-plugin-cgo" else ()
    )
    tags = ("-tags=" + ",".join(profile.case.tags),) if profile.case.tags else ()
    selection = ("-run", "^(" + "|".join(profile.tests) + ")$") if profile.tests else ()
    return (
        str(binary),
        "test",
        *flags,
        *tags,
        "-json",
        "-count=1",
        "-short",
        "-parallel=1",
        "-timeout=120s",
        *selection,
        *profile.case.patterns,
    )


def make_environment(
    profile: Replay, binary: Path, modules: Path, cache: Path, tmpdir: Path
) -> dict[str, str]:
    env = environment(profile.case, binary, modules, cache, 1)
    env.update(
        TMPDIR=str(tmpdir),
        GOMEMLIMIT=os.environ.get("GOMEMLIMIT", "256MiB"),
        GOGC=os.environ.get("GOGC", "25"),
    )
    if profile.case.id == "xxhash-plugin-cgo":
        env["GOEXPERIMENT"] = "none"
    return env


def policy_for(profile: Replay) -> Policy | None:
    if profile.case.id != "modernc-sqlite":
        return None
    return verify_policy(
        replace(
            profile.case,
            logos_policy=MODERNC_POLICY,
            logos_policy_sha256=MODERNC_POLICY_SHA256,
        ),
        ROOT,
    )


def prepare_work(
    profile: Replay, source: Source, stage: str, work: Path
) -> dict[str, object]:
    prepared = prepare_source(source, work, stage, policy_for(profile))
    if profile.case.id == "xxhash-plugin-cgo" and stage == "logos":
        if (
            PLUGIN_CONFIG.is_symlink()
            or file_hash(PLUGIN_CONFIG) != PLUGIN_CONFIG_SHA256
        ):
            raise ValueError(
                "plugin arena configuration differs from its pinned checksum"
            )
        content = PLUGIN_CONFIG.read_bytes()
        destination = work / "logos.toml"
        if destination.exists():
            raise ValueError("plugin profile must not overwrite an upstream logos.toml")
        destination.write_bytes(content)
        prepared.update(
            plugin_config_sha256=PLUGIN_CONFIG_SHA256,
            plugin_config_content=content.decode(),
            effective_source_digest=tree_hash(
                work, (*source.files, "logos.toml"), source.prefix
            ),
        )
    return prepared


def positive_timeout(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be positive and finite") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive and finite")
    return seconds


@dataclass(frozen=True)
class Options:
    go_oracle: Path
    go_candidate: Path
    logos: Path
    source_root: Path
    module_cache: Path
    cache_root: Path
    tmpdir: Path
    out: Path
    timeout: float


def arguments() -> Options:
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
    parser.add_argument(
        "--timeout",
        type=positive_timeout,
        default=900.0,
        help="whole-command deadline in seconds, including compilation (default: 900)",
    )
    args = parser.parse_args()
    return Options(
        cast(Path, args.go_oracle).resolve(),
        cast(Path, args.go_candidate).resolve(),
        cast(Path, args.logos).resolve(),
        cast(Path, args.source_root).resolve(),
        cast(Path, args.module_cache).resolve(),
        cast(Path, args.cache_root).resolve(),
        cast(Path, args.tmpdir).resolve(),
        cast(Path, args.out).resolve(),
        float(args.timeout),
    )


def source_identity() -> dict[str, str]:
    root = ROOT.parents[1]
    status = git(root, "status", "--porcelain", "--untracked-files=all", "--", "src")
    if status:
        raise ValueError(f"toolchain source contains local changes: {status}")
    return {
        "revision": git(root, "rev-parse", "HEAD"),
        "src_tree": git(root, "rev-parse", "HEAD:src"),
        "src_status": status,
    }


def identities(
    tools: dict[str, Path], options: Options, profile: Replay, phase: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for stage, binary in tools.items():
        env = make_environment(
            profile,
            binary,
            options.module_cache,
            options.cache_root / stage,
            options.tmpdir,
        )
        log = options.out / "tools" / f"{phase}-{stage}"
        version = execute((str(binary), "version"), ROOT, env, log, 30)
        identity = toolchain_identity(
            binary, env, ROOT, log.with_name(log.name + "-identity")
        )
        result[stage] = {
            "driver_sha256": file_hash(binary),
            "version_run": asdict(version),
            "version": Path(version.stdout).read_text().strip(),
            "toolchain": identity,
        }
        if (
            version.error
            or version.timed_out
            or version.returncode != 0
            or "tools" not in identity
        ):
            raise ValueError(
                f"cannot establish installed {stage} toolchain identity; see {log}"
            )
        if any(
            not obj(tool).get("sha256") for tool in obj(identity["tools"]).values()
        ) or identity.get("source_status"):
            raise ValueError(
                f"installed {stage} tools are missing or their source is dirty"
            )
    return result


def stable_identity(value: object) -> dict[str, object]:
    identity = obj(value)
    toolchain = obj(identity["toolchain"])
    return {
        "driver_sha256": identity["driver_sha256"],
        "version": identity["version"],
        **{
            key: toolchain.get(key)
            for key in (
                "goroot",
                "tools",
                "compiler_source_h1",
                "source_revision",
                "source_status",
            )
        },
    }


def main() -> int:
    options = arguments()
    profiles = load_profiles()
    tools = {
        "oracle": options.go_oracle,
        "go": options.go_candidate,
        "logos": options.logos,
    }
    for stage, binary in tools.items():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(
                f"{stage} must name an executable toolchain driver: {binary}"
            )
    for output in (options.out, options.tmpdir, options.cache_root):
        if output.is_relative_to(options.source_root) or output.is_relative_to(
            options.module_cache
        ):
            raise ValueError(
                f"runtime output must remain outside pinned input directories: {output}"
            )
    options.out.mkdir(parents=True, exist_ok=False)
    options.tmpdir.mkdir(parents=True, exist_ok=True)
    target = options.out / "report.json"
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "complete": False,
        "accepted": False,
        "scope": "five pinned backend profiles under stock Go, fork Go and LoGOs; selected SQL tests are not their full suites",
        "runner_sha256": file_hash(ROOT / "run.py"),
        "replay_sha256": file_hash(Path(__file__)),
        "manifests": {
            name: file_hash(ROOT / name)
            for name in ("corpus.json", "corpus-runtime.json")
        },
        "timeout_seconds": options.timeout,
        "test_timeout_seconds": 120,
        "jobs": 1,
        "case_count": len(profiles),
        "stages": DRIVERS,
        "cases": rows,
    }
    save_report(target, report)
    try:
        before_source = source_identity()
        report["source_before"] = before_source
        before_tools = identities(tools, options, profiles[0], "before")
        report["tools_before"] = before_tools
        save_report(target, report)
        for profile in profiles:
            source = verify_source(
                profile.case, options.source_root, options.module_cache, ROOT
            )
            policy = policy_for(profile)
            stages: list[dict[str, object]] = []
            row: dict[str, object] = {
                "profile": asdict(profile),
                "status": "running",
                "source_path": str(source.path),
                "source_digest": source.digest,
                "policy": None
                if policy is None
                else {
                    "path": str(policy.path),
                    "sha256": policy.sha256,
                    "content": policy.content.decode(),
                },
                "stages": stages,
            }
            rows.append(row)
            save_report(target, report)
            baseline: RuntimeResult | None = None
            for stage, binary in tools.items():
                work = options.out / "work" / profile.case.id / stage
                prepared = prepare_work(profile, source, stage, work)
                env = make_environment(
                    profile,
                    binary,
                    options.module_cache,
                    options.cache_root / stage,
                    options.tmpdir,
                )
                print(f"starting {profile.case.id}/{stage}", flush=True)
                run = execute(
                    make_command(profile, stage, binary),
                    work,
                    env,
                    options.out / "logs" / profile.case.id / stage,
                    options.timeout,
                )
                result = assess_runtime(run, stage, profile, baseline)
                stages.append(
                    {
                        "stage": stage,
                        "preparation": prepared,
                        "run": asdict(run),
                        "assessment": asdict(result.assessment),
                        "events": None
                        if result.events is None
                        else asdict(result.events),
                        "event_error": result.event_error,
                        "accepted": result.accepted,
                    }
                )
                if stage == "oracle":
                    baseline = result
                save_report(target, report)
                print(
                    f"finished {profile.case.id}/{stage}: {result.assessment.status}, {run.seconds}s",
                    flush=True,
                )
                shutil.rmtree(work)
            after_source = verify_source(
                profile.case, options.source_root, options.module_cache, ROOT
            )
            if after_source.digest != source.digest:
                raise ValueError(
                    f"pinned source changed during runtime replay: {profile.case.id}"
                )
            row.update(
                status="finished",
                source_after_digest=after_source.digest,
                accepted=all(item["accepted"] is True for item in stages),
            )
            save_report(target, report)
        after_tools = identities(tools, options, profiles[0], "after")
        after_source = source_identity()
        unchanged = before_source == after_source and all(
            stable_identity(before_tools[stage]) == stable_identity(after_tools[stage])
            for stage in DRIVERS
        )
        report.update(
            complete=True,
            tools_after=after_tools,
            source_after=after_source,
            unchanged=unchanged,
            accepted=unchanged and all(row["accepted"] is True for row in rows),
        )
    except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
        print(f"runtime replay incomplete: {exc}", flush=True)
    save_report(target, report)
    return 0 if report["accepted"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
