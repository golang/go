#!/usr/bin/env python3
# pyright: strict
"""Capture unlimited strict tools diagnostics; never accept the corpus here."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from run import (
    Case,
    Policy,
    all_files,
    decode,
    environment,
    execute,
    failure,
    file_hash,
    git,
    obj,
    prepare_source,
    save_report,
    string,
    strings,
    toolchain_identity,
    tree_hash,
    verify_policy,
    verify_source,
)

ROOT = Path(__file__).resolve().parent
POLICY_SHA256 = "005ba0707611025ef364004190601f0f103ab28ab97da9a6acd81e034d7f4d3f"


@dataclass(frozen=True)
class Options:
    logos: Path
    source_root: Path
    module_cache: Path
    cache_root: Path
    baseline: Path
    out: Path
    jobs: int = 2
    timeout: float = 600

    def __post_init__(self) -> None:
        if type(self.jobs) is not int or self.jobs <= 0:
            raise ValueError("jobs must be a positive integer")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")


def load_profile(root: Path) -> tuple[Case, Policy]:
    data = decode((root / "corpus-permissions.json").read_text())
    if not isinstance(data, list):
        raise TypeError("strict manifest must be an array")
    cases = [Case.parse(item) for item in cast(list[object], data)]
    selected = [case for case in cases if case.id == "tools"]
    if len(selected) != 1:
        raise ValueError("strict manifest must contain exactly one tools case")
    case = selected[0]
    expected: dict[str, object] = {
        "source_kind": "module",
        "source": "golang.org/x/tools@v0.52.0",
        "module": "golang.org/x/tools",
        "revision": "16517d951552611e7bd358c48ee1051fd0e57b0c",
        "version": "v0.52.0",
        "sum": "h1:ryxg4U63oaUYk7uOLTF37n43f4qgtGvqV4Ioit1BaEs=",
        "backend": "go-only",
        "cgo": 0,
        "tags": (),
        "patterns": ("./...",),
        "logos_policy": "policies/tools.toml",
        "logos_policy_sha256": POLICY_SHA256,
    }
    values = asdict(case)
    for key, value in expected.items():
        if values[key] != value:
            raise ValueError(f"original strict tools profile changed: {key}")
    policy = verify_policy(case, root)
    assert policy is not None
    return case, policy


def command(binary: Path) -> tuple[str, ...]:
    return str(binary), "build", "-ldflags=-e", "./..."


def identity_key(receipt: dict[str, object]) -> dict[str, object]:
    return {
        **receipt,
        "toolchain": {
            key: value
            for key, value in obj(receipt["toolchain"]).items()
            if key != "run"
        },
    }


def native_identity(binary: Path, env: dict[str, str], log: Path) -> dict[str, object]:
    repository = ROOT.parents[1]
    identity = toolchain_identity(binary, env, ROOT, log)
    save_report(log.with_suffix(".json"), identity)
    if (
        identity.get("goroot") != str(repository)
        or identity.get("source_revision") != git(repository, "rev-parse", "HEAD")
        or identity.get("source_status") != ""
    ):
        raise ValueError("native toolchain must identify this clean source checkout")
    tools = obj(identity["tools"])
    if any(
        not isinstance(obj(tools[name]).get("sha256"), str)
        for name in ("compile", "link", "asm", "cgo")
    ):
        raise ValueError("native executable identity is incomplete")
    go = binary.with_name("go.exe" if binary.suffix == ".exe" else "go")
    return {
        "drivers": {
            name: {"path": str(path), "sha256": file_hash(path)}
            for name, path in (("go", go), ("logos", binary))
        },
        "source_tree": git(repository, "rev-parse", "HEAD:src"),
        "toolchain": identity,
    }


def verify_baseline(
    path: Path,
    case: Case,
    env: dict[str, str],
    native: dict[str, object],
    effective_digest: str,
) -> str:
    """Bind the earlier three-driver build and its raw streams to this probe."""
    try:
        report = obj(decode(path.read_text()))
        if (
            report.get("complete") is not True
            or type(report.get("case_count")) is not int
            or report["case_count"] != 1
            or report.get("operation_override") != "build"
            or report.get("runner_sha256") != file_hash(ROOT / "run.py")
        ):
            raise ValueError(
                "baseline must be one completed build from the same runner"
            )
        rows = report["cases"]
        if not isinstance(rows, list) or len(cast(list[object], rows)) != 1:
            raise ValueError("baseline must contain exactly one case")
        row = obj(cast(list[object], rows)[0])
        if (
            Case.parse(row["case"]) != case
            or row.get("status") != "finished"
            or row.get("source_digest") != case.sum
        ):
            raise ValueError("baseline source/profile does not match strict tools")
        saved_policy = obj(row["logos_policy"])
        policy_path = Path(string(saved_policy["saved_path"]))
        if (
            not policy_path.resolve().is_relative_to(path.parent.resolve())
            or saved_policy.get("sha256") != case.logos_policy_sha256
            or file_hash(policy_path) != case.logos_policy_sha256
        ):
            raise ValueError("baseline strict policy bytes do not match")
        tools, versions = obj(report["tools"]), obj(report["versions"])
        if set(tools) != {"oracle", "go", "logos"}:
            raise ValueError("baseline needs distinct oracle, go, and logos identities")
        drivers, current = obj(native["drivers"]), obj(native["toolchain"])
        for name in ("go", "logos"):
            recorded = obj(obj(versions[name])["toolchain"])
            if tools[name] != drivers[name] or {
                k: v for k, v in recorded.items() if k != "run"
            } != {k: v for k, v in current.items() if k != "run"}:
                raise ValueError(f"baseline native identity changed: {name}")
        oracle = obj(tools["oracle"])
        oracle_path = Path(string(oracle["path"]))
        oracle_identity = obj(obj(versions["oracle"])["toolchain"])
        if (
            oracle_path.resolve() == Path(string(obj(drivers["go"])["path"])).resolve()
            or oracle_identity.get("goroot") == current.get("goroot")
            or oracle_identity.get("goroot") != str(oracle_path.parent.parent)
            or oracle.get("sha256") != file_hash(oracle_path)
        ):
            raise ValueError(
                "baseline stock oracle identity is missing, changed, or aliased"
            )
        stages_value = row["stages"]
        if not isinstance(stages_value, list):
            raise TypeError("baseline stages must be an array")
        stages = [obj(item) for item in cast(list[object], stages_value)]
        if sorted(string(stage["stage"]) for stage in stages) != [
            "go",
            "logos",
            "oracle",
        ]:
            raise ValueError("baseline needs exactly three build stages")
        native_prefix = (
            str(Path(string(obj(drivers["logos"])["path"])).parent) + os.pathsep
        )
        if not env["PATH"].startswith(native_prefix):
            raise ValueError("native PATH does not begin with the selected driver")
        inherited_path = env["PATH"].removeprefix(native_prefix)
        for stage in stages:
            name = string(stage["stage"])
            run = obj(stage["run"])
            binary = Path(string(obj(tools[name])["path"]))
            expected_env = {
                **env,
                "PATH": f"{binary.parent}{os.pathsep}{inherited_path}",
                "GOCACHE": str(Path(env["GOCACHE"]).parent / name),
            }
            expected_inputs: dict[str, object] = {"effective_source_digest": case.sum}
            if name == "logos":
                expected_inputs = {
                    "effective_source_digest": effective_digest,
                    "policy_sha256": case.logos_policy_sha256,
                }
            expected_cwd = (path.parent / "work" / case.id).resolve()
            if (
                strings(run["command"]) != (str(binary), "build", "./...")
                or Path(string(run["cwd"])).resolve() != expected_cwd
                or not expected_cwd.is_relative_to(path.parent.resolve())
                or run.get("environment") != expected_env
                or stage.get("inputs") != expected_inputs
                or run.get("timed_out") is not False
                or run.get("error") != ""
                or type(run.get("returncode")) is not int
                or run.get("returncode") not in (0, 1)
                or (
                    name != "logos"
                    and (
                        run["returncode"] != 0
                        or obj(stage["assessment"]).get("status") != "pass"
                    )
                )
            ):
                raise ValueError(f"baseline build/profile did not match: {name}")
            for stream in ("stdout", "stderr"):
                captured = Path(string(run[stream]))
                if not captured.resolve().is_relative_to(
                    path.parent.resolve()
                ) or file_hash(captured) != run.get(f"{stream}_sha256"):
                    raise ValueError(
                        f"baseline output hash/path changed: {name}/{stream}"
                    )
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError(f"baseline evidence is incomplete: {exc}") from exc
    return file_hash(path)


def capture(options: Options) -> int:
    out = options.out.resolve()
    for protected in (
        options.module_cache,
        options.source_root,
        options.cache_root,
        options.logos.parent.parent,
    ):
        root = protected.resolve()
        if out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError(f"evidence must not overlap inputs or caches: {root}")
    out.mkdir(parents=True, exist_ok=False)
    work, target = out / "work", out / "report.json"
    report: dict[str, object] = {
        "schema_version": 1,
        "purpose": "diagnostic-only; does not accept the corpus",
        "complete": False,
        "build_succeeded": False,
        "probe_sha256": file_hash(Path(__file__)),
        "runner_sha256": file_hash(ROOT / "run.py"),
        "jobs": options.jobs,
        "timeout_seconds": options.timeout,
        "inherited_gogc": {
            "present": "GOGC" in os.environ,
            "value": os.environ.get("GOGC"),
        },
    }
    save_report(target, report)
    try:
        case, policy = load_profile(ROOT)
        source = verify_source(case, options.source_root, options.module_cache, ROOT)
        inputs = prepare_source(source, work, "logos", policy)
        env = environment(
            case,
            options.logos,
            options.module_cache,
            options.cache_root / "logos",
            options.jobs,
        )
        report.update(case=asdict(case), source_digest=source.digest, inputs=inputs)
        saved_policy = out / "inputs/tools/logos.toml"
        saved_policy.parent.mkdir(parents=True)
        saved_policy.write_bytes(policy.content)
        report["policy"] = {"path": str(saved_policy), "sha256": policy.sha256}
        before = native_identity(options.logos, env, out / "native-before")
        report["native_before"] = before
        effective = string(inputs["effective_source_digest"])
        baseline_hash = verify_baseline(options.baseline, case, env, before, effective)
        report["baseline"] = {"path": str(options.baseline), "sha256": baseline_hash}
        save_report(target, report)
        invocation = execute(
            command(options.logos), work, env, out / "build", options.timeout
        )
        report.update(
            run=asdict(invocation), build_succeeded=invocation.returncode == 0
        )
        save_report(target, report)
        after = native_identity(options.logos, env, out / "native-after")
        report["native_after"] = after
        if identity_key(before) != identity_key(after):
            raise ValueError(
                "native source or executable identity changed during probe"
            )
        if (
            verify_source(case, options.source_root, options.module_cache, ROOT)
            != source
        ):
            raise ValueError("original pinned source changed during probe")
        if tree_hash(work, all_files(work), source.prefix) != effective:
            raise ValueError("private effective source changed during probe")
        if file_hash(policy.path) != policy.sha256:
            raise ValueError("original strict policy changed during probe")
        if (
            verify_baseline(options.baseline, case, env, after, effective)
            != baseline_hash
        ):
            raise ValueError("baseline report changed during probe")
        problem = failure(invocation)
        if problem is not None:
            raise ValueError(
                f"diagnostic capture failed: {problem.status}: {problem.reason}"
            )
        if invocation.returncode not in (0, 1):
            raise ValueError(f"unexpected build exit status: {invocation.returncode}")
        if "too many errors" in Path(invocation.stderr).read_text(errors="replace"):
            raise ValueError("diagnostic capture still reached an error limit")
        report["complete"] = True
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        subprocess.CalledProcessError,
    ) as exc:
        report["error"] = str(exc)
        return 2
    finally:
        save_report(target, report)
        if work.exists():
            shutil.rmtree(work)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "logos",
        "source-root",
        "module-cache",
        "cache-root",
        "baseline",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    binary = cast(Path, args.logos)
    return capture(
        Options(
            logos=binary.parent.resolve() / binary.name,
            source_root=cast(Path, args.source_root).resolve(),
            module_cache=cast(Path, args.module_cache).resolve(),
            cache_root=cast(Path, args.cache_root).resolve(),
            baseline=cast(Path, args.baseline).resolve(),
            out=cast(Path, args.out).resolve(),
            jobs=args.jobs,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
