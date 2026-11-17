# pyright: strict
"""Share exact test execution and identity checks across native regression slices."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tagged-method-facts"))
import check_ci as method_gate


@dataclass(frozen=True)
class Phase:
    check: method_gate.Check
    race: bool = False


@dataclass(frozen=True)
class FrozenSuite:
    inputs: tuple[str, ...]
    observed: tuple[str, ...]
    phases: tuple[Phase, ...]


def run(repo: Path, out: Path, verify: Callable[[Path], FrozenSuite]) -> bool:
    phases: list[dict[str, object]] = []
    report: dict[str, object] = {"accepted": False, "complete": False, "phases": phases}
    before: dict[str, object] | None = None
    suite: FrozenSuite | None = None
    identity_paths: tuple[str, ...] = ()
    try:
        suite = verify(repo)
        if not suite.phases:
            raise ValueError("frozen suite has no phases")
        if len({phase.check.label for phase in suite.phases}) != len(suite.phases):
            raise ValueError("duplicate phase labels")
        identity_paths = (
            *suite.inputs,
            *suite.observed,
            "logos-doc/validation/native_frozen_gate.py",
        )
        before = method_gate.identity(repo, identity_paths)
        method_gate.write_json(out / "before.json", before)
        if before["tracked_status"]:
            raise ValueError("tracked checkout files are modified")
        env = method_gate.environment(repo)
        method_gate.write_json(
            out / "environment.json",
            {
                key: env[key]
                for key in (
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
                if key in env
            },
        )
        for phase in suite.phases:
            check = phase.check
            overlay_flags: tuple[str, ...] = ()
            if check.overlay is not None:
                source, old = check.overlay
                overlay = out / f"{check.label}.overlay.json"
                method_gate.write_json(
                    overlay,
                    {
                        "Replace": {
                            str(method_gate.rooted(repo, source)): str(
                                method_gate.rooted(repo, old)
                            )
                        }
                    },
                )
                overlay_flags = (f"-overlay={overlay}",)
            race_flags = ("-race",) if phase.race else ()
            command = (
                str(repo / "bin/go"),
                "test",
                "-json",
                "-count=1",
                "-p=1",
                "-parallel=1",
                "-vet=off",
                "-timeout=10m",
                *race_flags,
                *overlay_flags,
                check.package,
                "-run",
                check.pattern(),
            )
            phase_env = {**env, "CGO_ENABLED": "1"} if phase.race else env
            phases.append(
                method_gate.capture(command, repo, out, check, phase_env, 660)
            )
        report["complete"] = len(phases) == len(suite.phases)
        report["accepted"] = all(phase["accepted"] is True for phase in phases)
    except (OSError, TypeError, ValueError) as error:
        report["rejection"] = str(error)
    finally:
        if before is not None and suite is not None:
            try:
                if verify(repo) != suite:
                    raise ValueError("frozen suite changed during execution")
                after = method_gate.identity(repo, identity_paths)
                method_gate.write_json(out / "after.json", after)
                if after != before:
                    raise ValueError("source, frozen input, or installed tool changed")
            except (OSError, TypeError, ValueError) as error:
                report["accepted"] = False
                report["identity_rejection"] = str(error)
        method_gate.write_json(out / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)
    return report["accepted"] is True and report["complete"] is True


def main(verify: Callable[[Path], FrozenSuite], description: str | None) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    repo: Path = args.repo.resolve(strict=True)
    out: Path = args.evidence.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raise SystemExit(0 if run(repo, out, verify) else 1)
