#!/usr/bin/env -S uv run --script
"""Run serialized M13 scalar benchmarks after all builds and gates are idle.

Run through uv with the primary Python interpreter. This runner changes no
repository source. Each invocation creates a new evidence directory.

By default, compare logosopt=0 and logosopt=1 with the installed compiler.
Pass both --baseline-compiler and --candidate-compiler to compare two compiler
binaries at -O=3 with logosopt=1 on both sides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

DEFAULT_REPO = Path.cwd()
DEFAULT_EVIDENCE = Path.cwd() / ".outline" / "optimize" / "M13-S2"
PACKAGE = "cmd/compile/internal/test/testdata/scalaropt"
BENCHMARKS = (
    "BenchmarkMaskedArithmetic",
    "BenchmarkRotatedField",
    "BenchmarkCorrelated",
    "BenchmarkFixedPointDispatch",
    "BenchmarkPackedDescriptors",
    "BenchmarkPackedHeaders",
    "BenchmarkCorrelatedDescriptors",
    "BenchmarkPhi01Descriptors",
    "BenchmarkBswapLateDispatch",
    "BenchmarkBswapNetworkByteFields",
    "BenchmarkAlignedBitCounts",
    "BenchmarkHeaderBitLengths",
    "BenchmarkSparsePopulationClasses",
)
MODES = (("baseline", 0), ("candidate", 1))


@dataclass(frozen=True)
class Sample:
    iterations: int
    ns_per_op: float
    bytes_per_op: float
    allocs_per_op: float

    @property
    def useful_seconds(self) -> float:
        return self.iterations * self.ns_per_op / 1e9


def invoke(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    output: Path,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(
            f"Command failed with status {result.returncode}; see {output}"
        )
    return result.stdout


def samples(text: str, name: str) -> list[Sample]:
    pattern = re.compile(rf"^{re.escape(name)}(?:-\d+)?\s")
    result: list[Sample] = []
    for line in text.splitlines():
        if pattern.match(line):
            fields = line.split()
            # SetBytes adds MB/s between latency and allocation metrics.
            # Read by unit so throughput cannot shift the required columns.
            metrics = {
                unit: float(fields[index - 1])
                for index, unit in enumerate(fields)
                if unit in ("ns/op", "B/op", "allocs/op") and index > 1
            }
            if len(metrics) != 3:
                raise ValueError(f"Unexpected benchmark fields: {line}")
            result.append(
                Sample(
                    int(fields[1]),
                    metrics["ns/op"],
                    metrics["B/op"],
                    metrics["allocs/op"],
                )
            )
    if not result:
        raise ValueError(f"No samples for {name}")
    return result


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def append_record(path: Path, record: dict[str, object]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("runtime", "compile", "all"), default="runtime"
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--hyperfine", type=Path, required=True)
    parser.add_argument("--baseline-compiler", type=Path)
    parser.add_argument("--candidate-compiler", type=Path)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument("--compile-package", action="append")
    parser.add_argument("--target-seconds", type=float, default=1.25)
    args = parser.parse_args()
    repo = cast(Path, args.repo).resolve()
    evidence = cast(Path, args.evidence).resolve()
    hyperfine = cast(Path, args.hyperfine).resolve()
    baseline_compiler = cast(Path | None, args.baseline_compiler)
    candidate_compiler = cast(Path | None, args.candidate_compiler)
    if (baseline_compiler is None) != (candidate_compiler is None):
        parser.error("--baseline-compiler and --candidate-compiler must be paired")
    paired_compilers = baseline_compiler is not None
    modes: tuple[tuple[str, int], ...] = (
        tuple((name, 1) for name, _ in MODES) if paired_compilers else MODES
    )
    target_seconds = cast(float, args.target_seconds)
    if target_seconds < 1:
        raise ValueError("G6 requires at least one second of useful work")
    evidence.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(
        tempfile.mkdtemp(prefix=f"scalaropt-{args.phase}-{stamp}-", dir=evidence)
    )
    print(f"Evidence: {out}", flush=True)
    env = dict(os.environ)
    env.update(
        GOROOT=str(repo),
        GOTOOLCHAIN="local",
        GOENV="off",
        GOFLAGS="",
        GOMAXPROCS="1",
        GOAMD64=env.get("GOAMD64", "v1"),
    )
    env["PATH"] = str(repo / "bin") + os.pathsep + env["PATH"]
    go = repo / "bin/go"
    cwd = repo / "src"
    tool_dir = Path(
        invoke(
            [str(go), "env", "GOTOOLDIR"], env=env, cwd=cwd, output=out / "tool-dir.txt"
        ).strip()
    )
    compiler = tool_dir / "compile"
    profiler = tool_dir / "pprof"
    if not profiler.is_file() or not os.access(profiler, os.X_OK):
        raise ValueError(
            "The selected toolchain lacks pprof; build cmd/pprof into its "
            "GOTOOLDIR before starting the campaign"
        )
    compilers: dict[str, Path] = {name: compiler for name, _ in modes}
    build_flags: dict[str, list[str]] = {
        name: [f"-gcflags={PACKAGE}=-d=logosopt={mode}"] for name, mode in modes
    }
    wrappers: list[Path] = []
    compiler_versions: dict[str, str] = {}
    tool_ids: dict[str, str] = {}
    difference = "only -d=logosopt=0 versus -d=logosopt=1"
    if baseline_compiler is not None and candidate_compiler is not None:
        compilers = {
            "baseline": baseline_compiler.resolve(strict=True),
            "candidate": candidate_compiler.resolve(strict=True),
        }
        wrapper_dir = out / "toolexec"
        wrapper_dir.mkdir(mode=0o700)
        # Bare wrapper names avoid -toolexec's non-shell quoting grammar.
        env["PATH"] = str(wrapper_dir) + os.pathsep + env["PATH"]
        for name, selected in compilers.items():
            if not selected.is_file() or not os.access(selected, os.X_OK):
                raise ValueError(f"Compiler is not an executable file: {selected}")
            wrapper = wrapper_dir / f"scalaropt-{name}-toolexec"
            wrapper.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                "tool=$1\nshift\n"
                'case "${tool##*/}" in\n'
                f'  compile) exec -a compile {shlex.quote(str(selected))} "$@" ;;\n'
                '  *) exec "$tool" "$@" ;;\nesac\n'
            )
            wrapper.chmod(0o700)
            wrappers.append(wrapper)
            build_flags[name] = [
                "-O=3",
                f"-toolexec={wrapper.name}",
                *build_flags[name],
            ]
            version = invoke(
                [str(wrapper), str(compiler), "-V=full"],
                env=env,
                cwd=cwd,
                output=out / f"{name}-compiler-version.txt",
            ).strip()
            fields = version.split()
            if len(fields) < 3 or fields[:2] != ["compile", "version"]:
                raise ValueError(f"Invalid compiler tool identity: {version}")
            tool_id = version
            if fields[-1].startswith("buildID="):
                tool_id = fields[-1].removeprefix("buildID=").rsplit("/", 1)[-1]
                if not tool_id:
                    raise ValueError(f"Compiler reports an empty build ID: {selected}")
            compiler_versions[name] = version
            tool_ids[name] = tool_id
        difference = (
            "baseline compiler versus candidate compiler; both use -O=3 and "
            "package -d=logosopt=1 with identical sources and optimization flags; "
            "toolexec selects the compiler"
        )
    invoke([str(go), "version"], env=env, cwd=cwd, output=out / "go-version.txt")
    invoke(
        [str(hyperfine), "--version"],
        env=env,
        cwd=cwd,
        output=out / "hyperfine-version.txt",
    )
    invoke(
        [
            str(go),
            "env",
            "-json",
            "GOOS",
            "GOARCH",
            "GOAMD64",
            "GOEXPERIMENT",
            "GOCACHE",
            "GOROOT",
        ],
        env=env,
        cwd=cwd,
        output=out / "go-env.json",
    )
    invoke(["uname", "-a"], env=env, cwd=cwd, output=out / "uname.txt")
    (out / "cpuinfo.txt").write_text(Path("/proc/cpuinfo").read_text())
    revision = invoke(
        ["git", "rev-parse", "HEAD"], env=env, cwd=repo, output=out / "revision.txt"
    ).strip()
    source_list = invoke(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "src/cmd/compile/internal/ssa/*.go",
            "src/cmd/compile/internal/base/debug.go",
            "src/cmd/compile/internal/test/testdata/scalaropt/*.go",
            "src/cmd/go/internal/work/*logos*.go",
            "src/cmd/go/internal/logos/settings/*.go",
        ],
        env=env,
        cwd=repo,
        output=out / "source-list.txt",
    )
    frozen = {
        str(path): digest(path)
        for path in [
            *(repo / name for name in sorted(set(source_list.splitlines()))),
            go,
            compiler,
            profiler,
            *compilers.values(),
            *wrappers,
            hyperfine,
            Path(__file__).resolve(),
        ]
    }
    (out / "source-hashes.json").write_text(
        json.dumps(frozen, sort_keys=True, indent=2) + "\n"
    )
    if (
        paired_compilers
        and frozen[str(compilers["baseline"])] != frozen[str(compilers["candidate"])]
        and tool_ids["baseline"] == tool_ids["candidate"]
    ):
        raise ValueError(
            "Different compiler binaries report the same -V=full tool identity; "
            "the Go build cache cannot distinguish them"
        )
    log = out / "log.jsonl"
    append_record(
        log,
        {
            "event": "freeze",
            "revision": revision,
            "source_hashes": frozen,
            "GOMAXPROCS": 1,
            "target_seconds": target_seconds,
            "environment": {
                name: env.get(name)
                for name in (
                    "GOOS",
                    "GOARCH",
                    "GOAMD64",
                    "GOEXPERIMENT",
                    "GOMAXPROCS",
                    "GOGC",
                    "GOMEMLIMIT",
                    "GODEBUG",
                    "GOCACHE",
                    "GOTMPDIR",
                    "TMPDIR",
                )
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "measurement_protocol": {
                "order": "baseline then candidate",
                "warmups_per_side": 3,
                "minimum_runs_per_side": 10,
                "profiles": "separate runs, excluded from timed samples",
            },
            "difference": difference,
            "compilers": {
                name: {
                    "path": str(selected),
                    "sha256": frozen[str(selected)],
                    "version": compiler_versions.get(name),
                    "tool_id": tool_ids.get(name),
                }
                for name, selected in compilers.items()
            },
            "runtime_build_flags": build_flags,
            "ranking": "parent supplies hot-path admission, lower bound, complexity cost, and keep/revert decision",
        },
    )

    def verify_frozen() -> None:
        for filename, expected in frozen.items():
            if digest(Path(filename)) != expected:
                raise RuntimeError(
                    f"Fingerprint changed during measurement: {filename}"
                )

    def measure(
        label: str, directory: Path, commands: list[list[str]], count: int
    ) -> None:
        verify_frozen()
        command = [
            str(hyperfine),
            "--warmup",
            "3",
            "--min-runs",
            "10",
            "--export-json",
            str(directory / "timing.json"),
        ]
        for (name, _), workload in zip(modes, commands, strict=True):
            command.extend(["--command-name", name, shlex.join(workload)])
        invoke(command, env=env, cwd=cwd, output=directory / "hyperfine.txt")
        parsed = cast(
            dict[str, object], json.loads((directory / "timing.json").read_text())
        )
        results = cast(list[dict[str, object]], parsed["results"])
        entries: list[dict[str, object]] = []
        for (name, _), result in zip(modes, results, strict=True):
            median = float(cast(float, result["median"]))
            deviation = float(cast(float, result["stddev"]))
            minimum = float(cast(float, result["min"]))
            entries.append(
                {
                    "mode": name,
                    "work_count": count,
                    "median_seconds": median,
                    "stddev_seconds": deviation,
                    "minimum_seconds": minimum,
                    "maximum_seconds": result["max"],
                    "noise_ok": deviation <= median * 0.2,
                    "minimum_batch_seconds_ok": minimum >= 1.0,
                }
            )
        ratio = float(cast(float, results[0]["median"])) / float(
            cast(float, results[1]["median"])
        )
        append_record(
            log,
            {
                "event": "measurement",
                "label": label,
                "work_count": count,
                "samples": entries,
                "baseline_over_candidate": ratio,
                "commands": commands,
            },
        )
        print(f"{label}: baseline/candidate = {ratio:.4f}", flush=True)
        if not all(bool(entry["noise_ok"]) for entry in entries):
            raise RuntimeError(f"Noise exceeds 20% of median; do not score {label}")
        if not all(bool(entry["minimum_batch_seconds_ok"]) for entry in entries):
            raise RuntimeError(
                f"A sample completed below one second; rerun BOTH sides in a fresh "
                f"campaign with larger --target-seconds; do not score {label}"
            )

    if args.phase in ("runtime", "all"):
        binaries: dict[str, Path] = {}
        for name, _ in modes:
            directory = out / name
            directory.mkdir()
            binary = directory / "scalaropt.test"
            invoke(
                [
                    str(go),
                    "test",
                    "-c",
                    "-o",
                    str(binary),
                    *build_flags[name],
                    PACKAGE,
                ],
                env=env,
                cwd=cwd,
                output=directory / "build.txt",
            )
            binaries[name] = binary
            frozen[str(binary)] = digest(binary)
            invoke(
                [str(binary), "-test.run=^TestScalar", "-test.count=1"],
                env=env,
                cwd=cwd,
                output=directory / "parity.txt",
            )
            append_record(
                log,
                {
                    "event": "binary",
                    "mode": name,
                    "sha256": frozen[str(binary)],
                    "bytes": binary.stat().st_size,
                    "parity": "passed",
                },
            )
        for bench in cast(list[str], args.benchmark or list(BENCHMARKS)):
            directory = out / bench
            directory.mkdir()

            def bench_command(
                name: str, duration: str, *, benchmark: str = bench
            ) -> list[str]:
                return [
                    str(binaries[name]),
                    "-test.run=^$",
                    f"-test.bench=^{benchmark}$",
                    f"-test.benchtime={duration}",
                    "-test.count=1",
                    "-test.benchmem",
                    "-test.cpu=1",
                ]

            initial = [
                samples(
                    invoke(
                        bench_command(name, "200ms"),
                        env=env,
                        cwd=cwd,
                        output=directory / f"calibrate-initial-{name}.txt",
                    ),
                    bench,
                )[0]
                for name, _ in modes
            ]
            count = max(
                1,
                math.ceil(
                    target_seconds * 1e9 / min(sample.ns_per_op for sample in initial)
                ),
            )
            for attempt in range(5):
                calibrated = [
                    samples(
                        invoke(
                            bench_command(name, f"{count}x"),
                            env=env,
                            cwd=cwd,
                            output=directory / f"calibrate-{attempt}-{name}.txt",
                        ),
                        bench,
                    )[0]
                    for name, _ in modes
                ]
                minimum_useful = min(sample.useful_seconds for sample in calibrated)
                if minimum_useful >= 1.0:
                    break
                count = math.ceil(count * target_seconds / minimum_useful)
            else:
                raise RuntimeError(f"Could not calibrate stable fixed work for {bench}")
            (directory / "iterations.txt").write_text(f"{count}\n")
            append_record(
                log,
                {
                    "event": "calibration",
                    "label": bench,
                    "work_count": count,
                    "samples": [asdict(sample) for sample in calibrated],
                },
            )
            measure(
                bench,
                directory,
                [bench_command(name, f"{count}x") for name, _ in modes],
                count,
            )
            for name, _ in modes:
                corroboration = samples(
                    invoke(
                        [
                            str(go),
                            "test",
                            "-run=^$",
                            f"-bench=^{bench}$",
                            f"-benchtime={count}x",
                            "-count=10",
                            "-benchmem",
                            "-cpu=1",
                            *build_flags[name],
                            PACKAGE,
                        ],
                        env=env,
                        cwd=cwd,
                        output=directory / f"corroborate-{name}.txt",
                    ),
                    bench,
                )
                if len(corroboration) != 10 or any(
                    sample.iterations != count or sample.useful_seconds < 1
                    for sample in corroboration
                ):
                    raise RuntimeError(
                        f"Corroboration violates fixed work or duration for {bench}/{name}"
                    )
                append_record(
                    log,
                    {
                        "event": "corroboration",
                        "label": bench,
                        "mode": name,
                        "work_count": count,
                        "median_ns_per_op": statistics.median(
                            sample.ns_per_op for sample in corroboration
                        ),
                        "samples": [asdict(sample) for sample in corroboration],
                    },
                )
                profile = directory / f"{name}.cpu.pprof"
                invoke(
                    [*bench_command(name, f"{count}x"), f"-test.cpuprofile={profile}"],
                    env=env,
                    cwd=cwd,
                    output=directory / f"profile-{name}.txt",
                )
                invoke(
                    [
                        str(go),
                        "tool",
                        "pprof",
                        "-top",
                        "-nodecount=20",
                        str(binaries[name]),
                        str(profile),
                    ],
                    env=env,
                    cwd=cwd,
                    output=directory / f"profile-top-{name}.txt",
                )
            verify_frozen()

    if args.phase in ("compile", "all"):
        packages = cast(
            list[str], args.compile_package or ["go/parser", "encoding/json"]
        )
        decoder = json.JSONDecoder()
        for package in packages:
            directory = out / ("compile-" + package.replace("/", "-"))
            directory.mkdir()
            text = invoke(
                [str(go), "list", "-export", "-deps", "-json", package],
                env=env,
                cwd=cwd,
                output=directory / "packages.json",
            )
            metadata: list[dict[str, object]] = []
            while text.strip():
                value, length = decoder.raw_decode(text.lstrip())
                metadata.append(cast(dict[str, object], value))
                text = text.lstrip()[length:]
            target = next(item for item in metadata if item["ImportPath"] == package)
            for field in ("CgoFiles", "CFiles", "SFiles", "SysoFiles", "EmbedFiles"):
                if target.get(field):
                    raise ValueError(
                        f"{package} needs unsupported direct-compile handling: {field}"
                    )
            importcfg = directory / "importcfg"
            importcfg.write_text(
                "".join(
                    f"packagefile {item['ImportPath']}={item['Export']}\n"
                    for item in metadata
                    if item.get("Export")
                )
            )
            source_dir = Path(cast(str, target["Dir"]))
            sources = [
                str(source_dir / filename)
                for filename in cast(list[str], target["GoFiles"])
            ]
            compile_inputs = {
                *(Path(filename) for filename in sources),
                importcfg,
                *(
                    Path(cast(str, item["Export"]))
                    for item in metadata
                    if item.get("Export")
                ),
            }
            input_hashes = {str(path): digest(path) for path in sorted(compile_inputs)}
            (directory / "input-hashes.json").write_text(
                json.dumps(input_hashes, sort_keys=True, indent=2) + "\n"
            )
            frozen.update(input_hashes)
            append_record(
                log,
                {
                    "event": "compile_input_freeze",
                    "label": package,
                    "hashes": input_hashes,
                    "rechecked": "before timing and after profiling, including imported archives",
                },
            )
            common = [
                "-p",
                package,
                "-std",
                "-complete",
                "-c=1",
                "-pack",
                "-importcfg",
                str(importcfg),
            ]
            batches: dict[str, Path] = {}
            compile_commands: dict[str, list[str]] = {}
            for name, mode in modes:
                # Standard-library O3 excludes variablemakethreshold, matching
                # logosForcedGcflags. Both compilers read the same archives.
                compile_flags = (
                    ["-d=inlbudget=400,inlrecursivebudget=80,logosopt=1"]
                    if paired_compilers
                    else [f"-d=logosopt={mode}"]
                )
                command = [
                    str(compilers[name]),
                    *common,
                    *compile_flags,
                    "-o",
                    str(directory / f"{name}.a"),
                    *sources,
                ]
                compile_commands[name] = command
                batch = directory / f"{name}-batch.sh"
                batch.write_text(
                    "#!/usr/bin/env bash\nset -euo pipefail\n"
                    "for ((i=0; i<$1; i++)); do\n  " + shlex.join(command) + "\ndone\n"
                )
                batches[name] = batch
            count = 1
            for attempt in range(5):
                elapsed: list[float] = []
                for name, _ in modes:
                    start = time.perf_counter()
                    invoke(
                        ["bash", str(batches[name]), str(count)],
                        env=env,
                        cwd=cwd,
                        output=directory / f"calibrate-{attempt}-{name}.txt",
                    )
                    elapsed.append(time.perf_counter() - start)
                if min(elapsed) >= target_seconds:
                    break
                count = math.ceil(count * target_seconds / min(elapsed))
            else:
                raise RuntimeError(f"Could not calibrate compile workload {package}")
            (directory / "iterations.txt").write_text(f"{count}\n")
            append_record(
                log,
                {
                    "event": "calibration",
                    "label": package,
                    "work_count": count,
                    "elapsed_seconds": elapsed,
                    "cache": "direct compiler; no action cache",
                },
            )
            measure(
                f"compile/{package}",
                directory,
                [["bash", str(batches[name]), str(count)] for name, _ in modes],
                count,
            )
            for name, _ in modes:
                cpu = directory / f"{name}.cpu.pprof"
                memory = directory / f"{name}.memory.pprof"
                command = compile_commands[name]
                invoke(
                    [
                        command[0],
                        f"-cpuprofile={cpu}",
                        f"-memprofile={memory}",
                        *command[1:],
                    ],
                    env=env,
                    cwd=cwd,
                    output=directory / f"profile-{name}.txt",
                )
                invoke(
                    [
                        str(go),
                        "tool",
                        "pprof",
                        "-top",
                        "-nodecount=25",
                        str(compilers[name]),
                        str(cpu),
                    ],
                    env=env,
                    cwd=cwd,
                    output=directory / f"profile-top-{name}.txt",
                )
                invoke(
                    [
                        str(go),
                        "tool",
                        "pprof",
                        "-top",
                        "-alloc_space",
                        "-nodecount=25",
                        str(compilers[name]),
                        str(memory),
                    ],
                    env=env,
                    cwd=cwd,
                    output=directory / f"allocations-top-{name}.txt",
                )
                invoke(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_rss",
                        str(directory / f"{name}-rss.json"),
                        *command,
                    ],
                    env=env,
                    cwd=cwd,
                    output=directory / f"rss-{name}.txt",
                )
            verify_frozen()
    append_record(log, {"event": "complete", "evidence": str(out)})
    print(f"Complete: {out}", flush=True)
    return 0


def measure_rss(arguments: list[str]) -> int:
    """A fresh helper process has exactly one child, so RSS is per compile."""
    output = Path(arguments[0])
    command = arguments[1:]
    start = time.perf_counter()
    result = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    output.write_text(
        json.dumps(
            {
                "command": command,
                "exit_code": result.returncode,
                "peak_rss_kib": usage.ru_maxrss,
                "user_seconds": usage.ru_utime,
                "system_seconds": usage.ru_stime,
                "wall_seconds": elapsed,
                "timing_use": "diagnostic only; excluded from primary timing samples",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_rss":
        raise SystemExit(measure_rss(sys.argv[2:]))
    raise SystemExit(main())
