#!/usr/bin/env python3
# pyright: strict
"""Replay pinned external sources against Go, loGos, and both lint drivers."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

STAGES = ("oracle", "go", "logos", "logos-lint", "gglint")
SUCCESS = {"pass", "clean", "findings"}
BLOCKERS = (
    "module lookup disabled by GOPROXY=off",
    "missing go.sum entry",
    "updates to go.mod needed",
    "requires go >=",
    "no required module provides package",
    "cannot find module providing package",
    'C compiler "',
    "executable file not found",
    "no space left on device",
)


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(k, str) for k in cast(dict[object, object], value)
    ):
        raise TypeError("expected a JSON object with string keys")
    return cast(dict[str, object], value)


def string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("expected an array of strings")
    return tuple(string(item) for item in cast(list[object], value))


def decode(text: str) -> object:
    return cast(object, json.loads(text))


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def all_files(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            p.relative_to(root).as_posix() for p in root.rglob("*") if not p.is_dir()
        )
    )


def tree_hash(root: Path, files: tuple[str, ...], prefix: str = "") -> str:
    """Go's dirhash.Hash1: hash the ordered content-hash/name summary."""
    digest = hashlib.sha256()
    for name in sorted(files):
        path = root / name
        if "\n" in name or path.is_symlink():
            raise ValueError(f"unsupported newline or symlink in source: {name}")
        full_name = f"{prefix}/{name}" if prefix else name
        digest.update(f"{file_hash(path)}  {full_name}\n".encode())
    return "h1:" + base64.b64encode(digest.digest()).decode("ascii")


def within(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes its root: {name}")
    return path


@dataclass(frozen=True)
class Case:
    id: str
    source_kind: str
    source: str
    module: str
    upstream: str
    revision: str
    version: str
    sum: str
    cohort: str
    backend: str
    cgo: int
    tags: tuple[str, ...]
    patterns: tuple[str, ...]
    operation: str
    logos_policy: str = ""
    logos_policy_sha256: str = ""

    @staticmethod
    def parse(value: object) -> Case:
        data = obj(value)
        cgo = data["cgo"]
        if type(cgo) is not int or cgo not in (0, 1):
            raise ValueError("cgo must be 0 or 1")
        case = Case(
            id=string(data["id"]),
            source_kind=string(data["source_kind"]),
            source=string(data["source"]),
            module=string(data["module"]),
            upstream=string(data["upstream"]),
            revision=string(data.get("revision", "")),
            version=string(data.get("version", "")),
            sum=string(data.get("sum", "")),
            cohort=string(data["cohort"]),
            backend=string(data["backend"]),
            operation=string(data["operation"]),
            cgo=cgo,
            tags=strings(data.get("tags", [])),
            patterns=strings(data.get("patterns", ["./..."])),
            logos_policy=string(data.get("logos_policy", "")),
            logos_policy_sha256=string(data.get("logos_policy_sha256", "")),
        )
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", case.id):
            raise ValueError(f"invalid case id: {case.id}")
        if case.source_kind not in ("git", "module", "local"):
            raise ValueError(f"unsupported source_kind: {case.source_kind}")
        if case.operation not in ("build", "test") or not case.patterns:
            raise ValueError(f"invalid operation or empty patterns: {case.id}")
        if any(not pattern or pattern.startswith("-") for pattern in case.patterns):
            raise ValueError("package patterns must not contain command flags")
        if any(not re.fullmatch(r"[A-Za-z0-9_.]+", tag) for tag in case.tags):
            raise ValueError("build tags must be individual tag names")
        if not case.source or not case.module:
            raise ValueError("source and module must be explicit")
        if case.source_kind == "git" and not re.fullmatch(
            r"[0-9a-f]{40}", case.revision
        ):
            raise ValueError("Git sources require a full commit pin")
        if case.source_kind != "git" and not re.fullmatch(
            r"h1:[A-Za-z0-9+/]{43}=", case.sum
        ):
            raise ValueError("module and local sources require a full h1 checksum")
        if case.source_kind == "module" and not case.version:
            raise ValueError("module sources require an exact version")
        if bool(case.logos_policy) != bool(case.logos_policy_sha256):
            raise ValueError("logos_policy and logos_policy_sha256 must be paired")
        if case.logos_policy:
            path = Path(case.logos_policy)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "policy path must remain within the manifest directory"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", case.logos_policy_sha256):
                raise ValueError("policy requires a full SHA-256 checksum")
        return case


@dataclass(frozen=True)
class Source:
    path: Path
    files: tuple[str, ...]
    digest: str
    prefix: str = ""


@dataclass(frozen=True)
class Policy:
    path: Path
    content: bytes
    sha256: str


def verify_policy(case: Case, root: Path) -> Policy | None:
    """Read only the exact reviewed capability grants pinned by the manifest."""
    if not case.logos_policy:
        return None
    relative = Path(case.logos_policy)
    path = within(root, case.logos_policy)
    if any(
        (root.joinpath(*relative.parts[:i])).is_symlink()
        for i in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("policy paths must not contain symlinks")
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != case.logos_policy_sha256:
        raise ValueError("policy contents differ from the pinned SHA-256 checksum")
    data = obj(cast(object, tomllib.loads(content.decode("utf-8"))))
    if set(data) != {"capabilities"}:
        raise ValueError("replay policy may contain only capability grants")
    grants = obj(data["capabilities"])
    if not grants or set(grants) - {
        "transitive_unsafe_allowlist",
        "transitive_net_allowlist",
        "transitive_exec_allowlist",
    }:
        raise ValueError("replay policy contains an unsupported capability grant")
    for value in grants.values():
        packages = strings(value)
        if not packages or len(set(packages)) != len(packages):
            raise ValueError("capability grants must be nonempty and unique")
        if any(
            not re.fullmatch(r"[A-Za-z0-9_~.!+-]+(?:/[A-Za-z0-9_~.!+-]+)*", p)
            or any(part in (".", "..") for part in p.split("/"))
            for p in packages
        ):
            raise ValueError("capability grants require exact package paths")
    return Policy(path, content, digest)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def verify_source(
    case: Case, source_root: Path, modules: Path, local_root: Path
) -> Source:
    prefix = ""
    if case.source_kind == "git":
        root = within(source_root, case.source)
        if git(root, "rev-parse", "HEAD") != case.revision:
            raise ValueError("Git HEAD does not match the pinned revision")
        if git(root, "remote", "get-url", "origin").removesuffix(
            ".git"
        ) != case.upstream.removesuffix(".git"):
            raise ValueError("Git origin does not match the pinned upstream")
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("Git source is not clean")
        files = tuple(git(root, "ls-files", "-z").rstrip("\0").split("\0"))
        digest = tree_hash(root, files)
    elif case.source_kind == "module":
        escaped = "".join(
            "!" + char.lower() if char.isupper() else char for char in case.module
        )
        root = within(modules, f"{escaped}@{case.version}")
        metadata = within(modules, f"cache/download/{escaped}/@v/{case.version}.info")
        info = obj(decode(metadata.read_text()))
        if info.get("Version") != case.version or (
            case.revision and obj(info.get("Origin")).get("Hash") != case.revision
        ):
            raise ValueError(
                "module metadata does not match version and origin revision"
            )
        if not case.sum.startswith("h1:"):
            raise ValueError("module requires a pinned h1 checksum")
        files = all_files(root)
        prefix = f"{case.module}@{case.version}"
        digest = tree_hash(root, files, prefix)
        if digest != case.sum:
            raise ValueError(f"module contents differ from pinned sum: {digest}")
    else:
        root = within(local_root, case.source)
        files = all_files(root)
        digest = tree_hash(root, files)
        if not case.sum or digest != case.sum:
            raise ValueError(f"local fixture differs from pinned sum: {digest}")
    if not files or not root.is_dir():
        raise ValueError("source directory is empty or missing")
    return Source(root, files, digest, prefix)


def copy_source(source: Source, target: Path) -> None:
    """Copy verified inputs only; never execute tests in an upstream checkout."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in source.files:
        destination = within(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.path / name, destination)
        destination.chmod((source.path / name).stat().st_mode | 0o200)
    if tree_hash(target, source.files, source.prefix) != source.digest:
        raise ValueError("source changed while copying verified inputs")


def prepare_source(
    source: Source, target: Path, stage: str, policy: Policy | None
) -> dict[str, object]:
    """Keep the verified upstream input separate from LoGOs replay permissions."""
    copy_source(source, target)
    if policy is None or stage not in ("logos", "logos-lint"):
        return {"effective_source_digest": source.digest}
    destination = target / "logos.toml"
    if destination.exists():
        raise ValueError("replay policy must not overwrite an upstream logos.toml")
    destination.write_bytes(policy.content)
    return {
        "policy_sha256": policy.sha256,
        "effective_source_digest": tree_hash(
            target, (*source.files, "logos.toml"), source.prefix
        ),
    }


@dataclass(frozen=True)
class Run:
    command: tuple[str, ...]
    cwd: str
    environment: dict[str, str]
    returncode: int | None
    timed_out: bool
    error: str
    seconds: float
    stdout: str
    stderr: str
    stdout_sha256: str
    stderr_sha256: str


def kill_and_reap(process: subprocess.Popen[bytes]) -> int:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass  # The command exited between cancellation and the kill.
    return process.wait()


def execute(
    command: tuple[str, ...], cwd: Path, env: dict[str, str], log: Path, timeout: float
) -> Run:
    log.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = log.with_suffix(".stdout"), log.with_suffix(".stderr")
    started = time.monotonic()
    code: int | None = None
    timed_out = False
    error = ""
    inherited = os.environ.copy()
    for name in ("GOROOT", "GOEXPERIMENT", "GODEBUG"):
        inherited.pop(name, None)
    inherited.update(env)
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=inherited,
                stdout=out,
                stderr=err,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            error = str(exc)
        else:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                code = kill_and_reap(process)
            except KeyboardInterrupt:
                kill_and_reap(process)
                raise
    return Run(
        command,
        str(cwd),
        env,
        code,
        timed_out,
        error,
        round(time.monotonic() - started, 3),
        str(stdout),
        str(stderr),
        file_hash(stdout),
        file_hash(stderr),
    )


@dataclass(frozen=True)
class Assessment:
    status: str
    reason: str = ""
    diagnostics: int = 0
    duplicate_diagnostics: int = 0
    normalized_sha256: str = ""


def failure(run: Run) -> Assessment | None:
    if run.timed_out:
        return Assessment("timeout", "deadline exceeded; process group terminated")
    if run.error:
        return Assessment("blocked", run.error)
    # Diagnostic messages may quote arbitrary source strings. Inspect transport
    # errors here; structured compiler errors are checked by the JSON decoder.
    text = "\n".join(
        line
        for line in Path(run.stderr).read_text(errors="replace").splitlines()
        if not line.lstrip().startswith("{")
    )
    if any(
        message in text
        for message in (
            "matched no packages",
            "no Go files in",
            "no non-test Go files in",
        )
    ):
        return Assessment("load-failed", "a requested pattern matched no packages")
    if any(message in text for message in BLOCKERS):
        return Assessment(
            "blocked",
            "dependency, toolchain, native compiler, or disk prerequisite missing",
        )
    if re.search(
        r"(?m)^(?:panic:|fatal error:|SIGSEGV|runtime: unexpected)|internal compiler error",
        text,
    ):
        return Assessment(
            "candidate-failed", "tool crashed or reported an internal compiler error"
        )
    return None


def assess_build(run: Run, stage: str, oracle: Assessment | None) -> Assessment:
    problem = failure(run)
    if problem is not None:
        if stage == "oracle" and problem.status == "candidate-failed":
            return Assessment("upstream-failed", problem.reason)
        return problem
    if stage == "oracle":
        return Assessment("pass" if run.returncode == 0 else "upstream-failed")
    if oracle is None or oracle.status != "pass":
        return Assessment(
            "uncompared",
            "oracle was not run or did not pass; inspect candidate exit status",
        )
    return Assessment("pass" if run.returncode == 0 else "candidate-failed")


def canonical(records: list[dict[str, object]], cwd: str) -> tuple[str, int]:
    # Sort a list, not a set: duplicate multiplicity remains observable.
    encoded = sorted(
        json.dumps(record, sort_keys=True, separators=(",", ":")).replace(
            cwd, "<source>"
        )
        if cwd
        else json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    digest = hashlib.sha256("\n".join(encoded).encode()).hexdigest()
    return digest, len(encoded) - len(set(encoded))


def logos_records(stderr: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    pending_header = False
    for line in stderr.splitlines():
        if not line.strip():
            continue
        if re.fullmatch(r"# \S+(?: \[\S+\])?", line):
            if pending_header:
                raise ValueError("package banner without diagnostic records")
            pending_header = True
            continue
        records.append(obj(decode(line)))
        pending_header = False
    if pending_header:
        raise ValueError("trailing package banner without diagnostic records")
    return records


def validate_compiler_error(record: dict[str, object]) -> None:
    """Check the untagged compiler-error shape emitted by base.logosRecord."""
    fields = {
        "schema_version",
        "severity",
        "rule_id",
        "file",
        "span",
        "message",
        "provenance",
        "autofix_candidates",
    }
    span = obj(record.get("span"))
    provenance = obj(record.get("provenance"))
    span_fields = {
        "start_line",
        "start_col",
        "end_line",
        "end_col",
        "byte_offset",
        "byte_len",
    }
    if (
        set(record) != fields
        or set(span) != span_fields
        or any(type(value) is not int or value < 0 for value in span.values())
    ):
        raise ValueError("malformed ordinary compiler-error fields")
    if (
        record["severity"] != "deny"
        or provenance != {"origin_expr": "", "origin_line": 0, "inferred_type": ""}
        or type(provenance.get("origin_line")) is not int
        or record["autofix_candidates"] != []
        or span["start_line"] != span["end_line"]
        or span["start_col"] != span["end_col"]
        or span["byte_len"] != 0
    ):
        raise ValueError("malformed ordinary compiler-error metadata")


def assess_lint(run: Run, tool: str, mode: str = "") -> Assessment:
    problem = failure(run)
    if problem is not None:
        return problem
    if run.returncode not in (0, 1):
        return Assessment("candidate-failed", "unexpected linter exit status")
    try:
        stdout = Path(run.stdout).read_text()
        stderr = Path(run.stderr).read_text()
        compiler_errors = False
        if tool == "logos-lint":
            if stdout.strip():
                raise ValueError("logos lint wrote unexpected stdout")
            records = logos_records(stderr)
            for record in records:
                if record.get("schema_version") != "1.0":
                    raise ValueError("unsupported diagnostic schema")
                rule = string(record.get("rule_id"))
                if not string(record.get("message")):
                    raise ValueError("empty diagnostic message")
                string(record.get("file"))
                obj(record.get("span"))
                if record.get("severity") not in ("warn", "deny", "forbid"):
                    raise ValueError("unknown diagnostic severity")
                if not rule:
                    validate_compiler_error(record)
                    compiler_errors = True
            expected = int(
                any(record["severity"] in ("deny", "forbid") for record in records)
            )
        else:
            document = obj(decode(stdout))
            if document.get("mode") != mode:
                raise ValueError("gglint report mode differs from requested mode")
            raw = document.get("findings")
            if not isinstance(raw, list):
                raise ValueError("gglint findings is not an array")
            records = [obj(item) for item in cast(list[object], raw)]
            for record in records:
                for field in ("rule", "path", "message"):
                    if not string(record.get(field)):
                        raise ValueError(f"empty gglint finding {field}")
                for field in ("line", "column"):
                    if (
                        type(record.get(field)) is not int
                        or cast(int, record[field]) < 1
                    ):
                        raise ValueError(f"invalid gglint finding {field}")
            if any(record["rule"] == "typecheck" for record in records):
                return Assessment(
                    "load-failed",
                    "typecheck findings prevent complete analysis",
                    len(records),
                )
            if any(
                not line.startswith("gglint: config ")
                for line in stderr.splitlines()
                if line.strip()
            ):
                raise ValueError("gglint wrote unexpected stderr")
            expected = int(bool(records))
        if run.returncode != expected:
            raise ValueError("exit status disagrees with diagnostic severity/count")
        if compiler_errors:
            return Assessment(
                "load-failed",
                "ordinary compiler errors prevent complete analysis",
                len(records),
            )
        digest, duplicates = canonical(records, run.cwd)
        if tool == "gglint" and duplicates:
            raise ValueError("gglint returned duplicate findings")
        return Assessment(
            "findings" if records else "clean",
            diagnostics=len(records),
            duplicate_diagnostics=duplicates,
            normalized_sha256=digest,
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        return Assessment("invalid-report", str(exc))


def compare_repeats(
    first: Assessment, second: Assessment, first_run: Run, second_run: Run
) -> Assessment:
    if first.status not in SUCCESS:
        return first
    if second.status not in SUCCESS:
        return second
    if (first.normalized_sha256, first_run.returncode) != (
        second.normalized_sha256,
        second_run.returncode,
    ):
        return Assessment(
            "nondeterministic", "diagnostics or exit status changed on repeat"
        )
    return first


def save_report(path: Path, report: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def toolchain_identity(
    binary: Path, env: dict[str, str], cwd: Path, log: Path
) -> dict[str, object]:
    run = execute(
        (str(binary), "env", "-json", "GOROOT", "GOTOOLDIR"), cwd, env, log, 30
    )
    identity: dict[str, object] = {"run": asdict(run)}
    if run.error or run.timed_out or run.returncode != 0:
        return identity
    locations = obj(decode(Path(run.stdout).read_text()))
    root = Path(string(locations["GOROOT"]))
    directory = Path(string(locations["GOTOOLDIR"]))
    identity["goroot"] = str(root)
    suffix = ".exe" if os.name == "nt" else ""
    identity["tools"] = {
        name: {
            "path": str(directory / (name + suffix)),
            "sha256": file_hash(directory / (name + suffix))
            if (directory / (name + suffix)).is_file()
            else None,
        }
        for name in ("compile", "link", "asm", "cgo")
    }
    compiler = root / "src" / "cmd" / "compile"
    if compiler.is_dir():
        identity["compiler_source_h1"] = tree_hash(compiler, all_files(compiler))
    if (root / ".git").exists():
        identity["source_revision"] = git(root, "rev-parse", "HEAD")
        identity["source_status"] = git(
            root, "status", "--porcelain", "--untracked-files=all", "--", "src"
        )
    return identity


def environment(
    case: Case, binary: Path, modules: Path, cache: Path, jobs: int
) -> dict[str, str]:
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": f"{binary.parent}{os.pathsep}{os.environ.get('PATH', '')}",
        "GOCACHE": str(cache),
        "GOMODCACHE": str(modules),
        "GOTOOLCHAIN": "local",
        "GOWORK": "off",
        "GOENV": "off",
        "GOPROXY": "off",
        "GOSUMDB": "off",
        "GOTELEMETRY": "off",
        "CGO_ENABLED": str(case.cgo),
        "GOMAXPROCS": str(jobs),
        "GOFLAGS": f"-mod=readonly -buildvcs=false -p={jobs}",
        **{
            key: os.environ[key]
            for key in (
                "GOOS",
                "GOARCH",
                "GOAMD64",
                "GOMEMLIMIT",
                "TMPDIR",
                "CC",
                "CXX",
                "CGO_CFLAGS",
                "CGO_CPPFLAGS",
                "CGO_CXXFLAGS",
                "CGO_LDFLAGS",
            )
            if key in os.environ
        },
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("corpus.json")
    )
    parser.add_argument(
        "--source-root", type=Path, required=True, help="parent of pinned Git checkouts"
    )
    parser.add_argument("--module-cache", type=Path, required=True)
    parser.add_argument("--go-oracle", type=Path, required=True)
    parser.add_argument("--go-candidate", type=Path)
    parser.add_argument("--logos", type=Path)
    parser.add_argument("--gglint", type=Path)
    parser.add_argument(
        "--out", type=Path, required=True, help="new evidence directory"
    )
    parser.add_argument(
        "--cache-root", type=Path, help="reusable isolated build caches"
    )
    parser.add_argument(
        "--only", action="append", default=[], help="case id; repeat to select cases"
    )
    parser.add_argument(
        "--stage", action="append", choices=STAGES, help="repeat to select stages"
    )
    parser.add_argument(
        "--mode", action="append", choices=("basic", "standard", "strict")
    )
    parser.add_argument(
        "--operation", choices=("build", "test"), help="override manifest operation"
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--jobs", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.timeout <= 0 or args.jobs <= 0:
        raise ValueError("timeout and jobs must be positive")
    manifest: Path = args.manifest.resolve()
    raw = decode(manifest.read_text())
    if not isinstance(raw, list):
        raise TypeError("manifest must be an array")
    cases = tuple(Case.parse(item) for item in cast(list[object], raw))
    ids = {case.id for case in cases}
    if not cases or len(ids) != len(cases) or set(args.only) - ids:
        raise ValueError("duplicate manifest id or unknown --only id")
    if args.only:
        cases = tuple(case for case in cases if case.id in args.only)
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    if report_path.exists():
        raise ValueError("evidence already exists; choose a new --out directory")
    modules: Path = args.module_cache.resolve()
    source_root: Path = args.source_root.resolve()
    caches: Path = (args.cache_root or out / "cache").resolve()
    binaries = {
        name: path.resolve() if path else None
        for name, path in (
            ("oracle", args.go_oracle),
            ("go", args.go_candidate),
            ("logos", args.logos),
            ("gglint", args.gglint),
        )
    }
    selected = args.stage or tuple(
        stage for stage in STAGES if stage != "gglint" or args.gglint is not None
    )
    stages = tuple(stage for stage in STAGES if stage in selected)
    modes: tuple[str, ...] = tuple(args.mode or ("basic", "standard", "strict"))
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "complete": False,
        "runner_sha256": file_hash(Path(__file__)),
        "manifest": str(manifest),
        "manifest_sha256": file_hash(manifest),
        "stages": stages,
        "modes": modes,
        "timeout_seconds": args.timeout,
        "operation_override": args.operation,
        "jobs": args.jobs,
        "sources": len({(case.source_kind, case.source) for case in cases}),
        "case_count": len(cases),
        "cases": rows,
        "tools": {
            name: {
                "path": str(path),
                "sha256": file_hash(path) if path.is_file() else None,
            }
            for name, path in binaries.items()
            if path is not None
        },
    }
    save_report(report_path, report)
    versions: dict[str, object] = {}
    report["versions"] = versions
    for name, binary in binaries.items():
        if binary is not None:
            env = environment(cases[0], binary, modules, caches / name, args.jobs)
            run = execute(
                (str(binary), "version"), manifest.parent, env, out / "tools" / name, 30
            )
            versions[name] = {
                "run": asdict(run),
                "text": Path(run.stdout).read_text(errors="replace").strip(),
            }
            if name != "gglint":
                versions[name] = {
                    **obj(versions[name]),
                    "toolchain": toolchain_identity(
                        binary, env, manifest.parent, out / "tools" / f"{name}-identity"
                    ),
                }
    oracle_binary = binaries["oracle"]
    assert oracle_binary is not None
    host_env = execute(
        (
            str(oracle_binary),
            "env",
            "-json",
            "GOOS",
            "GOARCH",
            "CC",
            "CXX",
            "GOVERSION",
        ),
        manifest.parent,
        environment(cases[0], oracle_binary, modules, caches / "oracle", args.jobs),
        out / "tools" / "go-env",
        30,
    )
    report["go_environment"] = {
        "run": asdict(host_env),
        "text": Path(host_env.stdout).read_text(errors="replace"),
    }
    save_report(report_path, report)
    for case in cases:
        row: dict[str, object] = {
            "case": asdict(case),
            "status": "running",
            "stages": [],
        }
        rows.append(row)
        save_report(report_path, report)
        work = out / "work" / case.id
        results: list[dict[str, object]] = []
        row["stages"] = results
        oracle: Assessment | None = None
        try:
            source = verify_source(case, source_root, modules, manifest.parent)
            row["source_path"], row["source_digest"] = str(source.path), source.digest
            policy = verify_policy(case, manifest.parent)
            if policy is not None:
                saved = out / "inputs" / case.id / "logos.toml"
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(policy.content)
                row["logos_policy"] = {
                    "source_path": str(policy.path),
                    "saved_path": str(saved),
                    "sha256": policy.sha256,
                }
            save_report(report_path, report)
            for stage in stages:
                tool = "logos" if stage == "logos-lint" else stage
                binary = binaries[tool]
                if binary is None:
                    results.append(
                        {
                            "stage": stage,
                            "assessment": asdict(
                                Assessment("blocked", "binary not supplied")
                            ),
                        }
                    )
                    continue
                for mode in modes if stage == "gglint" else ("",):
                    label = f"{stage}-{mode}" if mode else stage
                    inputs = prepare_source(source, work, stage, policy)
                    driver = binaries["oracle"] if stage == "gglint" else binary
                    assert driver is not None
                    env = environment(case, driver, modules, caches / tool, args.jobs)
                    flags = (f"-tags={','.join(case.tags)}",) if case.tags else ()
                    if stage == "gglint":
                        config = out / "config" / f"{case.id}.toml"
                        config.parent.mkdir(parents=True, exist_ok=True)
                        config.write_text(
                            f"[run]\ntests = true\nbuild_tags = {json.dumps(case.tags)}\n"
                        )
                        command = (
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
                        command = (
                            str(binary),
                            "lint",
                            "--json-diagnostics",
                            *flags,
                            *case.patterns,
                        )
                    else:
                        operation = args.operation or case.operation
                        test_flags = (
                            (
                                "-count=1",
                                "-short",
                                f"-timeout={max(1, int(args.timeout - 5))}s",
                            )
                            if operation == "test"
                            else ()
                        )
                        command = (
                            str(binary),
                            operation,
                            *test_flags,
                            *flags,
                            *case.patterns,
                        )
                    run = execute(
                        command, work, env, out / "logs" / case.id / label, args.timeout
                    )
                    if stage in ("gglint", "logos-lint"):
                        assessment = assess_lint(run, stage, mode)
                    else:
                        assessment = assess_build(run, stage, oracle)
                    entry: dict[str, object] = {
                        "stage": stage,
                        "mode": mode,
                        "inputs": inputs,
                        "run": asdict(run),
                        "assessment": asdict(assessment),
                    }
                    if stage == "oracle":
                        oracle = assessment
                    elif (
                        stage in ("gglint", "logos-lint")
                        and assessment.status in SUCCESS
                    ):
                        entry["repeat_inputs"] = prepare_source(
                            source, work, stage, policy
                        )
                        repeat = execute(
                            command,
                            work,
                            env,
                            out / "logs" / case.id / f"{label}-repeat",
                            args.timeout,
                        )
                        repeated = assess_lint(repeat, stage, mode)
                        entry["repeat"] = asdict(repeat)
                        entry["repeat_assessment"] = asdict(repeated)
                        assessment = compare_repeats(assessment, repeated, run, repeat)
                        entry["assessment"] = asdict(assessment)
                    results.append(entry)
                    print(
                        f"{case.id} {label}: {assessment.status} ({assessment.diagnostics} diagnostics)",
                        flush=True,
                    )
                    save_report(report_path, report)
            row["status"] = "finished"
        except (OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            row["status"], row["error"] = "blocked", str(exc)
            print(f"{case.id}: blocked: {exc}", flush=True)
        finally:
            if work.exists():
                shutil.rmtree(work)
        save_report(report_path, report)
    counts: dict[str, int] = {}
    for row in rows:
        if row["status"] != "finished":
            counts["blocked"] = counts.get("blocked", 0) + 1
        for entry in cast(list[dict[str, object]], row["stages"]):
            status = string(obj(entry["assessment"])["status"])
            counts[status] = counts.get(status, 0) + 1
    report["complete"], report["counts"] = True, counts
    save_report(report_path, report)
    return int(any(status not in SUCCESS for status in counts))


if __name__ == "__main__":
    raise SystemExit(main())
