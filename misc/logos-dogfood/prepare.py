#!/usr/bin/env python3
# pyright: strict
"""Fetch the pinned dogfood inputs and their module graphs without running code."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from run import (
    Case,
    Run,
    Source,
    all_files,
    copy_source,
    decode,
    execute,
    file_hash,
    obj,
    save_report,
    string,
    tree_hash,
    verify_policy,
    verify_source,
)

VERSION = "go1.27.4"
SUM = re.compile(r"h1:[A-Za-z0-9+/]{43}=")
MODULE = re.compile(r"[A-Za-z0-9_.~!+-]+(?:/[A-Za-z0-9_.~!+-]+)*")
MODULE_VERSION = re.compile(
    r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
)


class PreparationError(RuntimeError):
    """A pinned input could not be prepared without changing its source."""


@dataclass(frozen=True, order=True)
class Pin:
    path: str
    version: str

    @staticmethod
    def parse(path: object, version: object) -> Pin:
        module, release = string(path), string(version)
        if not MODULE.fullmatch(module) or any(
            part in (".", "..") for part in module.split("/")
        ):
            raise ValueError(f"invalid module path: {module}")
        if not MODULE_VERSION.fullmatch(release):
            raise ValueError(f"module requires an exact version: {module}@{release}")
        return Pin(module, release)

    @property
    def query(self) -> str:
        return f"{self.path}@{self.version}"


def relative_path(root: Path, name: str) -> Path:
    parts = name.split("/")
    if not name or any(part in ("", ".", "..") for part in parts) or "\\" in name:
        raise ValueError(f"source path must be a normalized relative path: {name}")
    path = Path(name)
    if path.is_absolute():
        raise ValueError(f"absolute source path: {name}")
    for index in range(1, len(parts) + 1):
        if root.joinpath(*parts[:index]).is_symlink():
            raise ValueError(f"symlink in source path: {name}")
    result = root.joinpath(*parts).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes its root: {name}")
    return result


def git_origin(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.strip("/")
    ):
        raise ValueError(f"Git origin must be an explicit HTTPS repository: {url}")
    return url


def json_records(text: str) -> tuple[dict[str, object], ...]:
    decoder = json.JSONDecoder()
    cursor = 0
    records: list[dict[str, object]] = []
    while cursor < len(text):
        if text[cursor].isspace():
            cursor += 1
            continue
        value, cursor = decoder.raw_decode(text, cursor)
        records.append(obj(cast(object, value)))
    return tuple(records)


def graph_pins(records: tuple[dict[str, object], ...], module: str) -> tuple[Pin, ...]:
    pins: set[Pin] = set()
    main = 0
    for record in records:
        if record.get("Error"):
            raise PreparationError(f"module graph error: {record['Error']}")
        if record.get("Main") is True:
            if record.get("Path") != module:
                raise PreparationError("main module differs from the manifest")
            main += 1
            continue
        target = obj(record["Replace"]) if "Replace" in record else record
        if target.get("Error"):
            raise PreparationError(f"replacement module error: {target['Error']}")
        if not target.get("Version"):
            raise PreparationError(
                "local filesystem replacements require a separate reviewed corpus input"
            )
        pins.add(Pin.parse(target.get("Path"), target.get("Version")))
    if main != 1:
        raise PreparationError(
            "module graph must contain exactly one pinned main module"
        )
    return tuple(sorted(pins))


def check_receipts(
    records: tuple[dict[str, object], ...], pins: tuple[Pin, ...], modules: Path
) -> None:
    expected = set(pins)
    seen: set[Pin] = set()
    for record in records:
        if record.get("Error"):
            raise PreparationError(f"module download failed: {record['Error']}")
        pin = Pin.parse(record.get("Path"), record.get("Version"))
        if pin not in expected or pin in seen:
            raise PreparationError(
                f"unexpected or duplicate download receipt: {pin.query}"
            )
        for name in ("Sum", "GoModSum"):
            if not SUM.fullmatch(string(record.get(name))):
                raise PreparationError(
                    f"missing checksum in {pin.query} receipt: {name}"
                )
        for name in ("Info", "GoMod", "Zip", "Dir"):
            path = Path(string(record.get(name))).resolve()
            available = path.is_dir() if name == "Dir" else path.is_file()
            if not path.is_relative_to(modules) or not available:
                raise PreparationError(
                    f"download path escapes or is missing: {pin.query} {name}"
                )
        seen.add(pin)
    if seen != expected:
        missing = ", ".join(pin.query for pin in sorted(expected - seen))
        raise PreparationError(f"missing module download receipts: {missing}")


def source_groups(cases: tuple[Case, ...]) -> tuple[tuple[Case, tuple[str, ...]], ...]:
    groups: dict[tuple[str, str], tuple[Case, list[str]]] = {}
    ids: set[str] = set()
    for case in cases:
        if case.id in ids:
            raise ValueError(f"duplicate case id: {case.id}")
        ids.add(case.id)
        key = case.source_kind, case.source
        if key not in groups:
            groups[key] = case, []
        first, names = groups[key]
        if source_identity(case) != source_identity(first):
            raise ValueError(f"conflicting pins for source: {case.source}")
        names.append(case.id)
    if not groups:
        raise ValueError("manifest must contain at least one case")
    return tuple((case, tuple(ids)) for case, ids in groups.values())


def source_identity(case: Case) -> tuple[str, ...]:
    return case.module, case.upstream, case.revision, case.version, case.sum


def reject_git_redirection() -> None:
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_SHALLOW_FILE",
        "GIT_CONFIG_PARAMETERS",
    ):
        if os.environ.get(name):
            raise ValueError(
                f"unset {name}: Git operations must stay in the new source directory"
            )


@dataclass(frozen=True)
class Preparation:
    go: Path
    out: Path
    environment: dict[str, str]
    timeout: float
    report: dict[str, object]
    commands: list[dict[str, object]]

    @property
    def modules(self) -> Path:
        return self.out / "modules"

    def invoke(self, argv: tuple[str, ...], cwd: Path, label: str) -> Run:
        result = execute(
            argv, cwd, self.environment, self.out / "logs" / label, self.timeout
        )
        self.commands.append(asdict(result))
        save_report(self.out / "preparation.json", self.report)
        if result.error or result.timed_out or result.returncode != 0:
            raise PreparationError(
                f"{label} failed: exit={result.returncode}, timeout={result.timed_out}, "
                f"error={result.error!r}; see {result.stderr}"
            )
        return result

    def download(
        self, pins: tuple[Pin, ...], label: str
    ) -> tuple[dict[str, object], ...]:
        if not pins:
            return ()
        result = self.invoke(
            (str(self.go), "mod", "download", "-json", *(pin.query for pin in pins)),
            self.out / "download-context",
            label,
        )
        records = json_records(Path(result.stdout).read_text())
        check_receipts(records, pins, self.modules)
        return records

    def fetch_git(self, case: Case) -> None:
        target = relative_path(self.out / "sources", case.source)
        target.mkdir(parents=True, exist_ok=False)
        self.invoke(
            ("git", "init", "--quiet", "--object-format=sha1", str(target)),
            self.out,
            f"{case.id}/git-init",
        )
        prefix = ("git", "-C", str(target))
        self.invoke(
            (*prefix, "remote", "add", "origin", git_origin(case.upstream)),
            self.out,
            f"{case.id}/git-origin",
        )
        self.invoke(
            (*prefix, "fetch", "--no-tags", "--depth=1", "origin", case.revision),
            self.out,
            f"{case.id}/git-fetch",
        )
        self.invoke(
            (*prefix, "checkout", "--detach", "--quiet", case.revision),
            self.out,
            f"{case.id}/git-checkout",
        )

    def dependencies(self, case: Case, source: Source) -> dict[str, object]:
        work = self.out / "work" / case.id
        copy_source(source, work)
        dependency_files: dict[str, str | None] = {}
        try:
            for name in ("go.mod", "go.sum"):
                path = work / name
                dependency_files[name] = file_hash(path) if path.is_file() else None
                if path.is_file():
                    saved = self.out / "inputs" / case.id / name
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, saved)
                    path.chmod(0o444)
            if dependency_files["go.mod"] is None:
                raise PreparationError("source has no pinned go.mod")
            result = self.invoke(
                (str(self.go), "list", "-mod=readonly", "-m", "-json", "all"),
                work,
                f"{case.id}/module-graph",
            )
            pins = graph_pins(
                json_records(Path(result.stdout).read_text()), case.module
            )
            receipts = self.download(pins, f"{case.id}/dependency-downloads")
            return {
                "dependency_files": dependency_files,
                "selected_modules": [asdict(pin) for pin in pins],
                "downloads": receipts,
            }
        finally:
            # This also catches a new go.sum or any other generated source file.
            unchanged = tree_hash(work, all_files(work), source.prefix) == source.digest
            if unchanged:
                shutil.rmtree(work)
            else:
                # Preserve the changed private copy for diagnosis, never for replay.
                raise PreparationError(
                    f"dependency preparation changed source bytes: {work}"
                )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("corpus.json")
    )
    parser.add_argument("--go-oracle", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="new private input and receipt directory",
    )
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    reject_git_redirection()
    manifest: Path = args.manifest.resolve()
    go: Path = args.go_oracle.resolve()
    out: Path = args.out.resolve()
    if args.timeout <= 0 or not go.is_file() or not os.access(go, os.X_OK):
        raise ValueError(
            "a positive timeout and an executable --go-oracle are required"
        )
    raw = decode(manifest.read_text())
    if not isinstance(raw, list):
        raise TypeError("manifest must be an array")
    cases = tuple(Case.parse(item) for item in cast(list[object], raw))
    groups = source_groups(cases)
    for case in cases:
        if case.source_kind == "git":
            relative_path(out / "sources", case.source)
            git_origin(case.upstream)
        elif case.source_kind == "module":
            Pin.parse(case.module, case.version)
        else:
            root = relative_path(manifest.parent, case.source)
            if out.is_relative_to(root):
                raise ValueError("preparation output must not be inside a local source")
        verify_policy(case, manifest.parent)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("sources", "modules", "cache", "download-context"):
        (out / name).mkdir()
    (out / "download-context" / "go.mod").write_text(
        "module example.invalid/logos-dogfood-download\n\ngo 1.27.4\n"
    )
    environment = {
        "PATH": f"{go.parent}{os.pathsep}{os.environ.get('PATH', '')}",
        "GOCACHE": str(out / "cache"),
        "GOMODCACHE": str(out / "modules"),
        "GOTOOLCHAIN": "local",
        "GOWORK": "off",
        "GOENV": "off",
        "GO111MODULE": "on",
        "GOFLAGS": "-mod=readonly -buildvcs=false -p=1",
        "GOPROXY": "https://proxy.golang.org",
        "GOSUMDB": "sum.golang.org",
        "GOPRIVATE": "",
        "GONOPROXY": "",
        "GONOSUMDB": "",
        "GOINSECURE": "",
        "GOTELEMETRY": "off",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_COUNT": "0",
        "GIT_ALLOW_PROTOCOL": "https",
        **{
            key: os.environ[key]
            for key in (
                "GOMAXPROCS",
                "GOMEMLIMIT",
                "GOGC",
                "GOOS",
                "GOARCH",
                "GOAMD64",
                "TMPDIR",
            )
            if key in os.environ
        },
    }
    commands: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "python": sys.version,
        "complete": False,
        "preparer_sha256": file_hash(Path(__file__)),
        "runner_sha256": file_hash(Path(__file__).with_name("run.py")),
        "manifest": str(manifest),
        "manifest_sha256": file_hash(manifest),
        "go": {"path": str(go), "sha256": file_hash(go)},
        "case_count": len(cases),
        "source_count": len(groups),
        "commands": commands,
        "sources": rows,
    }
    prep = Preparation(go, out, environment, args.timeout, report, commands)
    save_report(out / "preparation.json", report)
    try:
        version = prep.invoke((str(go), "version"), out, "go-version")
        version_text = Path(version.stdout).read_text().strip()
        report["go_version"] = version_text
        if not version_text.startswith(f"go version {VERSION} "):
            raise PreparationError(f"expected stock {VERSION}; got {version_text!r}")
        for case, ids in groups:
            row: dict[str, object] = {
                "case_ids": ids,
                "case": asdict(case),
                "status": "preparing",
            }
            rows.append(row)
            save_report(out / "preparation.json", report)
            if case.source_kind == "git":
                prep.fetch_git(case)
            elif case.source_kind == "module":
                row["source_download"] = prep.download(
                    (Pin.parse(case.module, case.version),),
                    f"{case.id}/source-download",
                )
            source = verify_source(case, out / "sources", prep.modules, manifest.parent)
            row.update(source_path=str(source.path), source_digest=source.digest)
            row.update(prep.dependencies(case, source))
            verified = verify_source(
                case, out / "sources", prep.modules, manifest.parent
            )
            if verified.digest != source.digest:
                raise PreparationError(
                    "original source changed during dependency preparation"
                )
            row["status"] = "ready"
            save_report(out / "preparation.json", report)
            print(f"{', '.join(ids)}: ready ({source.digest})", flush=True)
    except (
        OSError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
        PreparationError,
    ) as exc:
        if rows and rows[-1]["status"] == "preparing":
            rows[-1]["status"] = "failed"
        report["error"] = str(exc)
        save_report(out / "preparation.json", report)
        print(f"preparation failed: {exc}", flush=True)
        return 1
    report["complete"] = True
    save_report(out / "preparation.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
