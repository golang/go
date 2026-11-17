#!/usr/bin/env python3
# pyright: strict
"""Archive preserved reports and logs without running a compiler or rewriting results."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from collections import Counter
from pathlib import Path
from typing import cast

WORKSPACE = Path("/workspace/scratch/d8d410ec882d")
OUT = Path(__file__).resolve().parent
ROOTS = (
    "installed-audit",
    "corpus-review",
    "recovery-evidence",
    "corpus-verified-runtime",
    "corpus-vcs/runtime",
    "corpus-vcs/native-default",
    "corpus-verified-gglint",
)
EXCLUDED = {
    ".git",
    ".ruff_cache",
    "__pycache__",
    "work",
    "cache",
    "caches",
    "ejected",
    "dependency-prep",
    "dependency-work",
    "panic-generation",
}
SUFFIXES = {
    ".json",
    ".stdout",
    ".stderr",
    ".log",
    ".md",
    ".sha256",
    ".gz",
    ".py",
    ".sh",
    ".toml",
    ".txt",
}
HISTORICAL_HEAD = "22ab6f7a169e905d9fde5e345db101735a321486"
HISTORICAL_SOURCE = "8d74f82a97b8509400372ae4818525f19e7105ef"
CORRECTED_HEAD = "82c3817d309fdbf6092e4a5ad6f2f9e69aadd0dc"
BUNDLE_SHA256 = "844cb749d84f79db09004b1a92d0a4945823946359e714f10656fe9cc08fbad1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def decode(data: bytes) -> object:
    return cast(object, json.loads(data))


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return cast(dict[str, object], value)


def rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise TypeError("expected JSON array")
    return [obj(item) for item in cast(list[object], value)]


def snapshots() -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for relative in ROOTS:
        base = WORKSPACE / relative
        if not base.is_dir():
            raise FileNotFoundError(base)
        for folder, dirs, names in os.walk(base, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in EXCLUDED)
            for name in sorted(names):
                path = Path(folder) / name
                if (
                    path.suffix not in SUFFIXES
                    and path.name not in {"go.mod", "go.sum"}
                ) or path.is_symlink():
                    continue
                before = path.stat()
                data = path.read_bytes()
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ValueError(f"file changed during capture: {path}")
                result[path.relative_to(WORKSPACE).as_posix()] = data
    return result


def nested_bundle(payload: dict[str, bytes]) -> dict[str, bytes]:
    archive = payload["corpus-review/installed-acceptance.tar.gz"]
    if digest(archive) != BUNDLE_SHA256:
        raise ValueError("the completed installed-acceptance bundle changed")
    expected: dict[str, str] = {}
    for line in (
        payload["corpus-review/installed-acceptance-files.sha256"].decode().splitlines()
    ):
        sha256, path = line.split(maxsplit=1)
        expected[path] = sha256
    result: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for member in source.getmembers():
            if (
                not member.isfile()
                or member.name.startswith("/")
                or ".." in Path(member.name).parts
            ):
                raise ValueError(f"unexpected nested archive member: {member.name}")
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"missing nested archive data: {member.name}")
            data = stream.read()
            if member.name == "FILES.sha256":
                if data != payload["corpus-review/installed-acceptance-files.sha256"]:
                    raise ValueError(
                        "nested archive manifest differs from its original companion"
                    )
            elif expected.get(member.name) != digest(data):
                raise ValueError(f"nested archive hash mismatch: {member.name}")
            result[member.name] = data
    if set(expected) | {"FILES.sha256"} != set(result):
        raise ValueError(
            "nested archive member list differs from its original manifest"
        )
    return result


def verify_logs(all_files: dict[str, bytes]) -> list[dict[str, object]]:
    checked: list[dict[str, object]] = []

    def visit(value: object, origin: str, location: str) -> None:
        if isinstance(value, list):
            for i, item in enumerate(cast(list[object], value)):
                visit(item, origin, f"{location}[{i}]")
        elif isinstance(value, dict):
            node = cast(dict[str, object], value)
            for channel in ("stdout", "stderr"):
                path, recorded = node.get(channel), node.get(channel + "_sha256")
                if isinstance(path, str) and isinstance(recorded, str):
                    absolute = Path(path)
                    relative = (
                        absolute.relative_to(WORKSPACE).as_posix()
                        if absolute.is_absolute()
                        else path
                    )
                    if relative not in all_files:
                        raise ValueError(
                            f"recorded log is not archived: {origin} {relative}"
                        )
                    actual = digest(all_files[relative])
                    if recorded != actual:
                        raise ValueError(
                            f"recorded log hash mismatch: {origin} {relative}"
                        )
                    checked.append(
                        {
                            "report": origin,
                            "record": location,
                            "channel": channel,
                            "path": relative,
                            "sha256": actual,
                        }
                    )
            for key, item in node.items():
                visit(item, origin, f"{location}.{key}")

    for path, data in sorted(all_files.items()):
        if path.endswith(".json"):
            visit(decode(data), path, "$")
    return checked


def summarize(path: str, data: bytes) -> dict[str, object]:
    value = obj(decode(data))
    summary: dict[str, object] = {"path": path, "report_sha256": digest(data)}
    for key in (
        "complete",
        "accepted",
        "pass",
        "status",
        "source_stable",
        "tools_stable",
        "unchanged",
        "case_count",
    ):
        if key in value:
            summary[key] = value[key]
    if value.get("complete") is False:
        summary["archive_classification"] = (
            "interrupted; no completed campaign or final source check is inferred"
        )
    elif value.get("complete") is True:
        summary["archive_classification"] = (
            "recorded campaign completion; inspect failures and timeouts separately"
        )
    else:
        summary["archive_classification"] = (
            "diagnostic or provenance record without a campaign completion flag"
        )
    if "cases" in value:
        cases = rows(value["cases"])
        cells = [cell for case in cases for cell in rows(case.get("stages", []))]
        summary["observed_cases"] = len(cases)
        summary["finished_cases"] = sum(
            case.get("status") == "finished" for case in cases
        )
        summary["recorded_cells"] = len(cells)
        summary["assessment_counts"] = dict(
            sorted(
                Counter(
                    str(obj(cell.get("assessment", {})).get("status")) for cell in cells
                ).items()
            )
        )
        summary["pending_cases"] = [
            obj(case["case"]).get("id")
            for case in cases
            if case.get("status") != "finished"
        ]
    if "rows" in value:
        summaries: list[dict[str, object]] = []
        for row in rows(value["rows"]):
            item = {
                key: row[key]
                for key in (
                    "name",
                    "label",
                    "driver",
                    "pass",
                    "accepted",
                    "status",
                    "exact_match",
                    "assessment",
                )
                if key in row
            }
            if "run" in row:
                run = obj(row["run"])
                item["run"] = {
                    key: run[key]
                    for key in ("returncode", "timed_out", "seconds", "error")
                    if key in run
                }
            if "events" in row:
                events = obj(row["events"])
                item["events"] = {
                    key: len(cast(list[object], events[key]))
                    if isinstance(events[key], list)
                    else events[key]
                    for key in (
                        "started",
                        "passed",
                        "expected",
                        "exact_match",
                        "started_count",
                        "passed_count",
                        "missing_required_run_or_pass",
                        "failures",
                        "skips",
                    )
                    if key in events
                }
            summaries.append(item)
        summary["rows"] = summaries
    return summary


def make_tar(payload: dict[str, bytes]) -> bytes:
    target = io.BytesIO()
    with (
        gzip.GzipFile(
            fileobj=target, mode="wb", filename="", mtime=0, compresslevel=9
        ) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive,
    ):
        for name, data in sorted(payload.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mtime = member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    return target.getvalue()


def main() -> None:
    payload = snapshots()
    nested = nested_bundle(payload)
    all_files = dict(payload)
    for path, data in nested.items():
        if path in all_files and all_files[path] != data:
            raise ValueError(f"nested and outer historical snapshots differ: {path}")
        all_files[path] = data
    checked = verify_logs(all_files)
    reports = [
        summarize(path, data)
        for path, data in sorted(all_files.items())
        if path.endswith("report.json")
    ]
    index: dict[str, object] = {
        "schema_version": 1,
        "historical_logos_head": HISTORICAL_HEAD,
        "historical_logos_source_tree": HISTORICAL_SOURCE,
        "corrected_branch_head_not_tested_by_this_archive": CORRECTED_HEAD,
        "provenance_rule": "Original report bytes and recorded before/after identities are preserved. No final source check, exit status, completion flag or unobserved result has been added to an interrupted report.",
        "scope_roots": ROOTS,
        "excluded_directory_names": sorted(EXCLUDED),
        "selected_suffixes": sorted(SUFFIXES),
        "selected_additional_names": ["go.mod", "go.sum"],
        "nested_installed_acceptance": {
            "path": "corpus-review/installed-acceptance.tar.gz",
            "sha256": BUNDLE_SHA256,
            "verified_regular_files": len(nested),
        },
        "verified_recorded_log_references": len(checked),
        "verified_distinct_recorded_logs": len({str(item["path"]) for item in checked}),
        "reports": reports,
    }
    by_path = {str(report["path"]): report for report in reports}
    for path, cells in (
        ("corpus-vcs/native-default/report.json", 24),
        ("corpus-verified-gglint/report.json", 51),
    ):
        if (
            by_path[path].get("complete") is not False
            or by_path[path].get("recorded_cells") != cells
        ):
            raise ValueError(
                f"partial campaign advanced since the observed snapshot: {path}"
            )
    if by_path["installed-audit/upstream/report.json"].get("complete") is not False:
        raise ValueError("upstream report completion changed")
    modernc = obj(decode(payload["corpus-vcs/runtime/modernc-sql/report.json"]))
    modernc_rows = rows(modernc["rows"])
    if modernc.get("complete") is not True or [
        str(row["driver"]) for row in modernc_rows
    ] != ["oracle", "go", "logos"]:
        raise ValueError("unexpected modernc campaign shape")
    for row in modernc_rows[:2]:
        events = obj(row["events"])
        if (
            row.get("status") != "pass"
            or len(cast(list[object], events["passed"])) != 15
            or events.get("exact_match") is not True
        ):
            raise ValueError("modernc baseline execution proof changed")
    candidate = modernc_rows[2]
    if (
        obj(candidate["run"]).get("timed_out") is not True
        or obj(candidate["events"]).get("passed") != []
    ):
        raise ValueError("modernc candidate is not the recorded timeout")
    readme = (OUT / "ARCHIVE_README.md").read_bytes()
    payload["ARCHIVE_README.md"] = readme
    payload["ARCHIVE_INDEX.json"] = encode(index)
    payload["RECORDED_LOG_HASHES.json"] = encode({"references": checked})
    payload["archive-tool.py"] = Path(__file__).read_bytes()
    payload["FILES.sha256"] = "".join(
        f"{digest(data)}  {path}\n" for path, data in sorted(payload.items())
    ).encode()
    archive = make_tar(payload)
    if archive != make_tar(payload):
        raise ValueError("archive construction is not deterministic")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        members = source.getmembers()
        if [member.name for member in members] != sorted(payload):
            raise ValueError("archive member order or list changed")
        for member in members:
            stream = source.extractfile(member)
            if stream is None or stream.read() != payload[member.name]:
                raise ValueError(f"archive round trip failed: {member.name}")
    archive_path = OUT / "historical-22ab-evidence.tar.gz"
    archive_path.write_bytes(archive)
    manifest: dict[str, object] = {
        "archive": archive_path.name,
        "sha256": digest(archive),
        "git_blob_sha1": hashlib.sha1(
            f"blob {len(archive)}\0".encode() + archive
        ).hexdigest(),
        "compressed_bytes": len(archive),
        "regular_files": len(payload),
        "content_bytes": sum(map(len, payload.values())),
        "format": "USTAR with sorted regular members, mode0644 and zero timestamps/owners; gzip filename empty and mtime0",
        "deterministic_reconstruction_verified": True,
        "every_member_roundtrip_verified": True,
        "historical_logos_head": HISTORICAL_HEAD,
        "corrected_branch_head_not_tested_by_this_archive": CORRECTED_HEAD,
        "verified_recorded_log_references": len(checked),
        "verified_distinct_recorded_logs": len({str(item["path"]) for item in checked}),
        "nested_bundle_verified_regular_files": len(nested),
        "files": [
            {"path": path, "bytes": len(data), "sha256": digest(data)}
            for path, data in sorted(payload.items())
        ],
    }
    (OUT / "historical-22ab-evidence.json").write_bytes(encode(manifest))
    (OUT / "historical-22ab-evidence.sha256").write_text(
        f"{digest(archive)}  {archive_path.name}\n"
    )
    (OUT / "ARCHIVE_INDEX.json").write_bytes(encode(index))
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "files"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
