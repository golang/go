# pyright: strict
"""Preparation must preserve pins, source bytes, and failed-download evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from prepare import (
    Pin,
    Preparation,
    PreparationError,
    check_receipts,
    git_origin,
    graph_pins,
    json_records,
    reject_git_redirection,
    relative_path,
    source_groups,
)
from run import Case, Run, Source, all_files, file_hash, tree_hash


def case(root: Path) -> Case:
    return Case(
        id="fixture",
        source_kind="local",
        source="fixture",
        module="example.com/fixture",
        upstream="local",
        revision="",
        version="",
        sum=tree_hash(root, all_files(root)),
        cohort="stress",
        backend="go-only",
        cgo=0,
        tags=(),
        patterns=("./...",),
        operation="test",
    )


@given(st.lists(st.dictionaries(st.text(), st.integers()), max_size=20))
def test_json_receipts_are_a_sequence(records: list[dict[str, int]]) -> None:
    text = "\n\n".join(json.dumps(record) for record in records)
    assert json_records(text) == tuple(records)


def test_partial_receipt_stream_cannot_pass() -> None:
    with pytest.raises(json.JSONDecodeError):
        json_records('{"Path": "example.com/a"}\n{"Path":')


@pytest.mark.parametrize(
    "version", ["latest", "main", "upgrade", "v1", "v1.2", "v1.2.3/../x"]
)
def test_version_queries_are_not_pins(version: str) -> None:
    with pytest.raises(ValueError, match="exact version"):
        Pin.parse("example.com/a", version)


@pytest.mark.parametrize(
    "name", ["../elsewhere", "/absolute", "./local", "a/../b", "a//b", "a\\b"]
)
def test_source_paths_cannot_escape_or_alias(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        relative_path(tmp_path, name)


def test_source_paths_reject_symlink_components(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        relative_path(root, "link/source")


@pytest.mark.parametrize(
    "name",
    [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_SHALLOW_FILE",
        "GIT_CONFIG_PARAMETERS",
    ],
)
def test_ambient_git_state_cannot_redirect_fetches(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "/an/existing/repository")
    with pytest.raises(ValueError, match=name):
        reject_git_redirection()


@pytest.mark.parametrize(
    "origin",
    [
        "file:///tmp/repo",
        "ssh://git@example.com/repo",
        "https://user@example.com/repo",
        "https://example.com:443/repo",
        "https://example.com/repo?token=value",
    ],
)
def test_git_fetch_has_an_explicit_uncredentialed_https_origin(origin: str) -> None:
    with pytest.raises(ValueError):
        git_origin(origin)


def test_duplicate_variants_must_share_exact_source_identity(tmp_path: Path) -> None:
    original = case(tmp_path)
    variant = replace(original, id="fixture-purego", tags=("purego",))
    assert source_groups((original, variant)) == (
        (original, ("fixture", "fixture-purego")),
    )
    changed = replace(variant, sum="h1:" + "A" * 43 + "=")
    with pytest.raises(ValueError, match="conflicting pins"):
        source_groups((original, changed))


def test_graph_preserves_remote_replacement_pin() -> None:
    records: tuple[dict[str, object], ...] = (
        {"Path": "example.com/main", "Main": True},
        {
            "Path": "example.com/old",
            "Version": "v1.2.0",
            "Replace": {"Path": "example.com/new", "Version": "v2.1.0"},
        },
    )
    assert graph_pins(records, "example.com/main") == (
        Pin("example.com/new", "v2.1.0"),
    )


@pytest.mark.parametrize(
    "records",
    [
        (),
        ({"Path": "example.com/wrong", "Main": True},),
        (
            {"Path": "example.com/main", "Main": True},
            {"Path": "example.com/dep", "Error": {"Err": "unavailable"}},
        ),
        (
            {"Path": "example.com/main", "Main": True},
            {"Path": "example.com/dep", "Replace": {"Path": "../unreviewed"}},
        ),
    ],
)
def test_incomplete_or_unreviewed_graphs_fail(
    records: tuple[dict[str, object], ...],
) -> None:
    with pytest.raises(PreparationError):
        graph_pins(records, "example.com/main")


def receipt(modules: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "Path": "example.com/a",
        "Version": "v1.0.0",
        "Sum": "h1:" + "A" * 43 + "=",
        "GoModSum": "h1:" + "B" * 43 + "=",
    }
    for name in ("Info", "GoMod", "Zip"):
        path = modules / name
        path.write_text("verified by the fake download")
        record[name] = str(path)
    directory = modules / "source"
    directory.mkdir()
    record["Dir"] = str(directory)
    return record


def test_download_receipts_must_account_for_every_exact_pin(tmp_path: Path) -> None:
    record = receipt(tmp_path)
    pins = (Pin("example.com/a", "v1.0.0"),)
    check_receipts((record,), pins, tmp_path)
    with pytest.raises(PreparationError, match="missing module download"):
        check_receipts((), pins, tmp_path)
    with pytest.raises(PreparationError, match="duplicate"):
        check_receipts((record, record), pins, tmp_path)
    with pytest.raises(PreparationError, match="download failed"):
        check_receipts(({**record, "Error": "network failed"},), pins, tmp_path)
    with pytest.raises(PreparationError, match="escapes"):
        check_receipts(({**record, "Dir": str(tmp_path.parent)},), pins, tmp_path)


def test_failed_command_retains_its_report_and_raw_output(tmp_path: Path) -> None:
    commands: list[dict[str, object]] = []
    report: dict[str, object] = {"complete": False, "commands": commands}
    prep = Preparation(Path(sys.executable), tmp_path, {}, 10, report, commands)
    with pytest.raises(PreparationError, match="exit=7"):
        prep.invoke(
            (sys.executable, "-c", "import sys; print('partial receipt'); sys.exit(7)"),
            tmp_path,
            "failed-download",
        )
    assert len(commands) == 1
    assert commands[0]["returncode"] == 7
    assert Path(str(commands[0]["stdout"])).read_text() == "partial receipt\n"
    saved = json_records((tmp_path / "preparation.json").read_text())[0]
    assert saved["complete"] is False


@pytest.mark.parametrize("mutation", ["none", "go.mod", "new-go.sum"])
def test_graph_loading_preserves_original_and_copied_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    (original / "go.mod").write_text("module example.com/fixture\n\ngo 1.27.4\n")
    pinned = case(original)
    source = Source(original, all_files(original), pinned.sum)
    output = tmp_path / "preparation"
    output.mkdir()
    prep = Preparation(Path("/fake/go"), output, {}, 10, {}, [])
    commands: list[tuple[str, ...]] = []

    def invoke(self: Preparation, argv: tuple[str, ...], cwd: Path, label: str) -> Run:
        commands.append(argv)
        assert label == "fixture/module-graph"
        assert argv == ("/fake/go", "list", "-mod=readonly", "-m", "-json", "all")
        assert not (cwd / "go.mod").stat().st_mode & 0o222
        if mutation == "go.mod":
            path = cwd / "go.mod"
            path.chmod(0o644)
            path.write_text("module example.com/changed\n")
        elif mutation == "new-go.sum":
            (cwd / "go.sum").write_text("unexpected checksum\n")
        stdout = output / "graph.stdout"
        stderr = output / "graph.stderr"
        stdout.write_text(json.dumps({"Path": "example.com/fixture", "Main": True}))
        stderr.write_text("")
        return Run(
            argv,
            str(cwd),
            self.environment,
            0,
            False,
            "",
            0,
            str(stdout),
            str(stderr),
            file_hash(stdout),
            file_hash(stderr),
        )

    monkeypatch.setattr(Preparation, "invoke", invoke)
    if mutation == "none":
        result = prep.dependencies(pinned, source)
        assert result["selected_modules"] == []
        assert not (output / "work" / "fixture").exists()
    else:
        with pytest.raises(PreparationError, match="changed source bytes"):
            prep.dependencies(pinned, source)
        assert (output / "work" / "fixture").is_dir()
    assert len(commands) == 1
    assert tree_hash(original, all_files(original)) == source.digest
