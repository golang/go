# pyright: strict
"""A backend pass requires executed tests and a successful matching stock run."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest
import replay_backends
from hypothesis import given
from hypothesis import strategies as st
from replay_backends import (
    Replay,
    assess_runtime,
    load_profiles,
    make_command,
    make_environment,
    parse_events,
    policy_for,
    positive_timeout,
    prepare_work,
)

from run import Run, Source, all_files, file_hash, prepare_source, tree_hash

SQLITE_TESTS = ("TestOpen", "TestClose", "TestInsert")
MODERNC_TESTS = (
    "TestNamedParameters",
    "TestMemDB",
    "TestBinding",
    "TestBeginMode",
    "TestMultiStmtQueryStringRoundtrip",
    "TestExecReturningMultiRow",
    "TestMutexWakeOnce",
    "TestBinding/DB",
    "TestBinding/Prepare",
    "TestBeginMode/delete/deferred",
    "TestBeginMode/wal/deferred",
    "TestBeginMode/delete/immediate",
    "TestBeginMode/wal/immediate",
    "TestBeginMode/delete/exclusive",
    "TestBeginMode/wal/exclusive",
)


def profile(name: str) -> Replay:
    return next(item for item in load_profiles() if item.case.id == name)


def record(package: str, name: str, action: str) -> str:
    return json.dumps({"Package": package, "Test": name, "Action": action})


def executed(package: str, names: tuple[str, ...]) -> str:
    return "\n".join(
        record(package, name, action) for action in ("run", "pass") for name in names
    )


def run_record(tmp_path: Path, text: str, code: int = 0) -> Run:
    stdout, stderr = tmp_path / "run.stdout", tmp_path / "run.stderr"
    stdout.write_text(text)
    stderr.write_text("")
    return Run(
        ("fake-go", "test"),
        str(tmp_path),
        {},
        code,
        False,
        "",
        0.1,
        str(stdout),
        str(stderr),
        file_hash(stdout),
        file_hash(stderr),
    )


@pytest.mark.parametrize("count", (2, 3, 4))
def test_sqlite_requires_exactly_three_named_tests(tmp_path: Path, count: int) -> None:
    selected = profile("sqlite3-cgo")
    names = (*SQLITE_TESTS, "TestUnexpected")[:count]
    result = assess_runtime(
        run_record(tmp_path, executed(selected.case.module, names)),
        "oracle",
        selected,
        None,
    )
    assert result.accepted is (count == 3)


@pytest.mark.parametrize("count", (14, 15, 16))
def test_modernc_requires_all_fifteen_events(tmp_path: Path, count: int) -> None:
    selected = profile("modernc-sqlite")
    names = (*MODERNC_TESTS, "TestUnexpected")[:count]
    result = assess_runtime(
        run_record(tmp_path, executed(selected.case.module, names)),
        "oracle",
        selected,
        None,
    )
    assert result.accepted is (count == 15)


@pytest.mark.parametrize(
    "text", ("", '{"Action":"pass","Package":"example.com/empty"}')
)
def test_exit_zero_without_test_execution_fails(tmp_path: Path, text: str) -> None:
    result = assess_runtime(
        run_record(tmp_path, text), "oracle", profile("xxhash-asm-core"), None
    )
    assert result.assessment.status == "upstream-failed"
    assert not result.accepted


@pytest.mark.parametrize(
    "actions", (("run",), ("pass",), ("run", "pass", "pass"), ("run", "run", "pass"))
)
def test_unbalanced_test_events_fail(tmp_path: Path, actions: tuple[str, ...]) -> None:
    selected = profile("xxhash-asm-core")
    text = "\n".join(
        record(selected.case.module, "TestAll", action) for action in actions
    )
    assert not assess_runtime(
        run_record(tmp_path, text), "oracle", selected, None
    ).accepted


@pytest.mark.parametrize("action", ("skip", "fail"))
def test_skip_or_failure_cannot_be_hidden_by_pass(tmp_path: Path, action: str) -> None:
    selected = profile("xxhash-asm-core")
    text = (
        executed(selected.case.module, ("TestAll",))
        + "\n"
        + record(selected.case.module, "TestAll", action)
    )
    assert not assess_runtime(
        run_record(tmp_path, text), "oracle", selected, None
    ).accepted


def test_package_failure_is_rejected_even_with_zero_exit(tmp_path: Path) -> None:
    selected = profile("xxhash-asm-core")
    text = (
        executed(selected.case.module, ("TestAll",))
        + "\n"
        + json.dumps({"Action": "fail", "Package": selected.case.module})
    )
    assert not assess_runtime(
        run_record(tmp_path, text), "oracle", selected, None
    ).accepted


def test_plugin_requires_dynamic_test_in_its_package(tmp_path: Path) -> None:
    selected = profile("xxhash-plugin-cgo")
    text = executed(selected.case.module, ("TestAll", "TestDynamic"))
    assert not assess_runtime(
        run_record(tmp_path, text), "oracle", selected, None
    ).accepted
    text += "\n" + executed(selected.case.module + "/dynamic", ("TestDynamic",))
    text += "\n" + json.dumps(
        {"Action": "skip", "Package": selected.case.module + "/xxhsum"}
    )
    assert assess_runtime(run_record(tmp_path, text), "oracle", selected, None).accepted


@pytest.mark.parametrize(
    "candidate_names",
    (("TestAll",), ("TestAll", "TestOther"), ("TestAll", "TestAll", "TestSub")),
)
def test_candidate_must_match_stock_event_multiset(
    tmp_path: Path, candidate_names: tuple[str, ...]
) -> None:
    selected = profile("xxhash-asm-core")
    baseline = assess_runtime(
        run_record(tmp_path, executed(selected.case.module, ("TestAll", "TestSub"))),
        "oracle",
        selected,
        None,
    )
    assert baseline.accepted
    result = assess_runtime(
        run_record(tmp_path, executed(selected.case.module, candidate_names)),
        "go",
        selected,
        baseline,
    )
    assert result.assessment.status == "candidate-failed"
    assert not result.accepted


def test_package_identity_is_part_of_comparison(tmp_path: Path) -> None:
    selected = profile("xxhash-asm-core")
    baseline = assess_runtime(
        run_record(tmp_path, executed(selected.case.module, ("TestAll", "TestOther"))),
        "oracle",
        selected,
        None,
    )
    text = (
        executed(selected.case.module, ("TestAll",))
        + "\n"
        + executed("example.com/wrong", ("TestOther",))
    )
    assert not assess_runtime(
        run_record(tmp_path, text), "logos", selected, baseline
    ).accepted


def test_failed_stock_run_cannot_bless_candidate(tmp_path: Path) -> None:
    selected = profile("sqlite3-cgo")
    text = executed(selected.case.module, SQLITE_TESTS)
    baseline = assess_runtime(run_record(tmp_path, text, 1), "oracle", selected, None)
    result = assess_runtime(run_record(tmp_path, text), "logos", selected, baseline)
    assert not baseline.accepted
    assert result.assessment.status == "uncompared"
    assert not result.accepted


def test_missing_stock_run_cannot_bless_candidate(tmp_path: Path) -> None:
    selected = profile("sqlite3-cgo")
    result = assess_runtime(
        run_record(tmp_path, executed(selected.case.module, SQLITE_TESTS)),
        "go",
        selected,
        None,
    )
    assert result.assessment.status == "uncompared"
    assert not result.accepted


def test_completed_events_do_not_hide_process_timeout(tmp_path: Path) -> None:
    selected = profile("sqlite3-cgo")
    run = replace(
        run_record(tmp_path, executed(selected.case.module, SQLITE_TESTS)),
        timed_out=True,
    )
    assert assess_runtime(run, "oracle", selected, None).assessment.status == "timeout"


@pytest.mark.parametrize(
    "text",
    (
        '{"Action":',
        "[]",
        '{"Action":"run","Package":"p","Test":1}',
        '{"Action":"run","Test":"TestA"}',
        '{"Action":"unexpected","Package":"p","Test":"TestA"}',
    ),
)
def test_malformed_event_stream_cannot_pass(tmp_path: Path, text: str) -> None:
    result = assess_runtime(
        run_record(tmp_path, text), "oracle", profile("xxhash-asm-core"), None
    )
    assert result.assessment.status == "invalid-report"
    assert "line 1" in result.assessment.reason


@given(st.lists(st.sampled_from(("TestA", "TestB", "TestC")), min_size=1, max_size=8))
def test_event_order_does_not_erase_multiplicity(names: list[str]) -> None:
    lines = executed("example.com/tests", tuple(names)).splitlines()
    forward = parse_events("\n".join(lines))
    reverse = parse_events("\n".join(reversed(lines)))
    assert forward == reverse
    assert len(forward.started) == len(names)
    assert forward.started == forward.passed


def test_profiles_preserve_default_core_builds_and_match_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOEXPERIMENT", "jsonv2")
    binary = tmp_path / "go"
    for selected in load_profiles():
        env = make_environment(
            selected, binary, tmp_path / "modules", tmp_path / "cache", tmp_path
        )
        if selected.case.id == "xxhash-plugin-cgo":
            assert env["GOEXPERIMENT"] == "none"
        else:
            assert "GOEXPERIMENT" not in env
        for stage in ("oracle", "go", "logos"):
            command = make_command(selected, stage, binary)
            assert ("-O=1" in command) is (
                selected.case.id == "xxhash-plugin-cgo" and stage == "logos"
            )
            assert "-count=1" in command and "-json" in command
            assert ("-tags=purego" in command) is (
                selected.case.id == "xxhash-purego-core"
            )
        if selected.case.id.startswith("xxhash-"):
            assert not selected.tests
        else:
            assert selected.case.patterns == (".",)
    assert profile("xxhash-asm-core").case.cgo == 0
    assert profile("xxhash-purego-core").case.cgo == 0
    assert profile("xxhash-plugin-cgo").case.cgo == 1


def test_selected_sql_filters_retain_exact_scope() -> None:
    sqlite = profile("sqlite3-cgo")
    modernc = profile("modernc-sqlite")
    assert sqlite.tests == SQLITE_TESTS
    assert modernc.tests == MODERNC_TESTS[:7]
    assert sqlite.expected == SQLITE_TESTS
    assert modernc.expected == MODERNC_TESTS


def test_reviewed_policy_only_changes_modernc_logos_copy(tmp_path: Path) -> None:
    for selected in load_profiles():
        policy = policy_for(selected)
        if selected.case.id != "modernc-sqlite":
            assert policy is None
            continue
        assert policy is not None
        assert (
            policy.sha256
            == "5666c90bc1f25d5f0d7d0e5b53e80c2fa8e7ebe341286f7d7b0b1c5651513096"
        )
        original = tmp_path / "original"
        original.mkdir()
        (original / "go.mod").write_text("module example.com/input\n")
        source = Source(original, ("go.mod",), tree_hash(original, ("go.mod",)))
        for stage in ("oracle", "go", "logos"):
            work = tmp_path / stage
            preparation = prepare_source(source, work, stage, policy)
            assert (work / "logos.toml").exists() is (stage == "logos")
            assert ("policy_sha256" in preparation) is (stage == "logos")
        assert not (original / "logos.toml").exists()


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf", "-inf"))
def test_timeout_must_be_positive_and_finite(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_timeout(value)


def test_positive_timeout_retains_fractional_deadline() -> None:
    assert positive_timeout("0.25") == 0.25


@pytest.mark.parametrize("stage", ("go", "logos"))
def test_matching_candidate_passes(tmp_path: Path, stage: str) -> None:
    selected = profile("sqlite3-cgo")
    text = executed(selected.case.module, SQLITE_TESTS)
    baseline = assess_runtime(run_record(tmp_path, text), "oracle", selected, None)
    result = assess_runtime(run_record(tmp_path, text), stage, selected, baseline)
    assert baseline.accepted and result.accepted


@pytest.mark.parametrize(
    "name,names", (("sqlite3-cgo", SQLITE_TESTS), ("modernc-sqlite", MODERNC_TESTS))
)
@pytest.mark.parametrize("change", ("name", "package"))
def test_sql_exactness_rejects_equal_count_substitution(
    tmp_path: Path, name: str, names: tuple[str, ...], change: str
) -> None:
    selected = profile(name)
    package = selected.case.module
    text = executed(package, names[:-1])
    text += "\n" + executed(
        "example.com/wrong" if change == "package" else package,
        ("TestWrong" if change == "name" else names[-1],),
    )
    assert not assess_runtime(
        run_record(tmp_path, text), "oracle", selected, None
    ).accepted


def test_baseline_must_belong_to_same_profile(tmp_path: Path) -> None:
    assembly = profile("xxhash-asm-core")
    purego = profile("xxhash-purego-core")
    text = executed(assembly.case.module, ("TestAll",))
    baseline = assess_runtime(run_record(tmp_path, text), "oracle", assembly, None)
    assert baseline.accepted
    result = assess_runtime(run_record(tmp_path, text), "logos", purego, baseline)
    assert result.assessment.status == "uncompared"
    assert not result.accepted


def test_arena_opt_out_exists_only_in_logos_plugin_copy(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    (original / "go.mod").write_text("module example.com/input\n")
    source = Source(original, ("go.mod",), tree_hash(original, ("go.mod",)))
    for selected in load_profiles():
        for stage in ("oracle", "go", "logos"):
            work = tmp_path / selected.case.id / stage
            prepared = prepare_work(selected, source, stage, work)
            configured = selected.case.id == "xxhash-plugin-cgo" and stage == "logos"
            assert ("plugin_config_sha256" in prepared) is configured
            assert prepared["effective_source_digest"] == tree_hash(
                work, all_files(work)
            )
            if configured:
                assert (work / "logos.toml").read_bytes() == b"[build]\narena = false\n"
                assert (
                    prepared["plugin_config_sha256"]
                    == "2ae10a1fe10a8e72ac145fc13e7692747bd06c626ec91a403e9c19d4a4a30e21"
                )
                assert prepared["plugin_config_content"] == "[build]\narena = false\n"
            elif selected.case.id != "modernc-sqlite" or stage != "logos":
                assert not (work / "logos.toml").exists()
    assert all_files(original) == ("go.mod",)


def test_plugin_profile_does_not_overwrite_upstream_config(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    (original / "go.mod").write_text("module example.com/input\n")
    (original / "logos.toml").write_text("[build]\narena = true\n")
    files = all_files(original)
    source = Source(original, files, tree_hash(original, files))
    with pytest.raises(ValueError, match="overwrite"):
        prepare_work(profile("xxhash-plugin-cgo"), source, "logos", tmp_path / "work")


def test_plugin_profile_rejects_changed_config_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    (original / "go.mod").write_text("module example.com/input\n")
    source = Source(original, ("go.mod",), tree_hash(original, ("go.mod",)))
    changed = tmp_path / "changed.toml"
    changed.write_text("[build]\narena = true\n")
    monkeypatch.setattr(replay_backends, "PLUGIN_CONFIG", changed)
    with pytest.raises(ValueError, match="checksum"):
        prepare_work(profile("xxhash-plugin-cgo"), source, "logos", tmp_path / "work")
