# pyright: strict
"""Package and build records must not conceal failures behind passing tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from replay_backends import assess_runtime
from test_replay_backends import executed, profile, run_record


@pytest.mark.parametrize(
    "event",
    (
        {"Action": "unexpected", "Package": "p"},
        {"Action": "run", "Package": "p"},
        {"Action": "start", "Package": ""},
        {"Action": "pass"},
        {"Action": "skip"},
        {"Action": "build-output", "Package": "p"},
    ),
)
def test_invalid_package_event_cannot_hide_after_passing_tests(
    tmp_path: Path, event: dict[str, str]
) -> None:
    selected = profile("xxhash-asm-core")
    text = executed(selected.case.module, ("TestAll",)) + "\n" + json.dumps(event)
    result = assess_runtime(run_record(tmp_path, text), "oracle", selected, None)
    assert result.assessment.status == "invalid-report"
    assert "line 3" in result.assessment.reason
    assert not result.accepted


@pytest.mark.parametrize("package", ("example.com/broken", ""))
def test_build_failure_cannot_hide_after_passing_tests(
    tmp_path: Path, package: str
) -> None:
    selected = profile("xxhash-asm-core")
    text = executed(selected.case.module, ("TestAll",))
    text += "\n" + json.dumps({"Action": "build-fail", "ImportPath": package})
    result = assess_runtime(run_record(tmp_path, text), "oracle", selected, None)
    assert result.assessment.status == "upstream-failed"
    assert not result.accepted


@pytest.mark.parametrize("action", ("start", "output", "pass", "skip", "build-output"))
def test_valid_package_and_build_records_keep_runtime_pass(
    tmp_path: Path, action: str
) -> None:
    selected = profile("xxhash-asm-core")
    event = {"Action": action, "Output": "ordinary output\n"}
    event["ImportPath" if action == "build-output" else "Package"] = (
        selected.case.module
    )
    text = executed(selected.case.module, ("TestAll",)) + "\n" + json.dumps(event)
    assert assess_runtime(run_record(tmp_path, text), "oracle", selected, None).accepted


def test_global_build_output_keeps_runtime_pass(tmp_path: Path) -> None:
    selected = profile("xxhash-asm-core")
    event = {
        "Action": "build-output",
        "ImportPath": "",
        "Output": "ordinary global output\n",
    }
    text = executed(selected.case.module, ("TestAll",)) + "\n" + json.dumps(event)
    assert assess_runtime(run_record(tmp_path, text), "oracle", selected, None).accepted
