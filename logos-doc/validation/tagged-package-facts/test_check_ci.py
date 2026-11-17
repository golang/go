# pyright: strict
"""Adversarial controls for the exact tagged-fact CI acceptance oracle."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import check_ci as gate
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def event(action: str, name: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {"Action": action, "Package": gate.PACKAGE}
    if name is not None:
        value["Test"] = name
    return value


def transcript(*, negative: bool = False) -> list[dict[str, object]]:
    names = gate.NEGATIVE_NAMES if negative else gate.CANDIDATE_NAMES
    outcome = "fail" if negative else "pass"
    rows = [event("start")]
    for root in sorted(name for name in names if "/" not in name):
        rows.append(event("run", root))
        for child in sorted(name for name in names if name.startswith(root + "/")):
            rows.append(event("run", child))
            if negative:
                rows.append(
                    {
                        **event("output", child),
                        "Output": gate.NEGATIVE_DIAGNOSTIC
                        + ' TP="0" RTP="0", want true/"0"/"0"\n',
                    }
                )
            rows.append(event(outcome, child))
        rows.append(event(outcome, root))
    rows.append(event(outcome))
    return rows


def parse(rows: list[dict[str, object]]) -> gate.Events:
    return gate.parse_events("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("negative", (False, True))
def test_accepts_exact_success_and_named_negative(negative: bool) -> None:
    gate.accept(parse(transcript(negative=negative)), int(negative), negative=negative)


@pytest.mark.parametrize("negative", (False, True))
@pytest.mark.parametrize("stderr", (b"warning\n", b" ", b"\xff"))
def test_nonempty_stderr_is_rejected(negative: bool, stderr: bytes) -> None:
    with pytest.raises(ValueError, match="nonempty stderr"):
        gate.accept(
            parse(transcript(negative=negative)),
            int(negative),
            negative=negative,
            stderr=stderr,
        )


def test_slash_names_have_no_implicit_parent_events() -> None:
    assert len(gate.CANDIDATE_NAMES) == 150
    assert len(gate.NEGATIVE_NAMES) == 2
    assert gate.NEGATIVE_TEST in gate.CANDIDATE_NAMES
    assert (
        "TestFactScopeTaggedFunctionAssertions/Project@simd128"
        not in gate.CANDIDATE_NAMES
    )
    assert "TestFactScopeLeavesCompilerSymbolsLazy/local·1" not in gate.CANDIDATE_NAMES
    assert "TestCompileImportedLocalType/generic_method" not in gate.CANDIDATE_NAMES
    invented = "TestFactScopeTaggedFunctionAssertions/Project@simd128"
    with pytest.raises(ValueError, match="extra="):
        gate.accept(
            parse(transcript() + [event("run", invented), event("pass", invented)]),
            0,
            negative=False,
        )


@settings(max_examples=50, derandomize=True)
@given(st.sampled_from(sorted(gate.CANDIDATE_NAMES)), st.sampled_from(("run", "pass")))
def test_missing_or_duplicate_named_event_is_rejected(name: str, action: str) -> None:
    rows = transcript()
    selected = next(
        row for row in rows if row.get("Test") == name and row["Action"] == action
    )
    rows.remove(selected)
    with pytest.raises(ValueError, match="missing="):
        gate.accept(parse(rows), 0, negative=False)
    with pytest.raises(ValueError, match="extra="):
        gate.accept(parse(transcript() + [selected]), 0, negative=False)


def test_equal_count_wrong_name_is_rejected() -> None:
    rows = transcript()
    for row in rows:
        if row.get("Test") == "TestCompileSIMDTaggedFuncFacts/Float32sFromArch@simd128":
            row["Test"] = "TestCompileSIMDTaggedFuncFacts/Float32sFromArch@simd256"
    with pytest.raises(ValueError, match="test runs"):
        gate.accept(parse(rows), 0, negative=False)


@pytest.mark.parametrize(
    "extra",
    (
        event("skip", gate.NEGATIVE_TEST),
        event("fail", gate.NEGATIVE_TEST),
        event("skip"),
        event("fail"),
        {"Action": "build-fail", "ImportPath": gate.PACKAGE},
        {"Action": "build-fail", "ImportPath": ""},
    ),
)
def test_failure_or_skip_never_passes(extra: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        gate.accept(parse(transcript() + [extra]), 0, negative=False)


@pytest.mark.parametrize("exit_code", (1, 2, -9, 124))
def test_candidate_exit_must_be_zero(exit_code: int) -> None:
    with pytest.raises(ValueError, match="exit/package"):
        gate.accept(parse(transcript()), exit_code, negative=False)


@pytest.mark.parametrize("exit_code", (0, 2, -9, 124))
def test_negative_exit_must_be_one(exit_code: int) -> None:
    with pytest.raises(ValueError, match="exit/package"):
        gate.accept(parse(transcript(negative=True)), exit_code, negative=True)


def test_negative_requires_its_exact_named_diagnostic() -> None:
    rows = transcript(negative=True)
    for row in rows:
        if row["Action"] == "output":
            row["Output"] = "unrelated failure\n"
    with pytest.raises(ValueError, match="missing-metadata diagnostic"):
        gate.accept(parse(rows), 1, negative=True)
    rows.append(
        {**event("output", gate.NEGATIVE_ROOT), "Output": gate.NEGATIVE_DIAGNOSTIC}
    )
    with pytest.raises(ValueError, match="missing-metadata diagnostic"):
        gate.accept(parse(rows), 1, negative=True)


def test_negative_rejects_build_failure_and_wrong_named_failure() -> None:
    with pytest.raises(ValueError):
        gate.accept(
            parse([{"Action": "build-fail", "ImportPath": gate.PACKAGE}]),
            1,
            negative=True,
        )
    rows = transcript(negative=True)
    for row in rows:
        if row.get("Test") == gate.NEGATIVE_TEST:
            row["Test"] = f"{gate.NEGATIVE_ROOT}/Project@simd128/mask=1"
    with pytest.raises(ValueError, match="test runs"):
        gate.accept(parse(rows), 1, negative=True)


@pytest.mark.parametrize("negative", (False, True))
def test_empty_stream_does_not_satisfy_a_gate(negative: bool) -> None:
    with pytest.raises(ValueError, match="test runs"):
        gate.accept(gate.parse_events(""), int(negative), negative=negative)


@pytest.mark.parametrize(
    "invalid",
    (
        {"Action": "run", "Package": gate.PACKAGE},
        {"Action": "unexpected", "Package": gate.PACKAGE},
        {"Action": "start", "Package": gate.PACKAGE, "Test": gate.NEGATIVE_TEST},
        {"Action": "run", "Package": "another/package", "Test": gate.NEGATIVE_TEST},
        {"Action": "run", "Package": gate.PACKAGE, "Test": ""},
        {"Action": "output", "Package": gate.PACKAGE, "Output": 17},
        {"Action": "build-output", "ImportPath": None, "Output": "diagnostic"},
        {
            "Action": "build-output",
            "ImportPath": "",
            "Output": "diagnostic",
            "Test": gate.NEGATIVE_TEST,
        },
    ),
)
def test_invalid_event_shape_is_rejected(invalid: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse(transcript() + [invalid])


@pytest.mark.parametrize(
    ("raw", "error_type"),
    (
        ("[]\n", TypeError),
        ("null\n", TypeError),
        ("\n", ValueError),
        ("not json\n", ValueError),
        ('{"Action":"start","Action":"pass"}\n', ValueError),
        ('{"Action":NaN}\n', ValueError),
    ),
)
def test_malformed_json_is_rejected(raw: str, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        gate.parse_events(raw)


def test_benign_build_and_package_output_are_allowed() -> None:
    rows: list[dict[str, object]] = [
        {"Action": "build-output", "ImportPath": "", "Output": "global build output\n"},
        {
            "Action": "build-output",
            "ImportPath": gate.PACKAGE,
            "Output": "package build output\n",
        },
        {**event("output"), "Output": "package output\n"},
        *transcript(),
    ]
    gate.accept(parse(rows), 0, negative=False)


@pytest.mark.parametrize("changed", tuple(path for path, _ in gate.FROZEN_PINS))
def test_frozen_source_digest_mismatch_is_rejected(
    tmp_path: Path, changed: str
) -> None:
    repository = Path(__file__).resolve().parents[3]
    for relative, _ in gate.FROZEN_PINS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    assert len(gate.verify_frozen(tmp_path)) == len(gate.FROZEN_PINS)
    target = tmp_path / changed
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen source digest mismatch"):
        gate.verify_frozen(tmp_path)


@pytest.mark.parametrize("raw", ("[]", "null"))
def test_capture_retains_status_for_malformed_json(tmp_path: Path, raw: str) -> None:
    root, out = tmp_path / "source", tmp_path / "evidence"
    (root / "src").mkdir(parents=True)
    out.mkdir()
    result = gate.capture(
        (sys.executable, "-c", f"print({raw!r})"), root, out, "candidate", 5
    )
    assert result["accepted"] is False
    assert result["rejection"] == "JSON event is not an object"
    assert (out / "candidate.jsonl").read_text(encoding="utf-8") == raw + "\n"
    assert (out / "candidate.stderr.txt").read_bytes() == b""
    retained = gate.json_object(
        (out / "candidate.status.json").read_text(encoding="utf-8")
    )
    assert retained["accepted"] is False
    assert retained["exit_code"] == 0
    assert retained["stdout_sha256"] == gate.digest(out / "candidate.jsonl")
