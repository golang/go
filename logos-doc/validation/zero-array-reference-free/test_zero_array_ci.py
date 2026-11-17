# pyright: strict
"""Check the zero-array gate boundaries without executing Go."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import zero_array_ci as gate


def transcript(check: gate.method_gate.Check) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"Action": "start", "Package": check.package}]
    result = "fail" if check.negative is not None else "pass"

    def emit(name: str) -> None:
        rows.append({"Action": "run", "Package": check.package, "Test": name})
        for child in sorted(
            candidate
            for candidate in check.names
            if candidate.rpartition("/")[0] == name
        ):
            emit(child)
        if check.negative is not None and name == check.negative[0]:
            rows.append(
                {
                    "Action": "output",
                    "Package": check.package,
                    "Test": name,
                    "Output": f"    frozen_test.go:1: {check.negative[1]}\n",
                }
            )
        rows.append({"Action": result, "Package": check.package, "Test": name})

    for root in sorted(name for name in check.names if "/" not in name):
        emit(root)
    rows.append({"Action": result, "Package": check.package})
    return rows


def accept(
    rows: list[dict[str, object]], check: gate.method_gate.Check, code: int
) -> None:
    text = "".join(json.dumps(row) + "\n" for row in rows)
    gate.method_gate.accept(gate.method_gate.parse_events(text, check), check, code)


def test_frozen_inventory() -> None:
    repo = Path(__file__).resolve().parents[3]
    assert len(gate.frozen(repo)) == 22
    assert len(gate.type_names()) == 67
    assert len(gate.fact_names()) == 15
    assert (
        sum(len(check.names) for check in gate.CHECKS if check.negative is None) == 82
    )


@pytest.mark.parametrize(
    "check", gate.CHECKS, ids=[check.label for check in gate.CHECKS]
)
def test_exact_event_transcript(check: gate.method_gate.Check) -> None:
    accept(transcript(check), check, 1 if check.negative is not None else 0)


@pytest.mark.parametrize(
    "check", gate.CHECKS[2:], ids=[check.label for check in gate.CHECKS[2:]]
)
def test_missing_leaf_is_rejected(check: gate.method_gate.Check) -> None:
    leaf = max(check.names, key=lambda name: (name.count("/"), name))
    rows = [row for row in transcript(check) if row.get("Test") != leaf]
    with pytest.raises(ValueError, match="test runs"):
        accept(rows, check, 0)


@pytest.mark.parametrize(
    "check", gate.CHECKS[:2], ids=[check.label for check in gate.CHECKS[:2]]
)
@pytest.mark.parametrize("code", (0, 2, 124, -9))
def test_negative_exit_must_be_one(check: gate.method_gate.Check, code: int) -> None:
    with pytest.raises(ValueError, match="exit/package"):
        accept(transcript(check), check, code)


@pytest.mark.parametrize(
    "check", gate.CHECKS[:2], ids=[check.label for check in gate.CHECKS[:2]]
)
def test_negative_diagnostic_must_belong_to_leaf(check: gate.method_gate.Check) -> None:
    rows = transcript(check)
    root = next(name for name in check.names if "/" not in name)
    for row in rows:
        if row["Action"] == "output":
            row["Test"] = root
    with pytest.raises(ValueError, match="exact diagnostic"):
        accept(rows, check, 1)


def test_candidate_wrong_package_is_rejected() -> None:
    check = gate.CHECKS[2]
    rows = transcript(check)
    rows[0]["Package"] = gate.LOGOS
    with pytest.raises(ValueError, match="unexpected package"):
        accept(rows, check, 0)


def test_candidate_skip_is_rejected() -> None:
    check = gate.CHECKS[3]
    rows = transcript(check)
    for row in rows:
        if row.get("Test") == f"{gate.FACTS}/ReturnZero" and row["Action"] == "pass":
            row["Action"] = "skip"
    with pytest.raises(ValueError):
        accept(rows, check, 0)


def test_negative_crash_is_rejected() -> None:
    check = gate.CHECKS[0]
    rows = transcript(check)
    rows.insert(
        -1, {"Action": "output", "Package": check.package, "Output": "panic: crash\n"}
    )
    with pytest.raises(ValueError, match="crash"):
        accept(rows, check, 1)


def test_manifest_change_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / gate.DIRECTORY
    directory.mkdir(parents=True)
    (directory / "FROZEN.sha256").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        gate.frozen(tmp_path)


@pytest.mark.parametrize(
    "relative",
    (
        gate.ORIGINAL_FIXTURE,
        "src/cmd/compile/internal/logos/constfacts_zero_array_test.go",
        "src/cmd/compile/internal/logos/compile_method_tagged_facts_test.go",
        "src/simd/testdata/iface/iface.go",
        f"{gate.DIRECTORY}/old-const_logos.go.txt",
    ),
)
def test_frozen_input_change_is_rejected(tmp_path: Path, relative: str) -> None:
    repo = Path(__file__).resolve().parents[3]
    for path in gate.frozen(repo):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / path).read_bytes())
    destination = tmp_path / relative
    destination.write_bytes(destination.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="frozen zero-array input changed"):
        gate.frozen(tmp_path)


def test_replacement_manifest_preserves_original_inputs() -> None:
    repo = Path(__file__).resolve().parents[3]
    directory = repo / gate.DIRECTORY
    original = directory / "FROZEN.sha256"
    assert (
        original.read_bytes()
        == (directory / "api-compatibility/original-FROZEN.sha256.txt").read_bytes()
    )
    legacy = {
        relative: expected
        for expected, relative in (
            line.split("  ", 1) for line in original.read_text().splitlines()
        )
    }
    replacement = {
        relative: expected
        for expected, relative in (
            line.split("  ", 1)
            for line in (directory / "REPLACEMENT.sha256").read_text().splitlines()
        )
    }
    assert len(legacy) == 21 and len(replacement) == 54
    assert {
        path: replacement[path] for path in legacy if path != gate.TYPE_FIXTURE
    } == {path: value for path, value in legacy.items() if path != gate.TYPE_FIXTURE}
    assert replacement[gate.ORIGINAL_FIXTURE] == legacy[gate.TYPE_FIXTURE]
    assert replacement[gate.TYPE_FIXTURE] == gate.method_gate.digest(
        repo / gate.TYPE_FIXTURE
    )
    assert replacement[f"{gate.DIRECTORY}/FROZEN.sha256"] == gate.MANIFEST_SHA256
    assert gate.ORIGINAL_FIXTURE in gate.frozen(repo)
    assert gate.TYPE_FIXTURE not in gate.frozen(repo)
    assert len(gate.replacement(repo)) == 55
    assert gate.suite(repo).inputs == gate.replacement(repo)


def test_exact_fixture_api_correction() -> None:
    repo = Path(__file__).resolve().parents[3]
    original = (repo / gate.ORIGINAL_FIXTURE).read_bytes()
    active = (repo / gate.TYPE_FIXTURE).read_bytes()
    gate.verify_fixture_api(original, active)
    assert active == original.replace(
        b"NewInterfaceType(nil, nil).Complete()", b"NewInterfaceType(nil, nil)"
    ).replace(
        b"NewInterfaceType(nil, embeds).Complete()", b"NewInterfaceType(nil, embeds)"
    )


@pytest.mark.parametrize(
    "change", ("assertion", "name", "whitespace", "only-first", "unchanged")
)
def test_fixture_api_rejects_every_other_change(change: str) -> None:
    repo = Path(__file__).resolve().parents[3]
    original = (repo / gate.ORIGINAL_FIXTURE).read_bytes()
    active = (repo / gate.TYPE_FIXTURE).read_bytes()
    if change == "assertion":
        active = active.replace(b'{"zero", 0, true}', b'{"zero", 0, false}')
    elif change == "name":
        active = active.replace(b'"function"', b'"renamed-function"')
    elif change == "whitespace":
        active += b"\n"
    elif change == "only-first":
        active = original.replace(
            b"NewInterfaceType(nil, nil).Complete()", b"NewInterfaceType(nil, nil)"
        )
    else:
        active = original
    with pytest.raises(ValueError, match="exact two API suffix removals"):
        gate.verify_fixture_api(original, active)


@pytest.mark.parametrize(
    "call",
    (
        b"NewInterfaceType(nil, nil).Complete()",
        b"NewInterfaceType(nil, embeds).Complete()",
    ),
)
@pytest.mark.parametrize("change", ("missing", "duplicate"))
def test_original_api_calls_are_required_once(call: bytes, change: str) -> None:
    repo = Path(__file__).resolve().parents[3]
    original = (repo / gate.ORIGINAL_FIXTURE).read_bytes()
    original = (
        original.replace(call, call.removesuffix(b".Complete()"))
        if change == "missing"
        else original + b"\n// " + call
    )
    with pytest.raises(ValueError, match="exactly one known API call"):
        gate.verify_fixture_api(original, (repo / gate.TYPE_FIXTURE).read_bytes())


@pytest.fixture
def replacement_copy(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[3]
    for relative in gate.replacement(repo):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / relative).read_bytes())
    return tmp_path


@pytest.mark.parametrize(
    "relative",
    (
        gate.TYPE_FIXTURE,
        gate.ORIGINAL_FIXTURE,
        f"{gate.DIRECTORY}/FROZEN.sha256",
        f"{gate.DIRECTORY}/api-compatibility/original-FROZEN.sha256.txt",
        f"{gate.DIRECTORY}/api-compatibility/original-zero_array_ci.py.txt",
        f"{gate.DIRECTORY}/api-compatibility/hosted/zero-array/old-type-proof.jsonl",
        f"{gate.DIRECTORY}/api-compatibility/hosted/zero-array/types.jsonl",
        f"{gate.DIRECTORY}/api-compatibility/hosted/zero-array/report.json",
        f"{gate.DIRECTORY}/api-compatibility/hosted/unit-artifact.zip",
        f"{gate.DIRECTORY}/api-compatibility/provenance.json",
    ),
)
def test_replacement_input_change_is_rejected(
    replacement_copy: Path, relative: str
) -> None:
    target = replacement_copy / relative
    target.write_bytes(target.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="replacement input changed"):
        gate.replacement(replacement_copy)


def test_replacement_manifest_change_is_rejected(replacement_copy: Path) -> None:
    manifest = replacement_copy / gate.DIRECTORY / "REPLACEMENT.sha256"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="replacement manifest changed"):
        gate.replacement(replacement_copy)


def test_uncorrected_fixture_cannot_start_suite(replacement_copy: Path) -> None:
    original = (replacement_copy / gate.ORIGINAL_FIXTURE).read_bytes()
    (replacement_copy / gate.TYPE_FIXTURE).write_bytes(original)
    with pytest.raises(ValueError, match="replacement input changed"):
        gate.suite(replacement_copy)


@pytest.mark.parametrize(
    "check", gate.CHECKS, ids=[check.label for check in gate.CHECKS]
)
def test_retained_hosted_outcomes(check: gate.method_gate.Check) -> None:
    root = (
        Path(__file__).resolve().parents[3]
        / gate.DIRECTORY
        / "api-compatibility/hosted/zero-array"
    )
    events = gate.method_gate.parse_events(
        (root / f"{check.label}.jsonl").read_text(), check
    )
    code = 0 if check.label == "facts" else 1
    if check.label in ("old-type-proof", "types"):
        assert not events.runs
        with pytest.raises(ValueError, match="build failure or skipped test"):
            gate.method_gate.accept(events, check, code)
    else:
        gate.method_gate.accept(events, check, code)
