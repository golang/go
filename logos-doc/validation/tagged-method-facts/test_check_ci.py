# pyright: strict
"""Fail-closed controls for tagged-method CI evidence; no Go process is started."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import cast

import check_ci as gate
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def event(check: gate.Check, action: str, name: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"Action": action, "Package": check.package}
    if name is not None:
        result["Test"] = name
    return result


def transcript(check: gate.Check) -> list[dict[str, object]]:
    rows = [event(check, "start")]
    outcome = "pass" if check.negative is None else "fail"

    def visit(name: str) -> None:
        rows.append(event(check, "run", name))
        for child in sorted(check.names):
            if child.rpartition("/")[0] == name:
                visit(child)
        if check.negative is not None and name == check.negative[0]:
            rows.append(
                {
                    **event(check, "output", name),
                    "Output": f"    fixture.go:37: {check.negative[1]}\n",
                }
            )
        rows.append(event(check, outcome, name))

    for root in sorted(name for name in check.names if "/" not in name):
        visit(root)
    rows.append(event(check, outcome))
    return rows


def encoded(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def accept(rows: list[dict[str, object]], check: gate.Check, exit_code: int) -> None:
    gate.accept(gate.parse_events(encoded(rows), check), check, exit_code)


@pytest.mark.parametrize("check", gate.CHECKS, ids=lambda check: check.label)
def test_accepts_exact_positive_and_negative_transcripts(check: gate.Check) -> None:
    accept(transcript(check), check, int(check.negative is not None))


def test_inventory_matches_actual_nested_runs() -> None:
    facts, logos = gate.CHECKS[2:]
    assert len(facts.names) == 72
    assert len(logos.names) == 17
    assert (
        sum(
            not any(other.startswith(name + "/") for other in check.names)
            for check in (facts, logos)
            for name in check.names
        )
        == 78
    )
    assert len(gate.CHECKS[0].names) == 2
    assert len(gate.CHECKS[1].names) == 1
    assert "TestTaggedMethodObjectConstructors/Ω@simd128" in facts.names
    assert (
        "TestTaggedMethodObjectConstructors/Ω@simd128/const_alias_pointer"
        in facts.names
    )
    assert "TestCompileSIMDTaggedMethodFacts/fork-go" in logos.names
    assert "TestCompileSIMDTaggedMethodFacts/logos" in logos.names
    assert not any("VectorC" in name for name in logos.names)


@settings(max_examples=100, derandomize=True)
@given(
    st.sampled_from(sorted(gate.CHECKS[2].names | gate.CHECKS[3].names)),
    st.sampled_from(("run", "pass")),
)
def test_missing_or_duplicate_event_fails(name: str, action: str) -> None:
    check = next(item for item in gate.CHECKS[2:] if name in item.names)
    rows = transcript(check)
    selected = next(
        row for row in rows if row.get("Test") == name and row["Action"] == action
    )
    rows.remove(selected)
    with pytest.raises(ValueError):
        accept(rows, check, 0)
    with pytest.raises(ValueError):
        accept(transcript(check) + [selected], check, 0)


@pytest.mark.parametrize("check", gate.CHECKS, ids=lambda check: check.label)
@pytest.mark.parametrize("stderr", (b"warning\n", b" ", b"\xff"))
def test_stderr_fails(check: gate.Check, stderr: bytes) -> None:
    with pytest.raises(ValueError, match="nonempty stderr"):
        gate.accept(
            gate.parse_events(encoded(transcript(check)), check),
            check,
            int(check.negative is not None),
            stderr,
        )


@pytest.mark.parametrize(
    ("check", "code"),
    tuple(
        (check, code)
        for check in gate.CHECKS
        for code in (-9, 0, 1, 2, 124)
        if code != int(check.negative is not None)
    ),
)
def test_wrong_exit_fails(check: gate.Check, code: int) -> None:
    with pytest.raises(ValueError, match="exit/package outcome"):
        accept(transcript(check), check, code)


@pytest.mark.parametrize("check", gate.CHECKS[:2], ids=lambda check: check.label)
@pytest.mark.parametrize(
    "mutation", ("missing", "wrong owner", "substring", "panic", "skip", "wrong result")
)
def test_negative_requires_exact_named_failure(
    check: gate.Check, mutation: str
) -> None:
    rows = transcript(check)
    output = next(row for row in rows if row["Action"] == "output")
    if mutation == "missing":
        rows.remove(output)
    elif mutation == "wrong owner":
        output.pop("Test")
    elif mutation == "substring":
        output["Output"] = str(output["Output"]).rstrip() + " but unrelated\n"
    elif mutation == "panic":
        output["Output"] = str(output["Output"]) + "panic: unrelated crash\n"
    else:
        failure = next(row for row in rows if row["Action"] == "fail" and "Test" in row)
        failure["Action"] = "skip" if mutation == "skip" else "pass"
    with pytest.raises(ValueError):
        accept(rows, check, 1)


@pytest.mark.parametrize(
    "kind",
    (
        "build-fail",
        "failed-build field",
        "skip",
        "fail",
        "no start",
        "no result",
        "duplicate start",
    ),
)
def test_package_or_build_failure_fails(kind: str) -> None:
    check = gate.CHECKS[2]
    rows = transcript(check)
    if kind == "build-fail":
        rows.insert(0, {"Action": "build-fail", "ImportPath": ""})
    elif kind == "failed-build field":
        rows[-1]["FailedBuild"] = check.package
    elif kind in ("skip", "fail"):
        rows[-1]["Action"] = kind
    elif kind == "no start":
        rows.pop(0)
    elif kind == "no result":
        rows.pop()
    else:
        rows.insert(0, event(check, "start"))
    with pytest.raises(ValueError):
        accept(rows, check, 0)


@pytest.mark.parametrize(
    "mutation",
    ("wrong name", "wrong package", "early result", "foreign output", "unknown action"),
)
def test_event_identity_and_order_are_checked(mutation: str) -> None:
    check = gate.CHECKS[3]
    rows = transcript(check)
    selected = next(
        row
        for row in rows
        if row.get("Test") == "TestCompileSIMDTaggedMethodFacts/logos"
    )
    if mutation == "wrong name":
        selected["Test"] = "TestCompileSIMDTaggedMethodFacts/not-logos"
    elif mutation == "wrong package":
        selected["Package"] = "another/package"
    elif mutation == "early result":
        selected["Action"] = "pass"
    elif mutation == "foreign output":
        rows.append({**event(check, "output", "unexecuted/test"), "Output": "pass\n"})
    else:
        selected["Action"] = "unknown"
    with pytest.raises(ValueError):
        accept(rows, check, 0)


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "\n",
        "null\n",
        "[]\n",
        "not JSON\n",
        '{"Action":NaN}\n',
        '{"Action":"run","Action":"pass"}\n',
        '{"Action":"build-output","ImportPath":null,"Output":"x"}\n',
    ),
)
def test_empty_or_malformed_stream_fails(raw: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        gate.accept(gate.parse_events(raw, gate.CHECKS[2]), gate.CHECKS[2], 0)


def test_benign_build_output_is_retained() -> None:
    check = gate.CHECKS[2]
    rows: list[dict[str, object]] = [
        {"Action": "build-output", "ImportPath": "", "Output": "compiler note\n"},
        *transcript(check),
    ]
    accept(rows, check, 0)


def frozen_paths() -> tuple[str, ...]:
    repo = Path(__file__).resolve().parents[3]
    manifest = gate.json_object(
        (repo / gate.DIRECTORY / "frozen.json").read_text(encoding="utf-8")
    )
    paths = [f"{gate.DIRECTORY}/frozen.json", f"{gate.DIRECTORY}/README.md"]
    paths.append(gate.field(gate.mapping(manifest["real_source"]), "path"))
    for group in ("files", "existing_package_controls", "negative_controls"):
        entries = manifest[group]
        assert isinstance(entries, list)
        paths.extend(
            gate.field(gate.mapping(item), "path")
            for item in cast(list[object], entries)
        )
    return tuple(paths)


@pytest.mark.parametrize("changed", frozen_paths())
def test_every_frozen_input_is_checked(tmp_path: Path, changed: str) -> None:
    repo = Path(__file__).resolve().parents[3]
    for relative in frozen_paths():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
    assert len(gate.verify_frozen(tmp_path)) == 11
    target = tmp_path / changed
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen source digest mismatch"):
        gate.verify_frozen(tmp_path)


def test_paths_cannot_escape_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("payload", encoding="utf-8")
    (repo / "link").symlink_to(outside)
    for relative in ("../outside", "link", str(outside)):
        with pytest.raises(ValueError, match="inside the repository"):
            gate.rooted(repo, relative)


@pytest.mark.parametrize(
    "mode",
    ("valid", "malformed", "timeout", "wrong exit", "stderr", "missing executable"),
)
def test_capture_keeps_raw_logs_and_rejects_process_faults(
    tmp_path: Path, mode: str
) -> None:
    repo, out = tmp_path / "repo", tmp_path / "evidence"
    (repo / "src").mkdir(parents=True)
    out.mkdir()
    check = gate.CHECKS[3]
    payload = "null\n" if mode == "malformed" else encoded(transcript(check))
    script = f"import sys, time; sys.stdout.write({payload!r}); sys.stdout.flush()"
    if mode == "timeout":
        script += "; time.sleep(30)"
    elif mode == "wrong exit":
        script += "; sys.exit(2)"
    elif mode == "stderr":
        script += "; sys.stderr.write('unexpected warning\\n')"
    command = (
        (str(tmp_path / "missing"),)
        if mode == "missing executable"
        else (sys.executable, "-c", script)
    )
    result = gate.capture(
        command, repo, out, check, dict(os.environ), 0.1 if mode == "timeout" else 5
    )
    assert result["accepted"] is (mode == "valid")
    assert result["stdout_sha256"] == gate.digest(out / "logos.jsonl")
    assert result["stderr_sha256"] == gate.digest(out / "logos.stderr.txt")
    retained = gate.json_object((out / "logos.status.json").read_text(encoding="utf-8"))
    assert retained["accepted"] is (mode == "valid")
    if mode == "timeout":
        assert result["timed_out"] is True
    elif mode != "missing executable":
        assert (out / "logos.jsonl").read_text(encoding="utf-8") == payload


@pytest.mark.parametrize(
    "failure",
    ("none", "preflight", "negative", "positive", "source changed", "tool changed"),
)
def test_gate_requires_both_negatives_all_positives_and_stable_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    captured: list[str] = []
    identity_calls = 0

    def verify(_repo: Path) -> dict[str, str]:
        if failure == "preflight":
            raise ValueError("frozen source digest mismatch")
        return {"frozen-test.go": "same"}

    def identity(_repo: Path, _paths: tuple[str, ...]) -> dict[str, object]:
        nonlocal identity_calls
        identity_calls += 1
        return {
            "tracked_status": "",
            "source": "changed"
            if failure == "source changed" and identity_calls > 1
            else "same",
            "tool": "changed"
            if failure == "tool changed" and identity_calls > 1
            else "same",
        }

    def capture(
        _command: tuple[str, ...],
        _repo: Path,
        _out: Path,
        check: gate.Check,
        _env: dict[str, str],
        _timeout: float,
    ) -> dict[str, object]:
        captured.append(check.label)
        accepted = not (
            (failure == "negative" and check.label == "old-package-guard")
            or (failure == "positive" and check.label == "logos")
        )
        return {"accepted": accepted}

    def rooted(repo: Path, relative: str) -> Path:
        return repo / relative

    monkeypatch.setattr(gate, "verify_frozen", verify)
    monkeypatch.setattr(gate, "identity", identity)
    monkeypatch.setattr(gate, "capture", capture)
    monkeypatch.setattr(gate, "rooted", rooted)
    assert gate.run_gate(tmp_path, tmp_path) is (failure == "none")
    report = gate.json_object((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["accepted"] is (failure == "none")
    assert captured == (
        []
        if failure == "preflight"
        else [check.label for check in gate.CHECKS[:2]]
        if failure == "negative"
        else [check.label for check in gate.CHECKS]
    )
    assert (tmp_path / "after.json").is_file()
