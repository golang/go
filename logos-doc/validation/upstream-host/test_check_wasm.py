# pyright: strict
"""Reject missing, skipped, duplicated, or failed Wasm execution evidence."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import check_wasm as gate
import pytest


def event(action: str, name: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"Action": action, "Package": gate.PACKAGE}
    if name is not None:
        result["Test"] = name
    return result


def transcript(*, reverse: bool = False) -> list[dict[str, object]]:
    rows = [event("start")]
    for root in sorted(
        (name for name in gate.NAMES if "/" not in name), reverse=reverse
    ):
        rows.append(event("run", root))
        for child in sorted(
            (name for name in gate.NAMES if name.startswith(root + "/")),
            reverse=reverse,
        ):
            rows.extend((event("run", child), event("pass", child)))
        rows.append(event("pass", root))
    rows.append(event("pass"))
    return rows


def encoded(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


@pytest.mark.parametrize("reverse", (False, True))
def test_accepts_exact_inventory_in_either_map_order(reverse: bool) -> None:
    assert len(gate.NAMES) == 7
    assert len([name for name in gate.NAMES if "/" not in name]) == 3
    gate.accept(encoded(transcript(reverse=reverse)), b"", 0)


@pytest.mark.parametrize("name", sorted(gate.NAMES))
@pytest.mark.parametrize("action", ("run", "pass"))
def test_missing_or_duplicate_evidence_is_rejected(name: str, action: str) -> None:
    rows = transcript()
    selected = next(
        row for row in rows if row.get("Test") == name and row["Action"] == action
    )
    rows.remove(selected)
    with pytest.raises(ValueError):
        gate.accept(encoded(rows), b"", 0)
    with pytest.raises(ValueError):
        gate.accept(encoded(transcript() + [selected]), b"", 0)


@pytest.mark.parametrize("name", sorted(gate.NAMES))
@pytest.mark.parametrize("action", ("fail", "skip"))
def test_failure_and_skip_are_rejected(name: str, action: str) -> None:
    rows = transcript()
    next(row for row in rows if row.get("Test") == name and row["Action"] == "pass")[
        "Action"
    ] = action
    with pytest.raises(ValueError):
        gate.accept(encoded(rows), b"", 0)


@pytest.mark.parametrize(
    "case",
    (
        "empty",
        "wrong test",
        "wrong package",
        "missing start",
        "missing result",
        "duplicate result",
        "build failure",
        "failed build field",
        "early result",
    ),
)
def test_malformed_execution_is_rejected(case: str) -> None:
    rows = transcript()
    if case == "empty":
        rows.clear()
    elif case == "wrong test":
        rows[1]["Test"] = "TestWasmNotTheRequestedRoot"
    elif case == "wrong package":
        rows[0]["Package"] = "another/package"
    elif case == "missing start":
        rows.pop(0)
    elif case == "missing result":
        rows.pop()
    elif case == "duplicate result":
        rows.append(event("pass"))
    elif case == "build failure":
        rows.append({"Action": "build-fail", "ImportPath": gate.PACKAGE})
    elif case == "failed build field":
        rows[-1]["FailedBuild"] = gate.PACKAGE
    else:
        rows[1]["Action"] = "pass"
    with pytest.raises(ValueError):
        gate.accept(encoded(rows), b"", 0)


@pytest.mark.parametrize(
    "raw",
    (
        "[]\n",
        "null\n",
        "\n",
        "not JSON\n",
        '{"Action":NaN}\n',
        '{"Action":"pass","Action":"start"}\n',
    ),
)
def test_non_json_or_duplicate_keys_are_rejected(raw: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        gate.accept(raw, b"", 0)


@pytest.mark.parametrize("code", (-9, 1, 2, 124))
def test_nonzero_exit_is_not_a_pass(code: int) -> None:
    with pytest.raises(ValueError):
        gate.accept(encoded(transcript()), b"", code)


def test_stderr_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError):
        gate.accept(encoded(transcript()), b"unexpected host diagnostic", 0)


@pytest.mark.parametrize("level", ("package", "root", "child", "build"))
@pytest.mark.parametrize(
    "diagnostic",
    (
        "panic: unrelated crash\n",
        "fatal error: unexpected signal during runtime execution\n",
        "signal: killed\n",
        "SIGQUIT: quit\n",
        "panic: test timed out after 10m0s\n",
        "cmd/compile: signal: killed\n",
        "internal compiler error: unexpected state\n",
    ),
)
def test_crash_diagnostic_is_rejected(level: str, diagnostic: str) -> None:
    rows = transcript()
    if level == "build":
        rows.insert(
            0,
            {
                "Action": "build-output",
                "ImportPath": gate.PACKAGE,
                "Output": diagnostic,
            },
        )
    elif level == "package":
        rows.insert(1, {**event("output"), "Output": diagnostic})
    else:
        name = "TestWasmProgramRunsUnderNode"
        if level == "child":
            name += "/resume_points_everywhere"
        index = next(
            index
            for index, row in enumerate(rows)
            if row.get("Test") == name and row["Action"] == "run"
        )
        rows.insert(index + 1, {**event("output", name), "Output": diagnostic})
    with pytest.raises(ValueError, match="crash|timeout"):
        gate.accept(encoded(rows), b"", 0)


@pytest.mark.parametrize(
    "position",
    ("before start", "after completion", "build after completion", "test before run"),
)
def test_output_outside_its_lifetime_is_rejected(position: str) -> None:
    rows = transcript()
    output = {**event("output"), "Output": "ordinary output\n"}
    if position == "before start":
        rows.insert(0, output)
    elif position == "after completion":
        rows.append(output)
    elif position == "build after completion":
        rows.append(
            {
                "Action": "build-output",
                "ImportPath": gate.PACKAGE,
                "Output": "ordinary output\n",
            }
        )
    else:
        rows.insert(
            1,
            {
                **event("output", "TestWasmWrongMayYieldTraps"),
                "Output": "ordinary output\n",
            },
        )
    with pytest.raises(ValueError):
        gate.accept(encoded(rows), b"", 0)


def test_split_crash_text_is_rejected() -> None:
    rows = transcript()
    rows[1:1] = [
        {**event("output"), "Output": "pa"},
        {**event("output"), "Output": "nic: split diagnostic\n"},
    ]
    with pytest.raises(ValueError, match="crash|timeout"):
        gate.accept(encoded(rows), b"", 0)


@pytest.mark.parametrize(
    "changed",
    (
        "logos-doc/validation/upstream-host/frozen.json",
        "src/cmd/compile/internal/logos/wasm_test.go",
        "lib/wasm/wasm_exec.js",
        "lib/wasm/wasm_exec_node.js",
        "src/cmd/dist/test.go",
        "src/cmd/go/script_test.go",
        "src/run.bash",
    ),
)
def test_frozen_go_js_and_budget_inputs_are_verified(
    tmp_path: Path, changed: str
) -> None:
    repo = Path(__file__).resolve().parents[3]
    for relative in gate.verify_frozen(repo):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / relative, target)
    target = tmp_path / changed
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        gate.verify_frozen(tmp_path)


@pytest.mark.parametrize("mode", ("valid", "malformed", "timeout", "wrong exit"))
def test_capture_records_failures_without_starting_go(
    tmp_path: Path, mode: str
) -> None:
    (tmp_path / "src").mkdir()
    raw = "null\n" if mode == "malformed" else encoded(transcript())
    script = f"import sys, time; sys.stdout.write({raw!r}); sys.stdout.flush()"
    if mode == "timeout":
        script += "; time.sleep(30)"
    elif mode == "wrong exit":
        script += "; sys.exit(1)"
    result = gate.capture(
        (sys.executable, "-c", script),
        tmp_path,
        tmp_path,
        timeout=0.1 if mode == "timeout" else 5,
    )
    assert result["accepted"] is (mode == "valid")
    assert result["stdout_sha256"] == gate.digest(tmp_path / "wasm.jsonl")
    assert result["stderr_sha256"] == gate.digest(tmp_path / "wasm.stderr.txt")
    assert (tmp_path / "wasm.status.json").is_file()
