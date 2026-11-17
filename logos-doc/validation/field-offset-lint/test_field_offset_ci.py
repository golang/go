# pyright: strict
"""Check the frozen field inventory and slash-named event contract without Go."""

from __future__ import annotations

import json
from pathlib import Path

import field_offset_ci as gate
import pytest


def test_frozen_inventory_and_slash_named_events() -> None:
    repo = Path(__file__).resolve().parents[3]
    _, names, _, _ = gate.frozen(repo)
    check = gate.method_gate.Check("candidate", gate.PACKAGE, names)
    rows: list[dict[str, object]] = [{"Action": "start", "Package": gate.PACKAGE}]
    for root in sorted(name for name in names if "/" not in name):
        rows.append({"Action": "run", "Package": gate.PACKAGE, "Test": root})
        for child in sorted(name for name in names if name.startswith(root + "/")):
            for action in ("run", "pass"):
                rows.append({"Action": action, "Package": gate.PACKAGE, "Test": child})
        rows.append({"Action": "pass", "Package": gate.PACKAGE, "Test": root})
    rows.append({"Action": "pass", "Package": gate.PACKAGE})
    encoded = "".join(json.dumps(row) + "\n" for row in rows)
    gate.method_gate.accept(gate.method_gate.parse_events(encoded, check), check, 0)
    missing = "TestCompileArenaFieldOffset/offset_unsafe.lgo/address"
    partial = "".join(
        json.dumps(row) + "\n" for row in rows if row.get("Test") != missing
    )
    with pytest.raises(ValueError, match="test runs"):
        gate.method_gate.accept(gate.method_gate.parse_events(partial, check), check, 0)


def test_manifest_change_rejected_before_execution(tmp_path: Path) -> None:
    directory = tmp_path / gate.DIRECTORY
    directory.mkdir(parents=True)
    (directory / "frozen.json").write_text('{"required_names": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        gate.frozen(tmp_path)


def test_exact_negative_hierarchy_and_diagnostic() -> None:
    repo = Path(__file__).resolve().parents[3]
    _, _, name, diagnostic = gate.frozen(repo)
    root = "TestArenaFieldOffset"
    check = gate.method_gate.Check(
        "old-rule", gate.PACKAGE, frozenset((root, name)), (name, diagnostic)
    )
    rows: list[dict[str, object]] = [
        {"Action": "start", "Package": gate.PACKAGE},
        {"Action": "run", "Package": gate.PACKAGE, "Test": root},
        {"Action": "run", "Package": gate.PACKAGE, "Test": name},
        {
            "Action": "output",
            "Package": gate.PACKAGE,
            "Test": name,
            "Output": f"    rule_field_offset_test.go:173: {diagnostic}\n",
        },
        {"Action": "fail", "Package": gate.PACKAGE, "Test": name},
        {"Action": "fail", "Package": gate.PACKAGE, "Test": root},
        {"Action": "fail", "Package": gate.PACKAGE},
    ]
    encoded = "".join(json.dumps(row) + "\n" for row in rows)
    events = gate.method_gate.parse_events(encoded, check)
    gate.method_gate.accept(events, check, 1)
    for code in (0, 2, 124, -9):
        with pytest.raises(ValueError, match="exit/package"):
            gate.method_gate.accept(events, check, code)
    with pytest.raises(ValueError, match="exact diagnostic"):
        gate.method_gate.accept(
            gate.method_gate.parse_events(
                encoded.replace(diagnostic, "unexpected error"), check
            ),
            check,
            1,
        )
