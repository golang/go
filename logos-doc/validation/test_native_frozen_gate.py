# pyright: strict
"""Reject incomplete or changing evidence without starting a Go process."""

from __future__ import annotations

from pathlib import Path

import native_frozen_gate as gate
import pytest


@pytest.mark.parametrize("failure", ("none", "phase", "identity", "empty", "duplicate"))
def test_complete_stable_evidence_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    first = gate.Phase(
        gate.method_gate.Check("first", "example/p", frozenset(("TestOne",)))
    )
    second = gate.Phase(
        gate.method_gate.Check("second", "example/p", frozenset(("TestTwo",))),
        race=True,
    )
    phases = (first, second)
    if failure == "empty":
        phases = ()
    elif failure == "duplicate":
        phases = (first, first)
    calls: list[str] = []
    identities = 0

    def verify(_repo: Path) -> gate.FrozenSuite:
        return gate.FrozenSuite((), (), phases)

    def identity(_repo: Path, _paths: tuple[str, ...]) -> dict[str, object]:
        nonlocal identities
        identities += 1
        return {
            "tracked_status": "",
            "compiler": "changed"
            if failure == "identity" and identities == 2
            else "stable",
        }

    def capture(
        command: tuple[str, ...],
        _repo: Path,
        _out: Path,
        check: gate.method_gate.Check,
        env: dict[str, str],
        _timeout: float,
    ) -> dict[str, object]:
        calls.append(check.label)
        assert ("-race" in command) == (check.label == "second")
        assert env["CGO_ENABLED"] == ("1" if check.label == "second" else "0")
        return {"accepted": not (failure == "phase" and check.label == "first")}

    monkeypatch.setattr(gate.method_gate, "identity", identity)
    monkeypatch.setattr(gate.method_gate, "capture", capture)
    out = tmp_path / "evidence"
    out.mkdir()
    assert gate.run(tmp_path, out, verify) == (failure == "none")
    report = gate.method_gate.json_object((out / "report.json").read_text())
    assert report["accepted"] == (failure == "none")
    if failure in ("empty", "duplicate"):
        assert not calls
        assert report["complete"] is False
    else:
        assert calls == ["first", "second"]
        assert report["complete"] is True
