# pyright: strict
"""Check the real frozen SIMD inventories without executing Go."""

from __future__ import annotations

from pathlib import Path

import pytest
import simd_imports_ci as gate


def test_distinct_negative_and_positive_inventories() -> None:
    suite = gate.suite(Path(__file__).resolve().parents[3])
    assert len(suite.phases) == 5
    assert tuple(len(phase.check.names) for phase in suite.phases) == (2, 2, 11, 7, 19)
    assert tuple(phase.race for phase in suite.phases) == (
        False,
        False,
        True,
        False,
        False,
    )
    assert tuple(phase.check.negative is not None for phase in suite.phases) == (
        True,
        True,
        False,
        False,
        False,
    )
    compiled = suite.phases[-1].check.names
    for profile in ("fork-go", "logos"):
        assert f"TestCompileSIMDTransitiveImports/{profile}/type_only" in compiled
        assert f"TestCompileSIMDTransitiveImports/{profile}/internal_denied" in compiled


def test_manifest_change_rejected(tmp_path: Path) -> None:
    directory = tmp_path / gate.DIRECTORY
    directory.mkdir(parents=True)
    (directory / "frozen.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        gate.suite(tmp_path)
