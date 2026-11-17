# pyright: strict
"""Check exact package-name selection, including misleading prefix/variant names."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import partition_dist as gate
import pytest
from hypothesis import given
from hypothesis import strategies as st


def test_exact_cmd_go_name_has_a_complete_disjoint_complement() -> None:
    names = (
        "archive/tar",
        "cmd/go",
        "cmd/go/internal/cfg",
        "cmd/gofmt",
        "cmd/go:racebench",
        "cmd/internal/testdir:0_1",
        "crypto/...:purego",
        "internal/godebugs",
    )
    result = gate.partition("\n".join(names) + "\n")
    assert result.cmd_go == ("cmd/go",)
    assert result.complement == tuple(name for name in names if name != "cmd/go")
    assert not set(result.cmd_go).intersection(result.complement)
    assert set(result.cmd_go).union(result.complement) == set(names)
    assert gate.CMD_GO_FILTER == "^cmd/go$"
    assert gate.COMPLEMENT_FILTER == "!^cmd/go$"


@given(
    st.lists(
        st.sampled_from(
            ("fmt", "os", "cmd/go/internal/test", "cmd/gofmt", "cmd/go:racebench")
        ),
        min_size=1,
        unique=True,
    )
)
def test_partition_preserves_every_unrelated_dist_name(names: list[str]) -> None:
    result = gate.partition("\n".join((*names, "cmd/go")))
    assert result.all_names == (*names, "cmd/go")
    assert result.cmd_go == ("cmd/go",)
    assert result.complement == tuple(names)


@pytest.mark.parametrize(
    "text",
    (
        "",
        "\n",
        "cmd/go\n",
        "fmt\n",
        "cmd/go:racebench\nfmt\n",
        "cmd/go\ncmd/go\nfmt\n",
        "cmd/go\nfmt\nfmt\n",
        "cmd/go\n\nfmt\n",
        "cmd/go\n fmt\n",
        "cmd/go\nfmt\0\n",
    ),
)
def test_incomplete_or_ambiguous_inventory_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        gate.partition(text)


def test_cli_records_inventory_and_only_claims_a_partition(tmp_path: Path) -> None:
    inventory = tmp_path / "all-tests.txt"
    raw = b"cmd/go\ncmd/go/internal/cfg\ninternal/godebugs\n"
    inventory.write_bytes(raw)
    evidence = tmp_path / "partition"
    result = subprocess.run(
        (
            sys.executable,
            str(Path(gate.__file__)),
            "--inventory",
            str(inventory),
            "--evidence",
            str(evidence),
        ),
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0
    assert result.stderr == b""
    assert (evidence / "inventory.txt").read_bytes() == raw
    report: object = json.loads(
        (evidence / "partition.json").read_text(encoding="utf-8")
    )
    assert isinstance(report, dict)
    assert report["execution_claimed"] is False
    assert report["complete_union"] is True
    assert report["disjoint"] is True
    assert report["cmd_go"] == {
        "filter": "^cmd/go$",
        "timeout_scale": 4,
        "names": ["cmd/go"],
    }
