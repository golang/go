# pyright: strict
"""Check exact package-name selection, including misleading prefix/variant names."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

import partition_dist as gate
import pytest
from hypothesis import given
from hypothesis import strategies as st

FROZEN = Path(__file__).parent / "testdir-timeout"
FROZEN_SHA256 = "24f21c2b7f184d10249fbebb4a260a1340c6428c159bbf8a2b29ce1032e82596"
WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/logos.yml"


def test_exact_names_have_a_complete_disjoint_three_way_partition() -> None:
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
    assert result.complement == tuple(
        name for name in names if name not in ("cmd/go", "cmd/internal/testdir:0_1")
    )
    assert result.testdir == ("cmd/internal/testdir:0_1",)
    groups = (set(result.cmd_go), set(result.testdir), set(result.complement))
    assert all(
        not left.intersection(right)
        for i, left in enumerate(groups)
        for right in groups[i + 1 :]
    )
    assert set().union(*groups) == set(names)
    assert gate.CMD_GO_FILTER == "^cmd/go$"


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
    inventory = (*names, "cmd/go", "cmd/internal/testdir:0_1")
    result = gate.partition("\n".join(inventory))
    assert result.all_names == inventory
    assert result.cmd_go == ("cmd/go",)
    assert result.testdir == ("cmd/internal/testdir:0_1",)
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
        "cmd/go\nfmt\n",
        "cmd/go\ncmd/internal/testdir\nfmt\n",
        "cmd/go\ncmd/internal/testdir:0_1\n",
        "cmd/go\ncmd/internal/testdir:0_2\nfmt\n",
        "cmd/go\ncmd/internal/testdir:1_2\nfmt\n",
        "cmd/go\ncmd/internal/testdir:1_1\nfmt\n",
        "cmd/go\ncmd/internal/testdir:0_1\ncmd/internal/testdir:1_2\nfmt\n",
        "cmd/go\ncmd/internal/testdir:0_2\ncmd/internal/testdir:2_2\nfmt\n",
        "cmd/go\ncmd/internal/testdir:0_1\ncmd/internal/testdir:0_1\nfmt\n",
        "cmd/go\ncmd/internal/testdir:0_99999999999999999999999999999\nfmt\n",
    ),
)
def test_incomplete_or_ambiguous_inventory_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        gate.partition(text)


def test_cli_records_inventory_and_only_claims_a_partition(tmp_path: Path) -> None:
    inventory = tmp_path / "all-tests.txt"
    raw = b"cmd/go\ncmd/internal/testdir:0_1\ncmd/go/internal/cfg\ninternal/godebugs\n"
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
    assert report["schema"] == 2
    assert report["cmd_go"] == {
        "filter": "^cmd/go$",
        "timeout_scale": 4,
        "names": ["cmd/go"],
    }
    assert report["testdir"] == {
        "filter": gate.TESTDIR_FILTER,
        "timeout_scale": 2,
        "names": ["cmd/internal/testdir:0_1"],
    }
    assert report["complement"] == {
        "filter": gate.COMPLEMENT_FILTER,
        "timeout_scale": 1,
        "names": ["cmd/go/internal/cfg", "internal/godebugs"],
    }


def select(names: tuple[str, ...], pattern: str) -> tuple[str, ...]:
    negate = pattern.startswith("!")
    expression = re.compile(pattern.removeprefix("!"))
    return tuple(name for name in names if bool(expression.search(name)) != negate)


@given(st.integers(min_value=1, max_value=16), st.booleans())
def test_complete_shards_preserve_order_and_match_execution_filters(
    count: int, reverse: bool
) -> None:
    shards = tuple(f"cmd/internal/testdir:{index}_{count}" for index in range(count))
    names = ("fmt", "cmd/go", *shards, "cmd/go:racebench")
    if reverse:
        names = tuple(reversed(names))
    result = gate.partition("\n".join(names))
    assert result.testdir == (tuple(reversed(shards)) if reverse else shards)
    for selected, pattern in (
        (result.cmd_go, gate.CMD_GO_FILTER),
        (result.testdir, gate.TESTDIR_FILTER),
        (result.complement, gate.COMPLEMENT_FILTER),
    ):
        assert selected == select(names, pattern)


@pytest.mark.parametrize(
    "other",
    (
        "cmd/internal/testdir",
        "cmd/internal/testdir:race",
        "cmd/internal/testdir:00_1",
        "cmd/internal/testdir:0_01",
        "cmd/internal/testdir:0_0",
        "cmd/internal/testdir:0_1_extra",
        "cmd/internal/testdir:0_1:race",
        "cmd/internal/testdir:0_1/suffix",
        "cmd/internal/testdir/subpackage:0_1",
        "cmd/internal/testdirectory:0_1",
        "xcmd/internal/testdir:0_1",
    ),
)
def test_misleading_testdir_names_remain_in_the_complement(other: str) -> None:
    names = ("cmd/go", "cmd/internal/testdir:0_1", other)
    result = gate.partition("\n".join(names))
    assert result.complement == (other,)
    assert select(names, gate.TESTDIR_FILTER) == result.testdir
    assert select(names, gate.COMPLEMENT_FILTER) == (other,)


def test_frozen_hosted_inventory_and_failure_evidence() -> None:
    raw = (FROZEN / "frozen.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256
    manifest = cast(dict[str, object], json.loads(raw))
    for name, expected in cast(dict[str, str], manifest["files"]).items():
        assert hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() == expected
    inventory = (FROZEN / "evidence/all-tests.txt").read_text()
    result = gate.partition(inventory)
    assert len(result.all_names) == 666
    assert len(result.complement) == 664
    assert result.cmd_go == ("cmd/go",)
    assert result.testdir == ("cmd/internal/testdir:0_1",)
    assert (
        FROZEN / "evidence/cmd-go-exit.txt"
    ).read_text() == "driver=0\nconsole_capture=0\n"
    assert (
        FROZEN / "evidence/complement-exit.txt"
    ).read_text() == "driver=1\nconsole_capture=0\n"
    assert (
        "FAIL\tcmd/internal/testdir\t600.050s"
        in (FROZEN / "evidence/complement.log").read_text()
    )


def assert_workflow_profiles(workflow: str) -> None:
    lines = tuple(
        line.strip()
        for line in workflow.splitlines()
        if "./run.bash" in line and " -k " in line
    )
    profiles: dict[str, tuple[str, int]] = {}
    expression = re.compile(
        r"GO_TEST_TIMEOUT_SCALE=([0-9]+) \.\./\.github/scripts/logos-test-env\.sh "
        r"\./run\.bash --no-rebuild -k -run='([^']+)' 2>&1 \| "
        r'tee "\$evidence/([a-z-]+)\.log"'
    )
    for line in lines:
        match = expression.fullmatch(line)
        assert match is not None, line
        label = match.group(3)
        assert label not in profiles
        profiles[label] = (match.group(2), int(match.group(1)))
    assert profiles == {
        "cmd-go": (gate.CMD_GO_FILTER, 4),
        "testdir": (gate.TESTDIR_FILTER, 2),
        "complement": (gate.COMPLEMENT_FILTER, 1),
    }


def test_actual_workflow_executes_the_recorded_partition() -> None:
    workflow = WORKFLOW.read_text()
    assert_workflow_profiles(workflow)
    assert "    timeout-minutes: 90\n" in workflow
    testdir = re.search(
        r'(?ms)^      - name: "Run the complete upstream compiler fixture suite"\n'
        r"(.*?)(?=^      - name:|\Z)",
        workflow,
    )
    assert testdir is not None
    assert "        timeout-minutes: 25\n" in testdir.group(1)


@pytest.mark.parametrize("label", ("cmd-go", "testdir", "complement"))
def test_workflow_contract_rejects_selection_drift(label: str) -> None:
    workflow = WORKFLOW.read_text()
    lines = workflow.splitlines(keepends=True)
    matches = 0
    for index, line in enumerate(lines):
        if "./run.bash" in line and f'tee "$evidence/{label}.log"' in line:
            lines[index] = re.sub(r"-run='[^']+'", "-run='^fmt$'", line)
            matches += 1
    assert matches == 1
    with pytest.raises(AssertionError):
        assert_workflow_profiles("".join(lines))


def test_workflow_records_identity_around_all_partitions_after_ordinary_failure() -> (
    None
):
    workflow = WORKFLOW.read_text()
    before = 'bash ../.github/scripts/logos-upstream-identity.sh before "$GITHUB_WORKSPACE" "$evidence"'
    after = 'bash ../.github/scripts/logos-upstream-identity.sh after "$GITHUB_WORKSPACE" "$evidence"'
    assert workflow.count(before) == workflow.count(after) == 1
    assert workflow.index(before) < workflow.index("./run.bash --no-rebuild -list")
    for line in workflow.splitlines():
        if "./run.bash" in line and " -k " in line:
            assert workflow.index(before) < workflow.index(line) < workflow.index(after)
    step = re.search(
        r'(?ms)^      - name: "Verify upstream source and installed tools"\n'
        r"(.*?)(?=^      - name:|\Z)",
        workflow,
    )
    assert step is not None
    assert (
        "${{ !cancelled() && steps.smoke.outcome == 'success' && "
        "matrix.suite == 'upstream' && steps.upstream_inventory.outcome == 'success' }}"
    ) in step.group(1)
    assert "success()" not in step.group(1)
    assert after in step.group(1)
