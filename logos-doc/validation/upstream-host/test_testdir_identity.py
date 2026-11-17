# pyright: strict
"""Bind the top-level compiler fixtures as well as the compiler's src tree."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from test_upstream_identity import Workspace, workspace

__all__ = ["workspace"]


@pytest.mark.parametrize("tracked", (True, False), ids=("tracked", "untracked"))
def test_top_level_compiler_fixture_changes_are_rejected(
    workspace: Workspace, tracked: bool
) -> None:
    assert workspace.receipt("before").returncode == 0
    target = workspace.repo / "test" / ("input.txt" if tracked else "new.txt")
    target.write_text("changed compiler fixture\n")
    assert workspace.receipt("after").returncode != 0


def test_receipt_records_exact_head_src_and_test_trees(workspace: Workspace) -> None:
    assert workspace.receipt("before").returncode == 0
    expected = subprocess.run(
        (
            "git",
            "-C",
            str(workspace.repo),
            "rev-parse",
            "HEAD",
            "HEAD:src",
            "HEAD:test",
        ),
        env=workspace.env,
        capture_output=True,
        check=True,
    ).stdout
    assert (workspace.evidence / "source-before.txt").read_bytes() == expected


def test_original_src_only_checker_and_fixture_remain_archived() -> None:
    archive = Path(__file__).parent / "testdir-timeout/identity-src-only"
    expected = {
        "old-identity.sh.txt": "e62198a4cf88a683da1a225f1f1b652a9de0ffa09aaedba8c07a8ba16020c33b",
        "old-test_upstream_identity.py.txt": "c8243552645070c0d6b6bca381c9f1129ffea2cf6c9ee136c102238df86a3ecc",
    }
    for name, sha256 in expected.items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == sha256
