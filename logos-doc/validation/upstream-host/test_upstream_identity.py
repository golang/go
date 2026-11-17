# pyright: strict
"""Exercise receipts with inert files and Git; never execute the fake Go tools."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / ".github/scripts/logos-upstream-identity.sh"
)
TOOLS = (
    "bin/go",
    "bin/logos",
    *(f"pkg/tool/linux_amd64/{name}" for name in ("compile", "link", "asm", "cgo")),
)


@dataclass(frozen=True)
class Workspace:
    repo: Path
    evidence: Path
    env: dict[str, str]

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ("git", "-C", str(self.repo), *arguments),
            env=self.env,
            check=True,
            capture_output=True,
        )

    def receipt(self, phase: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("/bin/bash", str(SCRIPT), phase, str(self.repo), str(self.evidence)),
            cwd=self.repo / "src",
            env=self.env,
            check=False,
            text=True,
            capture_output=True,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    repo, evidence = tmp_path / "repository", tmp_path / "evidence"
    (repo / "src").mkdir(parents=True)
    (repo / "test").mkdir()
    evidence.mkdir()
    (repo / "src/input.txt").write_text("tracked source\n")
    (repo / "test/input.txt").write_text("tracked compiler fixture\n")
    for name in TOOLS:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"inert {name}; never executed\n")
    env = {
        "PATH": os.environ["PATH"],
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }
    result = Workspace(repo, evidence, env)
    result.git("init", "--quiet", "--initial-branch=main")
    result.git("config", "user.name", "Receipt test")
    result.git("config", "user.email", "receipt-test@example.invalid")
    result.git("add", "src/input.txt", "test/input.txt")
    result.git("commit", "--quiet", "-m", "Record inert test source")
    return result


def test_identical_receipts_from_src_working_directory(workspace: Workspace) -> None:
    assert workspace.receipt("before").returncode == 0
    assert workspace.receipt("after").returncode == 0
    for name in ("source", "tools"):
        suffix = "sha256" if name == "tools" else "txt"
        assert (workspace.evidence / f"{name}-before.{suffix}").read_bytes() == (
            workspace.evidence / f"{name}-after.{suffix}"
        ).read_bytes()
    assert (workspace.evidence / "source-before.status.txt").read_bytes() == b""
    assert (workspace.evidence / "source-after.status.txt").read_bytes() == b""
    lines = (workspace.evidence / "tools-before.sha256").read_text().splitlines()
    assert tuple(line.split("  ", 1)[1] for line in lines) == TOOLS


@pytest.mark.parametrize(
    "fault", ("tracked", "untracked", "head", "tool", "missing-tool")
)
def test_after_receipt_rejects_source_or_tool_changes(
    workspace: Workspace, fault: str
) -> None:
    assert workspace.receipt("before").returncode == 0
    original = (workspace.evidence / "tools-before.sha256").read_bytes()
    if fault == "tracked":
        (workspace.repo / "src/input.txt").write_text("changed source\n")
    elif fault == "untracked":
        (workspace.repo / "src/untracked.txt").write_text("new source\n")
    elif fault == "head":
        workspace.git(
            "commit", "--allow-empty", "--quiet", "-m", "Different HEAD, same src"
        )
    elif fault == "tool":
        (workspace.repo / "pkg/tool/linux_amd64/link").write_text("changed linker\n")
    else:
        (workspace.repo / "pkg/tool/linux_amd64/cgo").unlink()
    assert workspace.receipt("after").returncode != 0
    assert (workspace.evidence / "tools-before.sha256").read_bytes() == original
    assert (workspace.evidence / "source-after.txt").is_file()


def test_dirty_source_is_rejected_before_execution(workspace: Workspace) -> None:
    (workspace.repo / "src/input.txt").write_text("dirty before\n")
    assert workspace.receipt("before").returncode != 0
    assert (workspace.evidence / "source-before.status.txt").read_bytes()


def test_after_cannot_invent_a_missing_baseline(workspace: Workspace) -> None:
    assert workspace.receipt("after").returncode != 0
    assert not (workspace.evidence / "tools-before.sha256").exists()


def test_before_receipts_cannot_be_overwritten(workspace: Workspace) -> None:
    assert workspace.receipt("before").returncode == 0
    original = (workspace.evidence / "tools-before.sha256").read_bytes()
    (workspace.repo / "bin/go").write_text("changed driver\n")
    assert workspace.receipt("before").returncode != 0
    assert (workspace.evidence / "tools-before.sha256").read_bytes() == original
