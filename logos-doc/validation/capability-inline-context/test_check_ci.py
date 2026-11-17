from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, cast

import pytest

type Phase = Literal["negative", "candidate"]
type Event = dict[str, object]

SCRIPT = "TestScript/logos_cap_inline_context"
WITNESS = (
    "ERROR [capabilities::unauthorized_net]: Transitive dependency "
    "'example.com/inlinecapdep/safe' reaches 'net.sysSocket'\n"
)


class Checker(Protocol):
    def events(self, evidence: Path, phase: str) -> dict[str, object]: ...

    def run(self, mode: str, repo: Path, evidence: Path) -> bool: ...


@pytest.fixture
def checker() -> Checker:
    # Evaluate the repository-owned checker; no shell or Go command runs.
    runner = Path(__file__).with_name("check_ci.sh")
    parts = runner.read_text().split("<<'PY'\n")
    assert len(parts) == 2
    source, separator, _ = parts[1].partition("\nPY\n")
    assert separator
    module = ModuleType("inline_context_checker")
    exec(compile(source, str(runner), "exec"), module.__dict__)  # noqa: S102
    return cast(Checker, module)


def transcript(phase: Phase) -> list[Event]:
    result = "fail" if phase == "negative" else "pass"
    records: list[Event] = [
        {"Action": "start", "Package": "cmd/go"},
        {"Action": "run", "Package": "cmd/go", "Test": "TestScript"},
        {"Action": "run", "Package": "cmd/go", "Test": SCRIPT},
    ]
    if phase == "negative":
        records.append(
            {"Action": "output", "Package": "cmd/go", "Test": SCRIPT, "Output": WITNESS}
        )
    records.extend(
        [
            {"Action": result, "Package": "cmd/go", "Test": SCRIPT},
            {"Action": result, "Package": "cmd/go", "Test": "TestScript"},
            {"Action": result, "Package": "cmd/go"},
        ]
    )
    return records


def save(evidence: Path, phase: Phase, records: list[Event]) -> None:
    (evidence / f"{phase}.status.txt").write_text(
        "1\n" if phase == "negative" else "0\n"
    )
    (evidence / f"{phase}.stderr.log").write_bytes(b"")
    (evidence / f"{phase}.json").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records)
    )


@pytest.mark.parametrize("phase", ["negative", "candidate"])
def test_accepts_exact_transcripts(
    checker: Checker, tmp_path: Path, phase: Phase
) -> None:
    save(tmp_path, phase, transcript(phase))
    assert checker.events(tmp_path, phase)["exit"] == (1 if phase == "negative" else 0)


@pytest.mark.parametrize("origin", ["package", "root", "build"])
@pytest.mark.parametrize(
    "diagnostic",
    [
        "panic: unrelated crash\n",
        "SIGSEGV: segmentation violation\n",
        "SIGBUS: bus error\n",
        "signal: segmentation fault\n",
        "test timed out after 20m0s\n",
    ],
)
def test_rejects_crashes_outside_child(
    checker: Checker, tmp_path: Path, origin: str, diagnostic: str
) -> None:
    record: Event = {"Action": "output", "Package": "cmd/go", "Output": diagnostic}
    if origin == "root":
        record["Test"] = "TestScript"
    elif origin == "build":
        record = {"Action": "build-output", "ImportPath": "", "Output": diagnostic}
    save(tmp_path, "negative", transcript("negative") + [record])
    with pytest.raises(ValueError, match="tool failure"):
        checker.events(tmp_path, "negative")


@pytest.mark.parametrize("phase", ["negative", "candidate"])
def test_rejects_failed_build_metadata(
    checker: Checker, tmp_path: Path, phase: Phase
) -> None:
    records = transcript(phase)
    records[-1]["FailedBuild"] = "cmd/go"
    save(tmp_path, phase, records)
    with pytest.raises(ValueError, match="build"):
        checker.events(tmp_path, phase)


def test_requires_witness_in_the_child(checker: Checker, tmp_path: Path) -> None:
    records = transcript("negative")
    records[3]["Test"] = "TestScript"
    save(tmp_path, "negative", records)
    with pytest.raises(ValueError, match="witness"):
        checker.events(tmp_path, "negative")


@pytest.mark.parametrize(
    "malformed",
    [
        "[]",
        "null",
        '{"Action":"output","Action":"pass"}',
        '{"Action":"output","Package":"cmd/go","Test":null,"Output":"extra"}',
        '{"Action":"output","Package":"cmd/go","Output":"extra","Elapsed":NaN}',
    ],
)
def test_malformed_input_retains_rejection_status(
    checker: Checker, tmp_path: Path, malformed: str
) -> None:
    save(tmp_path, "candidate", transcript("candidate"))
    with (tmp_path / "candidate.json").open("a") as output:
        output.write(f"{malformed}\n")
    assert not checker.run("candidate", tmp_path, tmp_path)
    result: object = json.loads((tmp_path / "candidate.check.json").read_text())
    assert isinstance(result, dict)
    report = cast(dict[str, object], result)
    assert report["accepted"] is False
    assert "error" in report
