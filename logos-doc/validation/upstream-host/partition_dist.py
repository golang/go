# pyright: strict
"""Partition one unfiltered dist inventory without inventing or dropping tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

CMD_GO_FILTER = "^cmd/go$"
TESTDIR_FILTER = "^cmd/internal/testdir:(0|[1-9][0-9]*)_([1-9][0-9]*)$"
COMPLEMENT_FILTER = f"!^({CMD_GO_FILTER[1:-1]}|{TESTDIR_FILTER[1:-1]})$"


@dataclass(frozen=True)
class Partition:
    all_names: tuple[str, ...]
    cmd_go: tuple[str, ...]
    testdir: tuple[str, ...]
    complement: tuple[str, ...]


def partition(text: str) -> Partition:
    names = tuple(text.splitlines())
    if not names or any(
        not name or any(char.isspace() or char == "\0" for char in name)
        for name in names
    ):
        raise ValueError("inventory must contain nonempty test names, one per line")
    if len(set(names)) != len(names):
        raise ValueError("inventory contains duplicate test names")
    if names.count("cmd/go") != 1:
        raise ValueError("inventory must contain cmd/go exactly once")
    selected = tuple(name for name in names if name == "cmd/go")
    shards = tuple(
        (name, int(match.group(1)), int(match.group(2)))
        for name in names
        if (match := re.fullmatch(TESTDIR_FILTER, name)) is not None
    )
    if not shards:
        raise ValueError("inventory must contain the complete canonical testdir shards")
    count = shards[0][2]
    if len(shards) != count or any(total != count for _, _, total in shards):
        raise ValueError("testdir shards have missing or inconsistent counts")
    indices = {index for _, index, _ in shards}
    if len(indices) != count or any(index >= count for index in indices):
        raise ValueError("testdir shard indices must cover exactly 0 through N-1")
    testdir = tuple(name for name, _, _ in shards)
    separate = set(selected).union(testdir)
    complement = tuple(name for name in names if name not in separate)
    if not complement:
        raise ValueError("unfiltered inventory has an empty complement")
    assert len(selected) + len(testdir) + len(complement) == len(names)
    assert not separate.intersection(complement)
    assert separate.union(complement) == set(names)
    return Partition(names, selected, testdir, complement)


class Arguments(argparse.Namespace):
    inventory: Path
    evidence: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(namespace=Arguments())
    raw = args.inventory.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("dist inventory exceeds one MiB")
    result = partition(raw.decode("utf-8"))
    out = args.evidence.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "inventory.txt").write_bytes(raw)
    report = {
        "schema": 2,
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "all_names": result.all_names,
        "cmd_go": {"filter": CMD_GO_FILTER, "timeout_scale": 4, "names": result.cmd_go},
        "testdir": {
            "filter": TESTDIR_FILTER,
            "timeout_scale": 2,
            "names": result.testdir,
        },
        "complement": {
            "filter": COMPLEMENT_FILTER,
            "timeout_scale": 1,
            "names": result.complement,
        },
        "disjoint": True,
        "complete_union": True,
        "execution_claimed": False,
    }
    with (out / "partition.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"Partitioned {len(result.all_names)} dist names: cmd/go=1, "
        f"testdir={len(result.testdir)}, complement={len(result.complement)}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnicodeError, ValueError) as error:
        print(f"dist inventory rejected: {error}", file=sys.stderr)
        raise SystemExit(1) from error
