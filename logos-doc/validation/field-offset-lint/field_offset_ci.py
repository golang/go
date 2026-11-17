# pyright: strict
"""Run frozen field-offset controls with the installed native compiler."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_frozen_gate as native_gate

method_gate = native_gate.method_gate

DIRECTORY = "logos-doc/validation/field-offset-lint"
PRODUCTION = "src/cmd/compile/internal/logos/lint/rule_arena_pointer_escape.go"
PACKAGE = "cmd/compile/internal/logos/lint"
MANIFEST_SHA256 = "40fa23ddbefabafa4a55f8a58132f2eb311c95a8a49fbb46d424677396cba499"


def frozen(repo: Path) -> tuple[tuple[str, ...], frozenset[str], str, str]:
    manifest_path = f"{DIRECTORY}/frozen.json"
    path = method_gate.rooted(repo, manifest_path)
    if method_gate.digest(path) != MANIFEST_SHA256:
        raise ValueError("field-offset manifest changed")
    data = method_gate.json_object(path.read_text(encoding="utf-8"))
    entries, names = data["files"], data["required_names"]
    if not isinstance(entries, list) or not isinstance(names, list):
        raise TypeError("manifest files and required_names must be lists")
    paths = [manifest_path, f"{DIRECTORY}/README.md"]
    for value in cast(list[object], entries):
        entry = method_gate.mapping(value)
        relative = method_gate.field(entry, "path")
        if method_gate.digest(method_gate.rooted(repo, relative)) != method_gate.field(
            entry, "sha256"
        ):
            raise ValueError(f"frozen field-offset input changed: {relative}")
        paths.append(relative)
    recorded_names = cast(list[object], names)
    required = frozenset(method_gate.string(name) for name in recorded_names)
    if len(required) != 50 or len(recorded_names) != 50:
        raise ValueError("expected exactly 50 distinct named tests")
    return (
        tuple(paths),
        required,
        method_gate.field(data, "negative_name"),
        method_gate.field(data, "negative_diagnostic"),
    )


def suite(repo: Path) -> native_gate.FrozenSuite:
    paths, names, negative_name, diagnostic = frozen(repo)
    return native_gate.FrozenSuite(
        inputs=paths,
        observed=(PRODUCTION, f"{DIRECTORY}/field_offset_ci.py"),
        phases=(
            native_gate.Phase(
                method_gate.Check(
                    "old-rule",
                    PACKAGE,
                    frozenset(("TestArenaFieldOffset", negative_name)),
                    (negative_name, diagnostic),
                    (PRODUCTION, f"{DIRECTORY}/old-rule.go.txt"),
                )
            ),
            native_gate.Phase(method_gate.Check("candidate", PACKAGE, names)),
        ),
    )


if __name__ == "__main__":
    native_gate.main(suite, __doc__)
