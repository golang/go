# pyright: strict
"""Require frozen transitive SIMD imports, namespace, and installed-driver controls."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_frozen_gate as native_gate

method_gate = native_gate.method_gate
DIRECTORY = "logos-doc/validation/simd-transitive-imports"
MANIFEST_SHA256 = "6b2eecdbd386c1a31fc40193d9978d82c72f7b1402ae1cfa6475c15f75fd776a"
PRODUCTION = (
    "src/cmd/go/internal/work/logos_importcfg.go",
    "src/cmd/compile/internal/midway/rewrite.go",
    "src/cmd/compile/internal/midway/deepcopy.go",
    "src/cmd/compile/internal/midway/imports_logos.go",
)


def suite(repo: Path) -> native_gate.FrozenSuite:
    manifest_path = f"{DIRECTORY}/frozen.json"
    path = method_gate.rooted(repo, manifest_path)
    if method_gate.digest(path) != MANIFEST_SHA256:
        raise ValueError("SIMD import manifest changed")
    data = method_gate.json_object(path.read_text(encoding="utf-8"))
    entries, negatives = data["files"], data["negative_controls"]
    if not isinstance(entries, list) or not isinstance(negatives, list):
        raise TypeError("files and negative_controls must be lists")
    inputs = [manifest_path, f"{DIRECTORY}/README.md"]
    for value in cast(list[object], entries):
        entry = method_gate.mapping(value)
        relative = method_gate.field(entry, "path")
        if method_gate.digest(method_gate.rooted(repo, relative)) != method_gate.field(
            entry, "sha256"
        ):
            raise ValueError(f"frozen SIMD import input changed: {relative}")
        inputs.append(relative)
    phases: list[native_gate.Phase] = []
    for value in cast(list[object], negatives):
        entry = method_gate.mapping(value)
        name = method_gate.field(entry, "name")
        phases.append(
            native_gate.Phase(
                method_gate.Check(
                    method_gate.field(entry, "label"),
                    method_gate.field(entry, "package"),
                    frozenset((name.split("/", 1)[0], name)),
                    (name, method_gate.field(entry, "diagnostic")),
                    (
                        method_gate.field(entry, "source"),
                        method_gate.field(entry, "overlay"),
                    ),
                )
            )
        )
    if len(phases) != 2:
        raise ValueError("expected both archived-source negative controls")
    groups = method_gate.mapping(data["required_names"])
    for label, package, count, race in (
        ("work", "cmd/go/internal/work", 11, True),
        ("midway", "cmd/compile/internal/midway", 7, False),
        ("compiler", "cmd/compile/internal/logos", 19, False),
    ):
        value = groups[label]
        if not isinstance(value, list):
            raise TypeError("required names must be a list")
        values = cast(list[object], value)
        names = frozenset(method_gate.string(name) for name in values)
        if len(values) != count or len(names) != count:
            raise ValueError(f"unexpected {label} inventory")
        phases.append(
            native_gate.Phase(method_gate.Check(label, package, names), race=race)
        )
    return native_gate.FrozenSuite(
        inputs=tuple(inputs),
        observed=(*PRODUCTION, f"{DIRECTORY}/simd_imports_ci.py"),
        phases=tuple(phases),
    )


if __name__ == "__main__":
    native_gate.main(suite, __doc__)
