# pyright: strict
"""Verify frozen zero-array classifications and mutation masks with exact events."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import native_frozen_gate as native_gate

method_gate = native_gate.method_gate

DIRECTORY = "logos-doc/validation/zero-array-reference-free"
PRODUCTION = "src/cmd/compile/internal/types2/const_logos.go"
TYPES = "cmd/compile/internal/types2"
LOGOS = "cmd/compile/internal/logos"
MANIFEST_SHA256 = "b339088687a8d3f439680ac52dc91100c14ca097451762f6135a4fcdfdef11d6"
ARRAYS = "TestLogosReferenceFreeZeroArrays"
NESTING = "TestLogosReferenceFreeZeroArrayNesting"
PARAMS = "TestLogosReferenceFreeZeroArrayTypeParams"
FACTS = "TestGoMutationFactsZeroArrays"


def type_names() -> frozenset[str]:
    names = {ARRAYS, NESTING, PARAMS}
    for kind in (
        "pointer",
        "slice",
        "map",
        "channel",
        "function",
        "interface",
        "unsafe-pointer",
    ):
        parent = f"{ARRAYS}/{kind}"
        names.add(parent)
        names.update(
            f"{parent}/{length}" for length in ("unknown", "zero", "one", "two")
        )
    names.update(
        f"{NESTING}/{case}"
        for case in (
            "named-zero",
            "alias-zero",
            "named-marker-value",
            "zero-outer-live-inner",
            "live-outer-zero-inner",
            "live-outer-live-inner",
            "marker-only",
            "marker-and-scalar",
            "marker-and-pointer",
            "pointer-to-zero",
            "slice-of-zero",
            "map-of-zero",
            "channel-of-zero",
            "option-of-zero",
            "option-of-live",
        )
    )
    names.update(
        f"{PARAMS}/{case}"
        for case in (
            "unconstrained",
            "zero-unconstrained",
            "one-unconstrained",
            "nested-zero-unconstrained",
            "reference-set",
            "zero-reference-set",
            "zero-only-set",
            "reference-free-set",
            "live-alternative",
            "slice-alternative",
            "function-alternative",
            "scalar-set",
            "slice-of-zero-param",
            "pointer-to-zero-param",
        )
    )
    return frozenset(names)


def fact_names() -> frozenset[str]:
    return frozenset((FACTS,)) | frozenset(
        f"{FACTS}/{case}"
        for case in (
            "ReturnZero",
            "ReturnLive",
            "ForwardZero",
            "ForwardLive",
            "ReturnZeroPointers",
            "ReturnLivePointers",
            "ReturnPointerToZero",
            "ReturnSliceOfZero",
            "ReturnLiveSibling",
            "Value.Invoke",
            "(*Receiver).Return",
            "(*Receiver).Store",
            "GenericZero.Return",
            "GenericLive.Return",
        )
    )


CHECKS = (
    method_gate.Check(
        "old-type-proof",
        TYPES,
        frozenset((ARRAYS, f"{ARRAYS}/function", f"{ARRAYS}/function/zero")),
        (
            f"{ARRAYS}/function/zero",
            "LogosReferenceFree([0]func()) = false, want true",
        ),
        (PRODUCTION, f"{DIRECTORY}/old-const_logos.go.txt"),
    ),
    method_gate.Check(
        "old-mutation-proof",
        LOGOS,
        frozenset((FACTS, f"{FACTS}/ReturnZero")),
        (
            f"{FACTS}/ReturnZero",
            'ReturnZero: ParamMutated="1" ReceiverMutated=false, want "0" false',
        ),
        (PRODUCTION, f"{DIRECTORY}/old-const_logos.go.txt"),
    ),
    method_gate.Check("types", TYPES, type_names()),
    method_gate.Check("facts", LOGOS, fact_names()),
)


def frozen(repo: Path) -> tuple[str, ...]:
    manifest = f"{DIRECTORY}/FROZEN.sha256"
    path = method_gate.rooted(repo, manifest)
    if method_gate.digest(path) != MANIFEST_SHA256:
        raise ValueError("zero-array manifest changed")
    paths = [manifest]
    for line in path.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or len(expected) != 64 or relative in paths:
            raise ValueError("invalid frozen zero-array input")
        if method_gate.digest(method_gate.rooted(repo, relative)) != expected:
            raise ValueError(f"frozen zero-array input changed: {relative}")
        paths.append(relative)
    if len(paths) != 22:
        raise ValueError("expected exactly 21 frozen zero-array inputs")
    return tuple(paths)


def suite(repo: Path) -> native_gate.FrozenSuite:
    return native_gate.FrozenSuite(
        inputs=frozen(repo),
        observed=(
            PRODUCTION,
            f"{DIRECTORY}/zero_array_ci.py",
            ".github/workflows/logos.yml",
        ),
        phases=tuple(native_gate.Phase(check) for check in CHECKS),
    )


if __name__ == "__main__":
    native_gate.main(suite, __doc__)
