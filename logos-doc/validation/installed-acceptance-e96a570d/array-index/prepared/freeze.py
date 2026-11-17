from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1] / "logos-acceptance-fixes"
REVISION = "8afec95580ed24e593307fd9d30d626eff31e22d"
PREFIX = "src/cmd/compile/internal/logos/lint/"
FILES = (
    PREFIX + "registry.go",
    PREFIX + "rule_array_index_out_of_bounds.go",
    PREFIX + "rule_array_index_out_of_bounds_test.go",
    PREFIX + "lint_test.go",
    PREFIX + "rule_default_test.go",
    "test/logos/lint/array_index_out_of_bounds.lgo",
    "test/logos/lint/array_index_out_of_bounds_go.go",
    "src/cmd/go/testdata/script/logos_guarded_enum_index.txt",
)
REFERENCES = (
    "logos-doc/IMPLEMENTATION.md",
    "logos-doc/REDTEAM.md",
    "logos-doc/SPEC.md",
    "logos-doc/golangci.yml",
)


def git(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "--no-optional-locks", *arguments], cwd=REPOSITORY
    )


def save(relative: str, content: bytes) -> None:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(content)


def main() -> None:
    manifest: list[str] = []
    references: list[str] = []
    records: list[dict[str, str | int | bool]] = []
    contents: dict[str, bytes] = {}
    for path in (*FILES, *REFERENCES):
        content = git("show", f"{REVISION}:{path}")
        digest = hashlib.sha256(content).hexdigest()
        blob = git("rev-parse", f"{REVISION}:{path}").decode().strip()
        actual_blob = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content
        ).hexdigest()
        if actual_blob != blob:
            raise ValueError(f"Git blob mismatch: {path}")
        records.append({"path": path, "blob": blob, "sha256": digest,
                        "bytes": len(content), "execution_frozen": path in FILES})
        if path in FILES:
            contents[path] = content
            save("frozen/" + path, content)
            manifest.append(f"{digest}  {path}\n")
        else:
            references.append(f"{digest}  {path}\n")
    save("frozen-inputs.sha256", "".join(manifest).encode())
    save("reference-documents.sha256", "".join(references).encode())
    tests = contents[PREFIX + "rule_array_index_out_of_bounds_test.go"].decode()
    behavior, diagnostic = tests.split("func TestArrayIndexOutOfBoundsDiagnostic", 1)
    diagnostic, _ = diagnostic.split("func TestArrayIndexOutOfBoundsAllow", 1)
    names = re.findall(r'name:\s*"([^"]+)"', behavior)
    diagnoses = re.findall(r'name:\s*"([^"]+)"', diagnostic)
    if len(names) != 45 or len(diagnoses) != 2:
        raise ValueError("Frozen unit case census changed")
    leaves = [f"TestArrayIndexOutOfBounds/{name}/{language}"
              for name in names for language in ("x.go", "x.lgo")]
    leaves += [f"TestArrayIndexOutOfBoundsDiagnostic/{name}/{language}"
               for name in diagnoses for language in ("x.go", "x.lgo")]
    leaves += [f"TestArrayIndexOutOfBoundsAllow/{language}"
               for language in ("x.go", "x.lgo")]
    if len(leaves) != 96 or len(set(leaves)) != 96:
        raise ValueError("Expected 96 unique frozen unit leaves")
    save("expected-unit-leaves.txt", ("\n".join(leaves) + "\n").encode())
    source = contents[PREFIX + "rule_array_index_out_of_bounds.go"]
    original = b"proof.block(fn.Body.List) == arrayIndexReached"
    inverted = b"proof.block(fn.Body.List) != arrayIndexReached"
    if source.count(original) != 1 or source.count(inverted) != 0:
        raise ValueError("Mutation anchor is not unique")
    mutant = source.replace(original, inverted, 1)
    save("mutation/rule_array_index_out_of_bounds.go", mutant)
    record: dict[str, object] = {
        "revision": REVISION,
        "tree": git("rev-parse", REVISION + "^{tree}").decode().strip(),
        "files": records,
        "unit_leaf_count": len(leaves),
        "mutation": {"path": PREFIX + "rule_array_index_out_of_bounds.go",
                     "old": original.decode(), "new": inverted.decode(),
                     "original_sha256": hashlib.sha256(source).hexdigest(),
                     "mutant_sha256": hashlib.sha256(mutant).hexdigest()},
        "execution": "not started; final source/toolchain binding and Go lane pending",
    }
    save("frozen-provenance.json", (json.dumps(record, indent=2) + "\n").encode())
    print(json.dumps({"revision": REVISION, "frozen_files": len(FILES),
                      "unit_leaves": len(leaves), "go_commands_run": 0}))


if __name__ == "__main__":
    main()
