# Preserve the Go-only type projection

Slice M0-A-S21; attack #443; audit AUD-211.

## Current evidence boundary

The existing `go/types TestGenerate` assertion failed on source tree
`e237c399aae8f8065b724e8a0e27585686c7da18`: native loGo/s panic recording and
C-aware type-writer details leaked into the generated Go-only projection.
The public `go/types/builtins.go` and `typestring.go` files were already
correct and remain byte-identical.

One implementation candidate adds the C-writer method/field to the existing
projection allowlists, restores the upstream result-bearing panic block,
and moves only the loGo/s void-result recording into a private compiler hook.
The exact patch is `preparation/candidate-001.patch`; `candidate/` contains
the three source files. No language rule or public Go API changes.

The component gate used immutable installed e96 tools with the physical
candidate GOROOT. It validates generation from the candidate source. It
does not execute a compiler rebuilt with the new native panic hook.
Installed panic/C-ejection controls and the final combined acceptance gate
remain pending. Root's fresh combined make was running when this packet
was assembled; this packet claims no outcome for that build.

## Actual outcomes

| Run | Outcome | Seconds |
|---|---|---:|
| baseline-001-go-types-generate | Semantic RED: stale builtins and typestring projections | 4.765 |
| projection-component-001 | Setup failure before TestGenerate; generated build inputs absent | 0.542 |
| projection-component-002 | GREEN: unchanged TestGenerate | 1.879 |
| projection-method-mutation-001 | Caught removal of the logosCgoTypeName allowlist entry | 1.824 |
| projection-restored-001 | Restored candidate GREEN | 0.470 |

The setup failure reported undefined internal/buildcfg defaults. The next
run copied six unchanged, ignored generated source inputs from the installed
e96 tree. Their bytes and hashes are retained under `setup/` and
`provenance/projection-generated-build-inputs.json`. This was environment
repair, not a second implementation attempt or a semantic RED.

## Frozen inputs and mutation

`freeze/manifest.json` protects 23 complete public/source/test/fixture files
and the exact `TestGenerate` and `generate` assertion bodies. The frozen
public files and all native/ejected controls are copied under `freeze/`.
The three candidate hashes match `provenance/integration-source-001.json`.
That integration has source tree `33752c03819ed5afcb37f87633a3cac668e181a8`;
its installed behavior is outside this component proof.

The mutation is an external Go overlay removing one allowlist entry. It
leaves the assertion bodies unchanged and produces the expected typestring
diff. The original mutant source, original overlay JSON, and a portable
relative-path mutation patch are included. No void-hook mutation or newly
installed native gate is claimed here.

## Reproduction and integrity

Every `runs/` directory contains the complete original `inputs.json`,
`result.json`, stdout, and stderr. Raw bytes and historical pending states
were preserved. `specs/` and `recorders/` retain the executed command/env
specifications and their recorder implementation. Absolute historical paths
remain in those original records; `source-map.json` maps copied inputs to
relative paths in this packet. Recreate overlays for local checkout paths
when replaying; do not rewrite historical records.

The generation command is the unfiltered `TestGenerate` function with
`-count=1 -p=1 -parallel=1 -short -v -timeout=20m`. It reads actual files
through runtime.GOROOT, so a compile-only overlay cannot replace the physical
candidate source checkout. Use the pinned tool identities in the receipts.

`files.sha256` covers every payload file. `manifest.json` also covers that
checksum list. Binaries, caches, and build temporary directories are omitted.
