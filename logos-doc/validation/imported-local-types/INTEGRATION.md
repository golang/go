# Imported compiler symbols: M0-A-S17

Existing candidate `39c62004a991501a25f93eae7a63dc7c7fac5646` repairs an ordinary-Go
compiler error when an imported generic function or method declares a local
type that captures its type parameters. This integration preserves the single
production file and both frozen test files byte for byte. REDTEAM #418 remains
planned until the merged toolchain completes its acceptance gates.

## Source and binary provenance

The original corpus input is `golang.org/x/tools v0.52.0`, revision
`16517d951552611e7bd358c48ee1051fd0e57b0c`. Independent hashing of all 1,624
extracted module files gives the pinned digest
`h1:ryxg4U63oaUYk7uOLTF37n43f4qgtGvqV4Ioit1BaEs=`. The original comparison ran
`go build ./...` with cgo disabled and downloads off. The focused replay builds
24 byte-identical files from `go.mod`, `go.sum`, `cmd/digraph`, and
`internal/graph`; [upstream-input.sha256](upstream-input.sha256) records their
paths relative to that module. The external source is not duplicated here.

| Comparator | Source identity | Driver SHA-256 | Compiler SHA-256 |
|---|---|---|---|
| Stock Go | Pristine Go 1.27.4 | `de0ae3e082ce7599eae4fb69fa2ef1e43b8f31ffd3fce62cdf81930aa9005050` | `40307e5f5c65a4b4220bbf35dbe014348780589ea2964003f0bf318fa14640e1` |
| S3 fork | `584c7ece750d3e9508bb21c5a95d751dbf18cce2` | `3d9780d7f455d7b75920b5ce61f8cd34ed6a4f1c3bc539c394ee2b91d3ee2baf` | `85d01f687bcd896258299e4e0a74b21d2e817f1ab1393b20109bf96ba26954ee` |
| Existing repaired fork | Recorded build source `39c62004a991501a25f93eae7a63dc7c7fac5646` | `9625c880740e0b8ba821dbc5f3ab9ff16ceb727799e340be902a0b94ad900c8e` | `11647b40eb1e8d4442f1ef8409ff4e4acb230b6ffc3635365b96112df01be0fd` |

S3 is a component toolchain based on the earlier fork, not a rebuilt combined
integration. The repaired fork's six installed tool hashes match
[author/binaries.sha256](author/binaries.sha256). Its current source has no
changes under `src` after the recorded repair commit. The source and frozen
test hashes match that commit. Each independent candidate invocation checked
the driver and compiler identities before and after execution.

## Frozen independent replay

The five programs are the unchanged plain-Go cells of
`src/cmd/compile/internal/logos/compile_local_type_import_test.go`, including
separate importing modules and exact result assertions.

| Input | Stock Go | S3 fork | Existing repair |
|---|---|---|---|
| Generic method with a captured local type | Correct output | ICE: 1 implicit parameter | Correct output |
| Generic function with a captured local type | Correct output | ICE: 1 implicit parameter | Correct output |
| Nested locals capturing two parameters | Correct output | ICE: 2 implicit parameters | Correct output |
| Nongeneric local-type control | Correct output | Correct output | Correct output |
| Package-level generic-type control | Correct output | Correct output | Correct output |
| Exact upstream `cmd/digraph` build | Builds | ICE: 1 implicit parameter | Builds |

These are **18 commands: 15 program executions and three builds**. Successful
programs exit zero and print exactly `generic local ok\n`; successful builds
exit zero without output and produce an executable. The four S3 failures have
the expected internal compiler error. No command times out or exhausts memory.

The [independent records](independent/) preserve all commands, environment
overrides, times, exit codes, hashes, and raw stdout/stderr. The historical
[input manifest](independent/input-manifest.txt) was frozen before execution
and verified afterward; its paths are relative to the original replay root,
not this repository. Its two original test-file hashes identify the committed
fixtures, and its upstream hashes identify the pinned module subset.
All runs used private caches, `GOMAXPROCS=1`, `-p=1`, a 384 MiB soft heap limit,
`CGO_ENABLED=0`, local toolchain selection, and disabled downloads. Existing
source and module caches were read-only inputs. No compiler rebuild or
production edit was performed during this independent review or integration.

## Reviewed candidate evidence and attempt count

The retained author evidence covers **48 frozen cells**: 28 lazy-scope tests
and 20 compiler/runtime cells across plain Go, LoGOs handling Go, interoperation,
and structural overlays. [RED](author/red.log) records 16 forbidden lazy
lookups and 12 generic compiler errors; the other cells are controls.
[Installed GREEN](author/installed.log) explicitly runs all 48 cells without
skips. The [identifier-filter probe](author/scope-probe.log) fails 28 cells;
the [direct-readonly-lookup probe](author/readonly-probe.log) fails 12.
[Restoration](author/restored.log) passes the scope and existing metadata gates.
These complete author gates are reviewed records, not independent reruns here.
Raw logs retain their original bytes, including whitespace in the RED output.

This slice has used **one production candidate out of three**. The interrupted
bootstrap and cache setup failure are retained separately and did not change
the implementation. The [build observation](author/build-observation.json)
records the final dist completion banner and installed-tool observation; the
original shell exit status was not recovered. No exit status is invented.

The repair and frozen tests were locally committed when reviewed. The author's
validation directory was then untracked; the selected records here preserve
that reviewed snapshot. The public PR #66 head at review was
`3a87df9d1b60279aafcfd01e5692719f53d9a57c`, which did not contain this repair.

## Integration and required acceptance

The candidate's production preimage matched integration base
`198b36928fa826a2a8331efcaed883debb3def49`. Cherry-pick `0db13853` preserves its
production and frozen-test bytes; follow-up integration changes only tracking
and evidence. SPEC 5 and T-Sec5 state the invariant: resolve source declaration
identifiers before lazy lookup, retain legal private/Unicode declarations, and
reach method facts through declaring types. The importer assertion and export
format remain unchanged.

From the repository root, verify the committed inputs and records:

```sh
sha256sum -c logos-doc/validation/imported-local-types/frozen-tests.sha256
sha256sum -c logos-doc/validation/imported-local-types/production.sha256
sha256sum -c logos-doc/validation/imported-local-types/evidence.sha256
```

A fresh merged `make.bash`, complete affected compiler-package tests and vet,
and the configured upstream regression gate remain required. Existing CI owns
those gates. This report does not claim a merged installed-compiler pass,
complete external-corpus pass, or full upstream `run.bash` pass.
