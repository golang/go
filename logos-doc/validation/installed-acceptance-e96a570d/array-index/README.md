# Installed array-index acceptance at e96a570d

All focused gates passed on 2026-11-16. The external mutation produced the
two required unwanted-diagnostic failures, and both unchanged cells then
passed without the overlay. No source, frozen input, or installed compiler
was modified. This is the later execution record; `prepared/README.md`
preserves the preparation-only report exactly as it existed before execution.

## Exact source and installed toolchain

- Commit: `e96a570da271c43a80e23bb1a188aac054acff25`.
- Full tree: `0841c3d2a92b7d6091a5e248b82bd820326947a8`.
- Compiler source tree: `e237c399aae8f8065b724e8a0e27585686c7da18`.
- Installed go/logos SHA-256:
  `34a04896c6bdbedb2a8dbdb491b836b41e80efa47a255286733de021c776e6f9`.
- Installed compiler SHA-256:
  `b8d0742e328093b87fec566a68fb810d7043446ce0b575a81f304432bee87b6e`.

Root's fresh make completed with exit 0 in 1290.122 seconds. Its receipt is
copied to `provenance/fresh-make-result.json`; the installed identity is in
`provenance/identity-e96a570d.json`. The common make log remains in the
parent build evidence, identified by the receipt's SHA-256. The before/after
bindings in `execution/` match exactly and record a clean worktree, the eight
frozen source/test/script hashes, and the installed go/logos/compiler hashes.

## Observed gates

Times below are outer command durations, including compilation where needed.
Every command completed within its existing bound, without interruption.

| Stage | Seconds | Exit |
|---|---:|---|
| lint-full | 82.526 | 0 |
| fixture-completeness | 52.034 | 0 |
| installed-fixtures | 85.242 | 0 |
| guarded-enum-race | 151.036 | 0 |
| vet | 7.627 | 0 |
| gofmt | 0.032 | 0 |
| mutation | 5.292 | 1, expected probe failure |
| restored | 0.515 | 0 |

The complete lint package passed. Its required 96 leaves are exactly the
45 behavior cases in both Go and loGo/s, four diagnostic/span cells, and two
suppression cells listed in `prepared/expected-unit-leaves.txt`. All 96
executed and passed; no lint test was skipped. Fixture completeness passed.
The actual testdir harness ran both frozen array-index errorcheck files,
checking all eight expected diagnostics with no skipped leaves.

The race-enabled cmd/go script `logos_guarded_enum_index` ran and passed.
Its unchanged assertions require two native JSON lint findings, three valid
executions over all 256 enum values, and three bounds-panic controls. This
includes ordinary Go runtime behavior and explicit suppression in loGo/s.
The lint package's vet gate passed, and gofmt listed no frozen Go source.

## Mutation and restoration

The sole mutation changes the report condition from
`proof.block(fn.Body.List) == arrayIndexReached` to `!= arrayIndexReached`.
The external Go overlay recompiles the lint package. It does not replace the
installed compiler or edit its source. Both unchanged safe-guard cells,
`TestArrayIndexOutOfBounds/guardedReceiver/x.go` and its `.lgo` counterpart,
failed for the predicted unwanted `correctness::array_index_out_of_bounds`
deny diagnostic with `want []`. Missing tests, build errors, timeouts,
skips, and unrelated failures were not accepted as mutation evidence.
Both cells passed again without the overlay. The exact patch, complete
mutant, raw streams, and leaf checks are preserved.

## Evidence layout and reproduction

- `prepared/` preserves all preparation records, exact eight frozen inputs,
  the 96-leaf oracle, executed runner, external mutation, source review, and
  author-recorded RED provenance. Its historical checksum files are unchanged.
- `execution/` contains the eight commands' exact argv, selected environment,
  UTC times, bounds, raw stdout/stderr, exit/timeout status, stream hashes,
  leaf checks, before/after bindings, overlay map, and final summary.
- `provenance/` contains the common source/tool identity and fresh make receipt.
- `evidence.sha256` covers every portable file except itself.

The original test-first input and its provenance remain in `prepared/upstream/`.
The author reported four missing diagnostics before the rule existed. No
historical raw RED stream was recovered or reconstructed. That historical
RED and this fresh installed GREEN/probe/restoration are distinct records.

Raw logs retain the absolute paths used during this execution. The actual
cache was `final-postbuild-cache`, CC was `/usr/bin/cc`, CGO was enabled,
and real-home source temp was `/home/agent/logos-array-index-e96-7952kg8p`.
The runner used serial package compilation, GOMAXPROCS=1, GOGC=20,
GOMEMLIMIT=512MiB, ten-minute test timeouts, and an eleven-minute outer bound.
The cmd/go harness supplies its own child environment; this report does not
claim those heap settings reach every nested process.

To reproduce, copy `prepared/` outside any source worktree, use a fresh output
directory, and invoke its runner against a freshly built clean source commit
with the corresponding installed-tool hashes. The original command inputs
are recorded in `execution/`; do not substitute another installed compiler
while claiming this source identity.

The retained race test executable is intentionally excluded. Its original
path, SHA-256, size, and source/tool binding remain in
`execution/guarded-enum-race/retained-binary.json`. No cache, executable,
live source temp, or additional repository checkout is included.

This record closes the focused M9-A-S18 / AUD-204 / REDTEAM #440 installed
acceptance. It does not claim the complete language fixture, broad command,
full Go, or other final integration gates passed.
