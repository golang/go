# Compiler stability validation

Baseline: `b0bbbc05d5a7dee54e6501415163e78290f72231`, Go 1.27.4.
Tracking issue: [#53](https://github.com/gosuda/LoGOs/issues/53).

This change repairs observable compiler and ejection failures and removes a
duplicate inliner traversal. It preserves language features, optimization
tiers, and experiment defaults. The host-ISA test prerequisite is the unchanged
[PR #47](https://github.com/gosuda/LoGOs/pull/47), head
`949e633789fe5122c548d6c9f39c06fbb2769702`.

While the original PR checks were running, the target branch advanced to
`15ff36f8bee66047199f2061624b1dfafa875478`, landing the shared GC scan work in
[PR #26](https://github.com/gosuda/LoGOs/pull/26) and the host-ISA prerequisite.
A normal merge incorporates that current base. The only conflict was the spec
header: the current compiler baseline and the GC scan amendment are both
retained. The compiler production files and all frozen tests are unchanged by
this integration. The base's runtime metrics, arena race, and alternate
collector checks remain in the existing unit workflow. Final acceptance must
use the integrated tree; earlier green checks apply to the original base.

## Scope and proof obligations

| Area | Original failure | Repair and invariant |
|---|---|---|
| CTFE assignment | Destination/RHS trace `21` instead of `12`; base/index/RHS trace `321` instead of `123`; rebinding a map redirects an already evaluated destination; nil-map comma-ok fails | Capture destinations, evaluate sources, copy values, then store left to right. Map/slice headers keep their identity. Ordinary and comma-ok assignment share this path. |
| CTFE inline storage | A slice of an array keeps stale values after the array is assigned; overlapping copy and append expose shallow storage copies | Assign into existing array/struct slots. Snapshot aggregate sources before overlapping writes. Preserve reference fields, initialize grown capacity, and copy returned aggregates at the call boundary. |
| Tuple names | Equal line/column positions collide across files; a first package-ordinal candidate also collides across separately lowered build variants | Encode the physical basename injectively, distinguish repeated basenames in raw compiler input, and allocate groups within each file. |
| Spread defaults | A spread suppresses defaults even for fields it does not supply | Determine field coverage from the checked spread metadata and initialize only the remaining defaults. Explicit fields, spread fields, and defaults retain their precedence. |
| Ejection constraints | Word-sized enumeration fails at 63/64/65/128 custom tags and cannot disable a requested tag; a first solver candidate adds needless tags | Solve the parsed Boolean constraint, preserve an already satisfying requested assignment, and validate a complete witness with go/build. Pass the selected complete tag set to the helper, including empty. |
| Inliner | Package and function fixpoints duplicate the same batch traversal; an always-true protocol leaves a dead fallback | Share one batch pass while preserving SCC/discovery order, closure-use accounting, reassignment-oracle refresh, and devirtualization. |

For tuple names, a file basename is encoded byte for byte, without hashing or
punctuation removal. A local initialization map assigns occurrence numbers to
identical basenames in direct multi-directory compiler input. Ordinary package
files have unique basenames, so independent inactive-variant compilations retain
compatible names. Source positions remain diagnostic positions.

For CTFE, `storeValue` requires assignment-compatible types and already
snapshotted aggregate sources. It preserves inline destination slots; it does
not recursively clone slice or map referents. The return boundary must also copy
aggregate values before a later argument, RHS, composite element, or append
argument can mutate their source.

## Test-first evidence

The behavioral tests were written before each corresponding repair. Frozen
tests were retained through their repair and mutation checks. The two candidate
defects found during review received additive regressions before correction:
independent tuple-variant composition and preservation of a satisfying host tag
assignment. The storage repair likewise received an additive return-boundary
regression before its second production attempt.

| Regression | Executed baseline observation |
|---|---|
| Assignment phases | Wrong call traces, rebound-map stores, comma-ok destination ordering, and nil-map lookup errors. Fourteen source bodies also pass pristine Go 1.27.4. |
| Assignment failure phases | RHS-first and index-before-base paths produce the wrong panic. Literal string panic diagnostics also lost their text before the repair. |
| Inline storage | Twenty of twenty-four cases fail at both target word sizes. Four reference/empty/range-fresh controls pass. All twenty-four ordinary-Go bodies pass the pristine oracle. |
| Returned aggregates | The sequencing-only parent passes all seven final cases. The first storage candidate fails five explicit-return cases at both word sizes; the two named-return controls pass. All final cases pass pristine Go with and without inlining. |
| Tuple globals | Two files declare the same generated `_logos_tup3_5_0` name. The first candidate then redeclares `_logos_tup0_0` when independent variants are combined. |
| Spread defaults | Missing defaults, address literals, zero-valued spreads, explicit fields combined with defaults, empty spread sources, and fresh composite defaults fail. |
| Variant selection | Large tag conjunctions have no selected context; twelve tags cause 4096 MatchFile source opens; negated requested custom tags cannot be selected. |
| Host-tag preference | A candidate changes a satisfying `[b]` assignment to `[a,b]`, potentially selecting different dependency files. The additive test detects that change. |

An initial test spelling `!!a` was invalid build-constraint syntax and was
corrected to `!(!a)` before the selector repair. An initial spread fixture used
ordinary map-read comparisons where loGo/s requires option handling; it was
corrected before production changes. Neither diagnostic is counted as a
behavioral RED.

The return-boundary investigation also tried a named result whose slice escapes
and is mutated by a later RHS call in the same assignment. Pristine Go 1.27.4
returns different values with and without inlining for that probe. It is not a
frozen expected-value test. The final named-result controls complete the result
assignment before mutating the view; both oracle configurations agree. The five
explicit-return attacks were independently verified before their repair.

## Local evidence before integrated CI

The source-overlay checks use an existing built fork whose CTFE and syntax
sources were compared with the default baseline. They compile the edited
packages and exercise their frozen tests. They are preliminary package evidence,
not proof that a complete final toolchain was rebuilt.
The pull request's Checks tab records the subsequent clean-checkout builds and
complete regression results. The table below is the local publication snapshot.

| Gate | Current record |
|---|---|
| CTFE assignment | Value cases passed at target word sizes 4 and 8; failure-phase cases passed with the default target configuration. An isolated RHS-first mutation failed the frozen order assertions. |
| CTFE storage | Focused original storage and return-boundary checks passed. Direct storage replacement caused 38 failures; omitting the return copy caused 10 failures. Final uncached full CTFE package, including the fresh-cache differential oracle, passed in 200.685 seconds; serial vet passed. |
| Tuple lowering | Both file orders, repeated compilation, independent variants, punctuation-distinct filenames, and duplicate raw-input basenames passed. A constant group ordinal failed the frozen tests; clean restoration passed. |
| Full lowering package | The unchanged full short package suite passed in 180.590 seconds, followed by vet. An earlier run inherited unrelated ancestor Git metadata in temporary subprocess modules; moving the temporary directory outside that ancestry resolved the recorded setup failures. |
| Ejection package | Final uncached normal and race suites passed. The final rollback mutation failed the independent satisfiability oracle; restoration and vet passed. |
| Ejection integration | The frozen final driver passed native execution, all expected variant outputs, pristine Go 1.27.2 execution, and the exact-empty helper case. |
| Spread semantic integration and mutation | The frozen fixture passed through an isolated compiler in native and pristine-Go 1.27.4 ejected execution. Omitting default initialization caused the six predicted failures while evaluation/zero-value controls passed. Both clean executions passed again after restoration. |
| Inliner equivalence | Source review and the source-bound selected-fixture comparison passed at O0/O1/O2/O3/Os. Compiler diagnostics and assembly are byte-identical to the frozen old-inliner outputs. Final Go/loGo/s script execution remains required. |
| Complete rebuilt toolchain and full regression | Required on the integrated tree. Early local attempts encountered compiler termination; the serialized preliminary build ended with stale-tool targets after running while source and commits changed. A later frozen-source `make.bash -d` with one worker reached toolchain3, then its SSA compiler was killed and the build exited 2; the source tree and all frozen hashes remained unchanged. Clean CI builds of the original PR tree passed, followed by the focused compiler gates, unit/library-race checks, and host-ISA checks. The PR records the subsequent complete results and the new-base acceptance separately. |

The workflow `.github/workflows/compiler-stability.yml` builds the exact checked
out source and runs affected compiler packages, ejection race tests, all seven
new command scripts, the new compiler fixtures, formatting, vet, and the ledger
checker. Its commands preserve failure status through `pipefail`; validation
output, source identity, and evidence hashes are retained in the job logs. The
bootstrap uses repository-owned shell steps under the action ownership policy
recorded in [ci-bootstrap.md](../ci-bootstrap.md). In the combined stabilization candidate, the normal
`logos.yml` workflow owns the complete `./run.bash --no-rebuild` job. That job
selects and verifies GCC 14 on Ubuntu 24.04 for the existing C23 consumer, checks
the ledger and consumer prerequisites, and uses the supported timeout scale 8.
The full selection and all acceptance assertions remain unchanged.

The repository's existing unit/library-race, complete script, compiler-fixture,
lint, and host-ISA workflows remain applicable. A passing focused check does not
stand in for those jobs or the complete upstream gate.

The selected inliner comparison recorded compiler SHA-256
`e56f37d597e01123c092943d2bdfdda9297249222c7b7e08da6b129bc0879951`.
Its symbol table contains the new `inlinePassLogos` implementation. Both inliner
source files, the fixture, the compiler binary, and all frozen baseline outputs
had identical hashes before and after the five comparisons. This proves the
selected fixture was compiled by a binary containing the refactor; it does not
prove a complete final toolchain build for the other changed compiler packages.

## Reproduction

### Freeze-manifest succession

`frozen-tests.sha256` preserves the original repair's frozen bytes. The
subsequent acceptance follow-up uses `frozen-tests-v2.sha256`. Its only changes
to existing entries are the two variant-test call sites that now use the
equivalent test-only Linux/amd64 adapter. The adapter still supplies the exact
same host arguments to the unchanged production selector; all inputs and
semantic assertions remain unchanged. The successor also records the adapter
file, the target-feature test with checked cleanup, and the positive CTFE
map-field fixture and expected output.

The historical manifest is not rewritten. The follow-up's exact transformation
proof and lint/race/vet results are recorded in
[acceptance follow-ups](../acceptance-followups/README.md). The CTFE oracle
correction is a separate test-only change, with its original negative input
and the specification-based reason preserved there.

### Commands

From `src/`, with the rebuilt fork's `bin` directory first on PATH and
`UPROOT1272`, `UPROOT1273`, and `UPROOT1274` pointing to the pinned pristine trees:

```sh
./make.bash
go test -count=1 -short cmd/compile/internal/inline/... cmd/compile/internal/logos/... cmd/go/internal/logos/eject
go test -count=1 -short -race cmd/go/internal/logos/eject
go test -count=1 -timeout=20m cmd/go -run 'Script/^logos_(ctfe_assignment_.*|tuple_(package_names|variant_names)|spread_defaults|inline_batches|eject_variant_tags)$'
go test -count=1 -timeout=20m cmd/internal/testdir -run 'Test/logos/run/(ctfe_assignment_.*|const_func_map_field_store|spread_defaults|inline_batches)'
go vet cmd/compile/internal/inline/... cmd/compile/internal/logos/... cmd/go/internal/logos/eject
go test -count=1 -short cmd/internal/goparity -run '^TestRedteamLedgerRows$'
./run.bash --no-rebuild
```

## Performance claims

The selector regression counts calls through go/build.MatchFile's file-open
hook: **4096 before, 1 after** for the frozen twelve-tag conjunction. Both paths
also perform the initial direct header read. This is a deterministic work-count
reduction, not a measured wall-clock speedup. General Boolean satisfiability can
still require exponential search.

The inliner refactor removes 43 net production lines from the two affected files.
It does not claim faster compilation, faster generated programs, or different
optimization decisions. No optional runtime speed optimization is admitted by
this audit.

## Follow-up defects

- [#55: declaration-owned defaults](https://github.com/gosuda/LoGOs/issues/55).
  Reparsed default text binds names in the caller, which can silently change
  scalar values and interface dynamic types. Imported private defaults fail to
  resolve. Option-valued defaults also need correct lowering. A partial literal
  substitution would not preserve all of these semantics.
- [#56: inactive-only packages](https://github.com/gosuda/LoGOs/issues/56) is repaired by M24-A-S1; see [complete-inventory evidence](../ejection-completeness/README.md).
  The old plan omitted packages absent from host loading and removed their
  `.lgo` files during copying. The module census now discovers them, and real
  alternate compiler loads retain their emitted source and owned provenance.
- [#57: target feature contexts](https://github.com/gosuda/LoGOs/issues/57) is repaired by M24-A-S2; see [target-fact evidence](../ejection-target-facts/README.md) for the separately frozen and verified controls.
  The old selector synthesized `amd64.v3` as a custom tag while its actual
  tool tags remained at v1. The new real-helper controls pin the architecture
  environment, feature implications, and target-specific experiment facts.

The specification retains its full language and complete-module contracts.
Remaining defects are not converted into exclusions or weaker acceptance rules.
