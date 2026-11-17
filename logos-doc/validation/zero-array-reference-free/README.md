# Zero-length arrays and reference-free values

This slice fixes mutation summaries for values whose only reference-typed
descendants occur inside zero-length arrays. Such an array contains no
elements. It cannot retain a function, pointer, or other reference from its
element type. Positive-length arrays still require the existing element-type
proof, and pointers or slices of an empty array still carry references.

## Observed failure and exact source

The independent native unit run
[43951103189](https://github.com/gosuda/LoGOs/actions/runs/43951103189), job
`133593503956`, built head `9909feb26fe92b124f17c11fdf0022be7a643188`, source
tree `df267333b1d5c5727e2477215a6807cae52f8e18`.
`TestCompileSIMDTaggedMethodFacts/logos` emitted `ParamMutated:1` for Foo,
Bar and Qux across all four amd64 SIMD tags. The unchanged test requires
explicit zero masks. Its separate receiver flags matched the expected
true, true and false values.

The raw [JSONL events](observed-logos.jsonl), [phase report](observed-report.json),
and equal [before](observed-before.json) and [after](observed-after.json)
snapshots are retained verbatim. [Provenance](observed-provenance.json)
records the original job, artifact and file hashes. The original unit job
failed; these copied records are not a successful candidate run.

Both drivers also failed the normal consumer build because implicit SIMD
imports were missing. That dependency issue is separate from this mutation
summary repair. This slice does not change the consumer, its imports, the
real SIMD source, or the frozen integration assertions.

The isolated base is `b61a6cfeb33051791348597fbb69818f4e2ec6f7`, whose compiler
source tree is the same as the failing head. The original predicate source is
preserved as [old-const_logos.go.txt](old-const_logos.go.txt) for an exact Go
overlay negative control.

## Why the zero mask remains the oracle

The receiver is tracked separately from the parameter mask. The fact writer
skips the receiver before incrementing the mask bit, so bit zero means the
first explicit parameter. SPEC section 5 describes the two independent
summaries. The existing `TestEmitGoParamMutatedFacts` also expects the
receiver-writing `(*Counter).SetN` to have a zero parameter mask.

`simd.Float32s` contains a marker derived from `bridge.ZeroSized`, which holds
`[0]func(*ZeroSized) *ZeroSized`. Fixed-width amd64 SIMD types have analogous
zero-length function arrays in their v128, v256 and v512 markers. The current
`logosReferenceFreeAlong` array branch inspects the element type even when
the array contains no elements. As a result, it treats every corresponding
VectorC value as reference-bearing.

In Qux, returning that value marks its first parameter. In Foo and Bar, the
unknown Add parameter effect marks the argument for the same reason. A
zero-length array proof removes these invented parameter effects.

The receiver flags must stay true, true and false. After the return value
becomes reference-free, the scanner still visits the Add call. Its unknown
receiver effect reaches `markMentionWrite`, which deliberately bypasses the
reference-free check. Qux does not mention its receiver. The new producer
controls pin this distinction instead of changing any old assertion.

## Frozen controls

`FROZEN.sha256` pins the two new test files, this evidence, the old predicate,
the unchanged real SIMD source and integration test, the current mutation
writer, and existing generic/const preservation fixtures. Verify it from the
repository root before and after each gate.

| Test root | Leaf cases | Named run/pass events | Invariant |
| --- | ---: | ---: | --- |
| `TestLogosReferenceFreeZeroArrays` | 28 | 36 | Seven reference kinds at unknown, zero, one and two elements; only exactly zero bypasses the element proof |
| `TestLogosReferenceFreeZeroArrayNesting` | 15 | 16 | Aliases, named values and nested empty arrays are safe; live siblings, pointers, slices, maps, channels and positive reference arrays remain reference-bearing |
| `TestLogosReferenceFreeZeroArrayTypeParams` | 14 | 15 | Empty generic storage is safe; unknown or mixed reference-bearing type sets remain conservative |
| `TestGoMutationFactsZeroArrays` | 14 | 15 | Ordinary Go checking, mutation emission and codec round trip preserve explicit zero masks and independent receiver flags |

The focused positive suite has 71 leaves and 82 named tests in two packages.
The new tests compile against the old production APIs. They do not require
the proposed implementation to exist.

## Hosted execution requirements

No local Go build, test, compiler, analyzer or vet command was executed for
this slice. Source formatting is not an execution pass. The existing hosted
real-SIMD failure is observed RED evidence. RED, GREEN and mutation-probe
execution for the new focused tests remain pending on an independently built
hosted toolchain; the first production candidate is not a completed G1/G2 gate.

After a fresh `./make.bash`, run from `src` with the installed fork first on
PATH and record commands, raw JSONL, stderr, exits, and source/tool hashes:

```sh
go test -json -count=1 -p=1 -parallel=1 -timeout=10m \
  -run '^Test(LogosReferenceFreeZeroArray(s|Nesting|TypeParams)|GoMutationFactsZeroArrays)$' \
  cmd/compile/internal/types2 cmd/compile/internal/logos
```

Require all 82 named tests to run and pass, both packages to pass, exit zero,
and no skips, crashes, build failures or unexpected error output. The complete
existing real-SIMD tagged-method gate must also pass after the separate
implicit-import repair; do not accept only the in-memory type checks.

For RED and the mutation probe, map the absolute production path
`src/cmd/compile/internal/types2/const_logos.go` to the exact archived old
source in an overlay. Run each negative separately:

```sh
go test -json -count=1 -p=1 -parallel=1 -timeout=10m -overlay "$OVERLAY" \
  -run '^TestLogosReferenceFreeZeroArrays$/^function$/^zero$' \
  cmd/compile/internal/types2
go test -json -count=1 -p=1 -parallel=1 -timeout=10m -overlay "$OVERLAY" \
  -run '^TestGoMutationFactsZeroArrays$/^ReturnZero$' \
  cmd/compile/internal/logos
```

The first must fail exactly its root, function group and zero leaf with
`LogosReferenceFree` reporting false where true is required. The second must
fail exactly its root and ReturnZero child with `ParamMutated="1"` where
`"0"` is required. Both must exit one with a failing package and no build
failure, skip or crash. A generic nonzero process result is insufficient.

After removing the overlay, repeat the focused positive suite and the
unchanged generic/const-view controls. Existing `TestEmitGoParamMutatedFacts`,
`TestLogos`, compiler fixtures, affected-package vet, and the complete
`./run.bash --no-rebuild` regression floor remain required. Verify frozen
hashes and installed tool hashes again. No mutation may remain in the
candidate.
