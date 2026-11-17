# Proposed bounded exception for M6-S3b

Status: read-only plan against final commit
`e96a570da271c43a80e23bb1a188aac054acff25`. This review ran no Go command,
changed no repository source or frozen test, and did not reopen the slice.
REDTEAM #403, #404, and #405 remain blocked.

## Decision proposed to the integration owner

Permit **one additional production candidate**, explicitly recorded as
M6-S3b exception attempt 4. Do not reset the original three-attempt history
or count the dependent helper and blank-field work as fresh budgets.
The exception would cover declaration-bound default application and its
required producer-helper, facts, address, and generated-JSON integration.
It would require all original frozen attacks and an installed compiler gate.
If that candidate fails a required semantic gate, preserve its evidence,
remove the candidate, and stop again. An environmental interruption remains
an incomplete gate and must not be relabeled as a semantic implementation
failure or a pass.

This is a proposal, not approval to change code. IMPLEMENTATION G1 item 7
requires the current stop; the instruction to prepare this plan does not
waive it. The separate older Go-parity exceptions do not cover M6-S3b.

## What is established

SPEC lines 312-323 and Table T-104.5 require defaults to be checked in their
declaration scope, to be assignable, to contain only recursively constant
values/map keys, and to apply in loGo/s composite literals. Explicit and
spread values win, including zero/nil spread values; mutable composite
defaults remain fresh. Ordinary Go literals, `var x T`, and `new(T)` retain
zero initialization. These contracts are not weakened by this plan.

The retained M6-S3a validator has installed source45 evidence. It checks
unused declarations independently of construction. That evidence does not
establish default application, producer-private materialization, or helper
transport. Its original source and frozen validation script stay unchanged.

Current `structdefault.go` renders checked defaults to strings and reparses
them at construction. Imported descriptions are selected by the exported
named struct's name. `facts/write_default.go` exports only exported, nonalias
named types whose immediate underlying type is a struct. These are concrete
limits in the shipped representation, not a missing check in the validator.

The rejected candidate is exactly
`96a5a8cc599cbcc55263d32c4ceb2348abc6f0fd`. Its `fillStructDefaults` assigns
to `info.Types[lit]` and generated results. Native noder sets
`StoreTypesInSyntax: true` and leaves `Types` nil. Its capture planner also
tests `info.Types[e].Value` directly. These source facts match the recorded
installed nil-map panic. Direct map-backed tests could pass while the real
compiler failed.

Allocating an empty map is insufficient: `Info.TypeOf` falls back to syntax
facts only when `Types` is nil. An empty map can hide already checked facts.
The current lowerer already provides `typeConst(info, e)` in `build.go`,
which respects both representations, and `carryType` plus
`retyped`/`stashBodyType` patterns. Reuse those decisions; do not add a
second ad hoc constant classifier or replace the checker storage mode.

## Preconditions and postconditions for the candidate

**Preconditions.** The source expression has passed the retained declaration
validator. Uses, definitions, exact field identity, generic origin, and the
active type-fact store are available. Imported facts belong to the checked
producer package and its archive fingerprint. Generated nodes have one
owner and a declared recheck/type-information lifecycle.

**Postconditions.** Applying a default resolves the declaration's objects,
not same-spelled constructor names. Exact value types and generic arguments
survive. Private producer state is initialized inside its producer. Each
explicit initializer is evaluated once in the established order; defaults
fill only absent fields. Separate mutable composites do not share storage.
Blank defaults are validated but never emitted as `_:` keys or `value._`
assignments. Ordinary Go zero initialization and warning behavior remain.

Both map-backed and syntax-stored callers must obtain identical values,
types, diagnostics, and emitted Go. Any generated-node type update must
preserve the caller's active stores and expression flags. A transformation
that changes a pointer-context literal to a value literal must carry that
specific type change without manufacturing a global `Types` map. Preserve
the existing second-check/retyping lifecycle rather than making untyped
generated nodes appear checked.

## Implementation boundary to review before editing

Use a single checked declaration-origin plan before lowering individual
files. Local constants and types are bound at the declaration. Do not insert
runtime closure variables that change otherwise legal goto scope. Package
defaults execute through checked producer-owned functions; consumers must
not execute or reparse the descriptive `DefaultFact.Text` field.

Keep the producer, facts writer, helper validator, and generated JSON
initialization on that same plan. Preserve the existing helper contract
attacks for a nonvariadic `(T, [N]bool) T` function, exact generic arity and
constraints, matching type identity, and total-field-count mask. Reject
missing, duplicate, malformed, or incompatible helper mappings before
changing the import registry. A legacy description with no required helper
must yield a useful dependency-rebuild diagnostic, not consumer-side text
evaluation. Keep existing facts for unrelated features intact.

Do not blindly cherry-pick the rejected commit: it predates the accepted
spread-default and other integration repairs. Compare its reviewed mechanism
with current source, retain the current explicit/spread precedence and fresh
storage behavior, and record every production difference. The original
helper ABI and its tests are inseparable from the producer implementation.

## Freeze and first execution order

1. Keep `logos-doc/blocked/M6-S3b.patch` byte-identical. It has 14 file-diff
   entries: restoring the original mixed binding fixture removes the extracted
   validator copy to avoid duplicate declarations. The complete original
   manifest consists of 14 attack files, including the still-active
   `structdefault_blank_test.go`. `attack-inventory.json` verifies their hashes
   against the rejected commit. No assertion or timeout is relaxed.
2. In a new isolated worktree only after approval, restore the test-only patch.
   Separate public behavioral programs from tests that reference the absent
   proposed helper API. An undefined helper symbol is a harness/API-availability
   failure, not the required semantic RED. Reuse the unchanged source programs
   in native and ejected baseline probes and preserve their raw results.
3. Add a separate representation-parity harness before the candidate. It must
   run the same constructor programs with a populated `Types` map and with
   `StoreTypesInSyntax: true` plus nil `Types`. Include declaration constants,
   constant map keys, explicit and implicit pointer elements, and an unrelated
   previously checked expression whose type must remain visible after lowering.
   Keep the original attack files unchanged; record the new harness separately.
4. After the candidate source is frozen and built, run installed native/export
   and ejected cases first. In particular, run both
   `TestStructDefaultExportedConstantBindings` and
   `TestStructDefaultExportedPrivateFields`, without the historical
   `-skip '^TestStructDefaultExported'`. Run the exact 34-field workload next
   so a compiler panic stops the candidate before broad direct-unit GREEN can
   be mistaken for completion.
5. Only after that first installed gate passes, run the complete unchanged
   attack matrix, helpers/facts boundary checks, restoration probes, and final
   component/release gates. Preserve source/test/tool identities at each stage.

The first installed gate is deliberately earlier than in candidate 3. It can
falsify the hypothesis that fixing the storage assumptions is sufficient;
any remaining constructor, private-type, generic, or blank-materialization
failure keeps the exception candidate unaccepted.

## Required preserved gates

From the new candidate's `src`, with its freshly installed `bin` first on
PATH, run uncached component gates:

```sh
go test -count=1 -short cmd/compile/internal/logos -run '^TestStructDefault'
go test -count=1 -short cmd/compile/internal/logos/facts
go test -count=1 -short cmd/compile/internal/syntax cmd/compile/internal/types2
go test -count=1 -short cmd/compile/internal/logos
go test -count=1 -race cmd/go -run '^TestScript$/^logos_(struct_default_validation|m6_addr_defaults|eject_struct_default|struct_default_mixed_language|interop_struct_default|spread_defaults)$'
```

Also execute the unchanged `struct_default`, `struct_default_run`,
`struct_default_tag`, `unapplied_struct_default`, and `spread_defaults`
testdir leaves with exact component anchors. Retain ordinary-Go producer
consumers, zero/new controls, explicit overrides, zero/nil spreads, private
producer fields, alias and generic origins, inferred arrays, shadowed imports
and predeclared names, and fresh storage assertions. Ejected outputs must
execute with the pinned upstream oracles required by their scripts.

Generated default initialization also requires both JSON profiles: the full
relevant frontend and runtime suites, including blank/default fields without
a source constructor. The runtime/facts/command changes require their race
gates. Normal vet for changed packages, formatting, one completed final
`make.bash`, and the G2 regression floor remain required.

Perform a caught/restored mutation of the new storage-aware path using the
unchanged parity attacks, with a predicted semantic failure in syntax-storage
mode. Preserve the existing helper fill, mask-validation, and missing-mapping
attacks. Record whether a probe recompiles the component or the installed
compiler; do not imply the installed compiler changed when only a component
overlay was used. Restored installed gates use the unmodified candidate.

## Exact allocation workload

The original workload is recovered byte-for-byte under `historical/` from
the rejected Git commit. Its SHA-256 is
`6cb0f0c37c863e8382c3b5aa07e1e834c445e71735c4064c80dfe08d1b03f051`.
It contains 32 scalar defaults, one slice default, one map default, a noinline
constructor, 10,000 allocation repetitions, and checksum 557.

The old report records zero allocations under a different exact compiler
and `go1.27.4-X:logosarena`, and explicitly says that observation is not a new
pass criterion. The old raw measurement and panic logs were not recovered
by this review. Their published names/hashes remain historical attribution.
Run this exact source against paired baseline/candidate tools with identical
target and experiment settings. Require successful construction, correct
checksum, and a complete recorded comparison; do not invent a new allocation
threshold or claim a speed result. Candidate 3 produced no allocation result.

## Dependent obligations that remain visible

Nested default composition and exported anonymous default-bearing structs
inside containers remain unresolved in the current records. The old report
records direct `Inner{}` as 7 and an outer default containing `Inner{}` as 0;
that raw log is unavailable here. The current facts writer's named-struct-only
selection also explains why anonymous container provenance needs a separate
design decision. This review has not executed a new reproducer for either.

After the core block is cleared, freeze independent native/checker/ejection
programs for nested defaults and for an exported named slice or array whose
element is an anonymous default-bearing struct, with direct-construction,
explicit-zero, zero-init, ordinary-Go, and declaration-shadow controls. Keep
those programs independent of the producer-helper spelling. An exported
type-path provenance design must follow that evidence; it is not silently
included by resetting the core attempt budget. Until their gates pass, a
successful #403-405 repair still does not establish complete M6 behavior.

## Bookkeeping and resource boundary

Record the exception authorization, candidate number, exact base, preconditions,
frozen hashes, RED/GREEN/PROBE evidence, and installed identities in the owning
M6-S3b record. Keep all previous interruption/panic history. Promote #403-405
only when their actual gates pass together; preserve #402's separate validator
evidence. Any normative change must update SPEC and Table T in the same slice,
but a repair to the existing rule does not justify weakening the contract.

Use the serialized Go lane, task-owned cache, disk compilation temp, and a
real-home directory for executed tests that create modules. Check resources
before the fresh build; do not start another compiler while another lane is
compiling. No resource cleanup, build, test, or production action was performed
while creating this plan.
