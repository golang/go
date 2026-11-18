# Stronger compile-time optimization

Issue: [#24](https://github.com/gosuda/LoGOs/issues/24).
Baseline: `b0bbbc05d5a7dee54e6501415163e78290f72231`.
Scope: generated-program performance at explicit `-O=3`, for Go and LoGOs.

## Decision

Extend scalar facts in the existing SSA pipeline. Keep its representation,
worklist, constant evaluator, and cleanup passes. The initial five selected
techniques form three implementation units; the follow-up adds bounded
transfers within those units. The lower optimization tiers keep
their current behavior. The change adds no dependency, whole-program mode,
runtime dispatch, or second range-proof pass.

The decision is based on concrete missing behavior. Before this change,
`((x &^ 255) + (y &^ 255) + 37) & 255` still executes arithmetic, a masked
rotate still executes a rotate, and two predecessor-aligned pairs `(1, 3)`
and `(2, 2)` still require a runtime addition although both sums are `4`.
SCCP also omits bit-count intrinsics and conditional selection from its
constant lattice. These are gaps in existing analyses, not missing compiler
infrastructure.

## Twenty candidates

The ranking weighs demonstrated opportunity, semantic proof, compiler cost,
code growth, and maintenance in this Go fork. It is a design ranking, not a
benchmark of twenty implementations.

| Rank | Technique | Decision | Reason |
|---:|---|---|---|
| 1 | Carry-aware known-bit arithmetic | Select | Preserve bit facts through wrapping addition, subtraction, and negation; remove masked arithmetic and impossible conditions without duplicating code. |
| 2 | Known-bit rotations and projections | Select | Transform data and known masks together, with at most 64 count residues; remove a rotate/shift round trip when the input's high bits prove it is an identity. |
| 3 | Correlated constant-phi folding | Select | Preserve relationships between values chosen by the same incoming edge. Independent scalar lattices lose this information. |
| 4 | SCCP bit-intrinsic evaluation | Select | Let bit length and population count consume constants discovered after the final generic rewrite pass. Reuse existing evaluation rules. |
| 5 | SCCP Boolean and conditional-select propagation | Select | Select a known arm or meet possible arms, with monotone handling of cyclic phis. Keep effects in the existing memory graph. |
| 6 | Demanded-bit simplification | Defer | Could remove work whose output bits are unused, but requires use-sensitive reasoning and proof across multiple consumers. |
| 7 | Additional SCCP, range-proof, and cleanup rounds | Defer | Needs a demonstrated source-level gap and a compiler-cost budget. A second unrestricted prove pass conflicts with the current O3 contract. |
| 8 | Profile-guided call-site inlining | Retain existing | Go already supplies PGO. Real profiles are preferable to another unconditional inline-budget increase. |
| 9 | Stronger static devirtualization | Defer | Potentially unlocks inlining and escape analysis; first identify missed upstream cases and preserve typed-nil behavior. |
| 10 | Partial inlining and cold-path outlining | Defer | Needs a cost model and careful treatment of defer, recover, stack traces, and code size. |
| 11 | Constant-argument function cloning | Reject for this change | Previously withdrawn by the spec; adds code growth and export/cache complexity. |
| 12 | Broader scalar replacement of aggregates | Defer | Decomposition and escape analysis already exist. New cases need alias and address-observability proofs. |
| 13 | Load forwarding and stronger dead-store elimination | Defer | Requires memory-alias proofs across calls, atomics, panics, and explicit clears. |
| 14 | Extended loop-invariant code motion | Defer | Existing LICM is present. Broader hoisting must preserve zero-trip behavior, faults, and safepoints. |
| 15 | Bounded constant-trip loop unrolling | Defer | Requires target-aware profitability and a code-growth cap; explicit comptime field expansion already exists. |
| 16 | Loop unswitching and guarded versioning | Defer | Duplicates code and can change when guards execute. Needs a measured loop workload and safe guard placement. |
| 17 | Automatic vectorization | Separate workstream | Dependence analysis, aliases, tails, target instructions, and floating-point order warrant a separate implementation. |
| 18 | Whole-program or ThinLTO-style optimization | Defer | Requires coordinated export-data, object, cache, incremental-build, and linker changes. |
| 19 | Equality saturation or solver-guided superoptimization | Defer | Prefer offline discovery of proven rewrite rules before adding bounded search to every compilation. |
| 20 | Fast-math reassociation and assumed non-overflow | Reject | Conflicts with Go's defined integer overflow and this fork's fixed-order floating-point reductions. |

## Contracts and implementation

### Known bits

For every concrete input represented by the input facts, every output bit
marked known must match the concrete operation. Unknown data bits stay zero
in the abstract value. The existing `fold` normalizes narrow data and masks
by sign extension and handles cycles conservatively.

Addition propagates uncertainty from operand bits and possible carry changes.
Subtraction and negation use the same wrapping arithmetic with complemented
operands and a known carry-in. No signed or unsigned no-overflow assumption
is introduced. The constant-time transfer follows the tristate-number
construction proved in the [CGO 2022 paper](https://people.cs.rutgers.edu/~santosh.nagarakatte/papers/cgo-2022.pdf).
The implementation is a small standalone formula; it copies no kernel code.

Rotation uses the operand width, not the host integer width. Only count bits
below `log2(width)` matter, including negative counts. The analysis intersects
facts for the possible residues, with a hard maximum of 64. It performs no
enumeration of concrete data values.

The same facts can simplify a partial result. For a width `w` and a literal
count `0 <= k < w`, logically shifting `RotateLeft(x, k)` right by `k`
returns the low `w-k` bits of `x`. If its top `k` bits are known zero, the
whole pair is the identity. Replace it with the original input, allocating
no SSA value. Both counts must be literal integers with equal values and
the rotate and unsigned shift must have matching widths. Signed shifts,
unknown or unequal counts, and unknown or nonzero high bits are excluded.
The existing cached facts remain sound because the replacement is equivalent.

This projection rule addresses a measured gap: removing a packed header's
reserved-bit guard alone left a rotate followed by a reversing shift and
regressed the processing benchmark. The original workload and its negative
result are retained; the completed compiler is measured against the same
source. No instruction padding or benchmark-specific layout rule is added.

### Correlated values

Valid SSA aligns same-block phi arguments by predecessor index. Evaluate
an allowlisted integer or Boolean binary operation separately for each
aligned pair. Replace it only when all incoming edges produce the same
constant opcode and bits. A scalar constant may accompany a phi.

The pass accepts 2 through 8 predecessors. It inspects one phi level and
never unfolds a loop. A phi from another block, a nonconstant incoming value,
an excluded operation, or a differing result leaves the original operation
intact. Division, remainder, shifts, floating-point, pointers, memory, and
calls are excluded. Existing generic rules own width and comparison semantics.

The pass runs after generic CSE and before `phiopt`. It allocates at most
eight evaluator temporaries per candidate; `gcse deadcode` after `phiopt`
removes unused values. Running before `phiopt` preserves predecessor alignment
when an integer phi chooses zero or one. For example, the aligned pairs
`(0, 7)` and `(1, 6)` always sum to seven; converting the first phi to a
Boolean conversion too early hides that relationship. The follow-up moves
the existing traversal and adds no analysis round.

### Conditional constant propagation

The existing SCCP worklist now admits bit length and population count at
8, 16, 32, and 64 bits, Boolean conjunction/disjunction, and integer or Boolean
conditional selection. Intrinsic evaluation uses `computeLattice` and the
existing generic rules. A known condition selects one arm; an unknown
condition meets both arms. Top facts wait where needed so the lattice only
moves from top to a constant to bottom, including newly reachable backedges.

Changing a value never authorizes changing effects. Calls, memory order,
bounds and nil checks, and division-by-zero behavior remain observable.

### Closing scalar fact gaps

[Issue #50](https://github.com/gosuda/LoGOs/issues/50) extends the selected
techniques after source probes exposed three residual gaps in the first O3
implementation. The baseline for this follow-up is `7089362d`, with scalar
analysis enabled on both sides. It selects pass ordering, exact byte-swap
facts, and bounded bit-query results over a second analysis pass, a new fact
cache, or speculative rewrites. Each helper has one concern and bounded work;
the G6 single-candidate exception applies to these small transfers.

A byte swap is a fixed bit permutation. Applying that permutation to both
the abstract value and its known mask preserves every bit fact exactly.
The operation determines the permutation width: 16, 32, or 64. The result
type determines constant representation. These widths are distinct on the
Power10 `ReverseBytes16` intrinsic, which returns an SSA `uint` value. Bits
above the operation width are known zero. Existing narrow-result
normalization still owns canonical sign extension of SSA constants.

SCCP must also evaluate byte swaps because loop convergence can discover a
constant after known-bits analysis has finished. Its transfer maps Top to
Top, a constant to the corresponding width-correct constant, and Bottom to
Bottom. It preserves the original result type and lets the worklist revisit
users. A conflicting backedge therefore cannot freeze a transient constant.

For a width-masked abstract input, let `ones` be its known-one bits and
`possible` include every bit not known zero. Every represented input has
bit-length bounds `Len(ones)..Len(possible)`, population-count bounds
`PopCount(ones)..PopCount(possible)`, and trailing-zero bounds
`Ctz(possible)..Ctz(ones)`. Clamp trailing-zero counts of zero to the input
width. The transfer deliberately uses these zero-inclusive bounds for
`CtzNonZero` too, even when a separate path proof excludes zero.

For a result interval `[lo, hi]`, every bit above the highest set bit of
`lo XOR hi` is constant throughout that interval. Keep that common prefix
and mark the remaining bits unknown. This can expose a constant projection
without replacing a genuinely variable count. All operations use existing
cached facts and constant-time word operations; cyclic input facts remain
conservative. The new transfers allocate no analysis storage.

The source attacks require actual instruction removal for aligned trailing
counts, rotated header lengths, sparse population classes, byte-swap literals,
masked byte fields, late loop predicates, and zero/one correlated phis.
Dynamic controls must retain their operations. Independent concrete-value
oracles cover all input widths, both 32- and 64-bit count results, narrow
constant representation, overflow, changing loops, effects, and required
panics. Cross-compilation and codegen assertions cover 386 and both Power10
endian targets. Local 386 execution was unavailable; CI now selects both scalar
flag values for execution. Power10 coverage is compilation evidence.

Incremental measurements and the frozen-test record belong to
[FOLLOWUP.md](measurements/scalar-opt/FOLLOWUP.md). The original flag-off
versus flag-on measurements remain separate.

### Tier routing and limits

`-O=3` adds `-d=logosopt=1` through the existing settings registry. Both
drivers pass it to every package, including standard-library packages.
The existing standard-library variable-make exemption removes only its own
key. The flag participates in compiler action IDs through the existing path.
`logos.toml` can select O3 under `logos`; plain `go` still ignores that file.
O0, O1, O2, Os, and both command defaults retain their prior scalar analysis.

The debug flag also permits controlled baseline/candidate builds with all
other flags equal. No export format, runtime API, source syntax, ISA default,
escape-analysis budget, or prove-pass budget changes.

## Verification record

The attack suites were run before the implementation and failed on the
missing optimization. Negative controls passed. Source fixtures also showed
the baseline instruction sequences before the new compiler was installed.
The frozen attacks cover abstract-domain enumeration, integer widths and
wraparound, rotate-count residues, predecessor alignment, independently
controlled diamonds, loop-carried phis, unsupported operations, and bounds
at 7, 8, and 9 predecessors.

Build, integration, and mutation evidence is recorded in the
[validation record](measurements/scalar-opt/VALIDATION.md). The
[measurement record](measurements/scalar-opt/README.md) retains all paired
timings, including the initial regressions, with raw samples and a reproducible
runner. A removed instruction or a smaller binary alone is not a measured
runtime speedup. Synthetic kernels demonstrate their particular dataflow
patterns, not a whole-application gain.

## Primary references

- [Go integer-overflow semantics](https://go.dev/ref/spec#Integer_overflow).
- [CGO 2022: tristate-number abstract interpretation](https://people.cs.rutgers.edu/~santosh.nagarakatte/papers/cgo-2022.pdf).
- [LLVM KnownBits reference](https://llvm.org/doxygen/structllvm_1_1KnownBits.html).
- [LLVM transformation passes](https://llvm.org/docs/Passes.html).
- [LLVM InstCombine contributor guide](https://llvm.org/docs/InstCombineContributorGuide.html).
- [Go profile-guided optimization](https://go.dev/doc/pgo).
