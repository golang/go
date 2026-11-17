# Review of the legacy CTFE map-field fixture

## Verdict

`test/logos/compile/const_func_unsupported_store.lgo` has a stale negative
support-boundary expectation under the integrated contract. Accepting this
particular function does not violate the declared purity or storage rules.
The specified result is 4; successful compilation alone does not prove that
numeric result, so an acceptance correction must assert it.

This was a read-only review of the frozen integration source. No Go process,
source edit, or expectation edit was performed. The observation that the old
errorcheck fixture now compiles comes from root's hosted-failure report.

## Exact case and contract

The function creates a fresh local `S{M: map[string]int{}}`, assigns
`s.M["k"] = 4`, and returns `s.M["k"] ?? -1`. It does not access mutable
package state, protected constant-origin storage, pointers, I/O, goroutines,
dynamic map iteration, or any other excluded operation.

- `logos-doc/SPEC.md:444` lists the const-function purity exclusions.
- SPEC lines 448-451 require Go assignment ordering, destination identity,
  value-copy behavior, and reference sharing through slice and map fields.
- SPEC line 462 requires an error for a form the interpreter does not support;
  it does not permanently designate map-field stores as unsupported.
- Table T row T-201 (`logos-doc/IMPLEMENTATION.md:305`) has the same assignment
  and map-header obligations. M24-S1 and M24-S2 at lines 4000 and 4017 state the
  corresponding implementation invariants.

## Concrete source and history evidence

Commit `c3733e8adfc6c3839b50b11c1b985c5f56f92fa9`, "Preserve Go assignment
phases during CTFE", is the decisive change. Its parent handled map stores
only when the map base was a bare `syntax.Name`. A base such as `s.M` fell
through to `cannot assign in const evaluation`.

The new `indexPlace` evaluates the base first, recognizes its KMap type, and
captures that evaluated map header (`src/cmd/compile/internal/logos/ctfe/value.go`,
starting at line 1087). This is necessary for the recorded Go destination and
evaluation-order semantics. It accepts the fresh local map in this fixture.
NaN-key rejection remains at line 1105 and nil-map write rejection at line
1122. Imported writes and pointer dereferences retain separate impurity checks.

Commit `0cd8c0366759596caa8c653f521eebbdea2c1024` subsequently preserves inline
aggregate storage and reference sharing. Its frozen `TestAssignmentStorage`
includes `array_map_references`, which positively checks mutation through a
nested map header and expects 779 at both word sizes 4 and 8. The order and
error suites retain nil-map and failure-order rejection controls.

The legacy fixture is unchanged between baseline
`b0bbbc05d5a7dee54e6501415163e78290f72231` and actual hosted checkout
`f2ad58be8742b8ec350db8f06d9dc794ea3302e4`. Its SHA-256 is
`0997677cba3f15d6be766c190860622bbba3104a1482d4f4293a63640c0bf02a`.
The repository is shallow; `6c62c85c` is the retained history boundary, so this
review does not claim it was the fixture's original introduction commit.

The legacy file is not named in the current compiler-stability freeze manifest
or the M24-S1/S2 attack list. The frozen storage/order attacks and their exact
assertions must remain unchanged. This absence is not a claim that the legacy
file was never frozen in unavailable earlier history.

## Proposed disposition

Use a separate, explicitly documented test-oracle correction. Preserve the
original body and its recorded hash as history, and turn the newly supported
case into a positive assertion that the constant value is exactly 4. Do not
merely delete its ERROR annotation or add an artificial production rejection.

Retain genuine negative controls, including the existing out-of-range,
fractional, and NaN map-key cases in `const_func_map_store_key.lgo`, plus the
frozen nil-map failure cases. A separately added nil-map-field control would
test the adjacent rejection boundary without altering any frozen M24 attack.

After the compiler slot is released, require the corrected case's compile-time
value assertion and native/ejected behavior, the genuine rejection controls,
and the unchanged frozen CTFE gates. Record the current errorcheck failure as
an obsolete-oracle failure, not a production RED. Apply G1's separate
test-correction review explicitly; do not silently weaken a frozen attack or
relabel this as a new CTFE production repair.
