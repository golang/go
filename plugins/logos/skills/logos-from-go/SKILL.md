---
name: logos-from-go
description: Migrate a Go module or package to loGo/s. Use when the user says migrate, convert, or port Go code to loGo/s or .lgo files.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH, plus the Go toolchain for the baseline.
metadata:
  version: "0.20.0"
  milestone: "M20"
---

# Migrate Go to loGo/s

Move one Go module to loGo/s in order: inventory, baseline, convert, fix, verify. No converter ships yet; each step below is done by hand. Fall back to eject when migration stalls.

Coverage: the mapping table covers M1 error propagation, M3 nil forms, and M6 interpolation and chains. M2 unions, M4 properties, M5 const forms, and M12 capabilities add no mapping rows: those features have no Go equivalent to rewrite. Not shipped: M7–M11, M13–M21+.

## Procedure

1. Inventory. List each non-standard Go dependency with `logos list -deps`. Done when the list is written down and each entry has a verdict: keep (the `.lgo` package imports it as a Go package), replace with a loGo/s module, or replace with the standard library. A module holds one kind (all `.go` or all `.lgo`), and Go and loGo/s packages import each other per import, so a dependency never blocks migration.
2. Baseline. Run `go test ./...` in the Go module and save the full output to a file. Done when the log file exists and the exit code is recorded.
3. Convert. Rename every `.go` file to `.lgo` with `git mv`, including `_test.go` to `_test.lgo`. Then rewrite each `if err != nil { return ..., err }` pattern to `?` with the mapping table below. Done when no `.go` file remains and no convertible error check remains.
4. Fix. Run `logos build ./...`. Fix each diagnostic by its rule ID in the table below. Repeat until the build exits 0.
5. Verify. Run `logos build ./...` and `logos test ./...`. Done when both exit 0 and the same tests that passed in the baseline log still pass.
6. Fallback. When migration stalls, or the user needs plain Go, run `logos eject -o DIR` on the loGo/s module: eject writes a complete pure-Go copy of the module to DIR, and that plain-Go output works alongside the remaining Go code. Done when `go build ./...` and `go test ./...` both exit 0 inside DIR.

## Go to loGo/s mapping (M1)

| Go | loGo/s |
|---|---|
| `v, err := f()` then `if err != nil { return ..., err }` | `v := f()?` |
| `if err := f(); err != nil { return err }` | `f()?` |
| `defer` cleanup whose error must reach the caller | `defer f()?` |
| `if err != nil { cleanup() }` before return | `errdefer cleanup()` |
| `x_test.go` test file | `x_test.lgo` test file |

Rules for `?`: the call returns `error` last; the enclosing function returns `error` last; `?` ends an assignment, an expression statement, a `defer`, or an `errdefer`.

## Go to loGo/s mapping (M3)

Proven by `test/logos/run/option_basic.lgo`, `test/logos/run/option_chain.lgo`, `src/internal/types/testdata/logos/option_types.lgo`, `src/internal/types/testdata/logos/narrowing.lgo`, `src/internal/types/testdata/logos/map_index.lgo`.

| Go | loGo/s |
|---|---|
| `*T` that may be nil, guarded by `if p != nil` | `T?` with `?.`, `??`, and `!= nil` narrowing |
| `v, ok := m[k]` | unchanged; single-value `m[k]` reads as `V?`, so unwrap kept uses with `?? d` |

Rules: `nil` and `T` assign into `T?`; `T?` never converts back. After conversion, plain selections on the option fail until a `!= nil` proof narrows them. Results of `.lgo` callees are read only after the error check; results of Go callees keep their Go contracts.

## Go to loGo/s mapping (M6)

Proven by `test/logos/run/interp.lgo`, `test/logos/run/chains.lgo`, `src/internal/types/testdata/logos/interp.lgo`, `src/internal/types/testdata/logos/chains.lgo`.

| Go | loGo/s |
|---|---|
| `fmt.Sprintf("User %s has %d items", name, n)` | `$"User {name} has {n} items"` |
| filter loop appending matches, then a map loop over the result | `xs.filter(x => cond).map(x => f(x))`, fused into one pass |

Rules: interpolation operands are strings, integers, bools, float32/float64, `[]byte`, `error`, or `fmt.Stringer`; other types stay with `fmt.Sprintf`. Chains fuse consecutive `filter`/`map`/`take`/`drop`; terminal calls (`find`, `first`, `last`, `get`, `sum`, `count`, `any`, `all`, `sortBy`) end the pass. Lambda parameters are const views: rewriting them or the receiver inside the lambda is an error.

## Diagnostics

| Rule | Fix |
|---|---|
| `correctness::mixed_package` | Finish the rename: no package holds both `.go` and `.lgo`. |
| `correctness::misplaced_try` | Split the expression so `?` ends its own statement. |
| `correctness::errdefer_needs_error_result` | Add a final `error` result, or use `defer`. |
| `correctness::reserved_identifier` | Rename identifiers that start with `_logos_` or `Logos_`. |
| `correctness::duplicate_stem_filename` | Rename one of the two files that share a stem. |
| `correctness::not_an_option` | Apply `?.` and `??` to option types only. |
| `correctness::nil_deref_unproven` | Narrow with `!= nil`, select with `?.`, or unwrap with `??`. |
| `correctness::nil_to_nonnil` | Use `T?` for a pointer that may be nil. |
| `correctness::nonnil_uninitialized` | Give the pointer a value at declaration. |
| `correctness::result_before_error_check` | Read the result only after the error check. |
