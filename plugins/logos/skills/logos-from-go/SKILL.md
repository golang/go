---
name: logos-from-go
description: Migrate a Go module or package to loGo/s. Use when the user says migrate, convert, or port Go code to loGo/s or .lgo files.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH, plus the Go toolchain for the baseline.
metadata:
  version: "0.22.0"
  milestone: "M1"
---

# Migrate Go to loGo/s

Move one Go module to loGo/s in order: inventory, baseline, convert, fix, verify. Use `logos next` to automate file renaming and error-handling rewrites, with hand conversion as fallback. Fall back to eject when migration stalls.

Coverage: the mapping table covers M1 error propagation, M3 nil forms, M6 interpolation and chains, and M14 fixed-array SIMD. M2 unions, M4 properties, M5 const forms, and M12 capabilities add no mapping rows: those features have no Go equivalent to rewrite.

## Procedure

1. Inventory. List each non-standard Go dependency with `logos list -deps`. Done when the list is written down and each entry has a verdict: keep (the `.lgo` package imports it as a Go package), replace with a loGo/s module, or replace with the standard library. A module holds one kind (all `.go` or all `.lgo`), and Go and loGo/s packages import each other per import, so a dependency never blocks migration.
2. Baseline. Run `go test ./...` in the Go module and save the full output to a file. Done when the log file exists and the exit code is recorded.
3. Convert. Run `logos next` in the module root to preview changes in a unified diff without altering files. When ready, run `logos next --fix` to rewrite error returns to `?`, rename files to `.lgo`, and emit unhandled cases into `next_tasks.json`. Hand conversion remains the fallback if automated conversion leaves tasks.
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

## Go to loGo/s mapping (M14)

Proven by `SPEC.md` Sec 2.4, `src/internal/types/testdata/logos/simd_types.lgo`, `test/logos/run/simd.lgo`, and `test/logos/codegen/simd.lgo`.

Go packages have no language-level array operator syntax and use `simd/archsimd` with `GOEXPERIMENT=simd`. loGo/s supports element-wise operators (`+`, `-`, `*`, `/`, `&`, `|`, `^`) and reductions (`.sum()`, `.min()`, `.max()`, `.dot()`) on `[N]T` where `N` is a power of two, `T` is an integer or floating-point kind other than complex, and total array size is at most 64 bytes (such as `[4]float64`, `[8]float32`, `[16]uint8`, `[32]byte`, `[64]byte`).

## Diagnostics

These rules apply to both Go and `.lgo` lint targets, and diagnostics report original source positions, never ejected or generated Go.

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
| `correctness::arena_pointer_escape` | Do not store arena-allocated pointers into heap-visible locations. |
| `correctness::infinite_recursion` | Terminate recursive call chains with a base case. |
| `correctness::typed_nil_return` | Return untyped nil instead of typed nil pointer to interface result. |
| `correctness::unclosed_io_closer` | Close io.Closer resources with defer close. |
| `correctness::unbounded_channel_op` | Bound channel operations or handle cancellation with select. |
| `correctness::defer_erases_errdefer_state` | Do not overwrite error return values in defer ahead of errdefer. |
| `correctness::allow_forbidden_rule` | Do not attach an allow directive to a forbidden rule. |
| `concurrency::unsupervised_goroutine` | Move bare go f() statements into a go group block. |
| `concurrency::concurrent_multiword_race` | Protect multi-word structures across goroutines with mutex or atomic pointer. |
| `concurrency::reentrant_withlock_deadlock` | Do not call WithLock on the same Locked receiver inside its callback. |
| `slop::err_chain_reset` | Wrap errors with %w instead of formatting with %v or err.Error(). |
| `slop::discarded_error` | Check or return error instead of ignoring it, or use ? in functions returning error. |
| `complexity::cognitive_overflow` | Decompose complex functions to stay within threshold limits. |
| `perf::large_backing_array_pin` | Copy needed elements instead of slicing large backing arrays. |
| `perf::time_after_in_loop` | Do not call `time.After` inside a loop body; the compiler never hoists the timer. |
| `suspicious::side_effect_before_sort` | Avoid side-effect expressions in sort comparisons. |
| `portability::serialized_unsized_int` | Use sized integers (int32, int64) in serialized binary layouts. |
| `pedantic::exhaustive_union_switch` | Remove redundant default clause from exhaustive switch on closed union. |
