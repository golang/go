---
name: logos-from-go
description: Migrate a Go module or package to loGo/s. Use when the user says migrate, convert, or port Go code to loGo/s or .lgo files.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH, plus the Go toolchain for the baseline.
metadata:
  version: "0.1.0"
  milestone: "M1"
---

# Migrate Go to loGo/s

Move one Go module to loGo/s in order: inventory, baseline, convert, fix, verify. No converter ships yet; each step below is done by hand. Fall back to eject when migration stalls.

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

## Diagnostics

| Rule | Fix |
|---|---|
| `correctness::mixed_package` | Finish the rename: no package holds both `.go` and `.lgo`. |
| `correctness::misplaced_try` | Split the expression so `?` ends its own statement. |
| `correctness::errdefer_needs_error_result` | Add a final `error` result, or use `defer`. |
| `correctness::reserved_identifier` | Rename identifiers that start with `_logos_` or `Logos_`. |
| `correctness::duplicate_stem_filename` | Rename one of the two files that share a stem. |
