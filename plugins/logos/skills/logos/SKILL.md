---
name: logos
description: Write, read, build, test, and eject loGo/s code. Use when files end in .lgo, a logos.toml exists, the user names the logos command, or the user asks about loGo/s syntax, ? error propagation, errdefer, or eject to plain Go.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH.
metadata:
  version: "0.1.0"
  milestone: "M1"
---

# loGo/s

loGo/s is Go with checked error propagation. You write `.lgo` files. The `logos` command checks them, lowers them to plain Go, and builds them with the Go 1.27 backend.

## Procedure

1. Confirm the toolchain: run `logos version`. Done when it prints `logos version go1.27`. If the command is missing, stop and tell the user to build github.com/gosuda/loGos (`cd src && ./make.bash`) and put its `bin` directory on PATH.
2. Classify every module you touch (see "Module rules"). Done when each module holds one kind and each package holds one kind.
3. Write new code with the "Syntax" tables. Done when each error path uses a shipped form and no hand-written `if err != nil { return ..., err }` remains where `?` applies.
4. Apply "Best practices". Done when each practice holds.
5. Build and test: run `logos build ./...` and `logos test ./...`. Done when both exit 0. Fix each diagnostic by its rule ID in the "Diagnostics" table.
6. When the user needs plain Go, eject (see "Eject"). Done when `go build ./...` and `go test ./...` both exit 0 inside the ejected directory.

## Module rules

- One kind per module: a loGo/s module holds only `.lgo` packages (tests are `_test.lgo`); a Go module holds only `.go` packages. A package holds one kind. A workspace may hold both kinds.
- Imports cross kinds in both directions, per import: a `.lgo` package may import any Go package, and a Go package may import any loGo/s package. A Go package sees a loGo/s package through its plain-Go form, the one `logos eject` writes.
- Build with `logos`. Plain `go` ignores `.lgo` files, so it cannot build a loGo/s package or a Go package that imports one.
- `logos.toml` sits at the module root. It uses a strict TOML subset (tables, `key = value`, basic and literal strings, integers, booleans, string arrays, `#` comments; 1 MB cap). Unknown tables or keys are errors, reported as `logos.toml:<line>:<col>: <message>`. Keys shipped so far: `[project] name`, `edition`, `strict_files`; `[build] cgo`, `cgo_allowlist`.
- cgo is off by default in loGo/s modules (`[build] cgo = "forbid"`). Set `"restricted"` with `cgo_allowlist` globs, or `"allow"`, to use it.

## Syntax (M1)

Each row shows the loGo/s form and the plain Go it replaces.

| loGo/s | Plain Go |
|---|---|
| `v := f()?` | `v, err := f()` then `if err != nil { return zero, err }` |
| `f()?` | `if err := f(); err != nil { return err }` |
| `defer f()?` | `defer` that joins a non-nil cleanup error into the result with `errors.Join` |
| `errdefer f()` | cleanup that runs only on error return or panic |
| `errdefer f()?` | cleanup that runs only on error return or panic, and joins its error |

Rules for `?`:

- The operand is a call whose last result is exactly `error`. The enclosing function returns `error` last.
- `?` ends an assignment, an expression statement, a `defer`, or an `errdefer`. Anywhere else is an error.
- On error, each other result is its zero value, and the returned error is the call's error unchanged. Assignments in the statement run before the check.
- `defer f()?` leaves a nil result alone. `errdefer` also fires on panic and on `runtime.Goexit`. Deferred calls run last in, first out. Each call's function value, receiver, and arguments are fixed where written.
- `errdefer` needs a function whose last result is `error`. A name that starts with `_logos_` or `Logos_` is an error.

## Best practices

- Prefer `?` over hand-written error checks. Keep the happy path flat.
- Use `errdefer` for cleanup that matters only on failure (rollback, remove temp files). Use plain `defer` for cleanup that always runs.
- Use `defer f()?` when a cleanup failure must reach the caller. Ignore a cleanup error only with a comment that says why.
- Keep functions short enough that each `?` maps to one review step.

## Diagnostics

| Rule | Fix |
|---|---|
| `correctness::mixed_package` | Move the `.go` or the `.lgo` files to their own package. |
| `correctness::mixed_module` | Split the module so each module holds one kind. |
| `correctness::misplaced_try` | Split the expression so `?` ends its own statement. |
| `correctness::errdefer_needs_error_result` | Add a final `error` result, or use `defer`. |
| `correctness::reserved_identifier` | Rename identifiers that start with `_logos_` or `Logos_`. |
| `correctness::duplicate_stem_filename` | Rename one of the two files that share a stem. |
| `restriction::cgo_usage` | Turn on cgo in `logos.toml`, or drop the `import "C"`. |
| `portability::alias_extension` (warning) | Rename the alias-extension file to `.lgo` before publish. |

## Eject

1. Run `logos eject -o DIR` (or `logos build --emit-go=DIR`). DIR must not exist or must be empty. Done when the command exits 0.
2. Inside DIR, run `go build ./...` and `go test ./...` with an upstream Go 1.27 toolchain. Done when both exit 0.
3. Ejected files are plain Go with your comments kept. Read them with `references/lowering.md` when a line needs explanation.

## References

| File | Load when |
|---|---|
| `references/lowering.md` | Reading ejected code, or explaining what `?`, `defer f()?`, or `errdefer` do at run time |
