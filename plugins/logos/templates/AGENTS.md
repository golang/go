# AGENTS.md for a loGo/s repository

This repository holds loGo/s code (`.lgo` files) built with the `logos` toolchain.

## Module rules

- One kind per module: all `.go` or all `.lgo`. A workspace may hold both kinds.
- Imports cross kinds in both directions: a `.lgo` package may import any Go package, and a Go package may import any loGo/s package.
- Build with `logos`; plain `go` ignores `.lgo` files.
- `logos.toml` sits next to `go.mod` in the main module; unknown tables or keys are errors. Plain `go` reads no `logos.toml`; command-line `-O` and `--pedantic` override the file.

Rule IDs the compiler reports:

- M2 unions: `correctness::open_union_value`, `correctness::union_nonmember`, `correctness::union_member_method`, `correctness::nonexhaustive_switch`, `correctness::switch_arm_mismatch`, `correctness::switch_expr_needs_default`, `correctness::assertion_failed`.
- M3 options: `correctness::not_an_option`, `correctness::nil_deref_unproven`, `correctness::nonnil_uninitialized`, `correctness::nil_to_nonnil`, `correctness::result_before_error_check`.
- M4 properties: `correctness::property_requires_conformance`, `correctness::property_unsatisfied`, `correctness::property_needs_pointer`, `correctness::property_conflict`, `correctness::property_address`.
- M9 lints: `slop::err_chain_reset`, `slop::discarded_error`, `complexity::cognitive_overflow`, `perf::large_backing_array_pin`, `perf::time_after_in_loop`, `correctness::arena_pointer_escape`, `correctness::infinite_recursion`, `correctness::typed_nil_return`, `correctness::unclosed_io_closer`, `correctness::unbounded_channel_op`, `correctness::defer_erases_errdefer_state`, `correctness::allow_forbidden_rule`, `suspicious::side_effect_before_sort`, `portability::serialized_unsized_int`, `pedantic::exhaustive_union_switch`.
- M12 capabilities: `capabilities::unauthorized_unsafe`, `capabilities::unauthorized_exec`, `capabilities::unauthorized_reflect`, `capabilities::unauthorized_net`, `capabilities::unauthorized_cgo`.

## Commands

- `logos build ./...`: build. Done when it exits 0.
- `logos test ./...`: test. Done when it exits 0.
- `logos lint ./...`: lint code and reports findings.
- `logos next`: preview Go to loGo/s migration diff; `--fix` applies it.
- `logos context <pkg>#<symbol>`: print symbol signatures without bodies.
- `logos lsp`: start the language server proxy for editor integration. It serves over stdio through `gopls`, keeps lowered sources in memory, and merges compiler diagnostics.
- `logos eject -o DIR`: plain-Go copy in DIR; then `go build ./...` and `go test ./...` in DIR.

## Error propagation (M1)

| Form | Meaning |
|---|---|
| `x := f()?` | Return zero values and the error when the call's final `error` is non-nil. |
| `f()?` | Same, as a statement. |
| `defer f()?` | Join a non-nil cleanup error into the result. |
| `errdefer f()` | Run only on error return or panic. |
| `errdefer f()?` | As above, and join its error. |

`?` ends an assignment, an expression statement, a `defer`, or an `errdefer`. The call returns `error` last; the function returns `error` last.

## Agent skills

Install the official plugin: `claude plugin marketplace add gosuda/loGos`.
