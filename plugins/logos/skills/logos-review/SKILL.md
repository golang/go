---
name: logos-review
description: Review, audit, or check .lgo code or a loGo/s diff for rule violations. Use when the user says review, audit, or check loGo/s code, or asks whether .lgo code follows the compiler rules.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH.
metadata:
  version: "0.20.0"
  milestone: "M20"
---

# Review loGo/s code

Check `.lgo` code against the compiler rules. Each checklist item states the rule it checks. Report each finding with its rule ID, fix it, and end with a green build and test.

Coverage: M1 error propagation, M2 closed unions, M3 options and nil flow, M4 interface properties, M5 const composites with CTFE, read-only placement, const views, zero-copy conversions, const maps and `logos/rt.Freeze`, M6 interpolation, address-of, defaults, spread, tuples, lambdas, chains, leading-dot, yield and scope defer, M12 capabilities. Not shipped: M7–M11, M13–M21+. Do not demand those forms.

## Procedure

1. Confirm the toolchain: run `logos version`. Done when it prints `logos version go1.27`. If the command is missing, stop and tell the user to build github.com/gosuda/loGos (`cd src && ./make.bash`) and put its `bin` directory on PATH.
2. Read every changed `.lgo` file fully. Done when each file has one verdict per checklist item below.
3. Run each checklist item. Done when each finding carries its rule ID.
4. Fix every finding, then run `logos build ./...` and `logos test ./...`. Done when both exit 0.

## Checklist (M1)

1. Module kind: each module holds only `.lgo` or only `.go` packages, and no package mixes the two; imports may cross kinds in both directions. Findings: `correctness::mixed_package`, `correctness::mixed_module`.
2. `?` placement: `?` ends an assignment, an expression statement, a `defer`, or an `errdefer`, and nowhere else (no `a + f()?`, no `g(f()?)`, no `return f()?`, no `go f()?`). Finding: `correctness::misplaced_try`.
3. `?` types: the operand is a call whose last result is exactly `error`; the enclosing function returns `error` last; on error the other results are their zero values. No finding beyond items 2 and 4 when these hold.
4. `errdefer` only in functions whose last result is `error`; it also fires on panic and on `runtime.Goexit`. Finding: `correctness::errdefer_needs_error_result`.
5. Names and files: no identifier starts with `_logos_` or `Logos_`; no two files share a stem. Findings: `correctness::reserved_identifier`, `correctness::duplicate_stem_filename`.
6. cgo and publishing: cgo needs `[build] cgo` set in `logos.toml`; alias-extension files are renamed to `.lgo` before publish. Findings: `restriction::cgo_usage`, `portability::alias_extension` (warning).

## Checklist (M2)

Proven by the M2 run fixtures (`test/logos/run/union_basic.lgo`, `test/logos/run/union_assert.lgo`, `test/logos/run/switch_expr.lgo`), the types fixtures (`src/internal/types/testdata/logos/union_members.lgo`, `union_exhaustive.lgo`, `switch_expr_types.lgo`, `struct_assert.lgo`), and `src/cmd/go/testdata/script/logos_interop_union.txt`.

7. Unions: every union in a value position is closed (finite members, no `~` terms, no `comparable`); conversions, assertions, and case types name member types only; constants compared against a union convert to their default type first. Findings: `correctness::open_union_value`, `correctness::union_nonmember`.
8. Exhaustiveness: a type switch without `default` covers every member in declaration order and `nil`; a case naming a sub-union covers its members. A value switch expression has a `default` arm. Finding: `correctness::nonexhaustive_switch`, `correctness::switch_expr_needs_default`.
9. Switch-expression arms: all arms share one type (a typed arm fixes it; all-untyped arms fold by default type; `*T` plus `nil` gives `*T`). Finding: `correctness::switch_arm_mismatch`.
10. Inline assertions: each `type S struct : I` list names interfaces that `*S` implements; a struct reaches a property interface only through this list. Finding: `correctness::assertion_failed`.

## Checklist (M3)

Proven by the M3 run fixtures (`test/logos/run/option_basic.lgo`, `option_chain.lgo`, `option_result.lgo`, `option_size.lgo`, `option_nested.lgo`, `option_union.lgo`, `map_index.lgo`, `platform.lgo`), the types fixtures (`src/internal/types/testdata/logos/option_types.lgo`, `narrowing.lgo`, `error_first.lgo`, `nonnil_zero.lgo`, `map_index.lgo`), and `src/cmd/go/testdata/script/logos_interop_nil.txt`, `src/cmd/go/testdata/script/logos_interop_platform.txt`.

11. Options: `?.` and `??` apply to option types only; no plain method call, field selection, or dereference on an option without a `!= nil` narrowing proof. Findings: `correctness::not_an_option`, `correctness::nil_deref_unproven`.
12. Non-nil pointers: no `var p *T`, no nonzero `make`, no `new` that leaves a non-nil pointer without a value, no composite literal missing a non-nil field, no zero-clearing store, no `nil` assigned to a non-nil location. Findings: `correctness::nonnil_uninitialized`, `correctness::nil_to_nonnil`.
13. Narrowing and error-first: narrowing holds only in the guarded block, through `&&` chains, and after an early exit; assignment ends it; fields never narrow; address-taken and closure- or loop-assigned variables never narrow. Results of `.lgo` callees are read only after the error check; calls into other packages keep their own contracts. Finding: `correctness::result_before_error_check`.
14. Map reads: single-value `m[k]` is `V?`, so plain selection on it needs narrowing or `??`; `v, ok := m[k]`, stores, `++`, compound assignment, and `delete` keep Go meaning.

## Checklist (M4)

Proven by `test/logos/run/properties.lgo`, `src/internal/types/testdata/logos/property_iface.lgo`, `src/cmd/go/testdata/script/logos_interop_prop.txt`.

15. Properties: every struct that meets a property interface declares it with `: I`; each property has a field or method source in the declaring package; mutable properties cross the interface as pointers. Findings: `correctness::property_requires_conformance`, `correctness::property_unsatisfied`, `correctness::property_needs_pointer`.
16. Property conflicts and addresses: one property name has one type across the assert list; no `&iface.Prop`. Findings: `correctness::property_conflict`, `correctness::property_address`. No declared method starts with `Logos_`. Finding: `correctness::reserved_identifier`.

## Checklist (M5)

Proven by `test/logos/run/ctfe.lgo`, `test/logos/run/const_init_read.lgo`, `test/logos/run/rodata_write.lgo`, `test/logos/run/rodata_reflect.lgo`, `test/logos/run/rodata_named_map.lgo`, `test/logos/run/constmap_init.lgo`, `test/logos/run/const_map.lgo`, `test/logos/run/constmap_switch.lgo`, `test/logos/run/const_views.lgo`, `test/logos/run/constview_bound.lgo`, `test/logos/run/zerocopy.lgo`, `test/logos/run/freeze.lgo`, `src/internal/types/testdata/logos/ctfe.lgo`, `src/internal/types/testdata/logos/const_views.lgo`, `src/cmd/go/testdata/script/logos_interop_const.txt`.

17. Const views: no index or field assignment, `append`, `copy` into the view, `delete`, `clear`, slicing past `len`, non-const method call, or passing to a non-const parameter through a `const T` view; no assignment of a reference-carrying view to a plain type. Findings: `correctness::const_view_mutation`, `correctness::const_view_escape`.
18. CTFE: `const func` bodies hold no I/O, goroutine, package variable, `recover`, `uintptr`, map range, pointer, `defer`, `goto`, function literal, or method call; step and size limits hold; no overflow, NaN or duplicate map key, executed panic, or unsupported form. Findings: `correctness::const_func_impure`, `comptime::ctfe_step_limit`, `comptime::const_too_large`, `correctness::const_eval`.
19. Read-only placement and const maps: package-level composites need no dynamic init and are never written; const map keys are strings, numbers, bools, or structs or arrays of those; `range` assumes sorted order and lookups assume one validated probe, never speed. Findings: `comptime::constmap_key_type`, `comptime::constmap_seed_not_found`.
20. Zero-copy: `const []byte` to string conversions without an allocation need string or const-composite provenance; a view of a mutable slice converts with a copy. `rt.Freeze` results are views over a deep copy, never aliases.

## Checklist (M6)

Proven by `test/logos/run/interp.lgo`, `test/logos/run/addr_lit.lgo`, `test/logos/run/struct_default.lgo`, `test/logos/run/spread.lgo`, `test/logos/run/tuples.lgo`, `test/logos/run/lambda.lgo`, `test/logos/run/chains.lgo`, `test/logos/run/yield.lgo`, `test/logos/run/scope_defer.lgo`, `src/internal/types/testdata/logos/interp.lgo`, `src/internal/types/testdata/logos/addr_lit.lgo`, `src/internal/types/testdata/logos/struct_default.lgo`, `src/internal/types/testdata/logos/spread.lgo`, `src/internal/types/testdata/logos/tuples.lgo`, `src/internal/types/testdata/logos/lambdas.lgo`, `src/internal/types/testdata/logos/chains.lgo`, `src/internal/types/testdata/logos/yield.lgo`, `src/internal/types/testdata/logos/scope_defer.lgo`, `src/cmd/go/testdata/script/logos_m6_addr_defaults.txt`, `src/cmd/go/testdata/script/logos_tuples.txt`.

21. Interpolation and address-of: operands are strings, integers, bools, floats, `[]byte`, `error`, or `fmt.Stringer`; braces are escaped; `&` applies to any operand with a fresh variable per evaluation. Finding: `correctness::interp_type`.
22. Defaults and spread: defaults are constant and stand before the tag; `var`/`new` zero is intended where it appears; spread copies identical names and types with explicit fields winning; no Go literal is expected to carry defaults. Findings: `correctness::default_not_constant`, `suspicious::unapplied_struct_default` (warning), `correctness::spread_type_mismatch`.
23. Tuples and lambdas: tuples live in non-result positions with named access and destructuring; every lambda has a typed target with matching arity. Finding: `correctness::lambda_needs_target`.
24. Chains, yield, scope defer: chain lambdas never mutate the receiver or nest `?` where it cannot propagate; `yield` sits directly in a `Seq`/`Seq2` body with matching arity; `scope defer` cleanups run at block exit on every path. Findings: `correctness::nested_try_capture`, `correctness::misplaced_try`, `correctness::const_view_mutation`, `correctness::yield_outside_generator`.

## Checklist (M12)

Proven by `src/cmd/go/testdata/script/logos_cap_unsafe.txt` and `src/cmd/go/testdata/script/logos_cap_exec.txt`.

25. Capabilities: `logos.toml` `[capabilities]` severities are `deny`, `warn`, or `allow`; every dependency that reaches a protected sink (reflect constructors, process spawn, socket creation, `unsafe`, cgo entry) is allowlisted or removed. Findings: `capabilities::unauthorized_unsafe`, `capabilities::unauthorized_exec`, `capabilities::unauthorized_reflect`, `capabilities::unauthorized_net`, `capabilities::unauthorized_cgo`.

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
| `correctness::open_union_value` | Use a closed union (no `~` terms, no `comparable`) in value positions. |
| `correctness::union_nonmember` | Use a member type in the conversion, assertion, or case. |
| `correctness::nonexhaustive_switch` | Cover the listed members and `nil`, or add `default`. |
| `correctness::switch_arm_mismatch` | Give every switch-expression arm one common type. |
| `correctness::switch_expr_needs_default` | Add a `default` arm to the value switch. |
| `correctness::assertion_failed` | Implement the missing method on `*S`, or drop the interface from the list. |
| `correctness::not_an_option` | Apply `?.` and `??` to option types only. |
| `correctness::nil_deref_unproven` | Narrow with `!= nil`, select with `?.`, or unwrap with `??`. |
| `correctness::nonnil_uninitialized` | Give the pointer a value at declaration. |
| `correctness::nil_to_nonnil` | Use `T?` for a pointer that may be nil. |
| `correctness::result_before_error_check` | Read the result only after the error check. |
| `correctness::property_requires_conformance` | Add `: I` to the struct declaration. |
| `correctness::property_unsatisfied` | Add a field or method source for the property. |
| `correctness::property_needs_pointer` | Use the pointer type at the interface boundary. |
| `correctness::property_conflict` | Give the property one type across the interfaces. |
| `correctness::property_address` | Copy the property value instead of taking its address. |
| `correctness::const_view_mutation` | Read through the view; copy reference-free values out. |
| `correctness::const_view_escape` | Keep the view; copy reference-free values out. |
| `correctness::const_func_impure` | Move I/O, goroutines, package variables, `recover`, `uintptr`, map ranges, and pointers out of the `const func`. |
| `comptime::ctfe_step_limit` | Shrink the computation below 100,000 steps. |
| `comptime::const_too_large` | Shrink the value below 1 MB. |
| `correctness::const_eval` | Remove the overflow, NaN or duplicate key, panic, or unsupported form. |
| `comptime::constmap_key_type` | Use string, number, bool, or struct or array keys. |
| `comptime::constmap_seed_not_found` | Simplify the key set or split the map. |
| `correctness::interp_type` | Interpolate only string, integer, bool, float, `[]byte`, `error`, or `fmt.Stringer`. |
| `correctness::default_not_constant` | Give the field a constant default. |
| `suspicious::unapplied_struct_default` (warning) | Fill the literal, or accept the zero value. |
| `correctness::spread_type_mismatch` | Name the field explicitly, or align the types. |
| `correctness::lambda_needs_target` | Give the lambda a typed function target. |
| `correctness::nested_try_capture` | Hoist the `?` out of the nested chain, `??` fallback, or `?.` argument. |
| `correctness::yield_outside_generator` | Yield only directly inside an `iter.Seq`/`iter.Seq2` function. |
| `capabilities::unauthorized_unsafe` | Allowlist the dependency under `transitive_unsafe_allowlist`, or remove it. |
| `capabilities::unauthorized_exec` | Allowlist the dependency under `transitive_exec_allowlist`, or remove it. |
| `capabilities::unauthorized_reflect` | Allowlist the dependency under `reflect_allowlist`, or remove it. |
| `capabilities::unauthorized_net` | Allowlist the dependency under `transitive_net_allowlist`, or remove it. |
| `capabilities::unauthorized_cgo` | Allow cgo for the module with `[build] cgo = "allow"`, or remove the dependency. |
