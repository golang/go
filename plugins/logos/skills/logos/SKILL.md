---
name: logos
description: Write, read, build, test, and eject loGo/s code. Use when files end in .lgo, a logos.toml exists, the user names the logos command, or the user asks about loGo/s syntax, ? error propagation, errdefer, or eject to plain Go.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH.
metadata:
  version: "0.20.0"
  milestone: "M20"
---

# loGo/s

loGo/s is Go with checked error propagation. You write `.lgo` files. The `logos` command checks them, lowers them to plain Go, and builds them with the Go 1.27 backend.

Coverage: M1 error propagation, M2 closed unions, M3 options and nil flow, M4 interface properties, M5 const composites with CTFE, read-only placement, const views, zero-copy conversions, const maps and `logos/rt.Freeze`, M6 interpolation, address-of, defaults, spread, tuples, lambdas, chains, leading-dot, yield and scope defer, M12 capabilities. Not shipped: M7–M11, M13–M21+. Do not use those forms.

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
- `logos.toml` sits at the module root. It uses a strict TOML subset (tables, `key = value`, basic and literal strings, integers, booleans, string arrays, `#` comments; 1 MB cap). Unknown tables or keys are errors, reported as `logos.toml:<line>:<col>: <message>`. Keys shipped so far: `[project] name`, `edition`, `strict_files`; `[build] cgo`, `cgo_allowlist`; `[capabilities]` severity and allowlist keys (M12, below).
- cgo is off by default in loGo/s modules (`[build] cgo = "forbid"`). Set `"restricted"` with `cgo_allowlist` globs, or `"allow"`, to use it.
- Closed unions (M2): a Go importer sees a value-used union as its methods-only interface, the declaration `logos eject` writes (terms dropped, methods kept, with a `// loGo/s union: A | B` comment). A union used both as a type constraint and as a value type has no Go form; the Go import fails with `logos build: union U is used as a type constraint and as a value type; Go cannot express both`.
- Options and platform pointers (M3): declarations from Go packages are platform declarations for `.lgo` code. Call, field, and element results from Go are platform values: usable after the error check and comparable to nil. Storing a nil Go value into a declared non-nil location panics with `logos: nil from Go value stored into non-nil *T at <file>:<line>`. A Go importer sees `*T?` as a bare pointer (nil as none) or as `logos/rt.Option[T]`, and non-nil `*T` as a bare pointer. A Go caller built by `logos` gets the same nil-store panic on direct calls (calls through function values are not checked); code built by upstream `go` from ejected output has no checks.
- Cross-package facts (M3; proven by `src/cmd/go/testdata/script/logos_facts.txt`, `src/cmd/go/testdata/script/logos_facts_nil.txt`): option positions in exported declarations travel in the `__.LOGOS` archive member beside `__.PKGDEF`; only `.lgo` importers apply them. Rebuilding a dependency without its importers fails the import with `logos: stale facts for <module> (fingerprint mismatch)`. The linker skips the member.
- Properties (M4): a Go importer sees each property element as its `Logos_GetF`/`Logos_SetF` accessor methods.
- Const views (M5): a Go importer sees the plain type; views are erased.
- Capabilities (M12; proven by `src/cmd/go/testdata/script/logos_cap_unsafe.txt`, `src/cmd/go/testdata/script/logos_cap_exec.txt`): `logos.toml` holds a `[capabilities]` table with severity keys `direct_reflect`, `transitive_reflect`, `transitive_exec`, `transitive_net`, `transitive_unsafe` (each `deny`, `warn`, or `allow`; every key defaults to `deny` except `transitive_reflect`, which defaults to `warn`) and allowlist keys `reflect_allowlist`, `transitive_exec_allowlist`, `transitive_net_allowlist`, `transitive_unsafe_allowlist` (path globs). `logos` fails the build when a dependency reaches a protected sink (reflect constructors, process spawn, socket creation, `unsafe`, cgo entry) unless the dependency is allowlisted; `warn` prints and continues. Plain `go` runs no audit.

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

## Syntax (M2)

Proven by `test/logos/run/union_basic.lgo`, `test/logos/run/union_assert.lgo`, `test/logos/run/switch_expr.lgo`, `src/internal/types/testdata/logos/union_members.lgo`, `src/internal/types/testdata/logos/union_exhaustive.lgo`, `src/internal/types/testdata/logos/switch_expr_types.lgo`, `src/internal/types/testdata/logos/struct_assert.lgo`, `src/cmd/go/testdata/script/logos_eject_union.txt`, `src/cmd/go/testdata/script/logos_interop_union.txt`.

A closed union is an interface whose type set is a finite union of non-`~` terms. Methods are allowed when every member implements them.

| loGo/s | Plain Go |
|---|---|
| `type U interface { int \| string }` | methods-only interface plus generated membership functions |
| `U(x)` | conversion; allowed when `x` is a member or a closed union of members |
| `u == 1`, `u.(int)`, `case int:` | comparison, assertion, and case on a member type |
| `switch x.(type) { case A: ...; case B: ...; case nil: }` | type switch; without `default` it covers every member and `nil` |
| `r := switch n { case 1 => "one"; default => "many" }` | immediately invoked function returning the chosen arm |
| `type S struct : I { ... }` | struct plus `var _ I = (*S)(nil)` per asserted interface |

Rules:

- An open union (`~` terms, `comparable`) in a value position is an error. A conversion, assertion, or case outside the member set is an error. Constants convert to their default type before the membership check, and `u == nil` is allowed.
- A type switch without `default` must cover every member in declaration order and `nil`; a case naming a sub-union covers its members. A value switch expression without `default` is an error.
- Switch-expression arms share one type: a typed arm fixes it, all-untyped arms fold by default type, `*T` plus `nil` gives `*T`. The operand evaluates once; only the chosen arm evaluates. An exhaustive switch that matches nothing panics.
- `type S struct : I` asserts `*S` implements each listed interface. A struct satisfies a property interface only through this list, in any package. Identifiers starting with `Logos_` are reserved.
- `x.(U)` checks membership at run time: a non-member gives `ok == false` in comma-ok form and panics as a failed `interface conversion` (`<type> is not a member of <pkg>.U`) in single-value form. A Go caller that assigns a non-member still compiles (documented hole); loGo/s code rejects it.

## Syntax (M3)

Proven by `test/logos/run/option_basic.lgo`, `test/logos/run/option_chain.lgo`, `test/logos/run/option_result.lgo`, `test/logos/run/option_size.lgo`, `test/logos/run/option_nested.lgo`, `test/logos/run/option_union.lgo`, `test/logos/run/map_index.lgo`, `test/logos/run/platform.lgo`, `src/internal/types/testdata/logos/option_types.lgo`, `src/internal/types/testdata/logos/narrowing.lgo`, `src/internal/types/testdata/logos/error_first.lgo`, `src/internal/types/testdata/logos/nonnil_zero.lgo`, `src/internal/types/testdata/logos/map_index.lgo`, `src/cmd/go/testdata/script/logos_interop_nil.txt`, `src/cmd/go/testdata/script/logos_interop_platform.txt`, `src/cmd/go/testdata/script/logos_eject_option.txt`.

| loGo/s | Plain Go |
|---|---|
| `var x int?` | presence-bearing `logos/rt.Option[int]` (value plus present flag) |
| `var u *User?` | bare `*User` with nil as none; same size as `*User` |
| `u?.Name`, `u?.Greet()` | presence check returning an option of the selected type; none propagates |
| `x ?? d` | `x` when present, else `d`; right-associative |
| `m[k]` (single value) | option of the element type; none when the key is absent |
| `var p *User = q` | non-nil pointer; nil stores and zero values are compile errors |

Rules:

- `nil` and `T` assign into `T?`; `T?` never converts back to `T`. `(T?)?` nests and stays distinct from `T?`; `T??` is rejected, write `(T?)?`. Options whose payload admits nil as a valid value (slices, funcs, chans, interfaces) use `logos/rt.Option[T]` so present-nil stays distinct from absent.
- `?.` and `??` need an option on the left. A plain method call, field selection, or dereference on an option is an error until narrowing proves `!= nil`. `?.` flattens when the selected field is already an option.
- `*T` in `.lgo` code is non-nil: `var p *T`, arrays of non-nil pointers without initializers, `make` with nonzero length, `new` of a pointer or of a type with a non-nil field, composite literals missing a non-nil field, and zero-clearing stores are errors. Assigning `nil` to a non-nil location is an error. Use `T?` for a pointer that may be nil.
- Narrowing: `if x != nil` narrows locals and parameters in the block, through `&&` chains, and after an early exit (`return`, `panic`, `break`, `continue`) on `x == nil`. Assignment ends it. Fields never narrow (copy to a local first). Taking the address, or assigning in a closure or a loop body, disables narrowing for the whole function.
- Error-first guard: for a callee declared in `.lgo`, reading a result before the `err` check is an error; the fall-through of `if err != nil { exit }` or the inside of `if err == nil {}` discharges it. Calls into other packages keep their own contracts.
- Map reads: `m[k] ?? d` types as the element type. `v, ok := m[k]`, stores, `++`, compound assignment, and `delete` keep Go meaning.

## Syntax (M4)

Proven by `test/logos/run/properties.lgo`, `src/internal/types/testdata/logos/property_iface.lgo`, `src/cmd/go/testdata/script/logos_interop_prop.txt`.

| loGo/s | Plain Go |
|---|---|
| `type C interface { const Name string; HP int; Start() error }` | interface with `Logos_GetName`, `Logos_GetHP`, `Logos_SetHP` methods |
| `type Monster struct : Creature { Name string; HP int }` | struct plus generated `Logos_GetF`/`Logos_SetF` accessors |
| `c.HP -= 10` | receiver pinned once, then getter plus setter |

Rules:

- `const F T` is a read-only property; `F T` is read-write. Each property needs a source in the declaring package: an exported field, or `F() T` (plus `SetF(T)` for read-write). Embedded struct fields satisfy by Go selector depth; an equal-depth match is ambiguous. Read-only and read-write declarations of one name merge into read-write; one name with different types conflicts.
- Satisfaction needs the `: I` list even when the fields match; without a source the declaration fails. A mutable property needs a pointer: assigning a value with a mutable property to the interface is an error.
- Reads and writes through the interface reach the struct. Compound updates evaluate the receiver once. A method call on a struct-typed property writes back. Only declaring types gain accessors.
- Taking `&iface.Prop` is an error. Identifiers starting with `Logos_` are reserved in `.lgo`.

## Syntax (M5)

Proven by `test/logos/run/ctfe.lgo`, `test/logos/run/const_init_read.lgo`, `test/logos/run/rodata_write.lgo`, `test/logos/run/rodata_reflect.lgo`, `test/logos/run/rodata_named_map.lgo`, `test/logos/run/constmap_init.lgo`, `test/logos/run/const_map.lgo`, `test/logos/run/constmap_switch.lgo`, `test/logos/run/const_views.lgo`, `test/logos/run/constview_bound.lgo`, `test/logos/run/zerocopy.lgo`, `test/logos/run/freeze.lgo`, `src/internal/types/testdata/logos/ctfe.lgo`, `src/internal/types/testdata/logos/const_views.lgo`, `src/cmd/go/testdata/script/logos_interop_const.txt`.

| loGo/s | Plain Go |
|---|---|
| `const func Fib(n int) int { ... }` then `const fib20 = Fib(20)` | evaluated at compile time; the result is a real constant |
| `const primes = []int{2, 3, 5}` | package-level `var` with `//logos:rodata`, in read-only data |
| `const scores = map[string]int{"a": 1}` | generated lookup: `switch` up to 8 keys, perfect-hash table above |
| `scores["a"] ?? -1` | probing lookup that returns an option; absent keys read as none |
| `func sum(xs const []int) int` | function over the plain type; the view is erased |
| `var v const []int = xs` | plain slice assignment; reads stay usable, writes fail |
| `var b const []byte = const []byte(s)` | zero-copy conversion from a string; allocates nothing |
| `s2 := string(b)` | zero-copy conversion back to a string; allocates nothing |
| `f := rt.Freeze(&cfg)` | `const *Config` view over a deep copy |

Rules:

- `const func` bodies stay pure: no I/O, goroutine, package-variable read or write, `recover`, `uintptr`, map range, pointer, `defer`, `goto`, function literal, or method call. Violations are `[correctness::const_func_impure]`. Evaluation stops at 100,000 steps (`[comptime::ctfe_step_limit]`) and 1 MB values (`[comptime::const_too_large]`); an overflow, a NaN or duplicate map key, an executed panic, or an unsupported form reports `[correctness::const_eval]`. Integers use target widths (`1 << 32` is exact on 64-bit). A single-value map-index read is not constant. The step-limit knob has no `logos.toml` key yet.
- Read-only placement: package-level const composites land in read-only data (`RODATA`, or relro with pointers); the backing array of a const slice is a named symbol (`go tool nm` type `R`), and a write through `unsafe` faults. `reflect.DeepEqual` and `fmt.Println` accept them. Function-scope const composites stay writable stack values. A named map type reached through a const keeps dynamic initialization.
- Const maps: up to 8 keys lower to a `switch` (no `_tab` symbol); more keys lower to bucketed perfect-hash tables (`B = ceil(N/8)`) with constant-time metadata reads plus exactly one key/value-slot probe with full-key validation. `range` visits keys in sorted order; `len` is constant; use as a plain map materializes one copy. Keys are strings, numbers, bools, or structs or arrays of those, else `[comptime::constmap_key_type]`; `-0.0` canonicalizes to `+0.0` and `NaN` keys fail. Construction tries 8 salts of 125,000 seed trials (1,000,000 total); exhaustion reports `[comptime::constmap_seed_not_found]`. Lookup cost is report-only and slower than a runtime map: never claim a speedup.
- `const T` parses in parameter, receiver, result, variable, and field types. Reads through a view stay usable and yield const views (transitive). Reference-free values copy out of a view; `nil` converts into a view only when the element admits nil.
- Each of these through a view is an error: index or field assignment, `append`, `copy` into the view, `delete`, `clear`, slicing past `len`, calling a method whose receiver lacks `const` (a `const` pointer receiver is allowed), passing to a non-const `.lgo` parameter, and passing to a standard-library parameter without a non-mutating fact. Assigning a reference-carrying view to a plain type is an error. A non-constant bound past `len` on a `const []byte` view traps at run time.
- Zero-copy provenance: string to `const []byte` and back allocates nothing, as does a `const []byte` built from constant bytes converted back to a string. A `const []byte` view of a mutable slice converts with a copy: later writes to the source leave the string unchanged.
- `rt.Freeze` deep-copies its argument and returns a const view. Later writes to the source, including through a child pointer, leave the view unchanged.

## Syntax (M6)

Proven by `test/logos/run/interp.lgo`, `test/logos/run/addr_lit.lgo`, `test/logos/run/struct_default.lgo`, `test/logos/run/spread.lgo`, `test/logos/run/tuples.lgo`, `test/logos/run/lambda.lgo`, `test/logos/run/chains.lgo`, `test/logos/run/yield.lgo`, `test/logos/run/scope_defer.lgo`, `src/internal/types/testdata/logos/interp.lgo`, `src/internal/types/testdata/logos/addr_lit.lgo`, `src/internal/types/testdata/logos/struct_default.lgo`, `src/internal/types/testdata/logos/spread.lgo`, `src/internal/types/testdata/logos/tuples.lgo`, `src/internal/types/testdata/logos/lambdas.lgo`, `src/internal/types/testdata/logos/chains.lgo`, `src/internal/types/testdata/logos/yield.lgo`, `src/internal/types/testdata/logos/scope_defer.lgo`, `src/cmd/compile/internal/syntax/testdata/logos/interp_parse.lgo`, `src/cmd/compile/internal/syntax/testdata/logos/tuples.lgo`, `src/cmd/compile/internal/syntax/testdata/logos/spread.lgo`, `src/cmd/compile/internal/syntax/testdata/logos/yield_parse.lgo`, `src/cmd/compile/internal/syntax/testdata/logos/scope_defer_parse.lgo`, `src/cmd/compile/internal/syntax/testdata/logos/addr_default_parse.lgo`, `src/cmd/go/testdata/script/logos_m6_addr_defaults.txt`, `src/cmd/go/testdata/script/logos_tuples.txt`, `src/cmd/compile/internal/syntax/scanner_leadingdot_logos_test.go`.

| loGo/s | Plain Go |
|---|---|
| `$"User {name} has {n} items"` | one buffer sized from literal bytes plus operand bounds; exactly one allocation |
| `$"{{secret}}"` | literal braces; the operand text is never a format string |
| `p := &100`, `s := &"s"`, `n := &f()` | `new(expr)`; each evaluation yields a fresh variable |
| `type Cfg struct { Port int = 8080 }` | default before the tag, filled into `.lgo` composite literals |
| `s := State{A: 10, ..src()}` | fields with identical names and types copy; explicit fields win |
| `var t (int, error)` then `a, b := t` | struct-backed tuple; packed from calls, unpacked by assignment |
| `var f func(int) int = x => x + 1` | function literal typed against its target signature |
| `xs.filter(x => x > 2).map(x => 2 * x).take(5)` | one fused loop with the receiver evaluated once |
| `batch` newline `.filter(x => ...)` | leading-dot continuation: no semicolon before `.` plus a letter |
| `func Count(n int) iter.Seq[int] { for i := range n { yield i } }` | `return func(_logos_yield func(int) bool) { ... }` with each yield as an early-exit check |
| `scope defer note("unlock")` | guarding `defer` plus inline exit stanzas at each block exit |

Rules:

- Interpolation operands are strings, any integer, bools, float32/float64, `[]byte`, `error` (calls `Error()`), or `fmt.Stringer` (calls `String()`); other types report `[correctness::interp_type]`. `{{` and `}}` are literal braces. Operands evaluate once, left to right. An unmatched `{` is `syntax error: unterminated interpolation`; an unescaped `}` is a syntax error. Lowering imports `strconv` and `unsafe` implicitly; the capabilities audit reads declared imports only.
- `&expr` takes the address of any operand and lowers to `new(expr)`. Each evaluation is fresh (`*p = 1` leaves a second `&100` unchanged).
- Field defaults are constant expressions (or composite literals of constants) written before the tag; anything else reports `[correctness::default_not_constant]`. `Cfg{}`, `&Cfg{}`, and `Cfg{Host: "x"}` get the defaults, including through other packages via facts `default` records. `var c Cfg` and `new(Cfg)` stay zero and warn `[suspicious::unapplied_struct_default]`, which never fails the build. A struct literal in a Go package gets the zero value (documented hole).
- Spread copies fields with identical names and identical types; the same name with a different type reports `[correctness::spread_type_mismatch]`, unless an explicit field of that name wins. The source evaluates once (other operands bind to a temporary); a source with no copied fields still evaluates. `s = State{A: s.B, B: s.A, ..s}` swaps through one tuple assignment with no 4 KB `memmove`. The source must be a struct; `..` in an array literal or as a field key is an error.
- Tuples live in non-result positions: `var t (int, error)`, `chan (int, error)`, `[](x int, y int)`, map values. Named access is `t.x`; `t.0` is a syntax error. Destructuring is `a, b := t`; multi-value calls pack and unpack (`fmt.Println(pair())` prints both). Tuples with slice elements are not comparable. One `(` rule serves tuples, lambdas, and parenthesized expressions with no backtracking; a single element stays parenthesized. A Go importer sees the lowered struct.
- Lambdas need a typed function target, else `[correctness::lambda_needs_target]`; arity must match. Block bodies check against the target results. Closures capture like function literals. A lone `Name` without `=>` stays an operand; `(e)` without `=>` stays a `ParenExpr`.
- Chains on slices: `filter map find any all sortBy first last get take drop sum count`. Consecutive `filter`/`map`/`take`/`drop` fuse into one loop; the receiver evaluates once and lambda parameters are const views (rewriting the parameter or the receiver reports `const_view_mutation`). `find`/`first`/`last`/`get` return `T?`; `take`/`drop`/`get` clamp (`take(-5)` is empty, `get(-1)` is none). `sortBy` is stable with `NaN` first, as `cmp.Compare` orders. Float `.sum()` uses fixed pairwise order. `?` in an expression lambda returns from the enclosing function; a `?` nested in a lambda body, in a `??` fallback, or in a `?.` call argument reports `[correctness::nested_try_capture]`; a `?` mid-block in a block lambda reports `misplaced_try`. When produced elements implement `io.Closer`, failures close the produced prefix in reverse order.
- Leading-dot continuation: in `.lgo`, a newline followed by blanks and then `.` plus a letter or `_` inserts no semicolon. `x` newline `.5` still breaks; plain Go scanning is unchanged.
- `yield e` is allowed only directly inside a function returning `iter.Seq[T]` (one value) or `iter.Seq2[K, V]` (two values), else `[correctness::yield_outside_generator]`; a closure inside the generator may not yield. Falling off the end finishes the sequence; a valued `return` acts as bare. Early `break` in the consumer stops the body.
- `scope defer f()` runs when control leaves the enclosing block: last in, first out; arguments evaluated at the statement; every exit (`return`, `break`, `continue`, labeled forms, `goto` out of the block, panic) runs it first; a loop body runs it each iteration. Lowering is an armed-flag guarding `defer` plus inline exit stanzas (not a wrapper closure, so plain `defer` tails keep firing at function exit); a `goto` past the statement leaves it disarmed.

## Best practices

- Prefer `?` over hand-written error checks. Keep the happy path flat.
- Use `errdefer` for cleanup that matters only on failure (rollback, remove temp files). Use plain `defer` for cleanup that always runs.
- Use `defer f()?` when a cleanup failure must reach the caller. Ignore a cleanup error only with a comment that says why.
- Keep functions short enough that each `?` maps to one review step.
- Prefer `?.` and `??` over manual nil checks on options. Narrow once with `!= nil`, then use the value directly.
- Declare `T?` instead of commenting that a pointer may be nil. Keep `*T` for pointers that are never nil.
- Give each closed union its members in declaration order; keep exhaustive switches in the same order.
- Read through const views; copy reference-free values out. Convert to a string at the boundary instead of carrying a mutable slice.
- Put fixed lookup tables in `const` maps; `range` visits keys sorted. Never claim they are faster than runtime maps.
- Prefer `$"..."` over `fmt.Sprintf` for inline text, and chains over hand-written filter/map loops.
- Give struct fields constant defaults instead of constructor comments; remember `var` and `new` stay zero.
- Keep `yield` bodies straight-line; keep each `scope defer` next to the acquire it undoes.
- Keep dependencies off protected sinks, or allowlist them in `[capabilities]` with a comment that says why.

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

## Eject

1. Run `logos eject -o DIR` (or `logos build --emit-go=DIR`). DIR must not exist or must be empty. Done when the command exits 0.
2. Inside DIR, run `go build ./...` and `go test ./...` with an upstream Go 1.27 toolchain. Done when both exit 0.
3. Ejected files are plain Go with your comments kept. Value-used unions become methods-only interfaces with a `// loGo/s union: A | B` comment; options become plain pointers and `logos/rt.Option[T]`; property accessors stay as ordinary methods; const views are erased. Read them with `references/lowering.md` when a line needs explanation.
4. A union used both as a type constraint and as a value type has no Go form: `logos eject` fails instead of writing broken code.

## References

| File | Load when |
|---|---|
| `references/lowering.md` | Reading ejected code, or explaining what `?`, `defer f()?`, `errdefer`, unions, options, properties, or const views do at run time |
