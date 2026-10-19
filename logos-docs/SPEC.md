# loGo/s specification

- Status: Tailored spec, 2026-10-19, amended for first-class Go support and two-way interop (no features removed except the two import-ban diagnostics)
- Base: Go 1.27.2
- Original: `logos-docs/PLAN.md` at commit 57475ed, archived in `logos-docs/archive/`
- Source files: `.lgo` (canonical); aliases U+1F4DC (scroll), U+26A1 (lightning), and U+0295 U+25D4 U+03D6 U+25D4 U+0294 (gopher); `.go` files stay Go source files with Go language semantics.

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY follow RFC 2119. The text follows ASD-STE100.

## 0. Architectural axioms and transpilation contract

### 0.1 Upstream compatibility
The loGo/s toolchain MUST be one complete toolchain for Go packages and loGo/s packages [T-0.1][R-U1].
Go support MUST be first-class and MUST NOT be a legacy mode [T-0.1][R-U1].
Every improvement to code generation, runtime, GC, stack, SIMD, wasm, cgo, PGO, optimization level, and tooling MUST apply to Go packages and to loGo/s packages [T-0.1][R-U1].
That rule MUST hold under `logos` and under plain `go` [T-0.1][R-U1].
An improvement that this spec puts behind a `GOEXPERIMENT` flag MUST stay behind that flag as this spec states [T-0.1][R-U1].
The `go` command MAY differ from upstream Go 1.27.2 in performance, code generation, runtime, and tooling [T-0.1][R-U1].
The `go` command MUST keep Go language semantics, so every valid Go program MUST keep its Go meaning [T-0.1][R-U1].
The upstream test suite that `run.bash` runs MUST stay the regression floor [T-0.1].
The `go version` output and the `go help` output, including every help topic, MUST stay the upstream text [T-0.1].
Plain `go` MUST select improvements through flags and `GOEXPERIMENT` and MUST read no `logos.toml` [T-0.1][R-U1].
A flag that plain `go` gains MUST stay absent from the help text, so that text stays byte-identical to upstream [T-0.1][R-U1].
A default under plain `go` that differs from upstream MUST become the default only after `./run.bash` passes with it [T-0.1][R-U1].
The `logos` command MUST build Go modules with Go language semantics [T-0.1].
The `logos` command MUST apply no loGo/s type rules to Go modules [T-0.1].
The `logos build` command MUST run no lint on Go modules [T-0.1][R-U1].
The `logos lint` command MUST lint Go packages with the language-neutral rules only [T-0.1][T-102][R-U1].
A module MUST hold one kind only [T-0.1][R-U3].
A loGo/s module MUST hold only `.lgo` packages [T-0.1][R-U3].
A Go module MUST hold only `.go` packages [T-0.1][R-U3].
A workspace MAY hold Go modules and loGo/s modules side by side [T-0.1][R-U3].
A package MUST NOT mix user-authored `.go` files and `.lgo` files. Compiler-generated cgo `.go` outputs are the only exception. [T-0.1][A-CODEGEN]
The tool MUST reject a mixed package with `correctness::mixed_package` [T-0.1][R-U3].
The `correctness::mixed_package` text MUST name both file classes [T-0.1][R-U3].
The tool MUST reject a module with both kinds with `correctness::mixed_module` [T-0.1][R-U3].
The `correctness::mixed_module` text MUST name the Go packages and the loGo/s packages [T-0.1][R-U3].
A loGo/s package MAY import any Go package [T-0.1][R-U2].
A Go package MAY import any loGo/s package, except as the next sentence states [T-0.1][R-U2].
An import across kinds MUST NOT be an error, except the import of a dual-use union that Sec 2.3 names [T-0.1][R-U2].
Interop MUST work per import and MUST NOT need a package or a module that holds both kinds [T-0.1][R-U3].
A Go declaration that a `.lgo` file imports MUST be a platform declaration per Sec 2.2 [T-0.1][T-202][A-STD][R-U2].
See Sec 0.3 for the Go-facing signatures that a Go importer sees.
A workspace root `logos build ./...` MUST build every Go module and every loGo/s module of the workspace [T-0.1][R-U2].
The `logos` command alone MUST select the `.lgo` language. This covers file recognition, loGo/s syntax, loGo/s type rules, and loGo/s lints [T-0.1][A-LGOSEL].
Plain `go` MUST ignore `.lgo` files [T-0.1][A-LGOSEL].
Plain `go` MUST report the upstream `no Go files` result for a loGo/s-only package [T-0.1][A-LGOSEL].
A plain `go` build of a Go package that imports a loGo/s package MUST fail with a load error that names the imported package, because plain `go` ignores `.lgo` files [T-0.1][A-LGOSEL].
Packages from `golang.org/x` and other third-party Go modules MUST count as Go packages that `.lgo` code MAY import [A-XMOD][R-U2].
Code generators MUST emit `.lgo` in loGo/s modules [A-CODEGEN].
A loGo/s package MUST NOT hold a generated `.go` file except cgo outputs [A-CODEGEN].
A generator MAY write `.go` files into a Go package that a loGo/s package imports [A-CODEGEN][R-U2].
Project tools MUST use `tool` directives in `go.mod`, never a Go file in a loGo/s module [A-MODKIND][R-U3].

### 0.2 No new global keywords
The language MUST add no new global keywords [T-0.2].
See Sec 5.1 for contextual words and `const` positions.
Identifiers that start with `_logos_` or `Logos_` MUST be rejected in `.lgo` [T-203.2].
The compiler MUST report such identifiers with `correctness::reserved_identifier` [T-203.2].

### 0.3 Pure Go output through eject
The command `logos eject -o DIR` MUST write a complete pure-Go copy of the main module to DIR [T-0.3].
The command `logos build --emit-go=DIR` MUST be an alias of `logos eject -o DIR` [T-0.3].
DIR MUST NOT exist, or DIR MUST be empty [T-0.3].
The tool MUST refuse a DIR that exists and holds files [T-0.3].
Each `.lgo` file MUST become a `.go` file [T-0.3].
The output MUST keep all comments from the source [T-0.3].
Each non-standard loGo/s dependency module MUST be ejected too [T-0.3].
Each ejected dependency MUST land in `DIR/_logos_deps/<module path>@<version>/` [T-0.3].
The output `go.mod` MUST carry a `replace` line for each ejected module [T-0.3].
A Go dependency module MUST stay a `require` entry of the output `go.mod` [T-0.3][R-U2].
The support sources MUST become one shared module `github.com/gosuda/loGos/rt` [T-0.3].
The shared module MUST land in `DIR/_logos_deps/rt` [T-0.3].
The output MUST carry a `replace` line for the shared module [T-0.3].
The tool MUST remove `import "comptime"` during eject [T-0.3].
The output MUST NOT include a `//go:build !logos` line [T-0.3][R-U3].
The output MUST NOT emit `.gen.go` file names [T-0.3].
The output MUST NOT emit `_logos_strconv` imports [T-0.3].
The output MUST NOT emit `_logos_errors` imports [T-0.3].
Among support imports, lowered files MUST import only `_logos_rt "logos/rt"`, plus `_logos_unsafe "unsafe"` where needed, in addition to standard library imports [T-0.3].
The tool MUST NOT write ejected files into the source tree [T-0.3].
Ejected code MUST build with an upstream Go 1.27 toolchain [T-0.3].
A Go package that imports a loGo/s package MUST see each exported declaration with the Go-facing signature that `logos eject` emits for it [T-0.3][R-U2].

### 0.4 Design axiom for explicit code
Signatures and call sites MUST show side effects, mutability contracts, nilability states, and error paths [T-0.4].
The compiler MUST reject dangerous edge cases through type rules and `logos lint` rules [T-0.4].

## 1. Tier 1 specification (toolchain, linter, desugaring)

### 1.1 RFC-101: File loader, Unicode normalization, and cgo policy
The loader MUST recognize `.lgo` and the three alias extensions only in `logos` mode [T-101.1].
The loader MUST strip trailing U+FE0E and U+FE0F from the extension before matching [T-101.1].
NFC normalization MUST be the identity on all four extension strings [T-101.1].
The loader MUST run no normalizer for extensions [T-101.1].
Two files in one package whose names differ only by alias extension MUST fail the build [T-101.1].
The tool MUST report that failure with `correctness::duplicate_stem_filename` [T-101.1].
Module zips MUST reject alias extensions as non-letter symbols for publication [T-101.1].
The tool MUST report warning `portability::alias_extension` for each alias file when the main module path first element holds a dot [T-101.1].
When the main module is a loGo/s module, `CGO_ENABLED` MUST default to 0 [T-101.2].
The key `logos.toml [build] cgo` MUST accept `forbid`, `restricted`, or `allow` in a loGo/s main module [T-101.2].
The default for `[build] cgo` MUST be `forbid` in a loGo/s main module [T-101.2].
The value `restricted` MUST enable cgo only for packages that match `cgo_allowlist` globs [T-101.2].
In a loGo/s main module, a package with `import "C"` outside the allowlist MUST fail with `restriction::cgo_usage` [T-101.2].
In a loGo/s main module with no `logos.toml`, a package with `import "C"` MUST fail with `restriction::cgo_usage` and text `cgo is disabled` [T-101.2].
In a loGo/s main module, an explicit `CGO_ENABLED` in the environment MUST win over the default only when the `cgo` key is absent [T-101.2]. An explicit `[build] cgo = "forbid"` MUST stay enforced against the environment [T-101.2]. An explicit `[build] cgo = "restricted"` allowlist refusal MUST stay enforced against the environment [T-101.2].
When cgo is enabled, `CC` is unset, and the default C compiler is missing, `logos` and plain `go` MUST use `zig cc -target <zig-arch>-<zig-os>` for every cgo package, Go or loGo/s, when `zig` is in PATH [T-101.2][R-U1].
The `-musl` suffix MUST apply only for Linux targets [T-101.2].
The `[build] cgo` policy (`forbid`, `restricted`, `allow`) and its `forbid` default MUST apply to loGo/s main modules only [T-101.2][R-U1].
A Go main module MUST keep the upstream default for `CGO_ENABLED` under `logos`, and a `cgo` key in its `logos.toml` MUST have no effect [T-101.2][R-U1].
Plain `go` MUST keep the upstream default for `CGO_ENABLED` [T-101.2].
With cgo off, `net` and `os/user` in the standard library MUST use their pure-Go paths [T-101.2].

### 1.2 RFC-102: Compiler linter and rule set
The `logos lint` check MUST run inside the compiler on `.lgo` packages after the first type pass for syntax and type rules [T-102].
The `logos lint` check MUST run the language-neutral syntax and type rules on Go packages at the same point [T-102][R-U1].
The `logos lint` check MUST run inside escape analysis for escape rules, on Go packages and on `.lgo` packages [T-102][R-U1].
Lint overhead carries target: 2% of compile time, measured by BenchmarkLintOverhead [T-102].
`logos lint` MUST lint Go packages with the language-neutral rules only [T-102][R-U1].
A rule MUST count as language-neutral when a Go package can trigger it with Go syntax and types alone and the code it guards exists in Go packages. A loGo/s form inside its detection, such as a group child, a const view, or `?`, MUST stay inactive in Go [T-102][R-U1].
The language-neutral rules MUST be `correctness::typed_nil_return` (in Go, the `(*T)(nil)` form only), `correctness::arena_pointer_escape`, `correctness::infinite_recursion`, `correctness::unclosed_io_closer`, `correctness::unbounded_channel_op`, `concurrency::concurrent_multiword_race`, `concurrency::reentrant_withlock_deadlock`, `slop::err_chain_reset`, `slop::discarded_error`, `slop::todo_placeholder`, `complexity::cognitive_overflow`, `perf::large_backing_array_pin`, and `perf::time_after_in_loop` [T-102][R-U1].
The rules `correctness::nil_deref_unproven`, `correctness::defer_erases_errdefer_state`, `concurrency::unsupervised_goroutine`, `suspicious::unapplied_struct_default`, `suspicious::side_effect_before_sort`, `portability::serialized_unsized_int`, `portability::alias_extension`, and `pedantic::exhaustive_union_switch` MUST NOT apply to Go packages, because each depends on loGo/s syntax, types, or generated code [T-102][R-U1].
The rule `portability::serialized_unsized_int` guards the generated binary layout of Sec 12.1, which Go packages do not get [T-102][R-U1].
A Go package MUST get the default group severities, the `logos.toml [lints]` table, and the `//logos:allow(<rule>)` directive of this section, with the cap that the next line states [T-102][R-U1].
A language-neutral rule whose default severity is `forbid` MUST run at `deny` for a Go package [T-102][R-U1].
`//logos:allow(<rule>)` MUST silence such a rule in a Go package [T-102][R-U1].
A loGo/s package MUST keep the `forbid` severity of such a rule [T-102].
`logos build` MUST run no lint on a Go package [T-102][R-U1].
Third-party modules of either kind MUST get no code lints [T-102][R-U1].
Capability checks MUST apply to every module in the build [T-102].
`forbid` MUST NOT be lowered by configuration [T-102].
In a loGo/s package, `forbid` MUST ignore `//logos:allow(<rule>)` [T-102].
`deny` MUST be an error unless `//logos:allow(<rule>)` precedes the statement or declaration [T-102].
`warn` MUST print a warning and MUST NOT fail the build [T-102].
`allow` MUST stay silent [T-102].
In a loGo/s package, a `//logos:allow` directive on a `forbid` rule MUST itself be an error with `correctness::allow_forbidden_rule` [T-102].
In a Go package, a `//logos:allow` directive on a rule that the Go cap lowers from `forbid` to `deny` MUST silence that rule [T-102][R-U1].
The default group severities MUST be `correctness` forbid, `concurrency` forbid, `slop` deny, `complexity` deny, `perf` warn, `style` warn, `suspicious` warn, `portability` warn, `pedantic` allow [T-102].
The `logos.toml [lints]` table MUST control group severities [T-102].
The `logos.toml [lints.thresholds]` defaults MUST be 3 for nesting depth, 15 for cognitive complexity, 60 for function lines, 4 for function parameters, 4096 for comptime nodes, and 100000 for CTFE steps [T-102].
Configuration MAY raise `max_comptime_ast_nodes` and `max_ctfe_steps` above their defaults [T-102].
Validation and boundary tests MUST use the selected limits [T-102].
`logos build` and `logos lint` MUST accept `--pedantic` with default false [T-102].
The `--pedantic` flag MUST override `logos.toml` [T-102].
A default build MUST stay silent for the pedantic group [T-102].
`logos lint ./...` MUST report findings without building binaries [T-102].
`--json-diagnostics` MUST print one JSON object per line on stderr for each compiler and lint diagnostic in `.lgo` files and in Go files with schema version `1.0` [T-102][R-U1].
The compiler MUST enforce `correctness::nil_deref_unproven` as `forbid` against use of a nullable value before its guard [T-102].
The compiler MUST enforce `correctness::typed_nil_return` as `forbid` against return of a concrete nil pointer as an interface or `error` result without a dominating nil check [T-102].
The compiler MUST enforce `correctness::arena_pointer_escape` as `forbid` against storage of a stack or arena pointer outside its frame through an unsafe address [T-102].
The compiler MUST enforce `correctness::defer_erases_errdefer_state` as `forbid` against assignment of nil to the error result inside a plain deferred literal in a function with `errdefer` [T-102].
The compiler MUST enforce `correctness::infinite_recursion` as `forbid` against a function whose every path from entry calls itself with identical arguments before any return [T-102].
The compiler MUST enforce `concurrency::concurrent_multiword_race` as `forbid` against sharing of a mutable multi-word value with a child block plus a write on either side, unless the value is a const view or a `logos/sync` type [T-102].
The compiler MUST enforce `concurrency::reentrant_withlock_deadlock` as `forbid` against a `WithLock` call on the same receiver inside its own closure call graph [T-102].
The compiler MUST enforce `concurrency::unsupervised_goroutine` as `forbid` against a `go` statement in `.lgo` outside a `go group` block [T-102].
The compiler MUST enforce `slop::err_chain_reset` as `deny` against formatting of an `error` operand with a verb other than `%w` or against passing of `err.Error()` [T-102].
The compiler MUST enforce `slop::discarded_error` as `deny` against dropping of an `error` result, except callees with an always-nil-error fact and benign writers to standard output, standard error, byte buffers, and string builders [T-102].
The compiler MUST enforce `slop::todo_placeholder` as `deny` against TODO or FIXME comments and unimplemented panic stubs in non-test files [T-102].
The compiler MUST enforce `correctness::unclosed_io_closer` as `deny` against a local `io.Closer` with no deferred close and no transfer out of the function [T-102].
The compiler MUST enforce `correctness::unbounded_channel_op` as `deny` against a channel operation outside a select with a done case, unless the channel is locally made with constant capacity and send count stays within capacity [T-102].
The compiler MUST enforce `complexity::cognitive_overflow` as `deny` against nesting depth above 3, parameter count above 4, line count above 60, or cognitive score above 15 [T-102].
The cognitive score MUST be the sum over branch, loop, switch, select, conditioned case, goto, and labeled jump points of one plus nesting depth, plus one per change of boolean operator in a chain [T-102].
The compiler MUST enforce `suspicious::unapplied_struct_default` as `warn` against zero declaration of a type with field defaults [T-102].
The compiler MUST enforce `suspicious::side_effect_before_sort` as `warn` against a chain lambda with a call of unknown purity before `.sortBy()` [T-102].
The compiler MUST enforce `perf::large_backing_array_pin` as `warn` against escape of a subslice or substring of a backing array above 64 KB [T-102].
The compiler MUST enforce `portability::serialized_unsized_int` as `warn` against an `int` or `uint` field in a struct with JSON tags [T-102].
The compiler MUST enforce `portability::alias_extension` as `warn` for each alias-extension file per the module-path rule [T-101.1][T-102].
The compiler MUST enforce `perf::time_after_in_loop` as `warn` against `time.After` inside a loop body [T-102].
The compiler MUST enforce `pedantic::exhaustive_union_switch` as `allow` against a `default` clause in a switch over a closed union [T-102].
JSON output MUST offer autofix candidates for optional chaining with default, early-return guard, split of a misplaced `?`, verb fix to `%w`, and addition of `?` in an error-returning function [T-102].
JSON output for a Go package MUST offer only the verb fix to `%w`, because the other candidates name loGo/s syntax [T-102][R-U1].

### 1.3 RFC-103: Context and next tools
The command `logos context <pkg>#<Symbol>` MUST print the declaration plus one level of mentioned signatures with no function bodies [T-103].
`logos context` output MUST hold at most 100 lines [T-103].
`logos context` MUST serve `.lgo` and `.go` packages [T-103].
`logos context` MUST report an unknown symbol with text that names the package and the missing symbol [T-103].
`logos context` speed carries target: 50 ms, measured by BenchmarkContext [T-103].
The command `logos next` MUST convert a whole Go module to a loGo/s module [T-103][R-U3].
`logos next` MUST use byte-span splicing and MUST keep every comment and build line byte-identical outside rewrites [T-103].
`logos next` MUST keep each non-standard Go import of the module as an import of a Go package [T-103][R-U2].
`logos next` MUST change nothing while it only prints a diff [T-103].
`logos next --fix` MUST rename converted files and MUST rewrite matched patterns [T-103].
`logos next` MUST leave complex refactors unchanged and MUST list them in `next_tasks.json` [T-103].
`logos next` MUST rewrite an index-range append loop with a pure body to a map chain [T-103].
`logos next` MUST leave a loop with jump control or side effects unchanged and MUST list it in `next_tasks.json` [T-103].
No `logos fmt` command exists [T-103].
No command except `logos next` MUST rewrite `.lgo` files [T-103].

### 1.4 RFC-104: Syntactic sugar and desugaring
Error propagation with `?` MUST be a postfix operator [T-104.1].
`?` MUST appear only as the outermost operator of a single right-hand call in `=` or `:=`, an expression statement, a `defer` call, an `errdefer` call, or the body of an expression lambda [T-104.1].
Any other use of `?` MUST fail with `correctness::misplaced_try` and text `? must end an assignment, expression statement, defer, or errdefer` [T-104.1].
The operand MUST be a call whose last result type is identical to `error` [T-104.1].
A named type whose underlying type is `error` MUST NOT satisfy the operand rule [T-104.1].
The enclosing function MUST have a last result type identical to `error` [T-104.1].
`x := f()?` MUST bind the call results minus the error [T-104.1].
`x := f()?` where the call returns only `error` MUST fail [T-104.1].
Zero remaining results MUST be allowed only in an expression statement, a defer, or an errdefer [T-104.1].
Normal `:=` scope rules MUST apply to `x := f()?` with no new variable [T-104.1].
On error, every non-error result MUST take its zero value [T-104.1].
The returned error MUST be the call error value unchanged [T-104.1].
Assignments in the statement MUST run before the error check [T-104.1].
The error path MUST record the site through `logos/rt` when built by `logos` [T-104.1].
The error path MUST be a no-op for traces in ejected code [T-104.1].
`logos/rt.ErrorTrace()` MUST return recorded sites in propagation order for the calling goroutine [T-104.1].
A different error value MUST reset the trace [T-104.1].
The trace MUST keep the last 8 sites [T-104.1].
```go
func double(s string) (int, error) {
	n := strconv.Atoi(s)?
	return 2 * n, nil
}
```
`defer f()?` MUST join a non-nil result into the error result [T-104.2].
The join MUST keep the existing error when nil and MUST use errors join semantics otherwise [T-104.2].
`errdefer` MUST be allowed only in functions whose last result is `error` [T-104.2].
Other uses MUST fail with `correctness::errdefer_needs_error_result` [T-104.2].
`errdefer` MUST run when the frame panics, exits through Goexit, or holds a non-nil error result at pop time [T-104.2].
A typed-nil error MUST count as non-nil for `errdefer` [T-104.2].
The function, receiver, and arguments of a deferred call MUST be evaluated at the statement [T-104.2].
A panic while evaluating return values MUST count as failure for `errdefer` [T-104.2].
Deferred calls MUST run in LIFO order across `defer`, `defer?`, and `errdefer` [T-104.2].
```go
func copyFile(dst string, src string) (error) {
	f := os.Open(src)?
	defer f.Close()?
	return nil
}
```
`?.` and `??` MUST operate on option types `T?` [T-104.3].
A map index in `.lgo` MUST yield `V?` [T-104.3].
The form `v, ok := m[k]` MUST keep Go meaning [T-104.3].
`??` MUST bind less tightly than `||` [T-104.3].
`??` MUST group to the right [T-104.3].
See Sec 2.2 for `?.` evaluation and option operand checks.
```go
func greet(u *User?) (string, error) {
	return u?.Name ?? "Guest", nil
}
```
`$"..."` MUST accept operands of type string, any integer, bool, float32, float64, `[]byte`, `error`, or `fmt.Stringer` [T-104.4].
Any other operand type MUST fail with `correctness::interp_type` [T-104.4].
`{{` and `}}` MUST be literal braces [T-104.4].
Operands MUST be evaluated once from left to right [T-104.4].
The lowering MUST use one allocation with bound of literal bytes plus `len` of string and byte operands plus 20 per integer plus 24 per float plus 5 per bool [T-104.4].
The lowering MUST never regrow the buffer [T-104.4].
Operand text MUST NOT be read as a format string [T-104.4].
An unterminated `{` MUST be a syntax error [T-104.4].
```go
func msg(name string, n int) string {
	return $"User {name} has {n} items"
}
```
`&expr` for a non-addressable operand MUST type as a pointer [T-104.5].
Each `&expr` evaluation MUST yield a fresh variable [T-104.5].
A struct field default MUST have form `F T = constExpr` before the tag [T-104.5].
Defaults MUST apply in composite literals of the type, including address forms and partial literals [T-104.5].
A Go struct literal of an exported loGo/s struct MUST get zero values and MUST NOT get the field defaults; this is a documented hole [T-104.5][R-U2][A-GOBOUNDARY].
`var x T` and `new(T)` MUST stay zero and MUST trigger `suspicious::unapplied_struct_default` [T-104.5].
A non-constant default MUST fail with `correctness::default_not_constant` [T-104.5].
Spread `T{..src}` MUST copy fields with identical names and identical types [T-104.5].
The same name with a different type MUST fail with `correctness::spread_type_mismatch` [T-104.5].
Explicit fields MUST win over spread fields [T-104.5].
The source MUST be evaluated once [T-104.5].
Self-update `x = T{f: e, ..x}` MUST lower to one tuple assignment of the explicit fields [T-104.5].
Lambdas MUST have expression form `x => expr`, block form `x => { stmts }`, or tuple-parameter form `(a, b) => ...` [T-104.6].
A lambda MUST appear only where a function value of known parameter types is expected [T-104.6].
Parameter types MUST come from the target type [T-104.6].
A lambda without a target function type MUST fail with `correctness::lambda_needs_target` [T-104.6].
Slices MUST offer `filter`, `map`, `find`, `any`, `all`, `sortBy`, `first`, `last`, `get`, `take`, `drop`, `sum`, and `count` chain methods [T-104.6].
`find`, `first`, `last`, and `get` MUST return `T?` [T-104.6].
`take`, `drop`, and `get` MUST clamp indices without panic [T-104.6].
`sortBy` MUST be a stable sort by `cmp.Compare` of keys over ordered key types [T-104.6].
Float `.sum()` MUST use fixed pairwise order with no FMA contraction [T-104.6].
Float `.sum()` MUST be bit-identical across x86_64 and ARM64 builds [T-104.6].
Chained calls MUST fuse into one loop with the receiver evaluated once [T-104.6].
Lambda parameters in chains MUST be const views [T-104.6].
Mutation of the receiver inside a chain lambda MUST fail compilation [T-104.6].
`?` in an expression-lambda body MUST propagate from the enclosing function [T-104.6].
When produced elements implement `io.Closer`, partial failure MUST close already produced elements in reverse order [T-104.6].
A newline followed by spaces or tabs then `.` plus a letter or `_` MUST insert no semicolon in `.lgo` [T-104.6].
```go
func big(xs []int) []int {
	return xs.filter(x => x > 0).map(x => 2 * x).take(5)
}
```

### 1.5 RFC-105: Optimization tiers and synthetic PGO
`logos build` and `go build` MUST accept `-O=0`, `-O=1`, `-O=2`, `-O=3`, `-O=s`, and `-O=z` [T-105][R-U1].
A level MUST apply to every package of the build, Go or loGo/s [T-105][R-U1].
Under `logos`, the default level MUST come from `logos.toml [build] opt_level` with default `O2` [T-105].
Under plain `go`, the default level MUST be `O1` [T-105][R-U1].
The plain `go` default level MUST stay `O1` until `./run.bash` and the G6 comparison of `IMPLEMENTATION.md` both pass with `O2` as the default [T-105][T-0.1].
`[build] opt_level` MUST accept `O0`, `O1`, `O2`, `O3`, `Os`, and `Oz` only [T-105].
A bare spelling such as `2` MUST fail validation like an unknown level [T-105].
An invalid level MUST fail with text that lists the six accepted levels [T-105].
O0 MUST pass debug flags for no inline and no debug-info loss protection [T-105].
O1 MUST be the upstream baseline [T-105].
O2 MUST set inline budget 120 and variable-make stack buffer 256 bytes with standard synthetic PGO [T-105].
O3 MUST set inline budget 400 with 80 inside recursive cycles, clone specialization, variable-make buffer 4096 bytes, and aggressive synthetic PGO [T-105].
Os and Oz MUST set inline budget 15 with linker identical code folding [T-105].
Synthetic PGO MUST mark `?` error blocks, panics, and `os.Exit` paths unlikely through the intrinsic `logos/rt.cold()` [T-105].
A Go package MUST get the same marks for `os.Exit` paths, and for panic paths where the aggressive level marks them, with no source change and no new import [T-105][R-U1].
A Go package MUST NOT get a mark from a guess about `if err != nil` blocks, because Go has no `?` form to name an error block [T-105][R-U1].
The backend scale MUST be ternary with no numeric probability [T-105].
A cold successor MUST be placed last inside its function, never outlined to the tail of text [T-105].
Loop hotness MUST come from real profiles only, never from a synthetic power formula [T-105].
Generic code sharing MUST stay upstream shaped behavior with linker choice of one duplicate [T-105].
O3 MUST add no escape-analysis or prove-pass work budget [T-105].
`target_cpu` MUST accept `deterministic`, `baseline`, or `native` [T-105].
`target_cpu` MUST apply to every package that `logos` builds, Go or loGo/s [T-105][R-U1].
Plain `go` MUST read no `target_cpu` and MUST keep the upstream `GOAMD64` and `GOARM64` defaults [T-105][R-U1].
`deterministic` MUST set GOAMD64 v3 and GOARM64 v8.2 unless the environment already sets them [T-105].
`baseline` MUST set GOAMD64 v1 and GOARM64 v8.0 [T-105].
`native` MUST pick the host level [T-105].
The default binary MUST NOT run on x86 CPUs older than v3 [T-105].

## 2. Tier 2 specification

This section defines the loGo/s type system: compile-time constants, pointers and options, unions and property interfaces, tuples and fixed-array operators, comptime expansion, and group concurrency.

### 2.1 RFC-201: Const composites, CTFE, const views, and const maps

A `const` declaration MUST accept a composite literal (a struct, an array, a slice, or a map) and a call to a `const func`. [T-201]

```lgo
const primes = []int{2, 3, 5}
const httpPort = 8080
```

A `const func` MUST NOT perform I/O, spawn a goroutine, read a mutable package variable, call `recover`, convert a pointer to `uintptr`, range over a dynamic map, or use a pointer. [T-201]

The compiler MUST evaluate `const` composites and `const func` calls at compile time with target integer widths, a step limit from `max_ctfe_steps` with default 100,000, and a value cap of 1 MB. [T-201][T-102]

A run that exceeds the selected `max_ctfe_steps` limit MUST fail with `[comptime::ctfe_step_limit]`. [T-201][T-102]

Configuration MAY raise `max_ctfe_steps` above its default. [T-102]

Validation and boundary tests MUST use the selected limit. [T-102]

A constant value above 1 MB MUST fail with `[comptime::const_too_large]`. [T-201]

A `const func` that breaks a purity rule MUST fail with `[correctness::const_func_impure]` naming the reason. [T-201]

Pointer-free constant data MUST live in read-only memory. [T-201]

Constant data with pointers MUST live in read-only-after-relocation memory. [T-201]

A write to a constant composite through `unsafe` MUST fault at run time. [T-201]

`reflect.DeepEqual` and `fmt.Println` MUST accept constant composites. [T-201]

A package with a 1,000-key constant map MUST add no heap bytes at init, with its table symbol in read-only memory. [T-201][T-PERF]

A zero-time init claim is not measurable; the check above replaces it. [T-201][T-PERF]

See Sec 9.1 for const map lowering, iteration, and float keys.

`const []T`, `const *T`, and `const map[K]V` MUST form a qualifier checked at compile time and erased before code generation. [T-201]

A Go importer MUST see a constant view as its plain type: `const []T` as `[]T`, `const *T` as `*T`, and `const map[K]V` as `map[K]V`. [T-201][T-Sec6][R-U2]

A Go importer MUST see an exported `const` composite as a package-level variable of its plain type and a `const func` as a plain function. [T-201][T-Sec6][R-U2]

A write from Go into such a variable lies outside the loGo/s contract. The Go compiler does not report it, and the write MAY fault at run time because the data can sit in read-only memory. This is a documented hole. [T-201][T-Sec6][R-U2][A-GOBOUNDARY]

An index assignment through a constant view MUST fail with `[correctness::const_view_mutation]` naming the operation and the expression. [T-201]

A field assignment through `const *T` MUST fail with the same diagnostic. [T-201]

`append` to, `copy` into, `delete` from, and `clear` of a constant view MUST fail with the same diagnostic. [T-201]

A slice expression with a bound above `len` (such as `b[:cap(b)]`) MUST fail with the same diagnostic, with a run-time check when the bound is not constant. [T-201]

A call to a method with a pointer receiver that lacks `const` MUST fail with the same diagnostic. [T-201]

A method with a `const` pointer receiver (such as `func (r const *T) M()`) MUST be allowed. [T-201]

Passing a constant view to a non-constant parameter of an `.lgo` function MUST fail with the same diagnostic. [T-201]

Passing a constant view to a Go-mode parameter (the standard library or any Go package) MUST be allowed only when the parameter carries fact `ParamMutated=0`. [T-201][R-U2]

Assigning `const []*T` to `[]*T` MUST fail with `[correctness::const_view_escape]`. [T-201]

A reference-free constant view MUST be assignable to its plain type. [T-201]

A plain value MUST be assignable to its constant view. [T-201]

A `for _, v := range` loop over a constant slice MUST yield constant elements. [T-201]

Converting `string` to `const []byte` MUST be zero-copy. [T-201]

Converting `const []byte` to `string` MUST be zero-copy when provenance is a string or a constant composite. [T-201]

The same conversion MUST copy for any other provenance. [T-201]

`logos/rt.Freeze` MUST deep-copy its argument and return a `const` view. [T-201]

A later write to the source (or to a child object it points to) MUST NOT change the frozen view. [T-201]

### 2.2 RFC-202: Non-nil pointers, option types, and nil flow

`*T` in `.lgo` MUST be non-nil. [T-202]

`*T?` MUST denote an optional pointer. [T-202]

Assigning `nil` to `*T` MUST fail with `[correctness::nil_to_nonnil]`. [T-202]

`var p *T` MUST fail compilation. [T-202]

An array type with non-nil element pointers (such as `[3]*T`) MUST fail without initializers. [T-202]

`make([]*T, n)` with a non-constant `n` or a constant above 0 MUST fail compilation. [T-202]

`make([]*T, 0, n)` MUST be allowed. [T-202]

`new(S)`, `var s S`, and `S{}` MUST fail when `S` has a non-nil pointer field without an initializer, with `[correctness::nonnil_uninitialized]` naming the field and its type. [T-202]

Zero-clearing through `*p = S{}` MUST fail with the same diagnostic. [T-202]

`var t T` for a type parameter MUST be allowed. [T-202]

Pointers from Go-mode declarations (the standard library, every other Go package, and cgo-generated declarations) MUST be platform pointers. [T-202][A-STD][R-U2]

A `.lgo` file MAY import any Go package, and each declaration that it imports MUST be a platform declaration. [R-U2]

A type-parameter value instantiated with a non-nil pointer MUST also be a platform pointer. [T-202]

Dereferencing a platform pointer MUST be allowed. [T-202]

Comparing a platform pointer with `nil` MUST be allowed. [T-202]

A store of a platform value into a declared non-nil location MUST wrap the stored expression in `logos/rt.MustNonNil` with the source position. [T-202]

A nil value at such a store MUST panic with `logos: nil from Go value stored into non-nil *T at <file>:<line>`. [T-202]

`if x != nil` MUST narrow `x` from `T?` to `T` in the guarded block. [T-202]

An early exit of the form `if x == nil { return }` (also with `panic`, `break`, or `continue`) MUST narrow `x` after the statement. [T-202]

An `&&` chain with a nil test MUST narrow `x` in the rest of the chain. [T-202]

A later assignment MUST end narrowing. [T-202]

Taking the address of `x` MUST disable narrowing for `x`. [T-202]

Assigning `x` in a closure MUST disable narrowing for `x`. [T-202]

A nil test on a struct field MUST NOT narrow the field. [T-202]

Code MUST copy the field to a local first. [T-202]

For a callee declared in `.lgo`, reading a result before the `err == nil` proof MUST fail with `[correctness::result_before_error_check]`. [T-202]

A Go-mode callee (the standard library or any Go package, such as `io.Reader.Read`) MUST keep its own contract. [T-202][R-U2]

`nil` MUST be assignable to every `T?`. [T-202]

`T` MUST be assignable to `T?`. [T-202]

`T?` MUST NOT be assignable to `T`. [T-202]

`*T?` where `*T` is non-nil MUST lower to `*T` with nil as none and no extra storage. [T-202]

When nil is a valid payload of `T`, `T?` MUST carry explicit presence separate from payload nil. [T-202]

An interface-backed union option with no nil-capable member MUST use the interface nil value for absence. [T-202]

For every other `T` outside those two cases, `T?` MUST lower to `logos/rt.Option[T]`. [T-202]

A present nil payload MUST read as some with nil payload, distinct from none. [T-202]

`(T?)?` MUST lower to `logos/rt.Option[T?]`. [T-202]

The spelling `T??` MUST be a syntax error reporting `T?? is not allowed; use (T?)?`. [T-202]

`?.` on a non-option value MUST fail with `[correctness::not_an_option]`. [T-104.3]

`??` with a non-option left side MUST fail with the same diagnostic. [T-104.3]

A method call on an unnarrowed `T?` MUST fail with `[correctness::nil_deref_unproven]`. [T-202]

`??` MUST evaluate its right side only when the left side is none. [T-104.3]

`?.` MUST evaluate its receiver once. [T-104.3]

`?.` over a field that is already an option MUST flatten the result. [T-104.3]

See Sec 1.4 for `??` grouping, precedence, and single-value map reads.

`m[k] ?? d` MUST type as `V`. [T-104.3]

`v, ok := m[k]` MUST keep Go types. [T-104.3]

An absent key MUST read as none. [T-104.3]

A present zero value MUST read as some. [T-104.3]

A present nil slice payload MUST read as some with nil payload, distinct from none. [T-104.3][T-202]

Ejected and Go-facing signatures MUST use plain pointers and `rt.Option[T]` from the support module. [T-202][T-Sec6][R-U2]

A Go importer MUST see a non-nil `*T` and a `*T?` as `*T`, and MUST see every other `T?` as the lowered type of the rules above. [T-202][T-Sec6][R-U2]

```lgo
func greet(u *User?) string {
    return u?.Name ?? "anon"
}
```

### 2.3 RFC-203: Closed unions, property interfaces, inline assertions, and clone specialization

A closed union MUST be an interface whose type set is a finite union of non-`~` terms. [T-203.1]

It MAY also declare methods that every member implements. [T-203.1]

A closed union MUST be allowed in value positions (variables, parameters, results, fields, elements, keys, and values). [T-203.1]

A union with `~` terms or an unbounded set MUST fail with `[correctness::open_union_value]`. [T-203.1]

A conversion MUST be allowed when the source type is a member or a closed union of members. [T-203.1]

Any other conversion MUST fail with `[correctness::union_nonmember]` naming the type. [T-203.1]

A comparison between a union and a constant MUST convert the constant to its default type and require membership. [T-203.1]

An assertion or case type outside the member set MUST fail with `[correctness::union_nonmember]`. [T-203.1]

A comma-ok assertion on a non-member MUST yield `ok == false`. [T-203.1]

A single-value assertion on a non-member MUST panic with `interface conversion: <T> is not a member of <pkg>.<U>`. [T-203.1]

`case U:` MUST match only members. [T-203.1]

`nil` MUST go to `case nil`. [T-203.1]

An exhaustive `switch x.(type)` MUST cover every member and `nil`, or carry `default`. [T-203.1]

A missing member MUST fail with `[correctness::nonexhaustive_switch]` listing misses in declaration order. [T-203.1]

A case naming a sub-union MUST cover its members. [T-203.1]

A switch expression MUST lower to an immediately invoked function literal. [T-203.1]

Its result type MUST be the common type of the arms (identical types, the default type of untyped constants, or `T` when other arms are `nil`). [T-203.1]

Mismatched arms MUST fail with `[correctness::switch_arm_mismatch]`. [T-203.1]

A type switch expression without full coverage MUST fail with `nonexhaustive_switch`. [T-203.1]

A value switch expression without `default` MUST fail with `[correctness::switch_expr_needs_default]`. [T-203.1]

The switched value MUST evaluate once. [T-203.1]

Only the chosen arm MUST evaluate. [T-203.1]

The run-time representation MUST be the upstream interface. [T-203.1]

A two-tier tag layout MUST NOT be used. [T-203.1]

Reflection and `unsafe` MAY store non-members. [T-203.1]

The compiler MUST NOT check this case. [T-203.1]

A closed-union value MUST marshal through its member static helper. [T-203.1]

Unmarshalling MUST try members in declaration order into fresh scratch values. [T-203.1]

It MUST assign the target only on a single success. [T-203.1]

Otherwise it MUST return an ambiguity error naming the union and the members. [T-203.1]

An ejected value-used union MUST keep its methods and carry the comment `// loGo/s union: A | B`. [T-203.1]

A Go importer MUST see a value-used union as that method-only interface. [T-203.1][T-Sec6][R-U2]

A union used both as a constraint and as a value type MUST fail eject with `logos eject: union U is used as a type constraint and as a value type; Go cannot express both`. [T-203.1]

Such a union has no Go-facing form, and a Go import of it MUST fail with the same text. [T-203.1][R-U2]

```lgo
type U interface {
    int | string
}
func f(u U) int {
    return switch u.(type) {
    case int => 1
    case string => 2
    case nil => 0
    }
}
```

A property interface MUST be an interface with at least one `const F T` or `F T` element. [T-203.2]

A struct MUST satisfy a property interface only through an explicit `: I` declaration in any package. [T-203.2][R-U4]

For each property the declaring package MUST generate `Logos_GetF() T` from an exported field or from a method `F() T`. [T-203.2]

For each read-write property it MUST also generate `Logos_SetF(T)` from the field or from `SetF(T)`. [T-203.2]

A field-sourced getter MUST use a value receiver. [T-203.2]

A field-sourced setter MUST use a pointer receiver. [T-203.2]

A method-sourced accessor MUST keep the receiver kind of the forwarded method. [T-203.2]

Property elements MUST lower to these interface methods. [T-203.2]

A Go importer MUST see a property interface as the interface of its `Logos_GetF` and `Logos_SetF` accessor methods. [T-203.2][T-Sec6][R-U2]

A dynamic assertion such as `any(x).(I)` MUST work. [T-203.2]

Identifiers starting with `Logos_` or `_logos_` MUST be reserved in `.lgo`. [T-203.2]

A use of such an identifier MUST fail with `[correctness::reserved_identifier]`. [T-203.2]

A read-write property MUST require a pointer type. [T-203.2]

Using a value type for one MUST fail with `[correctness::property_needs_pointer]`. [T-203.2]

Taking the address of a property (`&c.HP`) MUST fail with `[correctness::property_address]`. [T-203.2]

A `: I` declaration without a field or method source for a property MUST fail with `[correctness::property_unsatisfied]`. [T-203.2]

The right fields without the declaration MUST fail with `[correctness::property_requires_conformance]` telling the writer to add `: I`. [T-203.2]

Conflicting property types across listed interfaces MUST fail with `[correctness::property_conflict]`. [T-203.2]

A non-struct type MUST NOT satisfy a property interface. [T-203.2]

A compound update through an interface MUST evaluate the receiver once and write the value back. [T-203.2]

A method call on a property value MUST use read-call-write-back around the call. [T-203.2]

Only a type with a `: I` list MUST gain accessor methods. [T-203.2]

`reflect` MUST count the generated accessors. [T-203.2]

A type without the declaration MUST see no method change. [T-203.2]

The backend chooses the instruction sequence; no exact-instruction claim holds. [T-203.2]

A 0ns timing claim for generated accessors is not measurable; timing depends on the CPU. [T-203.2][T-PERF]

Embedded fields MUST satisfy properties by Go selector depth rules. [T-203.2]

An ambiguous equal-depth selector MUST fail as an ambiguous selector. [T-203.2]

Read-only and read-write declarations of one name in one interface MUST merge into read-write. [T-203.2]

```lgo
type Creature interface {
    const Name string
    HP int
    Start() error
}
type Monster struct : Creature {
    Name string
    HP int
}
func (m Monster) Start() error {
    return nil
}
```

`type S struct : I, J { ... }` MUST assert that `*S` implements each listed interface at the declaration. [T-203.3]

A missing method MUST fail with `[correctness::assertion_failed]` naming the method. [T-203.3]

Each assertion MUST lower to `var _ I = (*S)(nil)`. [T-203.3]

An assertion list on a non-struct type MUST be a syntax error. [T-203.3]

At O2 and O3 the compiler MAY clone a same-package function and retype an interface parameter to the concrete argument type. [T-203.4]

Such a clone MUST apply only when the parameter serves solely as a method receiver or a passed-on argument. [T-203.4]

A `p == nil` test on the specialized parameter MUST fold to `false`. [T-203.4]

Specialization MUST stop at type depth 4. [T-203.4]

O2 MUST cap callees at 200 nodes. [T-203.4]

O3 MUST set no cost cap. [T-203.4]

A clone MUST carry the name `<f>·logosspec<N>`. [T-203.4]

The linker MUST strip that suffix in names written to pclntab, so traces print `<f>`. [T-203.4]

A parameter compared with `nil`, reassigned, or stored MUST block specialization for that call. [T-203.4]

A call site with more than 4 concrete types MUST NOT specialize. [T-203.4]

A `const []Creature` parameter receiving `[]*Monster` MUST specialize with no copy loop. [T-203.4]

No run-time call-site cache MUST exist. [T-203.4]

Function values MUST keep working. [T-203.4]

No dedup pass MUST run for clones. [T-203.4]

Unique package-local names MUST prevent duplicates. [T-203.4]

Specialization MUST apply to Go packages and to loGo/s packages. [T-203.4][R-U1]

### 2.4 RFC-204: Tuples and fixed-array SIMD

Tuple types `(a A, b B)` and `(A, B)` in non-result positions MUST lower to structs with exported fields (`T_a`, `T_b` / `T_0`, `T_1`). [T-204.1]

Exported field names MUST keep tuple identity across packages. [T-204.1]

A Go importer MUST see a tuple as that struct. [T-204.1][T-Sec6][R-U2]

`var t (int, error)`, `chan (int, error)`, slices of tuples, and map values with tuple types MUST be allowed. [T-204.1]

Named access such as `t.x` MUST work. [T-204.1]

Index-style access such as `t.0` MUST NOT be allowed. [T-204.1]

Code MUST use names or destructuring instead. [T-204.1]

Destructuring (`a, b := t`) MUST work. [T-204.1]

A multi-value call MUST unpack in calls and assignments. [T-204.1]

A tuple-typed variable MUST pack such values. [T-204.1]

`fmt.Println(pair())` MUST unpack. [T-204.1]

A tuple with non-comparable elements MUST NOT be comparable. [T-204.1]

```lgo
var t (id int64, err error)
a, b := t
_ = a
_ = b
```

Element-wise `+ - * / & | ^` MUST be allowed on `[N]T` with `N` a power of two, numeric `T`, and size at most 64 bytes. [T-204.2]

`.sum()`, `.min()`, `.max()`, and `.dot()` MUST be allowed on the same shapes. [T-204.2]

A larger array MUST fail with `[correctness::simd_array_too_large]`. [T-204.2]

A non-power-of-two length MUST fail compilation. [T-204.2]

With `GOEXPERIMENT=simd` on amd64 or arm64 the lowering MUST call `simd/archsimd`. [T-204.2]

Otherwise it MUST emit unrolled scalar code. [T-204.2]

Direct ISA lowering is replaced by this two-form rule. [T-204.2]

A Go package MUST get no array operator and no reduction method from this rule. It MAY use `simd/archsimd` under `GOEXPERIMENT=simd`. [T-204.2][R-U1]

Reductions MUST use one fixed pairwise order in both forms, giving bit-identical results. [T-204.2]

Every operator and reduction MUST match a scalar loop oracle. [T-204.2]

```lgo
var a, b [4]float64
c := a + b
_ = c.sum()
```

### 2.5 RFC-205: Comptime

`const if` MUST take a constant condition. [T-205]

A non-constant condition MUST fail with `[comptime::not_constant]`. [T-205]

Both branches MUST type-check. [T-205]

Only the live branch MUST reach lowering. [T-205]

`comptime.Fields[T]()`, `val.[f]`, `f.Name`, `f.Type`, `f.Tag.Get(k)`, `comptime.IsStruct[T]()`, `comptime.TypeOf[X]()`, and `comptime.Error(msg)` MUST be compile-time only. [T-205]

Any run-time use MUST fail with `[comptime::runtime_use]`. [T-205]

A generic function that uses them MUST become a comptime generic. [T-205]

Each instantiation MUST expand into a non-generic copy `<f>·logosct<N>` in the instantiating package. [T-205]

Copies MUST be memoized per type-argument list. [T-205]

An exported comptime generic MUST travel as source text in `__.LOGOS`. [T-205]

Expansion MUST visit fields in declaration order. [T-205]

It MUST include exported fields and unexported fields of a current-package `T`. [T-205]

It MUST exclude unexported fields of other packages. [T-205]

`val.[f]` MUST read and write the named field. [T-205]

`f.Tag.Get(k)` MUST constant-fold. [T-205]

A live `comptime.Error(msg)` MUST fail compilation with `msg`. [T-205]

`Resolve[f.Type](c)` MUST resolve per field. [T-205]

A panic in unrolled code MUST name the `.lgo` `const for` line in its trace. [T-205]

A circular instantiation MUST fail with `[comptime::circular_instantiation]` naming the chain. [T-205]

An expansion above the selected `max_comptime_ast_nodes` limit with default 4,096 nodes MUST fail with `[comptime::node_limit]`. [T-205][T-102]

Configuration MAY raise `max_comptime_ast_nodes` above its default. [T-102]

Validation and boundary tests MUST use the selected limit. [T-102]

A stack trace MUST print the base name without the `·logosct<N>` suffix. [T-205]

A shadowed `const switch` case MUST fail with `[comptime::unreachable_case]`. [T-205]

An exact case MUST rebind `T` and dependent variables in its block. [T-205]

An interface case MUST keep the concrete layout and gain the interface method set. [T-205]

`case int` MUST match `int` only. [T-205]

`case ~int` MUST match the underlying `int` (such as `UserID`). [T-205]

`[]*os.File` MUST never rebind as `[]io.Writer`. [T-205]

Cases MUST evaluate top-down in first-match order. [T-205]

Outside comptime generics a `const switch` MUST lower to a run-time type switch on `any((*T)(nil))` with `unsafe` rebinding for exact cases. [T-205]

Interface cases in that form MUST use assertion on the value. [T-205]

No match MUST return the input unchanged. [T-205]

Native and ejected output of the run-time form MUST agree. [T-205]

A comptime generic that marshals through `const for` MUST expand once per instantiation with its own static JSON helpers. [T-205]

`comptime.ViewAs[T](b)` MUST require a POD struct (no padding, no `bool`, no pointers, interfaces, slices, maps, or strings). [T-205]

Any other shape MUST fail with `[comptime::viewas_not_pod]` naming the offset. [T-205]

A short input MUST panic with a message of the form `logos: ViewAs: need <N> bytes, have <M>` naming the sizes. [T-205]

The view MUST keep `b` alive. [T-205]

Targets without unaligned loads MUST copy to an aligned temporary. [T-205]

```lgo
const if comptime.IsStruct[T]() {
    const for _, f := range comptime.Fields[T]() {
        _ = val.[f]
    }
}
```

### 2.6 RFC-206: Go group and logos/sync

`go group (ctx) { ... }` and `go group (ctx, limit = N) { ... }` MUST lower to `logos/rt.Group`. [T-206]

The group MUST cancel on the first child error or panic. [T-206]

It MUST join all children. [T-206]

It MUST bound concurrency with a semaphore when `limit` is present. [T-206]

`go group(x)` without a block MUST keep Go meaning as a call to a function named `group`. [T-206]

A child block MUST have type `func() error`. [T-206]

`?` inside a child MUST return to the group. [T-206]

`return` inside a child MUST fail with `[correctness::return_in_group_child]`. [T-206]

`go f(args)` inside a group body MUST form a child, sending an error-last result to the group. [T-206]

`go f()` outside any group body MUST fail with `[concurrency::unsupervised_goroutine]`. [T-206]

A child outside a group body MUST fail with `[correctness::child_outside_group]`. [T-206]

A `go group` statement in a function without an error result MUST fail with `[correctness::group_needs_error_result]`. [T-206]

The group error MUST propagate like `?`. [T-206]

The first failure in completion order MUST win. [T-206]

`Wait` MUST return only after all children exit. [T-206]

The context MUST be cancelled at join. [T-206]

A child panic MUST become `*logos/rt.PanicError` with the value and a stack. [T-206]

`NewGroup`, `Go`, `Wait`, and `PanicError.Error` MUST form the `Group` API (`NewGroup(ctx, limit)` returns the group and a child context; `Go` takes `func(context.Context) error`). [T-206]

A plain channel send, receive, or range directly in a child MUST lower to a `select` with `<-gctx.Done()`. [T-206]

A user `select` with `default` MUST stay unchanged. [T-206]

Cancellation MUST reach siblings blocked on channels. [T-206]

A child blocked in netpoll-backed network I/O MUST wake within 100 ms with a timeout error satisfying `os.IsTimeout`. [T-206]

A goroutine outside the group MUST stay blocked. [T-206]

Regular files and cgo calls MUST NOT wake this way. [T-206]

```lgo
func f(ctx context.Context) error {
    go group (ctx, limit = 8) {
        go { work()? }
    }
    return nil
}
```

`logos/sync.Locked[T]` MUST expose `WithLock(f func(*T))`. [T-206]

`WithLock` MUST hold `Lock`, defer `Unlock`, then call the function. [T-206]

It MUST unlock after a panic. [T-206]

A `WithLock` lambda MUST NOT return or store its argument. [T-206]

`logos/sync.Hot[T]` MUST expose `Load() const *T`, `Store(*T)`, and `Update(func(old *T) *T)` with a CAS loop. [T-206]

Concurrent updates MUST never expose a partial value. [T-206]

One million updates per second carries target: no memory growth, measured by the M8-S5 test. [T-206][T-PERF]

The WithLock overhead carries target: within 2 ns/op of manual `Lock` with `defer Unlock`, measured by BenchmarkWithLock. [T-206][T-PERF]

A 0ns read claim for `Hot.Load` is not measurable; BenchmarkHotLoad reports the cost of one atomic pointer load. [T-206][T-PERF]

Stores MUST run the hybrid write barrier. [T-206]

No retire list MUST exist. [T-206]

Values MUST stay GC-managed. [T-206]

```lgo
var c Locked[Config]
c.WithLock(p => { p.Port = 8080 })
```

## 3. Tier 3 specification

Every rule in this section MUST apply to Go packages and to loGo/s packages, under `logos` and under plain `go`, except a rule that depends on `.lgo` syntax or on a loGo/s language rule [T-0.1][R-U1].
A rule that sits behind a `GOEXPERIMENT` flag MUST stay behind it as this section states [T-0.1][R-U1].

### 3.1 RFC-301: Pluggable GC engine and tier-0 arena

The tier-0 arena MUST work under every `-gc` mode. [T-301.1]
It MUST stay off unless the build sets `GOEXPERIMENT=logosarena`. [T-301.1]
Under `logos`, `[build] arena` set to true MUST set that experiment for the build. [T-301.1][R-U1]
Under plain `go`, `GOEXPERIMENT=logosarena` alone MUST enable the arena. [T-301.1][R-U1]
The default value of `[build] arena` MUST be false. [T-301.1]
Each goroutine MUST lease one 64KB chunk while it holds arena objects. [T-301.1]
Objects that escape analysis proves do not outlive their frame, but that cannot live on the stack because of dynamic or large size, MUST allocate in the arena. [T-301.1]
Frame exit MUST restore the watermark. [T-301.1]
A loop back-edge MUST restore the watermark for iteration-local objects. [T-301.1]
The runtime MUST zero popped regions before reuse. [T-301.1]
Scan-region objects MUST carry a type word. [T-301.1]
The collector MUST scan them with the goroutine stack. [T-301.1]
Greentea MUST scale background workers by heap use. [T-301.2]
It MUST keep mark assist as backstop when heap use reaches 95% of goal. [T-301.2]
Under `GOEXPERIMENT=logosgen`, greentea MUST use a generational card table. [T-301.2]
Batch MUST collect stop-the-world only. [T-301.2]
It MUST start with `GOGC` off. [T-301.2]
It MUST trigger collection at 90% of `GOMEMLIMIT`. [T-301.2]
It MUST compile without write barriers. [T-301.2]
Epsilon MUST start with `GOGC` off and no memory limit. [T-301.2]
It MUST compile without write barriers. [T-301.2]
The GC code MUST stay linked under epsilon. [T-301.2]
The wasm collector MUST mark in slices during idle time of the JS event loop. [T-301.2]
Phase 1 MUST evacuate stop-the-world with pinning of conservatively referenced objects, cgo-pinned objects, and pointer-keyed map keys. [T-301.2]
Phase 2 MUST relocate concurrently with load barriers in zgc only. [T-301.2]
Pointer-keyed maps under zgc MUST use a runtime side-table identity hash of 32 bits. [T-301.2]
Address-hashed immix blocks MUST set the pin bit. [T-301.2]
Pinned blocks MUST NOT evacuate. [T-301.2]
Mode selection, defaults, and errors are defined in SPEC Sec 19. [T-301.2]
Go runtime states objects do not move, so zgc and immix stay research tracks until their gates pass. [T-301.2]

### 3.2 RFC-302: Smart stack engine

The variable-size `make` stack buffer MUST default to 32 bytes. [T-302]
At O2 it MUST be 256 bytes. [T-302]
At O3 it MUST be 4096 bytes. [T-302]
A slice that flows into a cgo call argument or an `unsafe.Pointer` conversion MUST NOT use the stack buffer. [T-302]
Under `GOEXPERIMENT=logosstack`, the runtime MUST choose the initial goroutine stack size from the per-PC exponential moving average of final stack sizes. [T-302]
Without history for a PC, it MUST fall back to the upstream global average. [T-302]

### 3.3 RFC-303: Hardware backend optimizations

Fault-suppressing masked tails MUST appear only in loops that M14 reductions over slices produce. [T-303]
Other code MUST carry no tail padding. [T-303]
BMI2 and `CMOV` code MUST come from existing `GOAMD64=v3` rules. [T-303]
Constant-count rotates MUST use BMI2 rotates at `GOAMD64=v3`. [T-303]
Variable-count rotates MUST stay on the existing shift and rotate path. [T-303]
In `.lgo`, `*` inside a `make` length or capacity MUST lower to `logos/rt.MulLen`. [T-303]
On overflow, `MulLen` MUST panic with `logos: make length overflow: w * h`. [T-303]
A Go package MUST keep the Go `make` length semantics. [T-303][R-U1]
LSE atomics MUST come from the `GOARM64=v8.2` default. [T-303]
Cache-line spacing MUST keep the upstream 128-byte constant. [T-303]
The runtime structs MUST carry no new padding. [T-303]
Preemption checks in long loops MUST fire every 4096 iterations, with the first check at iteration 4096. [T-303]
They MUST run only where async preemption is absent (`GOARCH=wasm`). [T-303]
Checks every 4096 iterations MUST carry target overhead below 0.01%, measured by `BenchmarkLogosLoopStride`. [T-303][T-PERF]
A call-free loop MUST NOT starve peers. [T-303]
A second goroutine MUST print before the deadline. [T-303]
`clear(b)` MUST reach a runtime memory clear. [T-303]
Dead-store elimination MUST NOT remove that clear. [T-303]
The rest of the WASM profile is defined in SPEC Sec 14. [T-Sec14]

### 3.4 RFC-304: Fast-CGo engine

A loGo/s package MUST accept cgo files beside other `.lgo` files. [T-304][A-CODEGEN]
A file that imports `C` MUST use Go syntax only. [T-304][A-CODEGEN]
LoGo/s syntax in such a file MUST fail with `[correctness::cgo_file_logos_syntax] files that import "C" must use Go syntax only`. [T-304][A-CODEGEN]
Pointers that C returns MUST count as platform pointers. [T-202]
Storing one into a non-nil location MUST check it. [T-202]
When the fast path is on, a C function that carries both `#cgo noescape` and `#cgo nocallback` MUST take the fast call path. [T-304][R-U1]
This rule MUST hold in Go packages and in loGo/s packages, under `logos` and under plain `go`. [T-304][R-U1]
Go code MUST need no new directive and no new syntax for the fast path. [T-304][R-U1]
A C function that carries only one of the two directives, or none, MUST keep the upstream call path. [T-304]
`//logos:cgo_fastcall` MUST be the `.lgo` spelling of the same pair and MUST mark a cgo call `noescape` and `nocallback`. [T-304]
One fast path MUST serve Go packages and loGo/s packages: the single `runtime.logosCgocallFast` trampoline and the one cgo pipeline of this section. [T-304][R-U1]
The fast path MUST skip P hand-off. [T-304]
It MUST record the call start. [T-304]
Sysmon MUST retake the P from a fast call under the same rule and the same threshold as from a syscall, as the upstream `retake` does, with no longer grace period. [T-304][R-U1]
Until the M20 gate passes, the fast path MUST stay behind `GOEXPERIMENT=logosfastcgo`. [T-304]
The M20 gate MUST cover race, SIGPROF, a blocking C call, every `-gc` mode whose M19 gate has passed, Go pointer validity, the `nocallback` panic, CPU profile attribution, and blocking parity. [T-304]
After the M20 gate passes and the speed rule and the blocking rule below hold, the fast path MUST be on by default. [T-304][R-U1]
`GOEXPERIMENT=nologosfastcgo` MUST turn the default off. [T-304][R-U1]
The speed rule MUST be a ratio of at least 1.05 of the same `BenchmarkFastcall`, built once with `GOEXPERIMENT=logosfastcgo` and once with `GOEXPERIMENT=nologosfastcgo`, measured under rule G6 of `IMPLEMENTATION.md`. [T-304]
Without that ratio the default MUST stay off. [T-304]
A fast-called C function MUST NOT block other goroutines measurably longer than a plain cgo call of the same function does, under the blocking rule below. [T-304][R-U1]
The blocking rule MUST be measured with a C function that sleeps 100 microseconds under `GOMAXPROCS=1`, as the delay of a ready goroutine against a plain cgo call. [T-304]
The blocking rule holds when the median delay with the fast path is at most 1.05 times the median delay of the plain cgo call, and the standard deviation of each series is at most 20 percent of its median. [T-304]
A series that exceeds the noise limit MUST count as blocked, never as passed. [T-304]
The default change MUST land as its own commit after the M20 milestone gate passes. [T-304]
The fast path MUST NOT change the cgo file layout, the generated file names, or the action cache key format. [T-304][R-U1]
Profiling with `SIGPROF` during fast calls MUST NOT crash. [T-304]
A C function marked `nocallback` that calls into Go MUST still panic on the fast path. It MUST NOT hang and MUST NOT corrupt state. [T-304][R-U1]
A CPU profile MUST attribute time in a fast-called C function to its Go call site. [T-304][R-U1]
Go pointer arguments MUST stay valid under every GC mode. [T-304]
Pointer arguments in slots 0 to 3 MUST stay valid during the call. [T-304]
They MUST read zero after return. [T-304]
POD and primitive-slice arguments MUST skip `_cgoCheckPointer`. [T-304]
Pin cost MUST be reported through `BenchmarkFastcall`. [T-PERF]
Pin cost MUST NOT gate a test pass; only the speed rule and the blocking rule above use their ratios. [T-PERF]

## 4. Red-team summary and resolutions

The original text of this section described implementation internals; it is archived in archive/red-team-summary.md. Section 17 holds the tailored resolutions. `REDTEAM.md` records their test status.

## 5. Lexer, parser, and syntax

### 5.1 Tokens and contextual words

The language MUST add no new global keywords [T-0.2]. The words `errdefer`, `yield`, `scope`, and `group` MUST act as contextual words valid only in `.lgo` files [T-0.2]. The word `errdefer` MUST start a statement only when an identifier or `func` follows it [T-0.2][T-104.2]. The word `yield` MUST start a statement only inside a generator function [T-0.2]. The word `scope` MUST appear only directly before `defer` [T-0.2]. The word `group` MUST appear only directly after `go` [T-0.2]. The word `const` MUST gain new positions before `for`, `if`, `switch`, and `func`, and those positions MUST remain syntax errors in plain Go [T-0.2]. Code MUST NOT declare `errdefer`, `yield`, `scope`, or `group` as ordinary names in those positions [T-0.2]. Outside those positions each word MUST keep its ordinary identifier meaning, so `errdefer := 1` MUST parse as an assignment [T-0.2][T-104.2]. The form `?` MUST scan as a postfix operator only in `.lgo` files [T-104.1]. The form `?.` MUST scan as one unit only when `?` joins `.` with no gap [T-104.3]. The form `??` MUST scan as one binary operator only when two `?` marks join with no gap [T-104.3]. The form `=>` MUST scan as one unit only when `=` joins `>` with no gap in `.lgo` files [T-104.6]. The form `$"..."` MUST scan as an interpolated string only when `$` joins an opening quote with no gap in `.lgo` files [T-104.4].

### 5.2 Grammar

New forms MUST follow this grammar [T-0.2][T-104.1][T-104.3][T-104.4][T-104.6][T-202][T-203.1][T-203.2][T-203.3]:

```
PostfixTry   ::= Operand "?"
OptionType   ::= Type "?"
Coalesce     ::= OrExpr ["??" Coalesce]
OptChain     ::= Operand ("?." IdentOrCall)*
Interp       ::= "$" QuotedStringWithBraces
Lambda       ::= ParamList "=>" (Expr | Block)
ChainCall    ::= SliceExpr ("." ChainMethod parens)*
TupleType    ::= "(" [Name] Type ("," [Name] Type)+ ")"
TupleLit     ::= "(" Expr ("," Expr)+ ")"
SwitchExpr   ::= "switch" [Init ";"] [Tag] "{" Arm+ "}"
Arm          ::= "case" ExprList "=>" Expr | "default" "=>" Expr
UnionDecl    ::= "type" Name "interface" "{" UnionTerms [";"] Methods [";"] "}"
UnionTerms   ::= Term ("|" Term)*
AssertDecl   ::= "type" Name "struct" ":" InterfaceList "{" Fields "}"
PropertyElem ::= ["const"] Name Type
ConstView    ::= "const" ("[]" Type | "*" Type | "map" "[" Type "]" Type)
ConstDecl    ::= "const" Name "=" (CompositeLit | ConstFuncCall)
ConstIf      ::= "const" "if" ConstExpr Block ["else" (Block | ConstIf)]
ConstFor     ::= "const" "for" [Index ","] Field ":=" "range" ComptimeFieldsCall Block
FieldProj    ::= Value ".[" Field "]"
ConstSwitch  ::= "const" "switch" TypeParam "{" ConstCase+ "}"
DeferTry     ::= "defer" Call "?"
ErrDefer     ::= "errdefer" (Call | Call "?" | FuncLitCall)
ScopeDefer   ::= "scope" "defer" Call
GoGroup      ::= "go" "group" "(" Expr ["," "limit" "=" Expr] ")" Block
GroupChild   ::= "go" "{" Stmts "}"
YieldStmt    ::= "yield" Expr
SpreadElem   ::= ".." Expr
DefaultField ::= Field Name Type "=" ConstExpr [Tag]
```

The type `T??` MUST remain a syntax error [T-202]. The type `(T?)?` MUST nest as an option of an option [T-202]. The union in `UnionDecl` MUST list a finite set of terms with no `~` term [T-203.1]. A struct MUST satisfy a property interface only through an explicit `: I` list [T-203.2][R-U4]. The operands `{expr}` inside `$"..."` MUST accept only string, integer, bool, float32, float64, `[]byte`, `error`, or `fmt.Stringer` values [T-104.4]. The texts `{{` and `}}` MUST mean literal braces [T-104.4].

```lgo
v := parse(s)?
f()?
defer closeAll()?
errdefer tx.Rollback()?
type S struct : Creature {
    Name string
    HP   int
}
```

### 5.3 Disambiguation and semicolon insertion

An opening `(` MUST parse its element list once and then branch on the next token [T-104.6]. A `:` after the first element MUST mean a named tuple type [T-204.1]. A `,` list followed by `=>` MUST mean a lambda, and each element MUST be a plain name [T-104.6]. A lone name followed by `=>` MUST mean a one-parameter lambda [T-104.6]. A lone `(e)` with no `=>` after it MUST mean a parenthesized expression [T-104.6]. The token `??` MUST bind less tightly than `||`, so `a || b ?? c` MUST parse as `(a || b) ?? c` [T-104.3]. Chained `??` MUST group to the right, so `a ?? b ?? c` MUST parse as `a ?? (b ?? c)` [T-104.3]. A newline after `?` MUST end the statement [T-104.1]. A newline after `?.` MUST NOT end the statement [T-104.3]. A newline followed by spaces or tabs and then `.` plus a letter or `_` MUST NOT insert a semicolon [T-104.6]. The form `x` newline `.5` MUST still insert a semicolon [T-104.6].

## 6. Type checker, facts, and representation guarantees

### 6.1 Cross-package facts

Cross-package facts MUST travel in the archive member `__.LOGOS` [T-Sec6]. The member MUST hold a text header of the package exported declarations in loGo/s syntax with nilability, const views, unions, tuples, and property interfaces [T-Sec6]. The member MUST hold one fact line per function for `ReturnsNonNil`, `AlwaysNilError`, `PureConst`, `ReceiverMutated`, the `ParamMutated` mask, and `MayYield` [T-Sec6]. The member MUST hold comptime generic sources [T-Sec6]. The facts MUST bind to the `__.PKGDEF` fingerprint, and a stale fingerprint MUST fail the dependent build [T-Sec6]. A Go importer of a loGo/s package MUST see the signatures that `logos eject` emits: a closed union value type as its methods-only interface with the terms dropped, `*T?` as the bare pointer when `*T` is non-nil and as `logos/rt.Option[T]` otherwise, a non-nil `*T` as a plain `*T`, a property interface as its `Logos_GetF` and `Logos_SetF` accessor methods, and a const view as its plain type [T-Sec6][R-U2]. A Go package MUST emit per-function fact lines only, namely `MayYield`, `AlwaysNilError`, and `ParamMutated`, with no declaration header [T-Sec6][R-U1]. It MUST write `__.LOGOS` only under `logos` or on `GOARCH=wasm` under either command, and plain `go` MUST write no `__.LOGOS` elsewhere [T-Sec6][T-Sec14]. When this toolchain compiles a Go package that calls a loGo/s function or method, the compiler MUST insert a `logos/rt.MustNonNil` check for each non-nil pointer parameter of the callee in a direct call, in a `go` statement, and in a `defer` statement, using the callee facts from `__.LOGOS` [T-Sec6][R-U2]. A call through a function value, a method value, or an interface has no such check; this is a documented hole [T-Sec6][R-U2][A-GOBOUNDARY]. A nil receiver that a Go caller passes to a method with a non-nil pointer receiver has no such check either; this is a documented hole [T-Sec6][R-U2][A-GOBOUNDARY]. Code that an upstream toolchain builds from eject output has no such check either; this is a documented hole, like reflection and `unsafe` [T-Sec6][R-U2][A-GOBOUNDARY]. A Go mutation through a const view and a Go store of a non-member into a closed union are documented holes, and the compiler does not check them [T-Sec6][R-U2][A-GOBOUNDARY].

### 6.2 Options, non-nil pointers, unions, and map index

The pointer type `*T` in `.lgo` code MUST mean non-nil [T-202]. The type `*T?` MUST mean an optional pointer [T-202]. A value of type `T` MUST convert to `T?`, and `nil` MUST convert to every `T?` [T-202]. A value of type `T?` MUST NOT convert to `T` without a check [T-202]. For `*T?` where `*T` is non-nil, the ejected Go type MUST be `*T` with `nil` as none [T-202][T-0.3]. An interface-backed union option with no nil-capable member MUST use the interface nil value for absence in ejected code [T-202][T-0.3]. For every other `T?` outside those two cases, the ejected Go type MUST be `logos/rt.Option[T]` [T-202][T-0.3]. When nil is a valid payload, the ejected form MUST carry explicit presence separate from payload nil [T-202][T-0.3]. A present nil payload MUST stay distinguishable from none in ejected code [T-202][T-0.3]. Ejected signatures MUST use plain pointers and `Option[T]` from the shared support module [T-202][T-0.3]. A Go importer MUST see the same Go types [T-202][T-Sec6][R-U2]. Zero values of non-nil pointers MUST fail compilation, including `var p *T`, arrays of non-nil pointers, `make` of a non-nil pointer slice with length above zero, and composite literals that leave a non-nil field empty [T-202]. Stores of `nil` into non-nil locations MUST fail compilation [T-202]. Narrowing after `x != nil`, after an early exit on `x == nil`, and through `&&` chains MUST apply to locals and parameters until their next assignment [T-202]. Narrowing MUST NOT apply to fields, to variables whose address escapes, or to variables that a closure assigns [T-202]. The error-first guard MUST apply only to callees declared in `.lgo` files, so standard-library and Go-package calls such as `io.Reader.Read` MUST keep their own contracts [T-202][A-STD][R-U2]. Pointers from Go-mode declarations and from type parameters instantiated with non-nil pointers MUST act as platform pointers: dereference and nil comparison MUST compile, and a store into a non-nil location MUST check at run time and panic with the `.lgo` file and line on nil [T-202][A-STD]. The operators `?.` and `??` MUST operate only on option types [T-104.3]. A single-value map read `m[k]` MUST type as `V?`, while `v, ok := m[k]` MUST keep Go meaning [T-104.3]. A closed union MUST be an interface whose type set is a finite union of non-`~` terms, and it MAY declare methods that every member implements [T-203.1]. Closed unions MUST be allowed in value positions in `.lgo` code [T-203.1]. Membership MUST be checked for assignments, conversions, comparisons with constants, assertions, and case types [T-203.1]. The forms `x.(U)` and `case U:` MUST check membership at run time [T-203.1]. An exhaustive `switch x.(type)` MUST cover every member and `nil` or hold `default` [T-203.1]. A switch expression MUST evaluate its operand once and only its chosen arm [T-203.1]. The result type of a switch expression MUST be the common type of its arms [T-203.1]. Ejected code for a value-used union MUST be a plain method-only interface plus a comment that names the member set [T-203.1][T-0.3]. A Go importer MUST see the same interface [T-203.1][T-Sec6][R-U2]. Reflection and `unsafe` MAY store a non-member in a closed union, and the compiler MUST NOT check that path [T-203.1]. Property reads and writes through an interface MUST call the generated accessors, and no timing guarantee MUST cover those calls [T-PERF].

### 6.3 Interface properties

A property interface MUST hold at least one `const F T` or `F T` element [T-203.2]. Satisfaction MUST be explicit: only a struct whose declaration lists `: I` MUST satisfy property interface `I`, in any package [T-203.2][R-U4]. For each property `F`, the declaring package MUST generate `Logos_GetF() T` and, for read-write properties, `Logos_SetF(T)` [T-203.2]. An exported field MUST source its accessors with a value-receiver getter and a pointer-receiver setter, and `F() T` or `SetF(T)` methods MUST source them by forwarding with the same receiver kind [T-203.2]. Property elements MUST lower to those interface methods, so dynamic assertions such as `any(x).(I)` MUST work [T-203.2]. A Go importer MUST see a property interface as the interface of those accessor methods [T-203.2][T-Sec6][R-U2]. Identifiers that start with `Logos_` or `_logos_` MUST remain reserved in `.lgo` code [T-203.2]. Read-write properties MUST need a pointer type [T-203.2]. The form `&c.HP` MUST fail compilation [T-203.2]. Compound updates through a property MUST evaluate the receiver once and write the result back [T-203.2]. Only types with a `: I` list MUST gain accessor methods, and `reflect` MUST count exactly those methods; types without the list MUST gain none [T-203.2][R-U4]. Non-struct types MUST NOT satisfy a property interface [T-203.2][R-U4]. The assertion `type S struct : I, J` MUST require `*S` to implement `I` and `J` at the declaration [T-203.3].

```lgo
type Creature interface {
    const Name string
    HP         int
    Start() error
}
type Monster struct : Creature {
    Name string
    HP   int
}
```

## 7. Eject and error-handling semantics

### 7.1 Error propagation, cleanup, and eject guarantees

See Sec 1.4 for error propagation, cleanup, and traces.
See Sec 0.3 for eject output, dependencies, and toolchain requirements.
Ejected output MUST be gofmt-clean. [T-0.3]

```lgo
func Transfer(db *DB, from int64, to int64, amt int64) (int64, error) {
    tx := db.Begin()?
    errdefer tx.Rollback()?
    defer tx.Close()?
    bal := tx.AddBalance(from, -amt)?
    tx.AddBalance(to, amt)?
    tx.Commit()?
    return bal, nil
}
```

### 7.2 Generic type switch lowering

The original text of this subsection described implementation internals; it is archived in archive/lowering-templates.md. The build steps are in IMPLEMENTATION.md, milestone M7.

## 8. Tier-3 runtime memory subsystem and GC internals

### 8.1 `g` and `p` struct additions

The original text of this section described implementation internals; it is archived in archive/runtime-memory.md. The build steps are in IMPLEMENTATION.md, milestones M1, M18, and M20. [T-Sec8]

### 8.2 Inline noscan bump allocator

The original text of this section described implementation internals; it is archived in archive/runtime-memory.md. The build steps are in IMPLEMENTATION.md, milestone M18.

### 8.3 Dual-ended side-stack arena lifecycle

An allocation that would push the noscan cursor past the scan cursor MUST link a fresh chunk or fall back to the heap. [T-301.1]
Allocation MUST still succeed in that case. [T-301.1]
A loop that allocates only iteration-local objects MUST NOT grow arena use across iterations. [T-301.1]
Recursion MUST return arena-allocated results to the caller intact. [T-301.1]
A value stored into `sync.Pool` MUST stay on the heap. [T-301.1]
A returned value that the caller keeps MUST allocate in the caller region. [T-301.1]
It MUST live until the caller exits. [T-301.1]
A goroutine parked with empty regions MUST hold no chunk. [T-301.1]
Idle parked goroutines MUST consume 0 bytes of arena memory. [T-301.1]
`runtime.SetFinalizer` on an arena object MUST panic with `runtime: SetFinalizer on loGo/s arena object`. [T-301.1]
Cgo pointer checks MUST NOT panic on arena addresses. [T-301.1]
Pointer-containing arena objects MUST keep their heap referents alive across `runtime.GC()`. [T-301.1]
After a frame exit, collection MUST find no stale pointers in popped scan memory. [T-301.1]

### 8.4 Pluggable GC write and load barrier dispatch

The `-gc` flag MUST select the barrier emitter at compile time. [T-301.2]
Under greentea built with `GOEXPERIMENT=logosgen`, pointer stores into mature spans MUST mark the card table. [T-301.2]
Under greentea, pointer loads MUST be plain loads with no barrier instructions. [T-301.2][T-PERF]
Under zgc, pointer loads MUST test the color and heal or remap on mismatch. [T-301.2]
Under zgc, pointer stores MUST be plain. [T-301.2]
Under immix, pointer stores MUST set the line mark. [T-301.2]
Under immix, pointer loads MUST be plain. [T-301.2]
Under wasm, pointer stores MUST keep the write barrier. [T-301.2]
Under batch and epsilon, pointer stores MUST be plain with no barrier. [T-301.2][T-PERF]
Pointer-keyed maps MUST keep working with the standard hash under greentea, batch, and epsilon. [T-301.2]
Under zgc they MUST use an identity hash that survives relocation. [T-301.2]
Under immix an address-hashed block MUST stay in place. [T-301.2]

### 8.5 `//logos:cgo_fastcall` trampoline

The original text of this section described implementation internals; it is archived in archive/runtime-memory.md. The build steps are in IMPLEMENTATION.md, milestone M20.

## 9. Const maps

### 9.1 Const map semantics

A const map with up to 8 keys MUST lower to a `switch` lookup [T-Sec9]. A const map with more keys MUST lower to a static perfect-hash table in read-only data plus a generated lookup function [T-Sec9]. A lookup MUST perform constant-time auxiliary metadata reads plus exactly one final key/value-slot probe with full-key validation [T-Sec9]. Construction MUST use at most 1,000,000 seed-candidate trials in total per const map [T-Sec9]. Exhaustion MUST fail with `comptime::constmap_seed_not_found` [T-Sec9]. A logarithmic fallback MUST NOT satisfy the contract [T-Sec9]. The 1, 8, 9, and 10,000-key cases in `test/logos/run/const_map.lgo` MUST be the acceptance check [T-Sec9]. That check MUST NOT promise success for every key set [T-Sec9]. A lookup of an absent key MUST return none [T-Sec9]. Iteration with `range` MUST visit keys in sorted order [T-Sec9]. The function `len` on a const map MUST be constant [T-Sec9]. Use of a const map as a `map[K]V` value MUST materialize a regular map once [T-Sec9]. A `-0.0` float key MUST canonicalize, and a `NaN` key MUST fail compilation [T-201][T-Sec9]. Lookup latency carries target: about 2.5ns versus about 14ns, measured by BenchmarkConstMapLookup [T-PERF]. Zero init cost and zero scan share MUST NOT be claimed; instead the table symbol MUST sit in read-only data and package init MUST add no heap bytes [T-PERF].

```lgo
const mimeMap = map[string]string{
    "txt": "text/plain",
    "html": "text/html",
}
v := mimeMap[k] ?? "unknown"
```

## 10. Configuration schema

Earlier drafts named this file bettergo.toml. [T-Sec10]

### 10.1 File location and parsing

The file `logos.toml` MUST sit next to `go.mod` in the main module. [T-Sec10]
Plain `go` MUST read no `logos.toml`. [T-Sec10][R-U1]
The `[build]`, `[lints]`, and `[capabilities]` tables MUST apply to modules of both kinds under `logos`. [T-Sec10][R-U1]
When the file is absent, the tool MUST use the default for every key. [T-Sec10]
The parser MUST accept tables. [T-Sec10]
The parser MUST accept key and value pairs, basic strings, literal strings, integers, booleans, string arrays, and hash comments. [T-Sec10]
Inputs larger than 1 MB MUST fail parsing. [T-Sec10]
The parser MUST reject unknown tables and unknown keys. [T-Sec10]
Each error MUST report file, line, and column as logos.toml:<line>:<col>: <message>. [T-Sec10]
Each key MUST take effect in the milestone that reads it. [T-Sec10]

### 10.2 Project table

The project table holds name, edition, and strict_files. [T-Sec10]
name is a string. [T-Sec10]
name MUST NOT be empty; an empty name fails with `logos.toml:<line>:<col>: project name must not be empty`. [T-Sec10]
name is informational: two builds that differ only in name MUST produce identical binaries. [T-Sec10]
edition is a string with default "2026". [T-Sec10]
edition selects the language rules of this spec, and "2026" is the only valid value. [T-Sec10]
Any other edition MUST fail with `logos.toml:<line>:<col>: unsupported edition "<value>" (want "2026")`. [T-Sec10]
strict_files is a string array. [T-Sec10]
strict_files lists the loGo/s extensions that the module enables, and its default is all four. [T-Sec10][T-101.1]
A source file whose loGo/s extension is not listed MUST fail with `[correctness::extension_not_enabled] <file>: extension <ext> is not enabled by logos.toml strict_files`. [T-Sec10][T-101.1]
Each entry MUST name one of the four source extensions: .lgo, U+1F4DC, U+26A1, or the gopher extension U+0295 U+25D4 U+03D6 U+25D4 U+0294. [T-Sec10][T-101.1]

### 10.3 Build table

cgo is a string with default "forbid" for a loGo/s main module. [T-Sec10][T-101.2]
It MUST have no effect in a Go main module. [T-101.2][R-U1]
It MUST be one of "forbid", "restricted", or "allow". [T-101.2]
With "forbid", a package that imports C fails with restriction::cgo_usage. [T-101.2]
With "restricted", logos enables cgo. [T-101.2]
With "restricted", each package that imports C MUST match cgo_allowlist. [T-101.2]
A package outside the list fails with restriction::cgo_usage. [T-101.2]
cgo_allowlist is a string array of path globs with default empty. [T-Sec10][T-101.2]
A double star matches any suffix. [T-Sec10][T-101.2]
An example entry is "example.com/c-sqlite/**". [A-EXAMPLES][T-101.2]
An explicit CGO_ENABLED value in the environment MUST win over the default only when the `cgo` key is absent [T-101.2]. It MUST NOT override an explicit `forbid` or a `restricted` allowlist refusal [T-101.2].
When cgo is on, CC is unset, and no C compiler is present, logos and plain go MUST use zig cc with the build target for every cgo package, Go or loGo/s, if zig is on PATH. [T-101.2][R-U1] The build target is the GOOS/GOARCH pair of the build, not the host. [T-101.2][R-U1]
With cgo off, net and os/user use their pure-Go paths. [T-101.2]
opt_level is a string with default "O2". [T-Sec10][T-105]
See Sec 1.5 for accepted `opt_level` values.
A `-O=<level>` flag on `logos build` MUST override the key. [T-105]
gc is a string with default "greentea". [T-Sec10][T-301.2]
It MUST be one of "greentea", "zgc", "immix", "wasm", "batch", or "epsilon". [T-Sec10][T-301.2]
On GOARCH=wasm the default is "wasm". [T-301.2]
synthetic_pgo MUST be one of "off", "standard", or "aggressive". [T-Sec10][T-105]
When the key is absent, O2 uses standard. [T-105]
When the key is absent, O3 uses aggressive. [T-105]
Other levels emit nothing. [T-105]
target_cpu is a string with default "deterministic". [T-Sec10][T-105]
It MUST be one of "deterministic", "baseline", or "native". [T-Sec10][T-105]
With "deterministic", logos sets GOAMD64 to v3 and GOARM64 to v8.2 unless the environment already sets them. [T-105]
With "native", logos uses the host level. [T-105]
See Sec 1.5 for the default binary CPU requirement.
arena is a boolean with default false. [T-Sec10]
A true value enables the Tier-0 arena experiment for Go packages and loGo/s packages under `logos`. [T-Sec10][R-U1]

### 10.4 Capabilities table

The capabilities table holds five severity keys and four allowlist keys. [T-Sec10]
direct_reflect is a string with default "deny". [T-Sec10]
transitive_reflect is a string with default "warn". [T-Sec10]
transitive_exec, transitive_net, and transitive_unsafe are strings. [T-Sec10]
Each severity key MUST be one of "deny", "warn", or "allow". [T-Sec10]
reflect_allowlist, transitive_exec_allowlist, transitive_net_allowlist, and transitive_unsafe_allowlist are string arrays of path globs with default empty. [T-Sec10]
An entry MAY name a Go module or a loGo/s module. [T-Sec10][R-U2]
Example entries use module paths such as "example.com/proto-runtime/**". [A-EXAMPLES][T-Sec10]
logos enforces the policy for every module in the build. [T-Sec10][T-Sec13]

### 10.5 Lint tables

The lints table holds one severity key per group: correctness, concurrency, slop, complexity, perf, style, suspicious, portability, and pedantic. [T-Sec10][T-102]
Each value MUST be one of "forbid", "deny", "warn", or "allow". [T-Sec10][T-102]
correctness and concurrency default to forbid. [T-Sec10][T-102]
slop and complexity default to deny. [T-Sec10][T-102]
perf, style, suspicious, and portability default to warn. [T-Sec10][T-102]
pedantic defaults to allow. [T-Sec10][T-102]
forbid cannot be lowered and ignores allow directives in a loGo/s package. [T-102]
deny fails the build unless a logos allow directive for the rule precedes the statement or declaration. [T-102]
warn prints a diagnostic. [T-102]
allow stays silent. [T-102]
The --pedantic flag overrides the pedantic key for logos build and logos lint. [T-Sec10]
The lints.thresholds table holds six integer keys. [T-Sec10]
Defaults are 3 for max_nesting_depth, 15 for max_cognitive_complexity, 60 for max_function_lines, 4 for max_function_params, 4096 for max_comptime_ast_nodes, and 100000 for max_ctfe_steps. [T-Sec10]
Configuration MAY raise max_comptime_ast_nodes and max_ctfe_steps above their defaults. [T-Sec10][T-102]
Validation and boundary tests MUST use the selected limits. [T-Sec10][T-102]
`logos lint` lints Go packages with the language-neutral rules of Sec 1.2 under these severities, with the cap that the next line states. [T-102][R-U1]
A language-neutral rule whose default severity is forbid runs at deny for Go packages, and `//logos:allow(<rule>)` silences it there. [T-102][R-U1]
`logos build` runs no lints on Go packages. [T-102][R-U1]
Third-party modules of either kind get no code lints. [T-102][R-U1]

## 11. End-to-end example

The normative example is test/logos/example/order_service.lgo. [T-Sec11]
It runs ProcessBatch against a fake database and ships an expected-output file. [T-Sec11]
The file uses only valid syntax, including leading-dot chains, block lambdas, and pointer assertion targets. [T-Sec11]
logos eject reproduces the same output from the ejected module. [T-Sec11]
The original text of this section is archived in archive/example-program.md. [T-Sec11]
The build steps are in IMPLEMENTATION.md, milestone M22. [T-Sec11]

## 12. Static serializer

### 12.1 Static marshaling

For each .lgo struct type with json tags, defaults, or non-nil pointer fields and no embedded fields, the lowering MUST generate a value-receiver MarshalJSON method. [T-Sec12]
Field kinds without static support skip generation for that type. [T-Sec12]
These kinds are interfaces, maps with non-string keys, and embedded structs. [T-Sec12]
Generated output MUST be byte-identical to encoding/json for the same value under both `goexperiment.jsonv2` and `!goexperiment.jsonv2`. [T-Sec12]
The default build MUST use the fork baseline with `goexperiment.jsonv2` on. [T-Sec12]
In both configurations, json.Marshal on an eligible value MUST return the same bytes as the generated method. [T-Sec12]
No call site is rewritten. [T-Sec12]
The generator emits fixed key literals first, computes a size bound, then fills one buffer from a single make. [T-Sec12]
The buffer is returned to the caller, so it escapes to the heap. [T-Sec12]
No size threshold and no caller buffer apply. [T-Sec12]
The fast path calls no reflect. [T-Sec12]
Only unmodeled fields fall back to reflective json.Marshal per field. [T-Sec12]
Each string field is read once into a local. [T-Sec12]
Concurrent mutation during marshal stays a data race. [T-Sec12]
Escaping and float formatting MUST follow encoding/json under the active `goexperiment.jsonv2` setting with portable Go and no SIMD requirement. [T-Sec12]
Supported tag options are name, omitempty, omitzero, string, and -. [T-Sec12]
HTML characters, U+2028, U+2029, invalid UTF-8, and float exponents encode per encoding/json. [T-Sec12]
NaN and infinities report UnsupportedValueError. [T-Sec12]
A closed-union value marshals to its dynamic member static bytes. [T-Sec12][T-203.1]
A comptime generic that marshals over comptime fields expands once per instantiation. [T-205]
Each instantiation carries its own static helpers. [T-205]
Strings without escapes take a single-copy path, with target: 99.9 percent of the corpus, measured by BenchmarkJsonEscapeScan. [T-PERF][T-Sec12]
The spec sets no CPU-clock budget for bound calculation. [T-PERF]
The benchmark reports the cost. [T-PERF]
The spec sets no nanosecond figure for integer length. [T-PERF]
BenchmarkJsonIntLen reports the cost. [T-PERF]

### 12.2 Static unmarshaling

For each eligible struct, the lowering generates pointer-receiver UnmarshalJSON. [T-Sec12]
Decoding fills defaults first, then parses. [T-Sec12]
A _logos_seen mask sized [(K+63)/64]uint64 tracks matched keys against a required-mask constant. [T-Sec12]
At the closing brace, a missing required non-nil field returns MissingFieldError naming type and field. [T-Sec12]
JSON null for a non-nil field returns the same error. [T-Sec12]
Keys MUST dispatch by switch up to 8 keys and by the Sec 9.1 const-map table above that, with constant-time auxiliary metadata reads plus exactly one final key/value-slot probe with full-key validation. [T-Sec12][T-Sec9]
Unknown fields are ignored. [T-Sec12]
Duplicate keys keep the last value. [T-Sec12]
Arrays grow incrementally with no pre-allocation cap. [T-Sec12]
Decoding through json.Decoder uses the same method. [T-Sec12]
No validate tags are needed. [T-Sec12]
Unmarshal of a closed union tries members in declaration order into fresh scratch values and counts successes. [T-Sec12][T-203.1]
It assigns the target only on a single success. [T-Sec12][T-203.1]
Ambiguity returns an error naming the union and the members. [T-Sec12][T-203.1]
Failed trials leave the target untouched. [T-Sec12][T-203.1]

### 12.3 Binary encoding

For each JSON-eligible struct, the lowering generates MarshalBinary, UnmarshalBinary, and AppendBinary. [T-Sec12.1]
They implement encoding.BinaryMarshaler, encoding.BinaryUnmarshaler, and encoding.BinaryAppender. [T-Sec12.1]
Output starts with version byte 0x01, then fields in declaration order. [T-Sec12.1]
All integer kinds encode as 8 bytes little-endian of the 64-bit value. [T-Sec12.1]
float32 encodes as 4 bytes little-endian of Float32bits. [T-Sec12.1]
float64 encodes as 8 bytes little-endian of Float64bits. [T-Sec12.1]
bool MUST encode as 0x00 or 0x01. [T-Sec12.1]
string and []byte encode as uvarint length plus raw bytes. [T-Sec12.1]
Slice and array encode as uvarint count plus elements. [T-Sec12.1]
Maps with string keys encode as uvarint count plus entries sorted by key. [T-Sec12.1]
Nested structs encode inline. [T-Sec12.1]
Optional values MUST encode as 0x00 when absent or 0x01 plus payload when present. [T-Sec12.1]
A field of type int or uint warns portability::serialized_unsized_int. [T-Sec12.1]
Sized kinds are preferred. [T-Sec12.1]
Tag options change nothing in binary output. [T-Sec12.1]
UnmarshalBinary rejects truncation, bad presence bytes, version mismatch, and trailing bytes with errors naming field and offset. [T-Sec12.1]
A count header MUST pre-allocate at most min(len(rest)/2, 1 MB / sizeof(elem)) elements and MUST grow past that only as input bytes arrive. [T-Sec12.1]

## 13. Capabilities auditor

### 13.1 Sink symbols

After deadcode, the linker MUST run a reachability audit from main.main and init functions. [T-Sec13]
The audit MUST run only in `logos` builds and MUST cover modules of both kinds. [T-Sec13][R-U1]
The audit follows relocations. [T-Sec13]
It adds synthetic edges for interface calls and reflective method lookups. [T-Sec13]
Protected sinks are reflect.ValueOf, reflect.TypeOf, reflect.New, syscall.Exec, syscall.ForkExec, os.StartProcess, os.startProcess, net.sysSocket, runtime.cgocall, and runtime.logosCgocallFast. [T-Sec13]
Packages that import unsafe are found through a compiler fact on the package graph. [T-Sec13]
The standard library is exempt. [T-Sec13]
A dependency that imports a sink package but never calls it passes as dead code. [T-Sec13]
Reflection through reflect.Value.Method counts as reachable. [T-Sec13]
On diamond graphs the report keeps the shortest chain. [T-Sec13]
Policy comes from logos.toml capabilities. [T-Sec10][T-Sec13]

### 13.2 Reachability report

A violation aborts linking with ERROR capabilities::unauthorized_exec naming the dependency and the sink. [T-Sec13]
The report lists the shortest call chain with file and line per frame. [T-Sec13]
For the allowlist keys that authorize each sink, see Sec 10.4. [T-Sec10][T-Sec13]
Unsafe violations use capabilities::unauthorized_unsafe with the shortest import chain. [T-Sec13]
cgo reachability reports capabilities::unauthorized_cgo unless the cgo policy allows the package. [T-Sec13][T-101.2]

## 14. Wasm profile

### 14.1 Selective resume points

The WASM profile MUST activate automatically for GOOS=js GOARCH=wasm and GOOS=wasip1 GOARCH=wasm. [T-Sec14]
It MUST apply to Go packages and to loGo/s packages, under `logos` and under plain `go`. [T-Sec14][R-U1]
It uses the installed toolchain with no external tool. [T-Sec14]
Functions with fact MayYield=0 compile without resume-point dispatch. [T-Sec14]
A function keeps MayYield=0 only with no defer, no indirect call, no call to MayYield=1 or unknown callees, no transitive runtime.gopark, and no channel, select, time.Sleep, Gosched, or blocking wasmimport call. [T-Sec14]
Pure imports keep MayYield=0. [T-Sec14]
Unknown wasmimport targets default to MayYield=1. [T-Sec14]
A wrong MayYield=0 MUST trap instead of hanging. [T-Sec14]
MayYield=1 functions MUST keep resume-point dispatch. [T-Sec14]
With `--wasm-features=stack-switching` on `logos build` or `go build` they MUST use stack switching instead. [T-Sec14][R-U1]
Pointers are 8 bytes and slice headers are 24 bytes. [T-Sec14]
alignof of int64, uint64, and float64 equals 8, so 64-bit atomics stay naturally aligned with no forcing code. [T-Sec14]
Deadcode removes unreachable functions, but type and method tables remain so reflect and encoding/json keep resolving. [T-Sec14]
Binary size carries target: 3x smaller than full Asyncify, measured by BenchmarkWasmAsyncifySize. [T-PERF][T-Sec14]
Execution speed carries target: 4x faster than full Asyncify, measured by BenchmarkWasmAsyncifySpeed. [T-PERF][T-Sec14]
The MayYield=0 share carries target: 90 percent, and the MayYield=1 share carries target: 10 percent, both measured by BenchmarkWasmSyncPureShare as shares, not guarantees. [T-PERF][T-Sec14]

### 14.2 Host interface and loader

`logos build` MUST emit <name>.loader.mjs beside the wasm file when the output is a `.wasm` file, for a Go main package and for a loGo/s main package. [T-Sec14][R-U1]
`go build` MUST emit the same loader for such an output when `--wasm-loader` is passed. [T-Sec14][R-U1]
The `--wasm-loader` flag on `go build` MUST stay absent from the help text, as Sec 0.1 states for flags that plain `go` gains. [T-Sec14][T-0.1]
Plain `go` MUST NOT emit the loader by default until `./run.bash` passes with it, as the flip rule of Sec 0.1 states. [T-Sec14][T-0.1]
After that flip, plain `go` MUST emit the loader by default, and `--wasm-loader=false` MUST turn it off. [T-Sec14][R-U1]
The loader MUST have identical bytes under both commands. [T-Sec14][R-U1]
Both commands MUST add the implicit `logos/rt` import for the loader helpers when they emit the loader for a `-o` output that names a `.wasm` file. [T-Sec14][R-U1]
The loader exports instantiateLogos(wasmBytes, imports). [T-Sec14]
The loader re-reads memory.buffer after every call into wasm. [T-Sec14]
Exported functions MUST return an int32 errCode. [T-Sec14]
String results are invalid. [T-Sec14]
On nonzero errCode the wrapper throws the message read from wasm memory. [T-Sec14]
The loader carries target: 1.4 KB, excluding wasm_exec.js, measured by BenchmarkWasmLoaderSize. [T-PERF][T-Sec14]
//logos:wasmexport f MUST behave as //go:wasmexport f. [T-Sec14]
//logos:wasmimport m f MUST behave as //go:wasmimport m f. [T-Sec14]

## 15. Backend peephole and SIMD passes

- Every pass in this section MUST apply to Go packages and to loGo/s packages, under `logos` and under plain `go`, except a rule about a loGo/s language construct: array operators, reduction methods, and `make` length checks. [T-303][R-U1]

### 15.1 Arm64 load and store pairing

- Adjacent loads and stores that meet the safety predicates MAY fuse into one paired instruction. [T-303]
- Fusion MUST NOT change program meaning, fault behavior, or atomic order. [T-303]
- Candidates split by a call, an atomic, or a block boundary MUST stay separate. [T-303]
- A pair MUST NOT read or write its own base register. [T-303]
- The two destinations of a pair MUST differ. [T-303]
- Offsets outside the encodable range MUST stay separate loads or stores. [T-303]
- With the default target_cpu the ARM64 target MUST use GOARM64=v8.2, so LSE atomics stay available. [T-303][T-105]

### 15.2 Amd64 masked tails, rotates, and checked lengths

- Reductions over slices MUST use fault-suppressing masked loads for tail elements. [T-303]
- Masked tails MUST NOT read past the end of the slice. [T-303]
- The build MUST NOT add guard padding for vector tails. [T-303]
- At GOAMD64=v3 constant-count rotates MAY use RORXQ or RORXL besides SARX, SHLX, and SHRX. [T-303]
- Variable-count rotates MUST stay on the existing path. [T-303]
- A ternary-shaped branch at GOAMD64=v3 MAY compile to a conditional move. [T-303]
- In `.lgo` files `*` inside a `make` length or capacity MUST lower to `logos/rt.MulLen`. [T-303]
- An overflowing `make` length MUST panic with `logos: make length overflow: <expr>`. [T-303]
- A negative `make` length MUST panic. [T-303]
- The call below panics with `logos: make length overflow: w * h` on overflow. [T-303]

```lgo
buf := make([]Pixel, w * h)
```

- An explicit `clear` of a slice MUST persist before return. [T-303]
- Cache-line spacing MUST keep the upstream 128-byte constant with no added padding. [T-303]

### 15.3 Long-loop preemption on wasm

- Back-edge checks MUST fire every 4096 iterations only where async preemption is absent (GOARCH=wasm). [T-303]
- A call-free loop MUST NOT starve its peers. [T-303]
- Check overhead has target: below 0.01%, measured by BenchmarkLogosLoopStride. [T-PERF]

## 16. JSON diagnostics schema

### 16.1 Emission

- `logos lint --json-diagnostics` and `logos build --json-diagnostics` MUST write one JSON object per line to stderr for every compiler and lint diagnostic in `.lgo` files and in Go files. [T-Sec16][R-U1]
- Each line MUST be a complete object with schema version `1.0`. [T-Sec16]

### 16.2 Top-level fields

- Field `schema_version` is a string and its value is `1.0`. [T-Sec16]
- Field `severity` is a string with one of `forbid`, `deny`, or `warn`. [T-Sec16]
- Level `allow` emits no diagnostic. [T-Sec16]
- Field `rule_id` is a string of the form `group::name` for a lint diagnostic, and the empty string for a compiler diagnostic that carries no rule ID, including each Go compiler diagnostic. [T-Sec16][R-U1]
- Field `file` is a string with the path of the source file. [T-Sec16]
- Field `message` is a string with the human-readable diagnostic text. [T-Sec16]
- Field `span` is an object with the line, column, and byte range. [T-Sec16]
- Field `provenance` is an object with the source of the inferred type. [T-Sec16]
- Field `autofix_candidates` is an array of fix objects. [T-Sec16]

### 16.3 Span object

- Object `span` holds six integers called `start_line`, `start_col`, `end_line`, `end_col`, `byte_offset`, and `byte_len`. [T-Sec16]

### 16.4 Provenance object

- Object `provenance` holds `origin_expr` as a string, `origin_line` as an integer, and `inferred_type` as a string. [T-Sec16]

### 16.5 Autofix objects

- Each candidate holds `description` as a string and `replacement` as a string. [T-Sec16]
- Candidates that insert text also hold `insert_before_line` as an integer. [T-Sec16]

## 17. Red-team ledger

Each entry from #1 to #182 states the tailored resolution of its original attack. Each entry from #183 on is new and comes from the two-way interop direction of 2026-10-19. `REDTEAM.md` records test paths, milestone ownership, and status; it does not duplicate these resolutions. The original ledger is preserved in `archive/red-team-ledger.md`.

- #1 Split-union pointer scalar overlap confusion: Unions MUST use the upstream interface representation. Ejected value unions MUST keep methods with a member comment. See Sec 2.3. [T-203.1]
- #2 String constBytes mutated via surviving alias: String to const byte views MUST allocate nothing. Views of other provenance to string MUST copy. See Sec 2.1. [T-201]
- #3 Rodata map breaks from runtime hash seed: Const maps MUST lower to a switch up to 8 keys. Larger maps MUST use a static perfect hash in read-only data. See Sec 9.1. [T-Sec9]
- #4 Side-stack arena growth across parked goroutines: A goroutine parked with empty arena regions MUST return its leased chunk. Such a goroutine MUST retain no arena capacity; its ordinary stack and runtime metadata still exist. See Sec 8.3. [T-301.1]
- #5 Cgo fastcall blocks OS thread forever: The fast path MUST record the call start. Sysmon MUST retake the P from a fast call under the same rule and threshold as from a syscall. See Sec 3.4. [T-304]
- #6 Block lambda return ambiguity inside map: Lambdas MUST allow expression and block forms with known target types. A ? in an expression lambda MUST propagate from the enclosing function. See Sec 1.4. [T-104.6]
- #7 Non-nil pointer bypass via arrays maps JSON: Zero values of non-nil pointers MUST fail with correctness::nonnil_uninitialized. Absent map and JSON values MUST check through V? and MissingFieldError. An absent key MUST read as none while a present nil payload MUST read as some. See Sec 2.2. [T-202][T-Sec12]
- #8 Go functions returning nil into lgo: Pointers from Go declarations, in the standard library and in every other Go package, MUST act as platform pointers. Stores into non-nil locations MUST panic with logos: nil from Go value at the source line. See Sec 2.2. [T-202][R-U2]
- #9 Text cold RVO corrupting debugger output: Cold successors MUST stay last inside their function. Traces MUST print base names without the `·logosspec<N>` or `·logosct<N>` suffix. See Sec 1.5, Sec 2.3, and Sec 2.5. [T-105][T-203.4][T-205]
- #10 Exhaustive union switch bypassed via default: A default over a closed union MUST report pedantic::exhaustive_union_switch when the pedantic group is enabled; that rule defaults to allow and is silent otherwise. Exhaustive switches MUST still list every member or default. See Sec 1.2. [T-102]
- #11 Async preemption migrating G mid-arena use: Arena objects in use MUST stay reachable across preemption. The bump sequence MUST be an async-preemption unsafe point. The collector MUST scan scan-region objects with the owning goroutine stack. See IMPLEMENTATION.md M18-S3. [T-301.1]
- #12 Inline bump allocator corrupting mark bits: The inline fast path MUST serve noscan objects up to 512 bytes. All other objects MUST take the slow path. See IMPLEMENTATION.md M18-S3. [T-301.1]
- #13 Concurrent multiword union write causing confusion: Unlocked sharing of a mutable multiword value with a child block plus a write on either side MUST fail with concurrency::concurrent_multiword_race. Const views and logos/sync types MUST pass. See Sec 1.2. [T-102]
- #14 If x nil promotion breaking redeclarations: Block scoping MUST stay standard Go. Outer variables MUST narrow after if x == nil return guards. See Sec 2.2. [T-202]
- #15 Auto-generification exploding binary size megamorphic: Specialization MUST stop at type depth 4. Sites with more than 4 types MUST keep the interface call. See Sec 2.3. [T-203.4]
- #16 Auto-clip subslicing breaking buffer reuse loop: Ordinary reslicing such as s = s[:0] MUST stay allowed. Only const views MUST enforce the len bound. See Sec 2.1. [T-201]
- #17 Tuple colliding with multi-return unpacking: Direct calls in multi-assignments MUST unpack. Single-variable bindings MUST pack into tuple values. See Sec 2.4. [T-204.1]
- #18 Named return shadowing corrupting defer close: A bare return that a shadowed named error result would change MUST fail as upstream Go reports it. errdefer MUST run on panic, Goexit, or non-nil error. See Sec 1.4. [T-104.2]
- #19 Comptime ViewAs unaligned trap padding garbage: ViewAs MUST accept only POD structs with no padding, no bool, no pointers, interfaces, slices, maps, or strings, and MUST keep the source alive. Short inputs MUST panic with logos: ViewAs need-have sizes. See Sec 2.5. [T-205]
- #20 Side-Stack object pointing to Call-Stack var: Locals whose addresses flow into arena objects MUST allocate in the arena. Stack growth MUST NOT invalidate arena pointers. See IMPLEMENTATION.md M18-S3. [T-301.1]
- #21 Slowloris exhausting pinned iouring buffers: The build MUST use standard netpoll with no per-connection pinned buffer. Ten thousand idle connections MUST grow the heap by less than 1 KB each. See IMPLEMENTATION.md M8-S5. [T-206]
- #22 AVX vector tail read crossing protection: Slice reductions MUST use fault-suppressing masked loads for tails. The build MUST add no guard padding elsewhere. See Sec 15.2. [T-303]
- #23 Accessing valid res when defer close fails: Reads of .lgo results before err == nil proof MUST fail with correctness::result_before_error_check. Standard library callees MUST keep their own contracts. See Sec 2.2. [T-202]
- #24 Non-deterministic map iteration inside const func: Dynamic map iteration in const funcs MUST fail compilation. Const map range MUST visit keys in sorted order. See Sec 2.1. [T-201][T-Sec9]
- #25 Transitive dependencies bypassing reflect ban unsafely: Transitive unsafe use MUST fail under transitive_unsafe severity deny unless its package matches transitive_unsafe_allowlist. Violations MUST list the shortest import chain. See Sec 10.4 and Sec 13.2. [T-Sec10][T-Sec13]
- #26 ARM64 weak ordering exposing Hot fields: Concurrent Hot updates MUST NOT expose partial values. Hot stores MUST run the hybrid write barrier. See Sec 2.6. [T-206]
- #27 WASM memory grow detaching JS views: The loader MUST re-read memory.buffer after every wasm call. Calls after growth MUST see current memory. See Sec 14.2. [T-Sec14]
- #28 RVO destination aliasing corrupting input reverse: Destination passing MUST NOT exist. Self-aliased calls such as arr = Reverse(arr) MUST return correct results. See IMPLEMENTATION.md M13-S3. [T-105][T-203.4]
- #29 ZGC colored pointers rejected by kernel: Every syscall and cgo entry MUST untag pointers. Color bits MUST survive addressing on ARM64 with TBI and on x86 with LAM or UAI. See Sec 19.2. [T-301.2]
- #30 Self-modifying inline cache causing cache thrashing: Devirtualization MUST stay static with no runtime writes. No call-site cache MUST exist. See Sec 2.3. [T-203.4]
- #31 Stream fusion over-executing side effects take: Fused take(5) loops MUST run map functions at most 5 times. Side effects before sortBy MUST warn suspicious::side_effect_before_sort. See Sec 1.4. [T-104.6][T-102]
- #32 Execution order chaos between defers: `scope defer` MUST run at block exit. `defer`, `defer?`, and `errdefer` MUST share one LIFO order at function exit. See Sec 20.1. [T-104.2][T-104.7]
- #33 Stack-first allocation exceeding its frame buffer: The stack buffer MUST hold 256 bytes at O2 and 4096 bytes at O3. Larger allocations MUST use the enabled arena when eligible, or the heap otherwise. See Sec 3.2. [T-302][T-301.1]
- #34 String interpolation allocating heap intermediate strings: Interpolation MUST evaluate each operand once, left to right, and MUST convert error and fmt.Stringer operands to strings before sizing. The bound MUST count those strings, so the lowering uses one allocation and the buffer MUST never regrow. See Sec 1.4. [T-104.4]
- #35 Go test race detector incompatibility with custom passes: Arena reuse under `-race` MUST report no false race. Watermark moves MUST stay visible to the race detector. See IMPLEMENTATION.md M18-S3. [T-301.1]
- #36 Overlapping union tags and infinite complement sets: A union with `~` terms MUST fail with `correctness::open_union_value`. A closed union MUST use the upstream interface representation. See Sec 2.3. [T-203.1]
- #37 Emit-go failing on exported const switch functions: An exact `const switch` case MUST lower to a run-time type switch with rebinding. An interface case MUST assert on the value. See Sec 2.5. [T-205]
- #38 Parser backtracking explosion on lambdas versus tuples: An opening `(` MUST parse its element list once and then branch. A `=>` after `)` MUST form a lambda. See Sec 5.3. [T-104.6]
- #39 Variadic ambiguity between one tuple and two args: A direct multi-value call in call position MUST unpack. A tuple-typed variable MUST pack the values. See Sec 2.4. [T-204.1]
- #40 Comparing unions and tuples containing slices: A tuple with slice, map, or func elements MUST fail as non-comparable. A closed union MUST use upstream interface equality. See Sec 2.4. [T-204.1][T-203.1]
- #41 Mutating source slice inside filter and map: Chains MUST treat lambda parameters as const views. Mutation of the receiver inside a chain lambda MUST fail compilation. See Sec 1.4. [T-104.6]
- #42 Defer close with error allocating join slice: `defer f()?` with nil result MUST leave the error result nil with no allocation. With a prior error it MUST join both errors. See Sec 1.4. [T-104.2]
- #43 Zero-sized union member representation: A closed union MUST use the upstream interface representation, including upstream handling of zero-sized members. See Sec 2.3. [T-203.1]
- #44 Cross-package nil and const inference slowing compilation: Cross-package facts MUST travel in `__.LOGOS` with a fingerprint. A stale fingerprint MUST fail the dependent build. See Sec 6.1. [T-Sec6]
- #45 Const switch rebinding slice of files as writers: An exact case MUST rebind the value to the case type. An interface case MUST keep the concrete layout and gain methods. See Sec 2.5. [T-205]
- #46 Side-stack arena hiding heap pointers from GC: The arena MUST hold a bottom-up noscan region and a top-down scan region. The collector MUST scan scan-region objects with the goroutine stack. See Sec 8.3. [T-301.1]
- #47 Const func purity broken by recover and addresses: A const func MUST NOT perform I/O, spawn a goroutine, read a mutable package variable, call recover, convert a pointer to uintptr, range over a dynamic map, or use a pointer. A violation MUST fail with correctness::const_func_impure naming the reason. See Sec 2.1. [T-201]
- #48 Auto-generification breaking function value assignment: The original function MUST stay callable through a function value. Each clone MUST carry a unique package-local name. See Sec 2.3. [T-203.4]
- #49 Optional union representation: Interface-backed union options with no nil-capable member MUST use the interface nil value for absence. An option whose payload can be nil MUST carry explicit presence as specified in Sec 2.2. Non-nilable value options and nested options MUST use `logos/rt.Option[T]` as specified in Sec 2.2. [T-202]
- #50 Const rodata slices rejected by reflection and print: reflect.DeepEqual and fmt.Println MUST accept constant composites, and the M5 tests MUST pin that behavior. See Sec 2.1. [T-201]
- #51 Infinite compile-time loop freezing the IDE: Compile-time evaluation MUST stop after the selected `max_ctfe_steps` limit with default 100,000 steps with `comptime::ctfe_step_limit`. Configuration MAY raise the limit. A value above 1 MB MUST fail with `comptime::const_too_large`. See Sec 2.1. [T-201][T-102]
- #52 Inlined defer inside loop exhausting stack slots: A `scope defer` in a loop body MUST run every iteration. It MUST use open-coded defers with no separate unwind table. See Sec 20.1. [T-104.7]
- #53 Recursive AST escaping local arena on return: A returned arena value that the caller keeps MUST allocate in the caller region. It MUST live until the caller exits. See Sec 8.3. [T-301.1]
- #54 Rodata constants crashing under address relocation: Pointer-free constant data MUST live in read-only memory. Constant data with pointers MUST live in read-only-after-relocation memory. See Sec 2.1. [T-201]
- #55 Ignored error leaving stale trace frames: A different error value MUST reset the trace. The trace MUST keep the last 8 sites. See Sec 1.4. [T-104.1]
- #56 Inlined scope defer evaluating args too late: The function, receiver, and arguments of a `scope defer` MUST evaluate at the statement. They MUST NOT re-evaluate at block exit. See Sec 20.1. [T-104.7]
- #57 Overlapping const switch cases shadowing each other: Cases MUST evaluate top-down in first-match order. A shadowed case MUST fail with `comptime::unreachable_case`. See Sec 2.5. [T-205]
- #58 Duplicate specialized clones colliding across packages: Clones MUST carry unique package-local names. No name-based dedup pass MUST pick among clones. At Os/Oz, the linker MAY fold content-identical functions (T-105). See Sec 2.3. [T-203.4][T-105]
- #59 Cgo pointer check panicking on arena addresses: Cgo pointer checks MUST accept arena addresses without panic. See Sec 8.3. [T-301.1]
- #60 Value struct with non-nil pointer zero-cleared: Zero-clearing through `*p = S{}` MUST fail with `correctness::nonnil_uninitialized`. A literal MUST initialize every non-nil field. See Sec 2.2. [T-202]
- #61 Nested option none versus some nil collision: The spelling `T??` MUST fail as a syntax error. `(T?)?` MUST lower to `Option[T?]`. See Sec 2.2. [T-202]
- #62 Mutating method called on const receiver: A call through a const view to a pointer-receiver method that lacks const MUST fail with correctness::const_view_mutation. A method with a const pointer receiver MUST allow the call. See Sec 2.1. [T-201]
- #63 panic leaking arena watermarks on unwind: Frame exit, including exit by panic, MUST restore the watermark. The restore MUST run as an emitted defer during unwinding. See Sec 3.1. [T-301.1]
- #64 concurrent mutation between interpolation sizing passes: Interpolation operands MUST evaluate once from left to right. See Sec 1.4. [T-104.4]
- #65 comptime fields exposing private fields across packages: Expansion MUST include unexported fields only for the current package. It MUST exclude unexported fields of other packages. See Sec 2.5. [T-205]
- #66 errdefer with unnamed returns failing in eject: Errdefer MUST work in functions with unnamed error results. See Sec 1.4. [T-104.2]
- #67 FMA contraction causing x86 ARM64 sum drift: Float sums MUST use fixed pairwise order with no implicit contraction. Results MUST stay bit-identical across x86_64 and ARM64. See Sec 1.4. [T-104.6]
- #68 group returning early while sibling reads stack: The group MUST cancel on first error. It MUST join all children before scope exit. See Sec 2.6. [T-206]
- #69 next stripping comments and build tags: Conversion MUST use byte-span splicing. It MUST keep every comment and build line byte-identical outside rewrites. See Sec 1.3. [T-103]
- #70 constant slice literal allocating on heap: Pointer-free constant data MUST live in read-only memory. See Sec 2.1. [T-201]
- #71 scan region dead pointers breaking barrier reuse: The runtime MUST zero popped regions before reuse. See Sec 3.1. [T-301.1]
- #72 ViewAs interior pointer confusing collector scan: The view MUST keep the backing bytes alive. A short input MUST panic with need and have byte counts in `logos: ViewAs` text. See Sec 2.5. [T-205]
- #73 Const-map seed search fails to terminate: Every const map with more than 8 keys MUST provide one-probe lookup with constant-time auxiliary metadata reads plus exactly one final key/value-slot probe with full-key validation. Construction MUST use at most 1,000,000 seed-candidate trials in total per const map. A logarithmic fallback MUST NOT satisfy the contract. Exhaustion MUST fail with `comptime::constmap_seed_not_found`. The 1, 8, 9, and 10,000-key cases in `test/logos/run/const_map.lgo` MUST be the acceptance check. That check MUST NOT promise success for every key set. Iteration MUST visit keys in sorted order. See Sec 9.1. [T-Sec9]
- #74 FuncForPC returning mangled specialized names: Traces MUST print the base name without the specialization suffix. See Sec 2.3. [T-203.4]
- #75 order dependent inference in nil switch arms: Switch results MUST take the common type of the arms. See Sec 2.3. [T-203.1]
- #76 const for skipping checks on unrolled bodies: Each unrolled body MUST pass full assignment and call checks. Pruned code MUST NOT skip verification. See IMPLEMENTATION.md M7-S2. [T-205]
- #77 plain defer clearing error after errdefer: A plain deferred literal MUST NOT assign nil to the error result with errdefer present. The compiler MUST reject such code with `correctness::defer_erases_errdefer_state`. See Sec 1.2. [T-102]
- #78 wasm32 alignment trapping 64-bit atomics: Wasm MUST align 64-bit types on 8 bytes. Atomics MUST stay naturally aligned with no forcing code. See Sec 14.1. [T-Sec14]
- #79 eject helper imports colliding with user names: Lowered files MUST import only the reserved support aliases. The output MUST NOT emit colliding helper imports. See Sec 0.3. [T-0.3]
- #80 cross layer regression matrix verification: The suite MUST pass the full std and cmd suites at O1, O2, and O3. See Sec 22.1. [T-Sec22]
- #81 concurrent collector racing scan zeroing: Collection after exit MUST find no stale pointers in popped scan memory. See Sec 8.3. [T-301.1]
- #82 split union needing custom GC bitmap: Unions MUST use the upstream interface representation. A two-tier tag layout MUST NOT be used. See Sec 2.3. [T-203.1]
- #83 escaping const byte to string zero copy mutation: Zero-copy to string MUST apply only with string or const composite provenance. All other provenances MUST copy. See Sec 2.1. [T-201]
- #84 eject evaluating chain receiver twice: Chains MUST fuse into one loop. The receiver MUST evaluate once before the loop. See Sec 1.4. [T-104.6]
- #85 typed nil error triggering errdefer: A typed-nil error MUST count as non-nil for errdefer. See Sec 1.4. [T-104.2]
- #86 const switch losing outer constraint methods: Interface cases MUST keep the concrete layout. They MUST gain the case interface method set. See Sec 2.5. [T-205]
- #87 errdefer skipped on panic or Goexit exit: Errdefer MUST run on panic and Goexit exits. A panic during return evaluation MUST count as failure. See Sec 1.4. [T-104.2]
- #88 fastcall corrupting callee saved registers: Fast calls MUST carry noescape and nocallback marks. Sysmon MUST retake the P from a fast call under the same rule and threshold as from a syscall. See Sec 3.4. [T-304]
- #89 overlapped scalar union leaving tail garbage: Unions MUST use the upstream interface representation. See Sec 2.3. [T-203.1]
- #90 recover inside emitted errdefer corrupting frames: Errdefer MUST stay skipped on success returns. See Sec 1.4. [T-104.2]
- #91 const switch ambiguity on defined int types: Exact cases MUST match only the named type. Tilde cases MUST match underlying types. See Sec 2.5. [T-205]
- #92 polymorphic recursion causing infinite specialization: Specialization MUST stop at type depth 4. See Sec 2.3. [T-203.4]
- #93 malicious read beyond len on const bytes: A bound above len MUST fail with `correctness::const_view_mutation`. Non-constant bounds MUST trap at run time. See Sec 2.1. [T-201]
- #94 Slow context extraction on large repositories: `logos context` MUST print the declaration and one level of referenced signatures without function bodies, capped at 100 lines. Latency has target: 50 ms, measured by `BenchmarkContext`. See Sec 1.3. [T-103][T-PERF]
- #95 Negative or huge take and drop indices: `take`, `drop`, and `get` MUST clamp indices without panic. See Sec 1.4. [T-104.6]
- #96 Early return skipping errdefer armed flag: `errdefer` MUST run when the frame holds a non-nil error, panics, or exits through Goexit. A panic while evaluating return values MUST count as failure. See Sec 1.4. [T-104.2]
- #97 Stale bytes in union struct variant slot: Union values MUST use the upstream interface representation. A two-tier tag layout MUST NOT be used. See Sec 2.3. [T-203.1]
- #98 DeepEqual on read-only composite constants: `reflect.DeepEqual` MUST accept constant composites. Go 1.27 already accepts them, so the test MUST pin that behavior. See Sec 2.1. [T-201]
- #99 Method call on an unchecked option: An ordinary method call on an option without a non-nil proof MUST fail with `[correctness::nil_deref_unproven]`. Code MAY instead use optional chaining or a nil guard. See Sec 2.2. [T-202]
- #100 Oversize fixed-array SIMD operator use: Element-wise operators MUST allow at most 64 bytes. A larger array MUST fail with `[correctness::simd_array_too_large]`. See Sec 2.4. [T-204.2]
- #101 Race false positives on reused arena chunks: Race builds MUST reset shadow memory on watermark moves. Reused chunks MUST report no false race. See IMPLEMENTATION.md M18-S3. [T-301.1]
- #102 CTFE integer width across compile targets: The compiler MUST evaluate CTFE integers with target widths. See Sec 2.1. [T-201]
- #103 Negative zero and NaN float map keys: A `-0.0` float key MUST canonicalize to `+0.0`. A `NaN` key MUST fail compilation. See Sec 9.1. [T-201][T-Sec9]
- #104 Profilers breaking on missing frame pointers: A cold successor MUST sit last inside its function. It MUST NOT outline to the tail of text. See Sec 1.5. [T-105]
- #105 Codesign rejecting custom cold text section: A cold successor MUST sit last inside its function. It MUST NOT use a separate cold section. See Sec 1.5. [T-105]
- #106 Finalizer registered on arena allocated object: `runtime.SetFinalizer` on an arena object MUST panic with `runtime: SetFinalizer on loGo/s arena object`. See Sec 8.3. [T-301.1]
- #107 Native CPU settings break reproducibility: The deterministic target MUST select GOAMD64=v3 and GOARM64=v8.2 unless the environment overrides them. Host-specific selection MUST require `target_cpu = "native"`. See Sec 1.5. [T-105]
- #108 Emit-go fallback for interface switch cases: Interface cases in run-time `const switch` MUST use assertion on the value. Native and ejected output MUST agree. See Sec 2.5. [T-205]
- #109 Conversion changes unrelated source text: `logos next` MUST preserve comments and build lines outside its edits. The toolchain MUST NOT add a `logos fmt` rewriting command. See Sec 1.3. [T-103]
- #110 Frozen view sharing mutable heap child: `logos/rt.Freeze` MUST deep-copy its argument and return a `const` view. A later write to the source or its child MUST NOT change the view. See Sec 2.1. [T-201]
- #111 Unsafe ARM64 load store pair fusion: Pairs MUST keep distinct destinations with encodable offsets only. Go 1.27 already enforces this and the test MUST pin it. See Sec 15.1. [T-303]
- #112 Generated files clashing with source files: Ejected output MUST NOT emit `.gen.go` file names. It MUST NOT emit a `//go:build !logos` line. See Sec 0.3. [T-0.3]
- #113 Inlined defer leaking resources in loops: A scope defer in a loop body MUST run every iteration. Cleanup on panic MUST use open-coded defers. See Sec 20.1. [T-104.7]
- #114 Field reordering breaking serialized field order: Expansion MUST visit fields in declaration order. `val.[f]` MUST read and write the named field. See Sec 2.5. [T-205]
- #115 Typed nil inside union interface value: A `nil` union value MUST go to `case nil`. Exhaustive switches MUST cover every member and `nil`. See Sec 2.3. [T-203.1]
- #116 Mutating value-boxed struct through interface: A read-write property MUST require a pointer type. Use of a value type MUST fail with `[correctness::property_needs_pointer]`. See Sec 6.3. [T-203.2]
- #117 Taking address of interface property value: Taking the address of a property MUST fail with `[correctness::property_address]`. See Sec 6.3. [T-203.2]
- #118 Embedded fields satisfying property contract depth: Embedded fields MUST satisfy properties by Go selector depth rules. An ambiguous equal-depth selector MUST fail as ambiguous. See Sec 6.3. [T-203.2]
- #119 Spread dropping mismatched same-name fields: Spread MUST copy fields with identical names and identical types. A different type MUST fail with `correctness::spread_type_mismatch`. See Sec 1.4. [T-104.5]
- #120 Synthetic getters polluting reflection method count: Only a type with a `: I` list MUST gain accessor methods. `reflect` MUST count the generated accessors. See Sec 6.3. [T-203.2]
- #121 Slice of pointers to interface slice copy: A `const []Creature` parameter receiving `[]*Monster` MUST specialize with no copy loop. No run-time call-site cache MUST exist. See Sec 2.3. [T-203.4]
- #122 Emoji file extensions with variation selectors: The loader MUST strip trailing U+FE0E and U+FE0F before matching. NFC normalization MUST be the identity on extension strings. See Sec 1.1. [T-101.1]
- #123 Default value overriding explicit JSON false: Decoding MUST fill defaults first, then parse. Duplicate keys MUST keep the last value. See Sec 12.2. [T-Sec12]
- #124 Specialized call breaking dynamic type assertion: A dynamic assertion in a specialized clone MUST give the same result as the unspecialized call. Function values MUST keep working. See Sec 2.3. [T-203.4]
- #125 errdefer rollback overwriting primary return error: For `errdefer f()?`, a rollback error MUST join the primary error with errors join semantics, as `defer f()?` does. A nil primary error MUST take the rollback error as is. See Sec 1.4. [T-104.2]
- #126 interpolation calling nextID twice in sizing: Interpolation operands MUST evaluate once from left to right. Each `{expr}` MUST bind to a local variable before sizing. See Sec 1.4. [T-104.4]
- #127 Interface default methods causing diamond ambiguity: Method bodies inside interfaces MUST fail compilation. Structs MUST satisfy property interfaces only through an explicit `: I` declaration. See Sec 6.3. [T-203.2]
- #128 Resolve failing before const for unrolls: Comptime generics MUST expand into non-generic copies in the instantiating package. `Resolve[f.Type](c)` MUST resolve per field in the copy. See Sec 2.5. [T-205]
- #129 Circular DI ServiceA ServiceB crashing compiler: A circular instantiation MUST fail with `[comptime::circular_instantiation]` naming the chain. The check MUST track the active instantiation stack. See Sec 2.5. [T-205]
- #130 MapTo failing on same-named distinct field types: Field copies MUST compare compile-time types through `f.Type`; `Resolve[f.Type](c)` resolves per field. A copy whose field types differ MUST fail compilation as an ordinary type mismatch. See Sec 2.5. [T-205]
- #131 Embedding read-only and read-write same property: Read-only and read-write declarations of one name MUST merge into read-write. The package MUST generate both getter and setter for it. See Sec 6.3. [T-203.2]
- #132 Compound property update evaluating receiver twice: A compound update through an interface MUST evaluate the receiver once and write the value back. See Sec 6.3. [T-203.2]
- #133 Non-ASCII module paths rejected by proxy: When the main module path's first element contains a dot, each alias-extension file MUST produce warning `portability::alias_extension`. Module zips reject alias extensions, so such files cannot publish. See Sec 1.1. [T-101.1][T-102]
- #134 O3 inlining inflating recursive stack frames: O3 MUST set inline budget 400 with budget 80 inside recursive cycles. Recursive functions MUST NOT inline above that budget. See Sec 1.5. [T-105]
- #135 defer Close violating LIFO order: Deferred calls MUST run in LIFO order across `defer`, `defer?`, and `errdefer`. `defer f()?` MUST run in place on the standard LIFO stack. See Sec 1.4. [T-104.2]
- #136 go group sibling hanging on channel receive: Plain channel operations in a child MUST lower to a `select` with `<-gctx.Done()`. Cancellation MUST reach blocked siblings. See Sec 2.6. [T-206]
- #137 Call-free loop starving scheduler blocking GC: Back-edge checks MUST fire every 4096 iterations where async preemption is absent. A call-free loop MUST NOT starve its peers. See Sec 15.3. [T-303]
- #138 Arena pointer stored into sync Pool: A value stored into `sync.Pool` MUST stay on the heap. The compiler MUST keep such values out of the arena. See Sec 8.3. [T-301.1]
- #139 Malicious JSON array triggering huge preallocation: JSON decoders MUST grow arrays incrementally with no pre-allocation cap. Binary counts MUST pre-allocate at most `min(len(rest)/2, 1 MB / sizeof(elem))`. See Sec 12.2. [T-Sec12][T-Sec12.1]
- #140 Known hash seed enabling hash-flooding DoS: Large const maps MUST provide one-probe lookup from a static perfect-hash table in read-only data with constant-time auxiliary metadata reads plus exactly one final key/value-slot probe with full-key validation. Construction MUST use at most 1,000,000 seed-candidate trials in total per const map. A logarithmic fallback MUST NOT satisfy the contract. Exhaustion MUST fail with `comptime::constmap_seed_not_found`. See Sec 9.1. [T-Sec9]
- #141 Method call on property value losing mutation: A method call on a property value MUST use read-call-write-back around the call. The write MUST reach the struct. See Sec 6.3. [T-203.2]
- #142 Chain map leaking file descriptors on failure: Partial chain failure MUST close already produced `io.Closer` elements in reverse order. The fused loop MUST run that rollback before returning. See Sec 1.4. [T-104.6]
- #143 Buffered channel retaining dead pointer after receive: Go 1.27 already clears the received slot, and the test MUST pin that a received pointer becomes collectable. See IMPLEMENTATION.md M8-S5. [T-206]
- #144 Stack slice passed to CGo invalidated: A slice that flows into a cgo argument or an `unsafe.Pointer` conversion MUST NOT use the stack buffer. Such slices MUST allocate on the heap. See Sec 3.2. [T-302]
- #145 Field versus embedded method selector ambiguity: Embedded fields MUST satisfy properties by Go selector depth rules. An equal-depth match MUST fail as an ambiguous selector. See Sec 6.3. [T-203.2]
- #146 Self spread copying large struct each tick: Self-update `x = T{f: e, ..x}` MUST lower to one tuple assignment of the explicit fields. It MUST perform no full struct copy. See Sec 1.4. [T-104.5]
- #147 Background goroutine panic crashing server process: A child panic MUST become `*logos/rt.PanicError` with the value and a stack. The group MUST cancel siblings and return it. See Sec 2.6. [T-206]
- #148 Concurrent multiword interface slice tearing: Sharing a mutable multi-word value with a child plus a write MUST fail with `concurrency::concurrent_multiword_race`. Const views and `logos/sync` types MUST stay allowed. See Sec 1.2. [T-102]
- #149 Dead-store elimination deleting clear before return: Go 1.27 lowering already keeps `clear(b)` as a runtime memory clear; the test MUST pin its survival before return. Dead-store elimination MUST NOT remove that clear. See Sec 3.3. [T-303]
- #150 Triple nested const for exploding compiler AST: An expansion above the selected `max_comptime_ast_nodes` limit with default 4,096 nodes MUST fail with `[comptime::node_limit]`. An expansion at or below the selected limit MUST compile. Configuration MAY raise the limit. See Sec 2.5. [T-205][T-102]
- #151 Group sibling stuck in blocking socket read: A child blocked in netpoll-backed network I/O MUST wake within 100 ms with a timeout error. Goroutines outside the group MUST stay blocked. See Sec 2.6. [T-206]
- #152 Zero-copy string crashing after munmap: `const []byte` to `string` MUST be zero-copy only from string or const-composite provenance. Any other provenance MUST copy to the Go heap. See Sec 2.1. [T-201]
- #153 Hot Store leaking retire-list memory: No retire list MUST exist for `Hot[T]`. Stores MUST run the hybrid write barrier with GC-managed values. See Sec 2.6. [T-206][T-PERF]
- #154 Global 16-byte alignment inflating struct padding: The toolchain MUST NOT change upstream alignment rules. `Hot[T]` MUST hold its value through `atomic.Pointer[T]` only. See Sec 2.6. [T-206]
- #155 SIGURG arriving mid-bump double allocating address: The noscan fast path MUST be an inline add-compare sequence with a slow-path call. The sequence MUST be marked as an async-preemption unsafe point. See IMPLEMENTATION.md M18-S3. [T-301.1]
- #156 Caller reading hash before Close error check: For callees declared in .lgo the compiler MUST fail reads before the err proof with [correctness::result_before_error_check]. Standard-library callees MUST keep their own contract, so use of n first stays allowed. See Sec 2.2. [T-202]
- #157 Unconditional self-recursion in const switch branch: The compiler MUST fail a function whose every path calls itself with identical arguments before return with [correctness::infinite_recursion]. See Sec 1.2. [T-102]
- #158 Substring pinning large backing array in heap: Escape analysis MUST warn with [perf::large_backing_array_pin] when a subslice or substring of a backing array above 64 KB escapes. See Sec 1.2. [T-102]
- #159 Moving collector breaking pointer-keyed map lookup: Pinned blocks MUST NOT evacuate. Pointer-keyed maps under zgc MUST use a side-table identity hash that survives relocation. See Sec 8.4. [T-301.2]
- #160 Collection moves live cgo arguments: Go pointer arguments MUST remain valid throughout a fast call under every GC mode. Temporary pin records MUST be released after return without clearing the caller's argument values. See Sec 3.4. [T-304]
- #161 Removing mark assist causing out-of-memory failure: Greentea background goal MUST follow 0.25 below u 0.80, linear rise to 0.75 below u 0.95, and 0.75 at or above u 0.95. Mark assists MUST still run when u reaches 0.95 or more. See Sec 19.1. [T-301.2]
- #162 Strict lint breaking third-party module libraries: Third-party modules of either kind MUST get no code lints. Capability checks MUST apply to every module in the build. See Sec 1.2. [T-102][R-U1]
- #163 Buffered channel send flagged as unbounded operation: The compiler MUST allow a channel operation outside a select with a done case only for a locally made channel with constant capacity and send count within capacity. Other operations MUST fail with [correctness::unbounded_channel_op]. See Sec 1.2. [T-102]
- #164 Benign writer flagged as discarded error: The compiler MUST allow always-nil-error callees and benign writers to standard output, standard error, byte buffers, and string builders. Other dropped errors MUST fail with [slop::discarded_error]. See Sec 1.2. [T-102]
- #165 Fastcall overflowing goroutine stack or handler crash: The fast path MUST switch stacks like asmcgocall and run the C call on the system stack. Profiling during the call MUST NOT crash the process. See IMPLEMENTATION.md M20-S1. [T-304]
- #166 Batch mode dying on unexpected memory spike: Batch MUST trigger automatic collection when heap use reaches 90 percent of `GOMEMLIMIT`, or of physical memory when `GOMEMLIMIT` is unset; that collection MUST stop the world. Stores under batch MUST be plain with no barrier. See Sec 3.1. [T-301.2]
- #167 Nested WithLock causing self-deadlock: The compiler MUST fail a WithLock call on the same receiver inside its own closure call graph with [concurrency::reentrant_withlock_deadlock]. See Sec 1.2. [T-102]
- #168 ZGC crashing on CPUs lacking address tagging: Without hardware tagging loads MUST test the color and unmask to the canonical address. Every syscall and cgo entry MUST untag pointers. See Sec 19.2. [T-301.2]
- #169 Make length integer overflow allocating tiny buffer: In .lgo a * inside a make length or capacity MUST lower to logos/rt.MulLen. On overflow MulLen MUST panic with logos: make length overflow: w * h. See Sec 15.2. [T-303]
- #170 Time.After inside loop leaking timers: The compiler MUST warn with [perf::time_after_in_loop] against time.After inside a loop body. It MUST NOT hoist timers across iterations. See Sec 1.2. [T-102]
- #171 ZGC multi-mapping inflating memory reporting: ZGC MUST use a single mapping with software unmasking. It MUST NOT multi-map pages. See Sec 19.2. [T-301.2]
- #172 Hoisted timer firing from undrained channel: The compiler MUST warn with [perf::time_after_in_loop] against time.After inside a loop body. It MUST NOT hoist timers, so no drain bug can occur. See Sec 1.2. [T-102]
- #173 SIGPROF during fastcall causing stack panic: Profiling with SIGPROF during a fast call MUST NOT crash the process. See Sec 3.4. [T-304]
- #174 Const-if dead branch triggering unused lint: Both const-if branches MUST type-check. Only the live branch MUST reach lowering. See Sec 2.5. [T-205]
- #175 Finalizer resurrection during moving collection: Finalizer and cleanup attachment MUST exclude the containing span from evacuation until span reuse without making that exclusion a GC root. Cleanup argument and closure references MUST remain valid when moved. Concurrent moving mode MUST remain unavailable until attachment and relocation are synchronized and verified. See IMPLEMENTATION.md M19-S5. [T-301.2][T-0.1]
- #176 Panic inside WithLock leaving mutex locked: WithLock MUST hold Lock, defer Unlock, then call the function. It MUST unlock after a panic. See Sec 2.6. [T-206]
- #177 User input causing format injection in interpolation: Operand text MUST NOT be read as a format string. User braces MUST print literally. See Sec 1.4. [T-104.4]
- #178 Spawning many goroutines exhausting memory in group: A go group with limit = N MUST bound concurrency with a semaphore. A go statement outside any group body MUST fail with [concurrency::unsupervised_goroutine]. See Sec 2.6. [T-206]
- #179 NaN breaking float sortBy ordering: sortBy MUST be a stable sort by cmp.Compare of keys. NaN keys MUST order deterministically. See Sec 1.4. [T-104.6]
- #180 Init order affecting const composite tables: The compiler MUST evaluate const composites at compile time with target widths. Pointer-free data MUST live in read-only memory before any init runs. See Sec 2.1. [T-201]
- #181 Long loop exhausting arena before return: A loop back-edge MUST restore the watermark for iteration-local objects. A loop with only iteration-local allocation MUST NOT grow arena use across iterations. See Sec 8.3. [T-301.1]
- #182 Unsized int overflowing in serialized struct: The compiler MUST warn with [portability::serialized_unsized_int] on an int or uint field in a struct with JSON tags. Sized kinds MUST be preferred. See Sec 1.2. [T-102]
- #183 loGo/s package unable to use a Go package from another module: A loGo/s package that imports a Go package from another module MUST build and run. See Sec 0.1. [T-0.1][R-U2]
- #184 Go package unable to use a loGo/s package from another module: A Go package that imports a loGo/s package from another module MUST build and run under `logos`. Plain `go` ignores `.lgo` files, so a plain `go` build of that Go module MUST fail with a load error that names the imported package. See Sec 0.1. [T-0.1][R-U2][A-LGOSEL]
- #185 Union used as constraint and value type crossing into Go: Such a union has no Go-facing form. A Go import of it MUST fail with the eject text. See Sec 2.3. [T-203.1][R-U2]
- #186 Go caller passing nil to a non-nil pointer parameter: When this toolchain compiles the Go caller, it MUST insert `logos/rt.MustNonNil` and MUST panic with the call position. A call through a function value, a method value, or an interface, and code that an upstream toolchain builds from eject output, stay documented holes. See Sec 6.1. [T-Sec6][A-GOBOUNDARY]
- #187 Wasm loader missing for a Go main package: `logos build` MUST emit the loader for a Go main package with a `.wasm` output, and `go build` MUST emit the same loader when `--wasm-loader` is passed. See Sec 14.2. [T-Sec14]
- #188 nocallback C function calling into Go on the fast path: A C function marked `nocallback` that calls into Go MUST panic on the fast path. It MUST NOT hang and MUST NOT corrupt state. See Sec 3.4. [T-304]
- #189 CPU profile losing fast-called C time: A CPU profile MUST attribute time in a fast-called C function to its Go call site. See Sec 3.4. [T-304]

## 18. Upstream file manifest

The build rules for upstream files live in `IMPLEMENTATION.md` under rule G3 and the per-slice file lists. [T-Sec18]
The original text of this section is archived in `archive/file-manifest.md`.

## 19. Pluggable GC state machines and memory layouts

The build MUST select the collector with `-gc=MODE` on `logos build` or `go build`, or with `logos.toml [build] gc` under `logos`. [T-301.2][R-U1]
MODE MUST be one of `greentea`, `zgc`, `immix`, `wasm`, `batch`, or `epsilon`. [T-301.2]
Any other mode MUST fail with `logos build: unknown -gc mode "x" (want greentea, zgc, immix, wasm, batch, epsilon)`. [T-301.2]
Under `logos`, the default MUST be `greentea`, or `wasm` when `GOARCH` equals `wasm`. [T-301.2]
Plain `go` MUST keep the upstream collector and pacing until `./run.bash` passes with a changed default, as the flip rule of Sec 0.1 states. [T-301.2][T-0.1]
Until phase 1 passes `go test -short std` with the experiment on, `-gc=immix` MUST fail with `logos build: -gc=immix requires GOEXPERIMENT=logosgcmoving phase 1, which is not complete`. [T-301.2]
Until phase 2 passes, `-gc=zgc` MUST fail with `logos build: -gc=zgc requires GOEXPERIMENT=logosgcmoving phase 2, which is not complete`. [T-301.2]
Batch MUST run no collection until heap use reaches 90% of `GOMEMLIMIT`. [T-301.2]
When `GOMEMLIMIT` is unset, the trigger MUST be 90% of physical memory. [T-301.2]
It MUST then run a stop-the-world collection. [T-301.2]
Programs compiled without write barriers MUST produce correct heaps under batch collection. [T-301.2]
Epsilon MUST NOT start a collection. [T-301.2]
Marking MUST stay single-threaded on wasm. [T-301.2]
Mark slices MUST run during idle time of the JS event loop. [T-301.2]
The heap MUST split into a low permanent region and a high scratch region. [T-301.2]
Growth MUST extend only the high region. [T-301.2]
The modes of this section MUST apply to Go packages and to loGo/s packages. [T-301.2][R-U1]
The plain `go` default MUST keep the upstream pacing until `./run.bash` passes with the `logos` default. [T-301.2][T-0.1][R-U1]
Under plain `go`, each `logos build:` message above MUST use the prefix `go build:`. [T-301.2]

### 19.1 Engine 1: Enhanced Green Tea GC

A pointer store into a span MUST set the mark bit and the dirty flag without dereferencing the pointer. [T-301.2]
Workers MUST drain dirty spans in monotonic address order. [T-301.2]
A single-page 8KB span MUST scan its grey objects sequentially in one pass. [T-301.2]
The sequential span scan MUST use hardware prefetch on amd64. [T-301.2]
Under `GOEXPERIMENT=logosgen`, the card table MUST hold one byte per 512 bytes of heap. [T-301.2]
A minor cycle MUST trigger when the heap grows 50% since the last cycle. [T-301.2]
A span MUST count as mature after surviving 2 cycles. [T-301.2]
A minor cycle MUST skip clean mature spans without reading their mark bits. [T-301.2]
A full cycle MUST follow every 4 minor cycles. [T-301.2]
Background CPU goal MUST be 0.25 when u is below 0.80. [T-301.2]
It MUST rise linearly from 0.25 to 0.75 when u is at least 0.80 and below 0.95. [T-301.2]
It MUST stay 0.75 when u is at or above 0.95. [T-301.2]
Here u means live heap over heap goal. [T-301.2]
Assists MUST still run when u reaches 0.95 or more. [T-301.2]
Until the generational gate passes, `-gc=greentea` builds MUST ignore the experiment. [T-301.2]
In that case `logos` MUST print `note: logosgen disabled: ` plus the first failing package. [T-301.2]

### 19.2 Engine 2: Colored-pointer compactor

On ARM64 with TBI and on x86 with LAM or UAI, color bits MUST survive addressing. [T-301.2]
Without hardware tagging, loads MUST test the color. [T-301.2]
On mismatch they MUST take the self-heal path. [T-301.2]
They MUST unmask to the canonical address. [T-301.2]
Unmask latency carries target: 1 clock, measured by BenchmarkLogosZgcUnmask. [T-PERF]
Every syscall and cgo entry MUST untag pointers. [T-301.2]
Evacuation MUST target pages over 35% fragmented. [T-301.2]
Pages at or below 35% MUST NOT evacuate. [T-301.2]
The forwarding table MUST live off-heap in page metadata headers. [T-301.2]
Source pages MUST return to the OS with `madvise(DONTNEED)` right after relocation completes, before self-heal drains. [T-301.2]
Conservatively referenced objects, cgo-pinned objects, and pointer-keyed map keys MUST pin. [T-301.2]
They MUST NOT move. [T-301.2]

### 19.3 Engine 3: Immix line and block throughput engine

Each 32KB block MUST split into 256 128B lines. [T-301.2]
Line marks MUST live off-heap in block metadata. [T-301.2]
The fast path MUST bump-allocate inside the current free hole. [T-301.2]
Hole search carries target: 3ns, measured by BenchmarkLogosImmixHoleScan. [T-PERF]
Blocks with the pin bit set MUST NOT evacuate. [T-301.2]
Unpinned blocks over 50% dead lines MUST evacuate into fresh blocks during mark. [T-301.2]
`unique` and `weak` handles MUST stay valid across evacuation. [T-301.2]

## 20. Unwind tables for scope defer and Locked

### 20.1 Scope defer

For the `scope` token rule, see Sec 5.1. [T-0.2]
`scope defer f()` MUST run `f` at block exit. [T-104.7]
Scope defers in one block MUST run last-in-first-out. [T-104.7]
The function, receiver, and arguments MUST evaluate at the statement. [T-104.7]
`return`, `break`, `continue`, labeled `break`, and `goto` out of the block MUST run scope defers first. [T-104.7]
A panic MUST run scope defers during unwinding. [T-104.7]
A scope defer in a loop body MUST run every iteration. [T-104.7]
Cleanup on panic MUST use open-coded defers. [T-206][T-104.7]
No separate unwind table exists for scope defer. [T-206][T-104.7]

```lgo
{
    mu.Lock()
    scope defer mu.Unlock()
    work()
}
```

### 20.2 Locked WithLock

For `WithLock` locking and unlock-after-panic, see Sec 2.6. [T-206]
`WithLock` overhead carries target: within 2 ns/op of manual `Lock` with `defer Unlock`, measured by BenchmarkWithLock. [T-PERF][T-206]
The result MUST NOT gate a pass. [T-PERF][T-206]

This example imports `logos/sync` with the package name `sync`.

```lgo
var m sync.Locked[Config]

m.WithLock(func(c *Config) {
    c.Port = 8080
})
```

## 21. IDE protocol

### 21.1 Proxy model and position mapping

- `logos lsp` MUST act as a proxy over `gopls` for `.lgo` packages and for Go packages. [T-Sec21][R-U1]
- It MUST serve stdio JSON-RPC and rewrite URIs and positions between `.lgo` files and the lowered overlay. [T-Sec21]
- It MUST merge compiler JSON diagnostics into `publishDiagnostics`. [T-Sec21]
- It MUST report `logos lsp: gopls not found on PATH` when `gopls` is missing. [T-Sec21]
- Stock LSP clients reach `.lgo` files through the proxy with no editor-specific code. [T-Sec21]

### 21.2 Inlay hints for lowering

- Inlay hints MUST show narrowed types such as `/* narrowed: *User */`. [T-Sec21]
- Inlay hints MUST show union members at switch statements. [T-Sec21]
- Inlay hints MUST show allocation badges `[stack]`, `[arena]`, `[heap]`, and `[rodata]`. [T-Sec21]
- Allocation badges MUST appear for Go packages too. [T-Sec21][R-U1]

### 21.3 Hover previews

- Hover over `?`, `const for`, `const switch T`, or `$"..."` MUST show the lowered code. [T-Sec21]
- Hover over an interpolation MUST render `Fixed: <n>B + dynamic` with the fixed byte count. [T-Sec21]

### 21.4 Code actions and cross-kind workspaces

- Code actions MUST apply the compiler autofix candidates. [T-Sec21]
- In a mixed workspace the overlay MUST hold the lowered `.lgo` files only, and Go packages MUST stay on disk for `gopls`. [T-Sec21][R-U3]
- A definition request across kinds MUST resolve to the declaration in its own file with the positions of that file. It MUST NOT write files to disk. [T-Sec21][R-U2]
- A Go package that imports a loGo/s package MUST resolve that import through the lowered overlay. [T-Sec21][R-U2]

## 22. Verification suite

### 22.1 Upstream regression gate

- `logos test -O=1 std cmd`, `logos test -O=2 std cmd`, and `logos test -O=3 std cmd` MUST all pass without `-short` on the full unmodified `std` and `cmd` suites. [T-Sec22]
- The upstream test suite that `./run.bash` runs MUST pass under plain `go` as the regression floor for Go semantics. [T-Sec22][T-0.1]

### 22.2 Eject round-trip fuzzing

- The eject round-trip suite MUST run each generated `.lgo` program with `logos run` and with `go run` after eject. [T-Sec22]
- It MUST compare stdout, exit code, and stderr with stack-frame lines removed. [T-Sec22]
- Routine runs MUST cover 1,000 generated programs and full runs MUST cover 10,000. [T-Sec22]

### 22.3 Red-team completeness

- Every row of `logos-docs/REDTEAM.md` MUST reach `tested` or `satisfied upstream` with its proving test. [T-Sec22]
- Runtime attack tests MUST run under `-race` and under each `-gc` mode whose gate has passed: greentea always, immix after its phase 1 gate, zgc after its phase 2 gate. [T-Sec22][T-301.2]
- A mode whose gate has not passed MUST be recorded as blocked for that attack, never as tested. [T-Sec22]

### 22.4 Zero-alloc and codegen checks

- A function marked `//logos:zeroalloc` that performs a heap allocation MUST fail with `[perf::zeroalloc_violation]` naming the function and the escape-analysis reason. [T-Sec22]
- The function below MUST fail with `[perf::zeroalloc_violation]`. [T-Sec22]

```lgo
//logos:zeroalloc
func pack(v int) []int {
    return []int{v}
}
```

- Codegen checks MUST pin the SIMD and fast-call emission. [T-Sec22]

### 22.5 Reproducibility and sizes

- Two builds from clean caches MUST yield identical SHA-256 digests for linux/amd64, linux/arm64, and js/wasm. [T-Sec22]
- Reproducibility MUST cover a second host when one is available. [T-Sec22]
- The wasm stripped size MUST be tracked in `test/logos/size/wasm_size.txt`. [T-Sec22]
- Growth above 1% against the recorded value MUST fail the gate. [T-Sec22]

### 22.6 End-to-end example exercise

- The suite MUST exercise `test/logos/example/order_service.lgo`. [T-Sec22]
- Ejected output MUST match native output for the example. [T-Sec22]

### 22.7 Interop suite

- The script `src/cmd/go/testdata/script/logos_interop.txt` MUST hold five cases, each with its own fixture. [T-Sec22][R-U2]
- Case 1: a loGo/s package that imports a Go package from another workspace module MUST build and run. [T-Sec22][R-U2]
- Case 2: a Go package that imports a loGo/s package from another workspace module MUST build and run under `logos`. [T-Sec22][R-U2][A-LGOSEL]
- In case 2, a plain `go build ./...` of that Go module MUST fail with a load error that names the imported package. [T-Sec22][A-LGOSEL]
- Case 3: a loGo/s package that imports `fmt` and `context` MUST build and run. [T-Sec22][R-U2]
- Case 4: `correctness::mixed_module` MUST name the Go packages and the loGo/s packages of the module. [T-Sec22][R-U3]
- Case 5: `logos build ./...` at a workspace root MUST build both modules, and each binary MUST run. [T-Sec22][R-U2]

## 23. Spec areas

### 23.1 Objective

loGo/s is a complete toolchain for Go and for the loGo/s language, which adds checked error propagation to Go. The command `logos eject` writes the pure-Go form of a loGo/s module.
Writers write `.lgo` files, the `logos` command checks and lowers them, and the Go 1.27 backend builds them.
The toolchain is one complete toolchain for Go and loGo/s: Go packages and loGo/s packages share one backend and one runtime, and each improvement serves both.

### 23.2 Commands

The toolchain build runs `cd src && ./make.bash` and puts `bin` on `PATH`, which provides `bin/go` and `bin/logos`.
Every slice gate runs `./make.bash`, the named `go test` commands with `-count=1`, a clean `gofmt -l` over changed `.go` files, and a clean `go vet` over changed packages.
Slices that touch concurrency, runtime, or the `go` command also run their tests with `go test -race -short`.
Each milestone gate runs `./run.bash --no-rebuild` first and then the milestone acceptance command.
New packages also pass golangci-lint v2 with the `standard` linter set.

### 23.3 Project structure

Upstream files hold hook lines only, for rebase economics.
A hook that selects loGo/s language behavior MUST sit behind a `logos` condition.
A hook for an improvement to code generation, runtime, cgo, or tooling MUST NOT depend on that condition, because the improvement serves Go packages under `go` too.
All logic lives in new files: `*_logos.go` beside the hook, or new packages under `cmd/compile/internal/logos`, `cmd/go/internal/logos`, `src/logos`, and `runtime`.
LoGo/s language behavior comes only from the `logos` command and from `.lgo` input files.
An improvement for Go packages comes from a build flag, from `GOEXPERIMENT`, or from a default that `./run.bash` accepted.

### 23.4 Code style

Changed `.go` files stay `gofmt` clean.
Exported API carries Go doc comments.
Compiler-made names use the `_logos_` prefix.
The lowering of `v := parse(s)?` reads as follows.

```go
v, _logos_e0 := parse(s)
if _logos_e0 != nil {
    _logos_rt.TraceErr(_logos_e0, &_logos_site0)
    return 0, _logos_e0
}
```

### 23.5 Testing strategy

Attack tests come first and fail red for the predicted reason, with the red output saved in the commit body.
Implementation follows without touching the attack tests in the same step, then green reruns the slice tests plus the slice gate with an empty test diff.
A mutation probe flips one condition or bound, shows at least one attack failing, and reverts, recorded in the commit body.
Each slice ships as one commit, with one violation test per error path and limit tests at N-1, N, and N+1.
Seams for the tests are `cmd/go` script tests, `syntax` testdata, `types2` testdata, `test/logos` run files, and unit tests beside new packages.

### 23.6 Boundaries

The team always runs the test-first protocol and the environment gates.
Ask first before changing `.go` package semantics, adding new exported standard-library API, or using network access.
The team never commits secrets, never edits `src/vendor` or `src/cmd/vendor`, and never weakens a test to reach green.
