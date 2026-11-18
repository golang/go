# LoGOs specification

- Status: Normative specification; compiler stability clarifications updated 2026-11-16
- Base: Go 1.27.4, as recorded in `VERSION`; language version Go 1.27; ejected-code floor Go 1.27.2
- GC scan amendment: 2026-11-16; shared region scan accounting and same-allocation lookup elision, with the alternate GC modes still withdrawn.
- Original: `logos-docs/PLAN.md` at commit 57475ed (`git show 57475ed:logos-docs/PLAN.md`)
- Source files: `.lgo` (canonical); aliases U+1F4DC (scroll), U+26A1 (lightning), and U+0295 U+25D4 U+03D6 U+25D4 U+0294 (gopher); `.go` files stay Go source files with Go language semantics.

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY follow RFC 2119. The text follows ASD-STE100. Bracketed tags trace each rule to rows in IMPLEMENTATION.md.

## 0. Architectural axioms and transpilation contract

### 0.1 Upstream compatibility
Every improvement to code generation, runtime, stack, SIMD, wasm, cgo, optimization level, and tooling MUST apply to Go packages and to LoGOs packages [T-0.1][R-U1].
That rule MUST hold under `logos` and under plain `go` [T-0.1][R-U1].
The following rules define their scope separately:

| Feature | Scope and controlling section |
|---|---|
| Fixed-array operators and reductions | LoGOs packages only; Sec 2.4 |
| `make` length-overflow lowering | `.lgo` files only; Sec 3.3 and Sec 13.2 |
| Slice-sum vector lowering | The `.sum()` chain call, which has no Go spelling; Sec 1.4 |
| Wasm loop-stride default | `logos` by default; plain `go` through explicit `--wasm-features`; Sec 13 and Sec 12.1 |
| `target_cpu` | `logos` builds; plain `go` retains upstream `GOAMD64` and `GOARM64` defaults; Sec 1.5 |
| Capabilities auditor | `logos` builds only; Sec 11.1 |
| `//logos:zeroalloc` | `logos lint` and `logos build`; `go build` ignores it in Go packages; Sec 17.2 |
| `logos` subcommands | `lint`, `eject`, `lsp`, `context`, and `next` have no plain-`go` counterpart. Shared flags and diagnostics remain governed by Sec 0.1, Sec 12, and Sec 14. |
| LoGOs syntax lowering | Options, unions, tuples, const views and composites, comptime, `go group`, and static JSON/binary forms have no Go spelling. Go importers see their Go-facing forms; Sec 0.3 and Sec 5. |
| Ejection | One-way conversion from LoGOs to Go; Sec 0.3 |

Alternate GC modes and synthetic PGO remain withdrawn and impose no rules.
An improvement that this spec puts behind a named `GOEXPERIMENT` flag MUST stay behind that flag. The `logos_settings_env.txt` script gates the arena and preemptibleloops experiment defaults [T-0.1][R-U1].
Note: the `go` command can differ from the pinned upstream Go release in performance, code generation, runtime, and tooling. No test asserts a difference.
The `go` command MUST keep Go language semantics, so every valid Go program MUST keep its Go meaning [T-0.1][R-U1].
The `go version` output and the `go help` output, including every help topic, MUST stay the upstream text [T-0.1].
Plain `go` MUST select improvements through flags and `GOEXPERIMENT` and MUST read no `logos.toml` [T-0.1][R-U1].
Test processes started by either `go test` or `logos test` MUST preserve the caller's inherited `GOEXPERIMENT`, including absence, explicit empty values, and explicit experiment tokens. The test-input cache MUST hash that inherited value and its presence. LoGo/s build defaults apply to compilation and MUST NOT rewrite this user-process environment. [T-0.1]
A flag that plain `go` gains MUST stay absent from the help text, so that text stays byte-identical to upstream [T-0.1][R-U1].
The `logos` command MUST build Go modules with Go language semantics [T-0.1].
Under `logos`, a Go-module build MUST set the `logos` build tag and, with the default arena enablement, the `goexperiment.logosarena` build tag [T-0.1].
An author `//logos:zeroalloc` directive in a Go file MUST be checked by `logos build` per Sec 17.2: a misplaced directive or a violation of its allocation and call requirements MUST fail with `[perf::zeroalloc_violation]`; a correctly placed directive whose function satisfies those requirements MUST NOT cause a build failure. The `go build` command MUST ignore the directive [T-0.1].
Under `logos` with the default arena enablement, `runtime.Version()` MUST carry the `X:logosarena` experiment suffix [T-0.1].
A Go call into a LoGOs function MUST get the `logos/rt.MustNonNil` check per Sec 5 [T-0.1].
These four MUST be the only differences from Go language semantics for Go modules under `logos` [T-0.1].
The `logos` command MUST apply no LoGOs type rules to Go modules [T-0.1].
The `logos build` command MUST run no default lint on Go modules; an author `//logos:zeroalloc` directive is the Sec 17.2 exception [T-0.1][T-102][R-U1].
The `logos lint` command MUST lint Go packages with the language-neutral rules only [T-0.1][T-102][R-U1].
A module MUST hold one kind only [T-0.1][R-U3].
A LoGOs module MUST hold only LoGOs packages [T-0.1][R-U3].
A Go module MUST hold only `.go` packages [T-0.1][R-U3].
A workspace MAY hold Go modules and LoGOs modules side by side [T-0.1][R-U3].
A package MUST NOT mix user-authored `.go` files and `.lgo` files. Compiler-generated cgo `.go` outputs are the only exception. [T-0.1][A-CODEGEN]
The tool MUST reject a mixed package with `correctness::mixed_package` [T-0.1][R-U3].
The `correctness::mixed_package` text MUST name both file classes [T-0.1][R-U3].
The tool MUST reject a module with both kinds with `correctness::mixed_module` [T-0.1][R-U3].
The `correctness::mixed_module` text MUST name the Go packages and the LoGOs packages of the module [T-0.1][R-U3].
Each module that contributes a loaded source package MUST be checked across all of its active packages, including unselected and unimported packages. [T-0.1][R-U3]
A duplicate-stem error in an unselected and unimported package MUST NOT abort the census; all of that package's active source-file kinds MUST still count. [T-0.1][R-U3]
A source-header failure or mixed-package error MUST still fail the census. [T-0.1][R-U3]
The census MUST honor active build tags, source overlays, nested module boundaries, and package-pattern directory exclusions. [T-0.1][R-U3]
The census MUST attribute vendored packages through `vendor/modules.txt`. [T-0.1][R-U3]
Default `logos list` MUST reject a census failure in a module-owned dependency, while `logos list -e` MUST retain structured package errors. [T-0.1][R-U3]
A synthesized command-line package without a resolved module has no module kind; its module-owned dependencies MUST still undergo the census. [T-0.1][R-U3]
A LoGOs package MAY import any Go package [T-0.1][R-U2].
A Go package MAY import any LoGOs package [T-0.1][R-U2].
An import across kinds MUST NOT be an error [T-0.1][R-U2].
Interop MUST work per import and MUST NOT need a package or a module that holds both kinds [T-0.1][R-U3].
A Go declaration that a `.lgo` file imports MUST be a platform declaration per Sec 2.2 [T-0.1][T-202][A-STD][R-U2].
See Sec 0.3 for the Go-facing signatures that a Go importer sees.
A workspace root `logos build ./...` MUST build every Go module and every LoGOs module of the workspace [T-0.1][R-U2].
The `logos` command alone MUST select the `.lgo` language. This covers file recognition, LoGOs syntax, LoGOs type rules, and LoGOs lints [T-0.1][A-LGOSEL].
Plain `go` MUST ignore `.lgo` files [T-0.1][A-LGOSEL].
Plain `go` MUST report the upstream `no Go files` result for a LoGOs-only package [T-0.1][A-LGOSEL].
A plain `go` build of a Go package that imports a LoGOs package, including `go build ./...` of its module, MUST fail with a load error that names the imported package, because plain `go` ignores `.lgo` files [T-0.1][A-LGOSEL].
Packages from `golang.org/x` and other third-party Go modules MUST count as Go packages that `.lgo` code MAY import [A-XMOD][R-U2].
A LoGOs package MUST NOT hold a generated `.go` file except cgo outputs [A-CODEGEN].
A generator MAY write `.go` files into a Go package that a LoGOs package imports [A-CODEGEN][R-U2].
Project tools SHOULD use `tool` directives in `go.mod`, never a Go file admitted to the build in a LoGOs module [A-MODKIND][R-U3].

### 0.2 No new global keywords
The language MUST add no new global keywords [T-0.2].
See Sec 4.1 for contextual words and `const` positions.
Identifiers that start with `_logos_` or `Logos_` MUST be rejected in `.lgo` [T-203.2].
The compiler MUST report such identifiers with `correctness::reserved_identifier` [T-203.2].

### 0.3 Pure Go output through eject
The command `logos eject -o DIR` MUST write a complete pure-Go copy of the main module to DIR [T-0.3].
The command `logos build --emit-go=DIR` MUST be an alias of `logos eject -o DIR` [T-0.3].
DIR MUST NOT exist, or DIR MUST be empty; the tool MUST refuse a DIR that exists and holds files [T-0.3].
Each `.lgo` file MUST become a `.go` file [T-0.3].
Completeness MUST include packages absent from the requested host build, including inactive-only tests and dependency modules reached only through an inactive variant [T-0.3].
Source discovery MUST respect package and module ownership boundaries: nested modules, `vendor`, `testdata`, dot-prefixed and underscore-prefixed paths, module ignore rules, and directory symlinks MUST NOT become additional package roots [T-0.3].
The physical copy MUST NOT contain a directory that holds its own `go.mod` (it is another module) unless the module's own `go.mod` has a local `replace` that names that directory, which the copy MUST keep so the replace still resolves. A kept module that lies below another nested module MUST be copied without the files of that other module [T-0.3]. Only a `go.mod` file marks a module, as in the go command: a directory named `go.mod` MUST be copied as data with the files beside it [T-0.3].
Each discovered inactive source MUST be type-checked and lowered by the ordinary compiler path under a context that selects it [T-0.3].
Inactive discovery, build-constraint selection, and helper compilation MUST use the requested source overlay. Overlay-deleted sources MUST remain absent, and inserted or replaced sources MUST use their effective contents [T-0.3].
Owned helpers MUST retain the package name selected by their source context; distinct package clauses in one directory MUST NOT be treated as declaration collisions [T-0.3].
Selection of an inactive file's build context MUST satisfy its filename and build constraints, including repeated and negated custom tags [T-0.3].
Selection MUST NOT impose a machine-word limit on the number of custom tags [T-0.3].
An inactive variant MAY disable a custom tag named by its constraint; unrelated custom tags from the requested build MUST remain in effect [T-0.3].
Within a candidate target context, a requested custom-tag set that already satisfies the file MUST remain unchanged [T-0.3].
Architecture feature and experiment facts used to select an inactive variant MUST describe its actual target configuration, including implied lower feature levels and target-specific experiment availability [T-0.3].
The helper MUST receive the selected architecture setting as its real `GO$GOARCH` environment value; eject MUST NOT synthesize architecture feature, compiler, release, or experiment tags as arbitrary user tags [T-0.3].
An explicitly supplied feature-shaped user tag MUST retain its ordinary Go build-tag meaning [T-0.3].
The output MUST keep all comments from the source, except for the cgo fastcall directive replacement defined below. [T-0.3]
C import declarations MUST retain their preamble attachment and content after ejection and formatting, except that each active `//logos:cgo_fastcall F` preamble marker MUST become `#cgo noescape F` and `#cgo nocallback F`. Generated Go line directives MUST stay outside the C preamble. Every emitted package clause MUST retain its physical Go filename so upstream cgo tools can reopen the file. Original declaration positions MUST resume after the imports, including a declaration on the same line as the package clause or C import. [T-0.3][T-304]
An ejected file that imports `C` MUST retain source-level C selectors and types. It MUST receive the same LoGOs checks and lowering as its native form, including checked non-nil stores with the original `.lgo` positions. Upstream cgo MUST regenerate its wrappers, pointer-argument checks and two-result errno calls from that source. Overlays MUST preserve the original logical source filename and the replacement text's line layout; source identity MUST be recovered once before checking. [T-0.3][T-202][T-304]
Owned composite constants containing C types MUST preserve their checked type identity and values after ejection. [T-0.3][T-201]
Each non-standard LoGOs dependency module MUST be ejected too [T-0.3].
Each ejected dependency MUST land in `DIR/_logos_deps/<module path>@<version>/` [T-0.3].
The output `go.mod` MUST carry a `replace` line for each ejected module [T-0.3].
A Go dependency module MUST stay a `require` entry of the output `go.mod` [T-0.3][R-U2].
The support sources MUST become one shared support module `github.com/gosuda/loGos/rt` [T-0.3][A-RTMOD].
The shared module MUST land in `DIR/_logos_deps/rt` [T-0.3].
The output MUST carry a `replace` line for the shared module [T-0.3].
The tool MUST remove `import "comptime"` during eject [T-0.3].
The output MUST NOT include a `//go:build !logos` line [T-0.3][R-U3].
The output MUST NOT emit `.gen.go` file names [T-0.3].
The output MUST carry the reserved `_logos_` import aliases that lowering writes, such as `_logos_unsafe "unsafe"` [T-0.3].
The output MUST NOT emit `_logos_errors` imports [T-0.3].
Among support imports, lowered files before eject MUST import only `_logos_rt "logos/rt"`, `_logos_json "logos/rt/json"`, plus `_logos_unsafe "unsafe"` where needed, in addition to standard library imports under reserved `_logos_` aliases. Ejected files MUST spell those support imports as `_logos_rt "github.com/gosuda/loGos/rt"` and `_logos_json "github.com/gosuda/loGos/rt/json"` (the eject rewrite in `src/cmd/go/internal/logos/eject/eject.go:supportPathEdit`).
The tool MUST NOT write ejected files into the source tree [T-0.3].
Ejected code MUST build with an upstream Go 1.27.2 toolchain [T-0.3].
A Go package that imports a LoGOs package MUST see each exported declaration with the Go-facing signature that `logos eject` emits for it [T-0.3][R-U2].
The output MUST NOT emit colliding helper imports. [T-0.3]

### 0.4 Design axiom for explicit code
Explicitness guides the rules below. Signatures and call sites show side effects, mutability contracts, nilability states, and error paths [T-0.4]. Each later section states its own testable rules [T-0.4].
The lint registry enumerates the enforced lint rules [T-0.4].

## 1. Tier 1 specification (toolchain, linter, desugaring)

### 1.1 RFC-101: File loader, Unicode normalization, and cgo policy
The loader MUST recognize `.lgo` and the three alias extensions only in `logos` mode [T-101.1].
The loader MUST strip trailing U+FE0E and U+FE0F from the extension before matching [T-101.1].
Building or importing a package with two active files whose names differ only by alias extension MUST fail [T-101.1].
The tool MUST report that failure with `correctness::duplicate_stem_filename` [T-101.1].
Module zips MUST reject alias extensions as non-letter symbols for publication [T-101.1].
The tool MUST report warning `portability::alias_extension` for each alias file when the main module path first element holds a dot [T-101.1].
When the main module is a LoGOs module, `CGO_ENABLED` MUST default to 0 [T-101.2].
The key `logos.toml [build] cgo` MUST accept `forbid`, `restricted`, or `allow` in a LoGOs main module [T-101.2].
The default for `[build] cgo` MUST be `forbid` in a LoGOs main module [T-101.2].
The value `restricted` MUST enable cgo only for packages that match `cgo_allowlist` globs [T-101.2].
In a LoGOs main module, a package with `import "C"` outside the allowlist MUST fail with `restriction::cgo_usage` [T-101.2].
In a LoGOs main module with no `logos.toml`, a package with `import "C"` MUST fail with `restriction::cgo_usage` and text `cgo is disabled` [T-101.2].
In a LoGOs main module, an explicit `CGO_ENABLED` in the environment MUST win over the default only when the `cgo` key is absent [T-101.2]. An explicit `[build] cgo = "forbid"` MUST stay enforced against the environment [T-101.2]. An explicit `[build] cgo = "restricted"` allowlist refusal MUST stay enforced against the environment [T-101.2].
When cgo is enabled, `CC` is unset, and the default C compiler is missing, `logos` and plain `go` MUST use `zig cc -target <zig-arch>-<zig-os>` for every cgo package, Go or LoGOs, when `zig` is in PATH [T-101.2][R-U1]. The target MUST be the GOOS/GOARCH pair of the build, not the host [T-101.2][R-U1].
The `-musl` suffix MUST apply only for Linux targets [T-101.2].
The `[build] cgo` policy (`forbid`, `restricted`, `allow`) and its `forbid` default MUST apply to LoGOs main modules only [T-101.2][R-U1].
A Go main module MUST keep the upstream default for `CGO_ENABLED` under `logos`, and a `cgo` key in its `logos.toml` MUST have no effect [T-101.2][R-U1].
Plain `go` MUST keep the upstream default for `CGO_ENABLED` [T-101.2].
With cgo off, `net` and `os/user` in the standard library MUST use their pure-Go paths, which is the upstream `CGO_ENABLED=0` behavior [T-101.2].

### 1.2 RFC-102: Compiler linter and rule set
The `logos lint` check MUST run inside the compiler on LoGOs packages for syntax and type rules [T-102].
The `logos lint` check MUST run the language-neutral syntax and type rules on Go packages at the same point [T-102][R-U1].
The `logos lint` check MUST run inside escape analysis for escape rules, on Go packages and on LoGOs packages [T-102][R-U1].
A rule MUST count as language-neutral when a Go package can trigger it with Go syntax and types alone and the code it guards exists in Go packages. A LoGOs form inside its detection, such as a group child, a const view, or `?`, MUST stay inactive in Go [T-102][R-U1]. The only exception is a `go` statement inside a `WithLock` callback that spawns a relock, which MUST NOT report under `concurrency::reentrant_withlock_deadlock` [T-102][R-U1].
The language-neutral rules MUST be `correctness::typed_nil_return` (in Go, the `(*T)(nil)` form only), `correctness::arena_pointer_escape`, `correctness::infinite_recursion`, `correctness::unclosed_io_closer`, `correctness::unbounded_channel_op`, `correctness::array_index_out_of_bounds`, `concurrency::concurrent_multiword_race`, `concurrency::reentrant_withlock_deadlock`, `correctness::withlock_leak`, `slop::err_chain_reset`, `slop::discarded_error`, `complexity::cognitive_overflow`, `perf::large_backing_array_pin`, `perf::time_after_in_loop`, `concurrency::unguarded_access`, `concurrency::nonatomic_access`, `concurrency::lock_requirement_unmet`, `correctness::layout_constant_mismatch`, `portability::atomic_plain_int64`, `correctness::linkname_mismatch`, `perf::arena_fallback`, and `pedantic::unsafe_outside_unsafe_file` [T-102][R-U1].
The compiler MUST enforce `concurrency::unguarded_access` as `forbid` against a read or write of a field that `//logos:guardedby` guards without its lock held [T-102].
The compiler MUST enforce `concurrency::nonatomic_access` as `forbid` against a non-`sync/atomic` access of a field that `//logos:atomic` marks [T-102].
The compiler MUST enforce `concurrency::lock_requirement_unmet` as `forbid` against a call whose `//logos:requires` set the caller does not hold [T-102].
The compiler MUST enforce `correctness::layout_constant_mismatch` as `forbid` against a constant declared with `//logos:sizeof`, `//logos:offsetof`, or `//logos:alignof` whose value differs from the computed layout [T-102].
The compiler MUST enforce `portability::atomic_plain_int64` as `warn` against a 64-bit `sync/atomic` call on the address of a plain `int64` or `uint64`, with an autofix to `atomic.Int64` or `atomic.Uint64` [T-102].
The compiler MUST enforce `correctness::linkname_mismatch` as `forbid` against a `//go:linkname local pkg.name` whose imported signature differs from the local one, when `pkg` is in the import graph [T-102].
The compiler MUST enforce `pedantic::unsafe_outside_unsafe_file` as `allow` against `import "unsafe"` in a file whose name does not end in `_unsafe.go` or `_test.go` [T-102].
The compiler MUST enforce `correctness::region_escape` as `forbid` against a `logos/arena.Region` value that leaves its local scope in a LoGOs package [T-102].
The compiler MUST report `perf::arena_fallback` when a `logos/arena` call escapes to the heap in a LoGOs package. In a Go package, `logos lint` MUST report it and `logos build` MUST NOT, because `logos build` runs no default lint on Go modules [T-102].
The directive grammar for these checks is `//logos:guardedby <mu>`, `//logos:atomic`, `//logos:requires <p.mu>`, `//logos:acquires <p.mu>`, `//logos:releases <p.mu>`, `//logos:zeroalloc [local]`, `//logos:sizeof T`, `//logos:offsetof T.f`, and `//logos:alignof T` [T-102]
The rules `correctness::nil_deref_unproven`, `correctness::defer_erases_errdefer_state`, `concurrency::unsupervised_goroutine`, `suspicious::unapplied_struct_default`, `suspicious::side_effect_before_sort`, `portability::serialized_unsized_int`, `portability::alias_extension`, and `pedantic::exhaustive_union_switch` MUST NOT apply to Go packages, because each depends on LoGOs syntax, types, or generated code [T-102][R-U1].
The rule `portability::serialized_unsized_int` guards the generated binary layout of Sec 10.3, which Go packages do not get [T-102][R-U1].
A Go package MUST get the default group severities, the `logos.toml [lints]` table, and the `//logos:allow(<rule>)` directive of this section [T-102][R-U1].
A language-neutral rule whose default severity is `forbid` MUST run at `deny` for a Go package [T-102][R-U1].
`//logos:allow(<rule>)` MUST silence such a rule in a Go package [T-102][R-U1].
A LoGOs package MUST keep the `forbid` severity of such a rule [T-102].
Third-party modules of either kind MUST get no code lints [T-102][R-U1].
Capability checks MUST apply to every module in the build [T-102].
`forbid` MUST NOT be lowered by configuration [T-102].
In a LoGOs package, `forbid` MUST ignore `//logos:allow(<rule>)` [T-102].
`deny` MUST be an error unless `//logos:allow(<rule>)` precedes the statement or declaration [T-102].
`warn` MUST print a warning and MUST NOT fail the build [T-102].
`allow` MUST stay silent [T-102].
In a LoGOs package, a `//logos:allow` directive on a `forbid` rule MUST itself be an error with `correctness::allow_forbidden_rule` [T-102].
The default group severities MUST be `correctness` forbid, `concurrency` forbid, `slop` deny, `complexity` deny, `perf` warn, `style` warn, `suspicious` warn, `portability` warn, `pedantic` allow [T-102].
The `logos.toml [lints]` table MUST control group severities [T-102].
Configuration MAY raise `max_comptime_ast_nodes` and `max_ctfe_steps` above their defaults, which Sec 8.5 lists [T-102][T-Sec8].
`logos build` and `logos lint` MUST accept `--pedantic` with default false [T-102].
`--pedantic` and `--pedantic=true` MUST NOT lower a stricter `logos.toml` severity [T-102]. `--pedantic` MUST raise the pedantic group from `allow` to `warn` and MUST keep a stricter configured severity, and `--pedantic=false` MUST set the pedantic group to `allow` [T-102].
A `logos.toml` value that lowers a `forbid` group MUST fail with `configuration::forbid_lowered` [T-102].
A Go package MUST NOT run any rule at `forbid`: a group configured `forbid` runs at `deny` there [T-102][R-U1].
A default build MUST stay silent for the pedantic group [T-102].
`logos lint ./...` MUST report findings without building binaries [T-102].
The compiler MUST enforce `correctness::nil_deref_unproven` as `forbid` against use of a nullable value before its guard [T-102].
The compiler MUST enforce `correctness::typed_nil_return` as `forbid` against return of a concrete nil pointer as an interface or `error` result without a dominating nil check [T-102].
`logos lint` MUST enforce `correctness::arena_pointer_escape` at default `deny` on Go and LoGOs packages against invalid `uintptr` reconstruction as `unsafe.Pointer` and standalone `reflect.SliceHeader` or `reflect.StringHeader` value misuse [T-102][R-U1].
The rule ID MUST remain unchanged so existing `//logos:allow(correctness::arena_pointer_escape)` directives keep their meaning [T-102].
The check MUST resolve conversions and reflection types by their checked types. It MUST accept immediate pointer-to-integer-to-pointer conversions with valid `+`, `-`, or `&^` arithmetic, including an offset on either side of `+`, and immediate conversions of the real `reflect.Value.Pointer` or `reflect.Value.UnsafeAddr` result [T-102].
Returning or storing an `unsafe.Pointer`, or using a converted address only as an integer, MUST NOT trigger this rule [T-102].
The rule MUST run only under `logos lint`, with the normal severity, directive, and JSON diagnostic policy. `logos build` MUST NOT run this rule [T-102].
This bounded syntax and type check MUST NOT claim a general proof of pointer bounds, object lifetime, or arbitrary unsafe behavior. Intrinsic const-view, region, and type checks MUST remain enforced during builds [T-102][T-201][T-301.1].
The compiler MUST enforce `correctness::defer_erases_errdefer_state` as `forbid` against assignment of nil to the error result inside a plain deferred literal in a function with `errdefer` [T-102].
The compiler MUST enforce `correctness::infinite_recursion` as `forbid` against a function whose every path from entry calls itself with identical arguments before any return [T-102].
A panic call used to prove termination or narrow an option MUST resolve to the predeclared builtin, including through parentheses and in functions with no results. A user function or variable named `panic` MUST NOT establish that proof [T-102][T-202].
The compiler MUST enforce `concurrency::concurrent_multiword_race` as `forbid` against sharing of a mutable multi-word value with a child block plus a write on either side, unless the value is a const view or a `logos/sync` type [T-102].
The compiler MUST enforce `concurrency::reentrant_withlock_deadlock` as `forbid` against a `WithLock` call on the same receiver inside its own synchronous closure call graph. A relock that a `go` statement inside the callback spawns MUST NOT report. [T-102].
The compiler MUST enforce `correctness::withlock_leak` as `deny` against a `WithLock` lambda that returns or stores its argument, sends it on a channel, or passes it to a goroutine [T-102].
The compiler MUST enforce `concurrency::unsupervised_goroutine` as `forbid` against a `go` statement in `.lgo` outside a `go group` block [T-102].
The compiler MUST enforce `slop::err_chain_reset` as `deny` against formatting of an `error` operand with a verb other than `%w` or against passing of `err.Error()` [T-102].
The compiler MUST enforce `slop::discarded_error` as `deny` against dropping of an `error` result, except callees with an always-nil-error fact, `fmt.Print*` calls, `fmt.Fprint*` calls to `os.Stdout` or `os.Stderr` or whose writer is a `*bytes.Buffer` or `*strings.Builder` (not an arbitrary `io.Writer`), `io.WriteString` calls to a `*bytes.Buffer` or `*strings.Builder`, `bytes.Buffer.Write*` and `strings.Builder.Write*` calls, and `Write` calls on a receiver that implements `hash.Hash` [T-102].
The compiler MUST enforce `correctness::unclosed_io_closer` as `deny` against a local `io.Closer` from a call other than `io.NopCloser`, with no deferred close, no direct `v.Close()` statement or return operand, no direct `v.Close()` value in an assignment or variable declaration, and no transfer out of the function. Assigning a close result, including in an `if`, `for`, or `switch` initializer, MUST count as closing the resource before that result is handled. This recognition MUST NOT descend into conditional expression operands or count `errdefer` as a guaranteed close [T-102].
The compiler MUST enforce `correctness::unbounded_channel_op` as `deny` against a channel operation outside a select with a done case, a default case, or a timeout case (a receive from `time.After` or from the `C` field of a timer or ticker that is a local of the function, created once with `time.NewTimer` or `time.NewTicker`, not stopped, and used in no other way; a one-shot timer counts once, or again after a `Reset` at the top of the loop body), unless the operation is a bare receive `<-x.Done()` where `Done` is a zero-argument selector returning a channel (not a send or range), the channel is locally made with constant capacity and send count stays within capacity, or the operation is written directly in a `go group` child block, outside any function literal or lambda inside it. In a group child or group stage segment lambda, the compiler MUST exempt receives directly in select-case operands. These operands include channel expressions, send values, and receive-assignment targets. Outside these group contexts, the compiler MUST NOT apply this exemption to select-case operands, even with a done case [T-102][T-206].
The compiler MUST enforce `correctness::array_index_out_of_bounds` as `deny` when a concrete boundary input reaches a fixed-array index through a fully understood function path [T-102]. The index MUST directly name an immutable parameter or scalar value receiver with a concrete integer type. The proof MUST test the array length when representable in the index's target type, and `-1` for a signed index; failures requiring other inputs are outside this rule [T-102]. It MUST match variables by checker object and follow comparisons with constants, short-circuit boolean guards, blocks, scalar or scalar-array declarations, and returns. Unknown calls, mutation or address exposure of the index parameter, captures, and unsupported control flow MUST stop the proof; slices, pointer-to-array bases, arithmetic or conversion indices, and local aliases are outside its scope [T-102]. A valid dominating guard MUST stay silent. A finding MUST name the failing input and array length, point to the index operand, and appear at most once per index expression even when both boundary inputs fail [T-102].
The compiler MUST enforce `complexity::cognitive_overflow` as `deny` against nesting depth above 3, parameter count above 4, line count above 60, or cognitive score above 15 [T-102].
The cognitive score MUST be the sum over branch, loop, switch, select, conditioned case, goto, and labeled jump points of one plus nesting depth, plus one per change of boolean operator in a chain [T-102].
The compiler MUST enforce `suspicious::unapplied_struct_default` as `warn` against zero declaration of a type with field defaults [T-102].
The compiler MUST enforce `suspicious::side_effect_before_sort` as `warn` against a chain lambda with a call of unknown purity before `.sortBy()` [T-102].
The compiler MUST enforce `perf::large_backing_array_pin` as `warn` against escape of a subslice or substring of a backing array above 64 KB [T-102].
The compiler MUST enforce `portability::serialized_unsized_int` as `warn` against an `int` or `uint` field in a struct with JSON tags [T-102].
The warnings of `suspicious::unapplied_struct_default`, `suspicious::side_effect_before_sort`, and `portability::alias_extension` MUST take their severity from the `logos.toml [lints]` group of the rule, and `//logos:allow(<rule>)` MUST silence each of them under `warn` and `deny` [T-102].
A `//logos:allow(portability::alias_extension)` directive MUST silence the rule for an alias-extension file only when it sits in the comment block directly above the package clause of that file [T-102].
The compiler MUST enforce `perf::time_after_in_loop` as `warn` against `time.After` inside a loop body [T-102].
A default over a closed union MUST report pedantic::exhaustive_union_switch when the pedantic group is enabled; that rule defaults to allow and is silent otherwise. [T-102]
JSON output MUST offer autofix candidates for optional chaining with default, early-return guard, split of a misplaced `?` where a split keeps the meaning of the statement, verb fix to `%w`, and addition of `?` in an error-returning function [T-102].
JSON output for a Go package MUST offer only the verb fix to `%w` among the previous sentence's candidates, because the other candidates name LoGOs syntax; it MAY also offer the `portability::atomic_plain_int64` wrapper fix. [T-102][R-U1]
A statement with several misplaced `?` MUST get one split candidate, which hoists each `?` into its own preceding statement in evaluation order. [T-102]
A statement with a misplaced `?` under the right operand of `&&`, `||`, or `??` MUST get no split candidate, because a split would run that operand when the left side decides the result; each diagnostic of that statement MUST carry a note that names this reason. [T-102]
A statement whose remaining call would return more values than the statement binds after a split, such as `v := f1(g1(step()?)?)` with a two-result `f1`, MUST get no split candidate. [T-102]
Exhaustive switches MUST still list every member or default. [T-102]
The compiler MUST allow always-nil-error callees and benign writers to standard output, standard error, byte buffers, and string builders. [T-102]
It MUST NOT hoist timers across iterations. [T-102]

### 1.3 RFC-103: Context and next tools
The command `logos context <pkg>#<Symbol>` MUST print the declaration plus one level of mentioned signatures with no function bodies [T-103].
`logos context` output MUST hold at most 100 lines [T-103].
`logos context` MUST serve `.lgo` and `.go` packages [T-103].
The package pattern MUST select exactly one package; zero or multiple matches MUST report the pattern and the match count. [T-103]
`logos context` MUST report an unknown symbol with text that names the package and the missing symbol [T-103].
The command `logos next --fix` MUST convert a whole Go module to a LoGOs module [T-103][R-U3].
`logos next` MUST use byte-span splicing and MUST keep every comment and build line byte-identical outside rewrites [T-103].
`logos next` MUST keep each non-standard Go import of the module as an import of a Go package [T-103][R-U2].
`logos next` MUST change nothing while it only prints a diff [T-103].
`logos next --fix` MUST rename converted files and MUST rewrite matched patterns [T-103].
`logos next` MUST leave complex refactors unchanged and MUST list them in `next_tasks.json` [T-103].
`logos next` MUST rewrite an index-range append loop with a pure body to a map chain [T-103].
`logos next` MUST leave a loop with jump control or side effects unchanged and MUST list it in `next_tasks.json` [T-103].
The toolchain MUST NOT add a `logos fmt` rewriting command. [T-103]
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
When the deferred call's result is nil, the join MUST keep the existing error; otherwise it MUST join both errors with errors join semantics. [T-104.2]
A `recover()` call inside `f` of `defer f()?` MUST return nil and MUST NOT stop a panic of the enclosing function, as in the ejected form that calls `f` from a deferred function literal. [T-104.2][T-0.3]
`errdefer` MUST be allowed only in functions whose last result is `error` [T-104.2].
Other uses MUST fail with `correctness::errdefer_needs_error_result` [T-104.2].
`errdefer` MUST run when the frame panics, exits through Goexit, or holds a non-nil error result at pop time [T-104.2].
A typed-nil error MUST count as non-nil for `errdefer` [T-104.2].
The function, receiver, and arguments of a deferred call MUST be evaluated at the statement [T-104.2].
An indexed function value MUST be captured at that statement, including when its callee expression is parenthesized. [T-104.2]
An `errdefer` MUST evaluate the index and function value even when cleanup is skipped after a successful return. [T-104.2]
A deferred generic method MUST capture its receiver at the statement, including when some or all method type arguments are explicit. A value receiver MUST copy the selected value, including a promoted receiver reached through a pointer; a pointer receiver MUST retain the selected pointer or implicit address. [T-104.2]
Explicit method type arguments MUST keep their bindings from that statement. Remaining type arguments MUST retain call-argument inference, including an inferred type that the caller cannot name. [T-104.2]
A `?.` call with an absent receiver MUST skip the call and MUST NOT evaluate its arguments, as an ordinary call and in a `defer` or `go` statement [T-104.3][T-104.2].
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
func greet(u *User?) string {
	return u?.Name ?? "anon"
}
```
`$"..."` MUST accept operands of type string, any integer, bool, float32, float64, `[]byte`, `error`, or `fmt.Stringer` [T-104.4].
Any other operand type MUST fail with `correctness::interp_type` [T-104.4].
`{{` and `}}` MUST be literal braces [T-104.4].
Operands MUST be evaluated once from left to right [T-104.4].
The lowering MUST use one allocation with a bound covering the literal bytes and the converted operands [T-104.4].
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
A default MUST be checked in its declaration's lexical scope and MUST be assignable to the field type, even when no composite literal constructs that type [T-104.5]. A later local type specification in the same declaration group MUST NOT change a default's bindings [T-104.5].
A default MUST be a constant expression or a composite literal whose values and map keys recursively satisfy this rule [T-104.5].
Defaults MUST apply in composite literals of the type, including address forms and partial literals [T-104.5].
An empty composite literal whose type's underlying kind is not a struct, an array, a slice, or a map MUST evaluate to the type's zero value [T-104.5].
A Go struct literal of an exported LoGOs struct MUST get zero values and MUST NOT get the field defaults; this is a documented hole [T-104.5][R-U2][A-GOBOUNDARY].
`var x T` and `new(T)` MUST trigger `suspicious::unapplied_struct_default` [T-104.5].
A non-constant default MUST fail with `correctness::default_not_constant` [T-104.5].
Spread `T{..src}` MUST copy fields with identical names and identical types [T-104.5].
The same name with a different type MUST fail with `correctness::spread_type_mismatch` [T-104.5].
Explicit fields MUST win over spread fields [T-104.5].
Defaults MUST fill only fields supplied by neither an explicit value nor a spread; a spread's zero value MUST suppress the default [T-104.5].
Mutable storage created by a slice or map default MUST NOT be shared between separate literal evaluations [T-104.5].
The source MUST be evaluated once [T-104.5].
Self-update `x = T{f: e, ..x}` MUST evaluate every explicit field expression before it writes any field of `x`, and MUST write only the explicit fields, so a field not listed keeps the value it holds when the assignment runs [T-104.5].
The self-update MUST perform no full struct copy [T-104.5].
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
Chained calls MUST fuse into one loop with the receiver evaluated once before the loop [T-104.6].
A chain lambda MUST NOT write its parameter or storage reached through it by assignment, compound assignment, increment or decrement, `copy`, `clear`, `append`, or `delete`; such a write MUST fail with `[correctness::const_view_mutation]`. This direct-write restriction does not prohibit calling methods on or passing on a chain-owned value under the ownership rules below, and does not make a borrowed receiver element owned [T-104.6].
Mutation of the receiver inside a chain lambda MUST fail compilation [T-104.6].
`?` in an expression-lambda body MUST propagate from the enclosing function [T-104.6].
`errdefer` in a chain's block lambda body MUST fail with `correctness::errdefer_needs_error_result`: the block lambda is the function that holds it, and it has no error result of its own [T-104.6][T-104.2].
`?` as the end of an assignment or as an expression statement in a chain's block lambda body, at any depth of nested blocks and of `if`, `for`, `switch`, `select`, `const if` and `const switch` bodies, MUST propagate from the enclosing function in a plain chain and MUST fail the element in a group stage segment [T-104.6].
Every `return` in a chain's block lambda body that holds `?`, at any depth including `const if` and `const switch` bodies, MUST take the element result alone, in a plain chain and in a group stage segment [T-104.6].
`defer f()?` in a chain's block lambda body MUST fail with `correctness::misplaced_try`: the element closure has no frame to hold the deferred call [T-104.6][T-104.2].
A `?`-chain in the right operand of `&&` or `||`, with the `?` inside a chain's expression lambda body or inside a statement of its block lambda body, MUST run only when the left operand does not decide the result [T-104.6].
A `?`-chain in a `for` condition or post statement, with the `?` inside a chain's expression lambda body or inside a statement of its block lambda body, MUST run on every iteration: the loop lowers to an infinite loop evaluating the hoisted computation at the top and the narrowed post at the bottom of each pass, with `continue` reaching the post computation [T-104.6].
A `?`-chain in a `switch` tag, with the `?` inside a chain's expression lambda body or inside a statement of its block lambda body, MUST run once before any guard [T-104.6].
A `?`-chain in a case guard, with the `?` inside a chain's expression lambda body or inside a statement of its block lambda body, MUST run only when earlier guards miss, in order: the lowering tests the guards in source order against one tag temporary, runs each test only while no earlier guard matched, records the matched case in a selector, and then switches on that selector with the case bodies and `fallthrough` unchanged [T-104.6].
A `?`-chain in a `go { }` child MUST return to the group with only the error result, never the enclosing function's results [T-206].
A `?`-chain, with the `?` inside an expression lambda body or inside a statement of a block lambda body, nested in a lambda body, in a `??` fallback, or in a `?.` call argument MUST fail with `correctness::nested_try_capture`: hoisting it to the enclosing function would unbind its captured parameters or run it when the source skips it [T-104.6].
When produced elements implement `io.Closer`, partial failure MUST close already produced elements in reverse order [T-104.6].
A fused chain MUST own only the `io.Closer` values its last `map` stage produces: a `map` produces a value when every return of its lambda, or its function value, yields a call result, a composite literal, `&T{}`, or `new(T)`, directly or through a variable local to the lambda whose every assignment is such an expression, so a call result is produced even when the callee returns one object for several elements, while a name, a field, or an index read, a captured variable, and the lambda argument returned unchanged are not produced, and a `map` whose result type is its argument type keeps the ownership its argument had [T-104.6].
A fused chain MUST close every owned value it does not return exactly once: where a `filter` or `drop` discards it, at the terminal for each value the terminal does not return, and, on a failure by `?` or on a panic, in reverse order of production including the value a `filter` predicate was examining, and it MUST return the first error [T-104.6].
A chain-owned value passed to a later `map` lambda MUST belong to that lambda, which MAY call its methods and MUST close it or pass it on; the chain MUST NOT close that transferred value. Passing a borrowed receiver element or another unowned value to a lambda MUST NOT transfer ownership [T-104.6].
The chain after a group stage MUST own the `io.Closer` values the last `map` of the segment produces, by the rules of a fused chain: `drop`, `take`, `count`, `any`, `all`, `find`, `first`, and a failure MUST close every produced value the caller does not receive [T-104.6][T-206].
A newline followed by spaces or tabs then `.` plus a letter or `_` MUST insert no semicolon in `.lgo` [T-104.6].
```go
func big(xs []int) []int {
	return xs.filter(x => x > 0).map(x => 2 * x).take(5)
}
```
`src.group(ctx, limit = N)` MUST open a parallel segment of a chain when `src` is a slice, an array, an `iter.Seq[T]`, a receive channel, or a named type of one of those, and has no field or method named `group` [T-104.6].
`ctx` MUST be a `context.Context`, and `limit = N` MUST follow it with `N` of type `int`; a missing limit, a positional limit, a limit of another type, or a third argument MUST fail compilation [T-104.6].
A call argument of the form `name = value` MUST fail with `[correctness::named_argument]`, except `limit = N` in a stage call [T-104.6].
`.unordered()` MUST be allowed only directly after `group` [T-104.6].
The segment MUST be every `map` and `filter` call directly after `group` or `.unordered()`, and a stage with none MUST fail compilation [T-104.6].
A segment operand that is not a lambda literal MUST be evaluated once, before the stage starts, on the calling goroutine, in source order after the source, `ctx`, and `limit` [T-104.6].
As a value, a stage chain MUST have type `(R, error)`, where `R` is the type the same chain has without the stage, and that form MUST be legal in a function with no error result; `?` on it MUST propagate the error as for any `?` [T-104.6].
As the range expression of a `for` statement with at most one iteration variable, a stage chain MUST yield its results one at a time, only `map` and `filter` MAY follow the stage, and the enclosing function MUST have an error result or the statement MUST fail with `[correctness::group_needs_error_result]` [T-104.6].
A function whose single result is `iter.Seq[T]` or `iter.Seq2[K, V]` MUST be a generator function [T-104.8].
`yield e` MUST be a statement, allowed only directly inside a generator function [T-104.8][T-0.2].
Inside a generator function, a statement that starts with the word `yield` followed directly by an operand start, `[` and `(` included, MUST be a yield statement whether the result of the function is spelled `iter.Seq`, `iter.Seq2`, or an alias of either, so that `yield[0] = 5` and `yield[0]++` MUST fail and `yield[0]()` MUST yield the value of `[0]()` read as an operand [T-104.8][T-0.2].
A `yield` elsewhere MUST fail with `correctness::yield_outside_generator`, and a statement outside a generator that starts with `yield` and is a valid Go statement as written, such as `yield (a)` or `yield <-ch`, MUST keep its Go meaning [T-104.8].
A generator MUST supply one value per `yield` for `iter.Seq[T]` and two values per `yield` for `iter.Seq2[K, V]` [T-104.8].
Each yielded value MUST be assignable to its element type [T-104.8].
A generator body MUST NOT run until the sequence the function returns is iterated, MUST run from its first statement once for each iteration of that sequence, and MUST pass each `yield e` to the consumer; an early `break` in the consumer MUST end the body at that `yield` with no further statement of it running [T-104.8].
A generator body MUST need no `return`: falling off the end ends the sequence [T-104.8].
A `return` with values inside a generator MUST act as a bare `return`: the values are discarded [T-104.8].
```go
func Count(n int) iter.Seq[int] {
	for i := range n {
		yield i
	}
}
```
A bare return that a shadowed named error result would change MUST fail as upstream Go reports it. errdefer MUST run on panic, Goexit, or non-nil error. [T-104.2]
Fused take(5) loops MUST run map functions at most 5 times. [T-104.6][T-102]
Side effects before sortBy MUST warn suspicious::side_effect_before_sort. [T-104.6][T-102]
Interpolation MUST evaluate each operand once, left to right, into a local variable, and MUST convert error and fmt.Stringer operands to strings before sizing. [T-104.4]
`defer f()?` with nil result MUST leave the error result nil with no allocation. [T-104.2]
In a loop outside a range-over-func body, `defer f()?` with nil result and the error result as its error operand MUST allocate nothing in steady state in a `logos` build: the error result stays on the stack and the record that holds the call comes from a pool of the processor. [T-104.2]
An allocation that evaluating the arguments of `f` causes, such as the variable of an `&x` argument that outlives its iteration, belongs to the argument and is not part of that rule. [T-104.2]
A garbage collection drops the shared pool, so a loop with more than 32 pending records on one processor MAY allocate records again after a collection. [T-104.2]
A `defer f()?` in the body of a range-over-func loop, and one whose error operand is a variable that shadows the error result, keep the closure of a plain `defer` and allocate it on every execution. [T-104.2]
Ejected Go allocates one closure per execution of a `defer f()?` in a loop and one box for the error result, because plain Go has no pooled record; outside a loop it allocates nothing. [T-104.2][T-0.3]
With a prior error it MUST join both errors. [T-104.2]
Errdefer MUST work in functions with unnamed error results. [T-104.2]
Errdefer MUST stay skipped on success returns. [T-104.2]
`errdefer` MUST run when the frame holds a non-nil error, panics, or exits through Goexit. [T-104.2]
For `errdefer f()?`, a rollback error MUST join the primary error with errors join semantics, as `defer f()?` does. [T-104.2]
A nil primary error MUST take the rollback error as is. [T-104.2]
`defer f()?` MUST run in place on the standard LIFO stack. [T-104.2]
The fused loop MUST run that rollback before returning. [T-104.6]
NaN sort keys MUST order deterministically. [T-104.6]

### 1.5 RFC-105: Optimization tiers
`logos build` and `go build` MUST accept `-O=0`, `-O=1`, `-O=2`, `-O=3`, and `-O=s` [T-105][R-U1].
A level MUST apply to every package of the build, Go or LoGOs [T-105][R-U1].
Under `logos`, the default level MUST come from `logos.toml [build] opt_level` with default `O2` [T-105].
Under plain `go`, the default level MUST be `O1` [T-105][R-U1].
`[build] opt_level` MUST accept `O0`, `O1`, `O2`, `O3`, and `Os` only [T-105].
A bare spelling such as `2` MUST fail validation like an unknown level [T-105].
An invalid level MUST fail with text that lists the five accepted levels [T-105].
O0 MUST pass `-N -l` for no inline and no debug-info loss protection [T-105].
O1 MUST be the upstream baseline [T-105].
O2 MUST set inline budget 120 and variable-make stack buffer 256 bytes for non-standard packages [T-105].
O3 MUST set inline budget 400 with 80 inside recursive cycles, and variable-make stack buffer 4096 bytes for non-standard packages [T-105].
A function in a recursive call-graph cycle MUST keep the upstream variable-make buffer of 32 bytes at every level, inlined bodies included [T-105][T-302].
A standard-library-provenance package (a standard library package or its external test package) MUST keep the default variable-make buffer of 32 bytes at every opt level; non-standard packages MUST use 256 bytes at O2 and 4096 bytes at O3 [T-105][T-302].
Os MUST set inline budget 15 [T-105].
O3 MUST add no escape-analysis or prove-pass work budget [T-105].
O3 MUST enable bounded scalar constant analysis for Go and LoGOs packages [T-105.1][R-U1].
O0, O1, O2, and Os MUST keep their existing scalar-analysis behavior [T-105.1].
The analysis MUST preserve defined integer overflow at each target width [T-105.1].
The analysis MUST preserve required calls, memory effects, nil and bounds checks, and panic behavior [T-105.1].
The analysis MUST NOT reassociate floating-point arithmetic or assume that signed overflow cannot occur [T-105.1].
O3 MAY eliminate arithmetic and conditions proved constant from known operand bits, including carries and rotation counts modulo the operand width [T-105.1].
O3 MAY propagate known bits through byte swaps and derive bit-length, trailing-zero-count, and population-count facts from partially known operands [T-105.1].
Each such transfer MUST use the operation's input width independently of its result type width and MUST perform bounded constant-time work without enumerating concrete input values [T-105.1].
Byte-swap facts MUST preserve zero extension when the result type is wider than the byte-swap operation [T-105.1].
Rotation-count analysis MUST examine no more than 64 count residues for one value [T-105.1].
O3 MAY replace a rotate-left followed by an unsigned right shift at the same literal count `k` with the original input when both operations have the same integer width, `0 <= k < width`, and the input's top `k` bits are proved zero [T-105.1].
O3 MAY fold an integer or Boolean operation whose constant inputs on each incoming control-flow edge produce the same result [T-105.1].
That correlated analysis MUST inspect no more than eight incoming edges and MUST NOT unfold loop-carried values [T-105.1].
The correlated analysis MUST run before normalization of integer zero/one phis destroys their predecessor alignment [T-105.1].
O3 MUST propagate scalar constants through byte-swap, bit-length, and population-count intrinsics, Boolean conjunction and disjunction, and integer or Boolean conditional selections [T-105.1].
Unknown or unsupported inputs MUST retain the original computation when no result can be proved [T-105.1].
`target_cpu` MUST accept `deterministic`, `baseline`, or `native` [T-105].
`target_cpu` MUST apply to every package that `logos` builds, Go or LoGOs [T-105][R-U1].
Plain `go` MUST read no `target_cpu` and MUST keep the upstream `GOAMD64` and `GOARM64` defaults [T-105][R-U1].
`deterministic` MUST set GOAMD64 v3 and GOARM64 v8.2 unless the environment already sets them [T-105].
`baseline` MUST set GOAMD64 v1 and GOARM64 v8.0 [T-105].
Host-specific selection MUST require `target_cpu = "native"`; `native` MUST pick the host level [T-105].
For a target architecture other than the architecture of the build host, `native` MUST select that architecture's baseline level (GOAMD64 v1, GOARM64 v8.0), because the host CPU says nothing about it and a build for another architecture MUST NOT vary with the host CPU [T-105].
An explicit GOAMD64 or GOARM64 in the environment MUST win over the level that `native` selects [T-105].
The default binary MUST NOT run on x86 CPUs older than v3 [T-105].
Self-aliased calls such as arr = Reverse(arr) MUST return correct results. [T-105]

## 2. Tier 2 specification

This section defines the LoGOs type system: compile-time constants, pointers and options, unions and property interfaces, tuples and fixed-array operators, comptime expansion, and group concurrency.

### 2.1 RFC-201: Const composites, CTFE, const views, and const maps

A `const` declaration MUST accept a composite literal (a struct, an array, a slice, or a map) and a call to a `const func`. [T-201]

```lgo
const primes = []int{2, 3, 5}
const httpPort = 8080
```

A `const func` MUST NOT perform I/O, spawn a goroutine, read a mutable package variable, call `recover`, convert a pointer to `uintptr`, range over a dynamic map, or use a pointer. [T-201]

The compiler MUST evaluate `const` composites and `const func` calls at compile time with target integer widths, a step limit from `max_ctfe_steps` with default 100,000, and a value cap of 1 MB. [T-201][T-102]

Compile-time assignments MUST preserve Go's operand evaluation order, destination identity, value-copy rules, and left-to-right store order. [T-201]
The evaluation and assignment phases, including failures while evaluating a destination operand and failures while storing a result, MUST follow Go assignment semantics. [T-201]
A two-value lookup on a nil map MUST return the element's zero value and `false`. [T-201]
Assignment to an array or struct MUST preserve its existing inline storage, including storage observed by a slice view. Value copies MUST remain independent; slice and map fields MUST retain Go reference sharing. The same rules apply to range assignment, `clear`, `copy`, and `append`. [T-201]

A run that exceeds the selected `max_ctfe_steps` limit MUST fail with `[comptime::ctfe_step_limit]`. [T-201][T-102]

Configuration MAY raise `max_ctfe_steps` above its default. [T-102]

A constant value above 1 MB MUST fail with `[comptime::const_too_large]`. [T-201]

A `const func` that breaks a purity rule MUST fail with `[correctness::const_func_impure]` naming the reason. [T-201]

A constant that is not representable in its required target type MUST fail with `[correctness::const_eval]` naming the cause. Non-constant fixed-width integer operations inside a `const func` MUST retain Go's wrapping arithmetic. [T-201]
A compile-time evaluation that inserts a NaN or duplicate map key, executes a panic, or reaches a form the interpreter does not support MUST fail with `[correctness::const_eval]` naming the cause. A panic with a string argument MUST retain that string in the diagnostic. [T-201]

On native targets, constant data with pointers MUST live in read-only memory. Native read-only placement is a separate defense against stores that reach that storage; it MUST NOT substitute for the checked-source rejection rules below or imply a portable run-time fault guarantee. [T-201]

A known `unsafe` derivation from constant-origin storage MUST preserve its readonly restriction and MUST fail statically if it would expose an ordinary writable reference, before native code generation or `logos eject` emits Go. A store that bypasses checked-source enforcement and reaches native read-only constant storage MUST fault at run time; this is not a promise for every target or every use of `unsafe`. [T-201]

A hand-written `//logos:rodata` directive on its own line in the comment block directly above a package-level `var` declaration or a spec of a `var (...)` group in a LoGOs package MUST place the variable in read-only memory, and a variable under it whose initializer needs dynamic initialization MUST fail compilation with `[correctness::rodata_dynamic_init]` at the variable. A `//logos:rodata` directive anywhere else (above a `const`, a `func`, a `type`, or the `var (` line of a group, inside a function body, after code on its line, or apart from its declaration by a blank line) MUST fail compilation with `[correctness::misplaced_directive]` at the directive. The other `//logos:` directives report a misplaced directive under the rule they serve (Sec 1.2, Sec 17.2), and no rule serves this one. [T-201]

A write through any alias to a variable placed by that directive MUST fault at run time. [T-201]

`reflect.DeepEqual` and `fmt.Println` MUST accept constant composites. [T-201]

A package with a 1,000-key unexported constant map MUST add no heap bytes at init, with its table symbol in read-only memory. [T-201]

See Sec 7 for const map lowering, iteration, and float keys.

`const []T`, `const *T`, and `const map[K]V` MUST form a qualifier checked at compile time and erased before code generation. [T-201]

A Go importer MUST see a constant view as its plain type: `const []T` as `[]T`, `const *T` as `*T`, and `const map[K]V` as `map[K]V`. [T-201][T-Sec5][R-U2]

A Go importer MUST see an exported `const` composite as a package-level variable of its plain type and a `const func` as a plain function. [T-201][T-Sec5][R-U2]

An exported const map is the one exception to the no-heap rule above: its package-level variable holds a built map, so only an unexported const map adds no heap bytes at init. [T-201][T-Sec5][R-U2]

A write from Go into such a variable lies outside the LoGOs contract. The Go compiler does not report it, and the write MAY fault at run time because the data can sit in read-only memory. This is a documented hole. [T-201][T-Sec5][R-U2][A-GOBOUNDARY]

An `&` over an inline field or an inline array element of a const composite, local or imported, MUST yield a pointer to a fresh variable that holds a copy. Only the copied slots are detached: a fresh outer struct or array copy MUST NOT make a nested pointer, slice, map, or other reference to protected constant storage writable. Writes to detached slots MUST NOT change the constant, and nested references MUST retain their readonly obligation. [T-201][T-104.5]

An `&` whose path traverses a slice element or a dereference inside a const composite, and a slice expression of an array inside a const composite, MUST fail with `[correctness::const_view_mutation]`; copying an enclosing value MUST NOT bypass these rules for references still reaching protected storage. [T-201]

A LoGOs read of an imported struct or other aggregate const composite that holds a map MUST preserve the readonly restriction on its reference descendants, including through a copied outer value; it MUST NOT yield a mutable deep copy per read. A write through a reference descendant that reaches protected constant storage MUST fail with `[correctness::const_view_mutation]`. The fresh-variable address rule above detaches only copied inline slots, not their nested references to protected storage. Bare const-map materialization under Sec 7 and the ordinary Go-importer boundary above are unchanged. [T-201][T-Sec5][T-Sec7][R-U2]

The checked-source invariant is that constant-origin values MUST NOT expose ordinary writable references to protected storage. The compiler MUST preserve this obligation through aliases, aggregate copies, and known `unsafe.SliceData`, `unsafe.Slice`, `unsafe.Add`, `unsafe.StringData`, and `unsafe.String` derivations. A qualification-erasing escape MUST fail statically with `[correctness::const_view_escape]`, and a forbidden mutation or address/slice operation MUST fail with `[correctness::const_view_mutation]`, before either native compilation or Go ejection. Qualifier erasure during lowering is permitted only after these checks succeed. [T-201]

A Go parameter's proven `ParamMutated=0` fact proves only the permitted call's non-mutation obligation, not independence of its returned aliases. A reference-carrying result that might alias a constant-origin argument MUST retain the readonly obligation unless result independence is separately proven; an unknown result MUST NOT become an ordinary writable reference merely because the parameter has that fact. [T-201][T-Sec5]

This invariant applies to checked LoGOs source, not arbitrary forged `unsafe` addresses. External ordinary Go importers remain subject to the documented Go boundary hole above, and modified ejected Go is outside checked-source enforcement. These boundaries do not weaken rejection of known flows in checked source and do not assert a universal static proof of unsafe behavior. The hand-written `//logos:rodata` directive's scope and alias-store contract above are unchanged. [T-201][T-Sec5][R-U2][A-GOBOUNDARY]

An index assignment through a constant view MUST fail with `[correctness::const_view_mutation]` naming the operation and the expression. [T-201]

A field assignment through `const *T` MUST fail with the same diagnostic. [T-201]

`append` to, `copy` into, `delete` from, and `clear` of a constant view MUST fail with the same diagnostic. [T-201]

A slice expression with a bound above `len` (such as `b[:cap(b)]`) MUST fail with the same diagnostic, with a run-time check when the bound is not constant. [T-201]

A call through a constant view to a method with a pointer receiver that lacks `const` MUST fail with the same diagnostic. [T-201]

A method with a `const` pointer receiver (such as `func (r const *T) M()`) MUST be allowed. [T-201]

A call of a method with a value receiver on a const composite that holds a map, or on an imported const composite, MUST fail with `[correctness::const_view_mutation]` unless the receiver type is reference-free or the method is proven readonly. [T-201]

For a local const composite without a map, a value-receiver call also needs no readonly proof when the actual receiver value has no live reference leaves and every result type is reference-free. Selector and index paths MUST preserve this value-level proof; an unknown index MUST be safe for every candidate receiver. This exception does not apply to imported const composites or pointer receivers.

A call of a method with a pointer receiver on a const composite, local or imported, with or without a map, MUST be accepted when the method is proven readonly and MUST fail with `[correctness::const_view_mutation]` otherwise. [T-201]

A bare const map materialized as a `map[K]V` value under Sec 7 MUST be a copy, and a write to that copy MUST NOT change what the const map reads as. This materialization rule MUST NOT make a map field or another reference descendant of a struct or aggregate const composite writable, or permit a mutable deep copy of that aggregate per read. [T-201][T-Sec7]

An interface method written `const M() T` (such as `const Name() string`) MUST be a readonly method obligation, distinct from a property element `const F T` (such as `const Name string`). [T-201]

The `const` on an interface method MUST NOT declare an accessor property and MUST NOT lower to `Logos_Get` or `Logos_Set` methods. [T-201]

A concrete type MUST satisfy an interface readonly method only with a method proven readonly; an unknown or mutating effect MUST fail satisfaction with `[correctness::const_view_mutation]` naming the method. [T-201]

A trusted imported `FnFact.Readonly` fact for the concrete method MUST count as the proof; an absent or old fact MUST NOT prove readonly. [T-201][T-Sec5]

Satisfaction MUST NOT depend on declaration order; a local proof finalized after method bodies MUST give the same legality, and the compiler MUST emit no code before the final satisfaction errors. [T-201]

Interface identity MUST distinguish the obligation: an ordinary interface with the same method name and signature MUST NOT acquire the readonly obligation, and an ordinary method set MUST NOT satisfy a readonly requirement without the proof above. [T-201]

A value of an interface with a readonly method MUST be assignable to the same method set without the obligation; the reverse direction MUST require the proof above. [T-201]

A dynamic call through a constant view to a method the interface declares `const` MUST be allowed. [T-201]

A dynamic call through a constant view to a method the interface does not declare `const` MUST fail with `[correctness::const_view_mutation]`. [T-201]

The explicit `const` receiver rule and the inference for ordinary concrete value receivers MUST stay unchanged. [T-201]

A method of a generic type MUST be proven readonly by the same proof, from its body and receiver type independently of any instantiation, and an instantiated method MUST carry the proof of its origin; such a method MUST satisfy a readonly method obligation, MUST be callable through a constant view, and MUST export its `FnFact.Readonly` fact exactly as a method of a type without type parameters does. [T-201][T-Sec5]

An array of length zero MUST count as reference-free regardless of its element type. A positive-length array MUST count as reference-free only if its element type is reference-free. A pointer, slice, map, channel, function, or interface remains reference-bearing when it refers to, holds, or returns an empty array. [T-201]

A value whose type is a type parameter MUST count as reference-free only when every type in its constraint's type set is, and a parameter whose type set has no specific types or admits a slice, map, pointer, channel, function, interface, or `unsafe.Pointer` type MUST count as bearing references; a method of a generic type that returns such a value read from its receiver, or writes through it, MUST fail the proof with `[correctness::const_view_mutation]` as the same method of a type without type parameters does. [T-201]

Receiver readonlyness MUST NOT imply parameter purity: a `FnFact.Readonly` fact proves only the receiver obligation, and a readonly-only or assertion-only fact MUST NOT prove that arguments are unmodified. An absent `ParamMutated` summary MUST be treated as unknown, distinct from an explicit proven-zero `ParamMutated=0` summary. [T-201][T-Sec5]

Ordinary generic assertion definitions, including `func As[T any](x any) T { return x.(T) }` and assertion methods on generic receivers, MUST remain legal. The compiler MUST record their assertion and type-switch effects and MUST require verified effects before manufacturing qualified targets for relevant function and receiver-type instantiations, including transitive wrappers and cross-package uses, independently of declaration order. An instantiation MUST NOT manufacture a readonly obligation erased from runtime interface identity; an unsafe instantiation MUST fail with `[correctness::const_view_mutation]`. [T-201][T-Sec5]

Qualified type arguments in assertion-relevant positions MUST require verified effects. Unknown effects MUST be conservative, not proven zero; rejection MUST be limited to arguments whose assertion could manufacture an erased readonly obligation. Ordinary concrete named types and nominal nesting, a named type whose type argument is itself an instantiated named type such as `StoreBox[StoreBox[Good]]`, MUST remain legal. Known-zero generics such as `Identity[Qualified]` and storage-only containers such as `StoreBox[Qualified]` MUST remain legal; those names alone MUST NOT exempt an imported generic whose effects are unknown. [T-201][T-Sec5]

Assertion-effect masks for function and receiver type parameters and mutation masks for value parameters MUST support arbitrary arity; positions at or above 64 MUST NOT be truncated into a false proof of safety. [T-201][T-Sec5]

A `const []I`, `const *I`, or `const map[K]I` where `I` is an interface MUST remain allowed as a container qualifier; container covariance MUST NOT imply method readonlyness and method readonlyness MUST NOT ban interface containers. [T-201]

Passing a constant view to a non-constant parameter of an `.lgo` function MUST fail with the same diagnostic. [T-201]

Passing a constant view to a Go-mode parameter (the standard library or any Go package) MUST be allowed only when the parameter carries fact `ParamMutated=0`. [T-201][R-U2]

Assigning `const []*T` to `[]*T` MUST fail with `[correctness::const_view_escape]`. [T-201]

A reference-free constant view MUST be assignable to its plain type. [T-201]

A plain value MUST be assignable to its constant view. [T-201]

A `for _, v := range` loop over a constant slice MUST yield constant elements. [T-201]

The equality operators `==` and `!=` MUST be defined on operands whose types carry a const-view or readonly-descendants qualification whenever the unqualified operand types are comparable and one unqualified operand type is assignable to the other. A comparison reads its operands; it MUST NOT be reported as a mutation through the view or as a qualification-erasing escape. [T-201]

A `?.` selector on an element reached through a constant view MUST resolve the field or method on the unqualified element type; a selected field keeps the element's qualification, and a selected method follows the receiver rules of this section. [T-201]

A method call on a readonly-descendants value that fails the reference-free or readonly proof of this section MUST fail with `[correctness::const_view_mutation]`, the same diagnostic as the same call through the enclosing constant view. [T-201]

Converting `string` to `const []byte` MUST be zero-copy and MUST allocate nothing. [T-201]

Converting `const []byte` to `string` MUST be zero-copy when provenance is a string or a constant composite. [T-201]

The same conversion MUST copy to the Go heap for any other provenance. [T-201]

`logos/rt.Freeze` MUST deep-copy its argument and return a `const` view. [T-201]

A later write to the source (or to a child object it points to) MUST NOT change the frozen view. [T-201]

`Freeze` MUST panic on a reachable non-nil `func`, `chan`, or `unsafe.Pointer`. It cannot isolate their referenced state. [T-201]

A nil `func`, `chan`, or `unsafe.Pointer` MUST remain valid and copy as nil. [T-201]

Reachability MUST include map keys and values, interface payloads, and slice capacity tails. [T-201]
Ordinary reslicing such as s = s[:0] MUST stay allowed. [T-201]
Only const views MUST enforce the len bound. [T-201]
Dynamic map iteration in const funcs MUST fail compilation. [T-201][T-Sec7]
Const map range MUST visit keys in sorted order. [T-201][T-Sec7]
Nil leaves MUST copy as nil. [T-201]
Pointer-free constant data MUST live in read-only memory before any init runs. [T-201]

### 2.2 RFC-202: Non-nil pointers, option types, and nil flow

`*T` in `.lgo` MUST be non-nil. [T-202]

`*T?` MUST denote an optional pointer. [T-202]

Option types MUST have the same type meaning in every explicit generic-argument slot, including the first argument of a function, method, or generic type literal. Parentheses MUST retain their binding: `*T?` is an optional pointer, while `*(T?)` is a non-nil pointer to an option. [T-202]

Assigning `nil` to `*T` MUST fail with `[correctness::nil_to_nonnil]`. [T-202]

`var p *T` MUST fail compilation. [T-202]

An array type with non-nil element pointers (such as `[3]*T`) MUST fail without initializers. [T-202]

`make([]*T, n)` with a non-constant `n` or a constant above 0 MUST fail compilation. [T-202]

`make([]*T, 0, n)` MUST be allowed. [T-202]

`new(S)`, `var s S`, and `S{}` MUST fail when `S` has a non-nil pointer field without an initializer, with `[correctness::nonnil_uninitialized]` naming the field and its type. [T-202]

Zero-clearing through `*p = S{}` MUST fail with `[correctness::nonnil_uninitialized]`. [T-202]

`var t T` for a type parameter MUST be allowed. [T-202]

Pointers from Go-mode declarations (the standard library, every other Go package, and cgo-generated declarations) MUST be platform pointers. [T-202][A-STD][R-U2]

A `.lgo` file MAY import any Go package, and each declaration that it imports MUST be a platform declaration. [R-U2]

A type-parameter value instantiated with a non-nil pointer MUST also be a platform pointer. [T-202]

Dereferencing a platform pointer MUST be allowed. [T-202]

Comparing a platform pointer with `nil` MUST be allowed. [T-202]

A store of a platform value into a declared non-nil location MUST wrap the stored expression in `logos/rt.MustNonNil` with the source position and the name of the stored type. [T-202]

A nil value at such a store MUST panic with `logos: nil from Go value stored into non-nil *T at <file>:<line>`, where `*T` is the string that `reflect.Type.String` returns for the stored type and `<file>:<line>` is the `.lgo` file and line of the store. [T-202][R-U2][A-STD]

`if x != nil` MUST narrow a local or parameter `x` from `T?` to `T` in the guarded block. [T-202]

An early exit of the form `if x == nil { return }` (also with `panic`, `break`, or `continue`) MUST narrow a local or parameter `x` after the statement. [T-202]

An `&&` chain with a nil test MUST narrow a local or parameter `x` in the rest of the chain. [T-202]

A later assignment MUST end narrowing. [T-202]

Taking the address of `x` MUST disable narrowing for `x`. [T-202]

Assigning `x` in a function literal, in a `go group` child block, or in a lambda of a chain that opens with `.group(...)` MUST disable narrowing for `x`. [T-202]

A function literal or a `go group` child block MUST NOT read `x` as narrowed when `x` is assigned anywhere besides its declaration. [T-202]

A short variable declaration that names a variable already declared in the same scope, such as a parameter, a named result, or a `var` variable, MUST count as an assignment to it. [T-202]

The operands of a `go` or `defer` statement, and the statements of a `go group` block outside its children, MUST read `x` as narrowed where they stand, because they run there. [T-202]

A nil test on a struct field MUST NOT narrow the field. [T-202]

Code MUST copy the field to a local first. [T-202]

For a callee declared in `.lgo`, reading a result before the `err == nil` proof MUST fail with `[correctness::result_before_error_check]`. [T-202]

An `err == nil` proof MUST remain valid for a captured result in a function literal, lambda, or `go group` child when that result is initialized at its declaration, is never assigned again, and never has its address taken. A result assigned anywhere besides its declaration or whose address is taken MUST NOT inherit that proof across the capture boundary. [T-202]

A return that passes a result and the err it came with, both unread and unchanged, MUST NOT count as a read of the result. [T-202]

Assigning a result to the blank identifier MUST NOT count as a read. [T-202]

A Go-mode callee (the standard library or any Go package, such as `io.Reader.Read`) MUST keep its own contract, so use of n first stays allowed. [T-202][R-U2]

`nil` MUST be assignable to every `T?`. [T-202]

`T` MUST be assignable to `T?`. [T-202]

`T?` MUST NOT be assignable to `T`. An assignment, argument, or result that needs `T` and receives an unnarrowed `T?` MUST fail with `[correctness::nil_deref_unproven]`. [T-202]

`*T?` where `*T` is non-nil MUST lower to `*T` with nil as none and no extra storage. [T-202]
For `*T?` where `*T` is non-nil, the ejected Go type MUST be `*T` with `nil` as none [T-202][T-0.3].

When nil is a valid payload of `T`, `T?` MUST carry explicit presence separate from payload nil. [T-202]
When nil is a valid payload, the ejected form MUST carry explicit presence separate from payload nil [T-202][T-0.3].

An interface-backed union option with no nil-capable member MUST use the interface nil value for absence. [T-202]
An interface-backed union option with no nil-capable member MUST use the interface nil value for absence in ejected code [T-202][T-0.3].

For every other `T` outside those two cases, `T?` MUST lower to `logos/rt.Option[T]`. [T-202]
For every other `T?` outside those two cases, the ejected Go type MUST be `logos/rt.Option[T]` [T-202][T-0.3].

A present nil payload MUST read as some with nil payload, distinct from none. [T-202]
A present nil payload MUST stay distinguishable from none in ejected code [T-202][T-0.3].

`(T?)?` MUST lower to `logos/rt.Option[T?]`. [T-202]

The spelling `T??` MUST be a syntax error reporting `T?? is not allowed; use (T?)?`. [T-202]

`?.` on a non-option value MUST fail with `[correctness::not_an_option]`. [T-104.3]

`??` with a non-option left side MUST fail with the same diagnostic. [T-104.3]

An ordinary method call on an unnarrowed `T?` MUST fail with `[correctness::nil_deref_unproven]`; a `?.` call is not an ordinary call. A dereference `*p` of an unnarrowed `*T?` MUST fail with the same diagnostic. [T-202]

`??` MUST evaluate its right side only when the left side is none. [T-104.3]

`?.` MUST evaluate its receiver once. [T-104.3]

`?.` over a field that is already an option MUST flatten the result. [T-104.3]

See Sec 1.4 for `??` grouping, precedence, and single-value map reads.

`m[k] ?? d` MUST type as `V`. [T-104.3]

`v, ok := m[k]` MUST keep Go types. [T-104.3]

An absent key MUST read as none. [T-104.3]

A present zero value MUST read as some. [T-104.3]

A present nil slice payload MUST read as some with nil payload, distinct from none. [T-104.3][T-202]

Ejected and Go-facing signatures MUST use plain pointers and `rt.Option[T]` from the shared support module. [T-202][T-Sec5][R-U2]

A Go importer MUST see a non-nil `*T` and a `*T?` as `*T`, and MUST see every other `T?` as the lowered type of the rules above. [T-202][T-Sec5][R-U2]

A `.lgo` importer MUST see each option that an imported declaration carries in a field, a property, a parameter, or a result, and MUST see it also when the declaration belongs to a generic type or is a generic function. [T-202][T-Sec5]

A `.lgo` importer MUST lower an option of an imported closed union by the rule of the package that declares the union, so that both packages use the bare interface for an option of a union with no nil-capable member and `logos/rt.Option[T]` for every other option. [T-202][T-Sec5]

```lgo
func greet(u *User?) string {
    return u?.Name ?? "anon"
}
```
A single-value map read `m[k]` MUST type as `V?`, while `v, ok := m[k]` MUST keep Go meaning [T-104.3].
Zero values of non-nil pointers MUST fail with correctness::nonnil_uninitialized. [T-202][T-Sec12]
Absent map and JSON values MUST check through V? and MissingFieldError. [T-202][T-Sec12]
Block scoping MUST stay standard Go. [T-202]
A local or parameter declared in the block of an `if x == nil` early exit, or in a block that encloses it, is an outer variable of that guard. An outer variable MUST stay narrowed after the guard, inside nested blocks too, until a nested declaration shadows it. [T-202]

### 2.3 RFC-203: Closed unions, property interfaces, and inline assertions

A closed union MUST be an interface whose type set is a finite union of non-`~` terms. [T-203.1]

It MAY also declare methods, either directly or through an embedded interface. [T-203.1]

Every member, after nested unions are flattened, MUST have each declared method in its method set; a pointer-receiver method does not count for a value member, and a promoted method counts. [T-203.1]

A listed member that lacks a declared method MUST fail with `[correctness::union_member_method]`, reported once per member. [T-203.1]

A closed union MUST be allowed in value positions in `.lgo` code (variables, parameters, results, fields, elements, keys, and values). [T-203.1]

A member type MAY hold the union itself by value in a field, element, or type-argument position: the value position stores a fixed-size interface word, so holding the union is not an invalid recursive type. [T-203.1]

A union with `~` terms or an unbounded set MUST fail with `[correctness::open_union_value]`. [T-203.1]

A conversion MUST be allowed when the source type is a member or a closed union of members. [T-203.1]
An assignment MUST require membership. [T-203.1]
An assignment of a value whose type is not a member MUST fail with `[correctness::union_nonmember]`. [T-203.1]

Any other conversion MUST fail with `[correctness::union_nonmember]` naming the type. [T-203.1]

A comparison between a union and a constant MUST convert the constant to its default type and require membership. A constant whose default type is not a member MUST fail with `[correctness::union_nonmember]`. [T-203.1]

A case type outside the member set MUST fail with `[correctness::union_nonmember]`. An assertion expression on a union value MUST compile and check membership at run time. [T-203.1]

A comma-ok assertion on a non-member MUST yield `ok == false`. [T-203.1]

A single-value assertion on a non-member MUST panic with `interface conversion: <T> is not a member of <pkg>.<U>`. [T-203.1]
The `<T>` of that panic text MUST be the string that `reflect.Type.String` returns for the asserted type, so it is qualified by the package name and never by the spelling or import alias written in the source. [T-203.1]

`case U:` MUST match only members. [T-203.1]

`nil` MUST go to `case nil`. [T-203.1]

An exhaustive `switch x.(type)` MUST cover every member and `nil`, or carry `default`. [T-203.1]

A missing member MUST fail with `[correctness::nonexhaustive_switch]` listing misses in declaration order. [T-203.1]

A case naming a sub-union MUST cover its members. [T-203.1]

Its result type MUST be the common type of the arms (identical types, the default type of untyped constants, or `T` when other arms are `nil`). [T-203.1]

Mismatched arms MUST fail with `[correctness::switch_arm_mismatch]`. [T-203.1]

A type switch expression without full coverage MUST fail with `[correctness::nonexhaustive_switch]`. [T-203.1]

A value switch expression without `default` MUST fail with `[correctness::switch_expr_needs_default]`. [T-203.1]

The switched value MUST evaluate once. [T-203.1]

Only the chosen arm MUST evaluate. [T-203.1]

A `?` in the init of a switch expression MUST evaluate the init once, at the position of the switch expression in the enclosing statement, after every call and receive operand to its left and before every operand to its right. [T-203.1][T-104.1]

When that `?` fails, the enclosing statement MUST run no later operand, assignment, call, or arm, and the function MUST return the error as a `?` statement at that position does, with the same zero results, `defer`, and `errdefer` behavior. [T-203.1][T-104.1]

The names such an init declares MUST be visible in the tag and the arms only, and the names declared in an enclosing `if`, `switch`, or `for` header before the switch expression MUST be visible in the init. [T-203.1]

In a `for` condition the init MUST run again for each test of the condition. [T-203.1][T-104.1]

A switch expression whose init ends in `?` MUST appear only in an assignment target or right-hand side, a `return` result, an ungrouped `var` declaration value, an expression statement, the call of a `defer` statement, the channel or value of a send statement, the range expression of a `for`, the init or condition of an `if` or `for`, or the init or tag of a `switch`, and the `for` or `switch` MAY carry a label [T-203.1][T-104.1].

From such a statement it MUST be reached only through parentheses, unary and binary operands, index and selector operands, composite literal elements, the function, receiver, and arguments of a call, the operand of `?`, interpolation holes, tuple elements, the base of `?.`, the left operand of `??`, and the init, the tag, the first case expression of each case, and the result of each arm of another switch expression [T-203.1][T-104.1].

Labels, `goto`, `break`, `continue`, and `fallthrough` around that statement MUST keep their meaning [T-203.1].

Any other position, among them a `for` post statement, the call of a `go` statement, the right operand of `??`, a case expression of a switch statement, a case expression after the first of its case, a case expression or arm of a type switch expression, a grouped `var` declaration, a `select` communication clause, and the body of a lambda, MUST fail with `correctness::misplaced_try` [T-203.1][T-104.1].

In the right operand of `&&` or `||` the init MUST run only when the left operand does not decide the result [T-203.1][T-104.1].

In an arm of another switch expression the init MUST run only when that arm is chosen, and in a case expression it MUST run only when no earlier case of that switch expression matched [T-203.1][T-104.1].

The run-time representation MUST be the upstream interface. [T-203.1]

A two-tier tag layout MUST NOT be used. [T-203.1]

Reflection MAY store non-members in a closed union. [T-203.1]

The compiler MUST NOT check this case. [T-203.1]

A closed-union value MUST marshal through its member static helper. [T-203.1]

Unmarshalling a closed union MUST try members in declaration order into fresh scratch values and count their successes. [T-203.1][T-Sec10]

It MUST assign the target only on a single success. [T-203.1]

Otherwise it MUST return an ambiguity error naming the union and the members. [T-203.1]

An ejected value-used union MUST keep its methods and carry the comment `// loGo/s union: A | B`. [T-203.1]

A Go importer MUST see a value-used union as that method-only interface. [T-203.1][T-Sec5][R-U2]

A union used both as a constraint and as a value type MUST fail eject with `logos eject: union U is used as a type constraint and as a value type; Go cannot express both`. [T-203.1]

Such a union has no Go-facing form, and a Go import of it MUST fail with `logos build: union U is used as a type constraint and as a value type; Go cannot express both`. [T-203.1][R-U2]

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

A property interface is an interface with at least one `const F T` or `F T` element.

A struct MUST satisfy a property interface through an explicit `: I` declaration in any package, unless every target property is supplied by promoted embedded accessor methods under the selector-depth rules below. Plain field or method sources, including a mixture of plain and promoted property sources, MUST still require `: I`. [T-203.2][R-U4]

For each property not already supplied by promoted embedded accessor methods, the declaring package MUST generate `Logos_GetF() T` from an exported field or from a method `F() T`; it MUST NOT generate redundant forwarding accessors for an already promoted property. [T-203.2]

For each read-write property not already supplied by promoted embedded accessor methods it MUST also generate `Logos_SetF(T)` from the field or from `SetF(T)`. [T-203.2]

A field-sourced getter MUST use a value receiver. [T-203.2]

A field-sourced setter MUST use a pointer receiver. [T-203.2]

A method-sourced accessor MUST keep the receiver kind of the forwarded method. [T-203.2]

A Go importer MUST see a property interface as the interface of its `Logos_GetF` and `Logos_SetF` accessor methods. [T-203.2][T-Sec5][R-U2]

A dynamic assertion such as `any(x).(I)` MUST work. [T-203.2]

Identifiers starting with `Logos_` or `_logos_` MUST be reserved in `.lgo`. [T-203.2]

A use of such an identifier MUST fail with `[correctness::reserved_identifier]`. [T-203.2]

A read-write property MUST require a pointer type. [T-203.2]

Using a value type for one MUST fail with `[correctness::property_needs_pointer]`. [T-203.2]

Taking the address of a property (`&c.HP`) MUST fail with `[correctness::property_address]`. [T-203.2]

A `: I` declaration without a field or method source for a property MUST fail with `[correctness::property_unsatisfied]`. [T-203.2]

A struct that has the right fields but no `: I` declaration MUST fail with `[correctness::property_requires_conformance]` telling the writer to add `: I`, unless every target property is supplied by promoted embedded accessor methods. [T-203.2]

Conflicting property types across listed interfaces MUST fail with `[correctness::property_conflict]`. [T-203.2]

A non-struct type MUST NOT satisfy a property interface. [T-203.2]

A write through a property interface (a plain assignment, a compound update, or a method call on a property value) MUST reach the struct. [T-203.2]

A compound update through an interface MUST evaluate the receiver once and write the value back. [T-203.2]

A method call on a property value MUST use read-call-write-back around the call. [T-203.2]

Only a type with a `: I` list MUST gain newly generated accessor methods; embedded accessor methods MUST still promote without that list. [T-203.2]

`reflect` MUST count the generated accessors. [T-203.2]

A type without the declaration MUST gain no newly generated methods; ordinary promotion of embedded methods is unchanged. [T-203.2]

Embedded fields, including property interfaces and their getter/setter accessor pairs, MUST promote properties by Go selector depth rules, with the same property selectors and conformance natively and after `logos eject`. Promotion MUST preserve the read-write property pointer requirement. [T-203.2]

An ambiguous equal-depth selector MUST fail as an ambiguous selector. [T-203.2]

Read-only and read-write declarations of one name in one interface MUST merge into read-write. [T-203.2]

```lgo
type Creature interface {
    const Name string
    HP int
    const Start() error
}
type Monster struct : Creature {
    Name string
    HP int
}
func (m Monster) Start() error {
    return nil
}
```

The `Monster.Start` value receiver above MUST satisfy the `const Start` obligation by the unchanged value-receiver inference, without an explicit `const` receiver. [T-201]

The `const Start` method obligation MUST stay distinct from the `const Name` property element: the method MUST NOT gain accessor methods and MUST NOT count as a property. [T-201][T-203.2]

A method `F() T` that sources a read-only property `const F T` MUST be proven readonly by the proof that a `const M` obligation requires, so that reading the property through a const view cannot write the receiver. [T-201][T-203.2]

A getter that writes its receiver, directly, through a nested field, or by calling a method that writes it, MUST fail the `: I` declaration with `[correctness::const_view_mutation]`; a trusted imported `FnFact.Readonly` fact counts as the proof, and a getter of a read-write property `F T` carries no such obligation. [T-201][T-203.2]

A getter that returns receiver state of a type that can alias it, a slice, map, pointer, channel, function, or interface value, or a struct or array holding one, MUST NOT be proven readonly, in the same package and across packages, and so MUST fail a `: I` declaration that needs the proof with `[correctness::const_view_mutation]`; a getter whose declared result is a const view is not such a getter, because a const view is not a mutable alias. [T-201][T-203.2]

An option `T?` holds what its payload `T` holds: a getter that returns receiver state of type `T?` follows the rule above for `T`, in the same package and across packages, so a getter of `int?` or `string?` MAY be proven readonly and a getter of `[]int?` or `*T?` MUST NOT be. [T-201][T-203.2][T-202]

`type S struct : I, J { ... }` MUST assert that `*S` implements each listed interface at the declaration. [T-203.3]

A missing method MUST fail with `[correctness::assertion_failed]` naming the method. [T-203.3]

Each assertion MUST emit a compile-time check that `*S` implements each listed interface. [T-203.3]

An assertion list on a non-struct type MUST be a syntax error. [T-203.3]
A readonly method `const M() T` MUST NOT count as a property element and MUST NOT satisfy a property requirement [T-201][T-203.2].
Unions MUST use the upstream interface representation, and a closed union MUST do so including upstream handling of zero-sized members. [T-203.1]

Method bodies inside interfaces MUST fail compilation. [T-203.2]

### 2.4 RFC-204: Tuples and fixed-array SIMD

Tuple types `(a A, b B)` and `(A, B)` in non-result positions MUST lower to structs with exported fields (`T_a`, `T_b` / `T_0`, `T_1`). [T-204.1]

Every explicit generic-argument slot MUST accept tuple types with the same field names and element types as a strict type position. Named fields MAY use `a A` or `a: A`; either spelling MUST retain names, container and nested tuple types, and parenthesized type binding in native and ejected code. Brackets on a value MUST retain indexing and slicing semantics, including tuple keys and their evaluation order. [T-204.1][T-202]

Exported field names MUST keep tuple identity across packages. [T-204.1]

A tuple and its lowered struct MUST have symmetric structural identity, including field names, element types, tags, and embedded-field status. A defined type over a tuple MUST follow Go's assignment and conversion rules: matching unnamed values MUST assign in both directions, while distinct defined types MUST require conversion. [T-204.1]

A Go importer MUST see a tuple as that struct. [T-204.1][T-Sec5][R-U2]

`var t (int, error)`, `chan (int, error)`, slices of tuples, and map values with tuple types MUST be allowed. [T-204.1]

Named access such as `t.x` MUST work. [T-204.1]

Named access MUST also work for defined tuple values and for `o?.x` when `o` is an option of that tuple type; optional selection MUST retain the usual option flattening rule. Positional tuple literals MUST check their elements against the target tuple type, including a defined tuple type or an option payload, without relaxing assignment between distinct defined types. [T-204.1]

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

Element-wise `+ - * / & | ^` MUST be allowed on `[N]T` with `N` a power of two, `T` an integer or floating-point kind other than complex (`int`, `uint`, and `uintptr` ride the 64-bit lane), and size at most 64 bytes. [T-204.2]

`.sum()`, `.min()`, `.max()`, and `.dot()` MUST be allowed on the same shapes; `a.dot(b)` takes one argument of `a`'s shape. [T-204.2]

A larger array MUST fail with `[correctness::simd_array_too_large]`. [T-204.2]

A non-power-of-two length MUST fail with `[correctness::simd_array_shape]`. [T-204.2]

With `GOEXPERIMENT=simd` on amd64 or arm64 the lowering MUST call `simd/archsimd` for element types that are a fixed-width numeric kind or an alias of one (`byte`, `rune`, `float32`, or `type f32 = float32`); `int`, `uint`, `uintptr`, and defined (non-alias) element types keep the scalar form, because their pointers do not match the archsimd loaders. [T-204.2]

Otherwise it MUST emit unrolled scalar code. [T-204.2]

On amd64 each array-operator vector form MUST run only when the `simd/archsimd` CPU feature check its methods need is true (`X86.AVX()` for 128-bit shapes and 256-bit float non-bitwise shapes, `X86.AVX2()` for 256-bit integer shapes and 256-bit float bitwise shapes, `X86.AVX512()` for 512-bit shapes and for 64-bit integer `*`) and MUST take the scalar form when it is false; the check MUST be omitted when the build's `GOAMD64` level guarantees the feature (`v3` for AVX and AVX2, `v4` for AVX512), and arm64 needs no check because its forms are baseline NEON. [T-204.2]

Direct ISA lowering is replaced by this two-form rule. [T-204.2]

A Go package MUST get no array operator and no reduction method from this rule. It MAY use `simd/archsimd` under `GOEXPERIMENT=simd`. [T-204.2][R-U1]

Reductions MUST use one fixed pairwise order in both forms, giving bit-identical results. [T-204.2]

Every operator and reduction MUST match a scalar loop oracle. [T-204.2]

```lgo
var a, b [4]float64
c := a + b
_ = c.sum()
```
A tuple with slice, map, or func elements MUST fail as non-comparable. [T-204.1][T-203.1]
A closed union MUST use upstream interface equality. [T-204.1][T-203.1]

### 2.5 RFC-205: Comptime

`const if` MUST take a constant condition. [T-205]

A non-constant condition MUST fail with `[comptime::not_constant]`. [T-205]

Both branches MUST type-check, except that a `const for` over the fields of a non-struct in a dead branch is not expanded and reports no `const_for_fields`. [T-205]

Only the live branch MUST reach lowering. [T-205]

The condition MUST end at the `{` that opens the branch, so a name or `true` before `{` is the condition and not a composite literal. [T-205]

`comptime.IsStruct[T]()` MUST be a constant `const if` condition, true when the underlying type of `T` is a struct. [T-205]

Inside a comptime generic its value MUST be taken per instantiation, and a condition built from it with `!`, `&&`, `||`, `==`, `!=` and constants MUST NOT fail `[comptime::not_constant]`. [T-205]

`comptime.SizeOf[T]()`, `comptime.AlignOf[T]()`, and `comptime.HasPointers[T]()` are compile-time constants describing the size, alignment, and pointer-containing status of `T`.

Each of them MUST equal what `unsafe.Sizeof`, `unsafe.Alignof`, and the garbage collector pointer map give for the Go type that the lowering emits for `T` on the target. That holds for every LoGOs type form: an option that the lowering keeps bare or wraps, a const view, a tuple, a closed union, and any struct, array, or option that holds them. [T-205]

A `const if` inside a comptime generic or a `const for` body MUST evaluate per instantiation or per field over `true`, `false`, constants, `comptime.IsStruct[X]()` with `X` a type argument, a type parameter or `f.Type`, and the connectives `!`, `&&`, `||`, `==`, `!=`. [T-205]

Any other non-constant condition MUST fail with `[comptime::not_constant]`, including a condition on `f.Name`, `f.Index`, `f.Exported` or the index variable. [T-205]

A `const for` in a dead `const if` branch MUST NOT fail with `[comptime::const_for_fields]`. [T-205]

`comptime.Fields[T]()`, `f.Type` used as a value, `comptime.IsStruct[T]()`, `comptime.TypeOf[X]()`, and `comptime.Error(msg)` MUST be compile-time only. `val.[f]`, `f.Name`, and `f.Tag.Get(k)` inside a `const for` body expand to per-field code; outside the body they are scope or projection errors, not run-time uses. [T-205]

Any run-time use of those compile-time-only forms MUST fail with `[comptime::runtime_use]`. A use of `val.[f]`, `f.Name`, or `f.Tag` outside a `const for` body fails with the scope or projection diagnostic. [T-205]

A generic function that uses them MUST become a comptime generic. [T-205]

Each instantiation MUST expand into a non-generic copy `<f>_logosct<N>` in the instantiating package. [T-205]

Copies MUST be memoized per type-argument list. [T-205]

An exported comptime generic MUST travel as source text in `__.LOGOS`. [T-205]

An exported comptime generic that names an unexported package-level identifier of its own package MUST fail with `[comptime::unexported_in_exported]` at that identifier in the defining package. [T-205]

The rule applies to every package that can be imported; `main` is exempt because it cannot be imported. [T-205]

A `const for` range MUST be `comptime.Fields[T]()` over a struct type. [T-205]

Any other range operand, and a non-struct `T`, MUST fail with `[comptime::const_for_fields]`. [T-205]

Expansion MUST visit fields in declaration order. [T-205]

It MUST include exported fields and unexported fields of a current-package `T`. [T-205]

It MUST exclude unexported fields of other packages. [T-205]

`val.[f]` MUST read and write the named field. [T-205]

`val.[f]` MUST name the field variable of an enclosing `const for` and project a struct value of that loop's type. [T-205]

Any other use MUST fail with `[comptime::const_for_projection]`. [T-205]

A `const for` index variable MUST NOT be assigned. [T-205]

An assignment to it MUST fail with `[comptime::const_for_assign]`. [T-205]

`f.Tag.Get(k)` MUST constant-fold when `k` is a compile-time constant: a literal, a named constant, an imported constant, or a constant expression. A run-time `k` cannot fold and stays an ordinary call. [T-205]

A live `comptime.Error(msg)` MUST fail compilation with `msg`. [T-205]

`comptime.Error(msg)` MUST take a constant string. [T-205]

A non-constant `msg` MUST fail with `[comptime::error_not_constant]`. [T-205]

`Resolve[f.Type](c)` MUST resolve per field. [T-205]

A panic in unrolled code MUST name the `.lgo` `const for` line in its trace. [T-205]

A circular instantiation MUST fail with `[comptime::circular_instantiation]` naming the chain. [T-205]

A comptime generic MUST be instantiated with concrete type arguments. [T-205]

An instantiation with a type parameter or a call inside an expansion without explicit type arguments MUST fail with `[comptime::generic_targ]`. [T-205]

A method of a generic type MUST NOT be a comptime generic. [T-205]

A method of a generic type that uses `const for` over a type parameter MUST fail with `[comptime::generic_method]`. [T-205]

An expansion above the selected `max_comptime_ast_nodes` limit with default 4,096 nodes MUST fail with `[comptime::node_limit]`. [T-205][T-102]

A stack trace MUST print the base name without the `_logosct<N>` suffix. [T-205]

A shadowed `const switch` case MUST fail with `[comptime::unreachable_case]`. [T-205]

An exact case MUST rebind `T` and dependent variables in its block. [T-205]

An interface case MUST keep the concrete layout and gain the interface method set. [T-205]

`case int` MUST match `int` only. [T-205]

`case ~int` MUST match the underlying `int` (such as `UserID`). [T-205]

A `~` case whose type is not a basic type MUST fail with `[comptime::tilde_not_basic]`. [T-205]

A `const switch` tag that is not a type MUST fail with `[comptime::switch_not_type]`. [T-205]

A `const switch` case that is not a type MUST fail with `[comptime::case_not_type]`. [T-205]

A `const switch` case MUST name exactly one type. A `default` arm and a list of types MUST be syntax errors. [T-205]

An unlabeled `break` or `continue` directly in a `const switch` case MUST fail with `[comptime::const_switch_jump]`, because the cases lower into generated switches. [T-205]

`[]*os.File` MUST never rebind as `[]io.Writer`. [T-205]

Cases MUST evaluate top-down in first-match order. [T-205]

Outside comptime generics a `const switch` MUST pick its case at run time from the type argument `T` alone, never from the dynamic type of a value. In an exact case, a rebound slice, map, or pointer MUST still refer to the storage of the caller's value. [T-205]

Inside an interface case, a call of an interface method on a value of type `T` MUST run the method of that value, whether `T` is a concrete type or an interface type. [T-205]

No match MUST return the input unchanged. [T-205]

Native and ejected output of the run-time form MUST agree. [T-205]

`comptime.ViewAs[T](b)` MUST require a POD struct (no padding, no `bool`, no pointers, interfaces, slices, maps, or strings). [T-205]

Any other shape MUST fail with `[comptime::viewas_not_pod]` naming the offset. [T-205]

A short input MUST panic with a message of the form `logos: ViewAs: need <N> bytes, have <M>` naming the sizes. [T-205]

The view MUST read multi-byte fields in the byte order of the host. [T-205]

The view MUST stay valid while it is reachable: on a target with unaligned loads, where it addresses `b`, it MUST keep `b` alive. [T-205]

Targets without unaligned loads MUST copy to an aligned temporary. [T-205]

```lgo
const if comptime.IsStruct[T]() {
    const for _, f := range comptime.Fields[T]() {
        _ = val.[f]
    }
}
```
An exact case MUST rebind the value to the case type. [T-205]
Expansion MUST include unexported fields only for the current package. [T-205]
Exact cases MUST match only the named type. [T-205]
Tilde cases MUST match underlying types. [T-205]
Native and ejected output MUST agree. [T-205]
Comptime generics MUST expand into non-generic copies in the instantiating package. [T-205]
`Resolve[f.Type](c)` MUST resolve per field in the copy. [T-205]
The check MUST track the active instantiation stack. [T-205]
Field copies MUST compare compile-time types through `f.Type`. [T-205]
A copy whose field types differ MUST fail compilation as an ordinary type mismatch. [T-205]
An expansion at or below the selected limit MUST compile. [T-205][T-102]

### 2.6 RFC-206: Go group and logos/sync

`go group (ctx) { ... }` and `go group (ctx, limit = N) { ... }` MUST lower to `logos/rt.Group`. [T-206]

The group MUST cancel on the first child error or panic. [T-206]

It MUST join all children before scope exit. [T-206]

It MUST bound concurrency with a semaphore when `limit` is present. [T-206]

`go group(x)` without a block MUST keep Go meaning as a call to a function named `group`. [T-206]

`?` inside a child MUST return to the group. [T-206]

`return` inside a child MUST fail with `[correctness::return_in_group_child]`. [T-206]

`go f(args)` inside a group body MUST form a child, sending an error-last result to the group. [T-206]

The function value and arguments MUST be evaluated in the calling goroutine before the child is submitted to the group, including when the concurrency limit would block submission. [T-206]

A `go f(args)` statement or a `go { ... }` child written inside a child MUST form a child of the same group, and spawning it MUST NOT block the spawning child on the concurrency limit: the new child MUST start once a slot is free, and `Wait` MUST wait for it. [T-206]

`go f()` outside any group body MUST fail with `[concurrency::unsupervised_goroutine]`. [T-206]

A child outside a group body MUST fail with `[correctness::child_outside_group]`. [T-206]

A `go group` statement in a function without an error result MUST fail with `[correctness::group_needs_error_result]`. [T-206]

The group error MUST propagate like `?`. [T-206]

The first failure in completion order MUST win. [T-206]

`Wait` MUST return only after all children exit. [T-206]

The context MUST be cancelled at join. [T-206]

A child panic MUST become `*logos/rt.PanicError` with the value and a stack. [T-206]

`NewGroup`, `Go`, `GoNested`, `Wait`, and `PanicError.Error` MUST form the `Group` API (`NewGroup(ctx, limit)` returns the group and a child context; `Go` and `GoNested` take `func(context.Context) error`, and `GoNested` never blocks its caller on the concurrency limit). [T-206]

A plain channel send, receive, or range directly in a child MUST lower to a `select` with `<-gctx.Done()`. [T-206]

A user `select` with `default` MUST stay unchanged. [T-206]

Cancellation MUST reach siblings blocked on channels. [T-206]

When the context argument of a `go group` statement is a plain identifier, that identifier MUST denote the group context everywhere inside the block, in the children and in the statements between them, as if the block began with `g, ctx := rt.NewGroup(ctx, limit)`, and MUST denote the outer context again after the block; the names the block declares MUST end with the block. [T-206]

The operands of a `return` in the block body, outside any child, MUST be evaluated where the statement stands, with the identifier of the previous sentence denoting the group context; the join MUST follow, and the function MUST return those values unless the join reports an error, in which case it MUST return that error. A statement after the last child MUST run before the join. [T-206]

The parent body of a `go group` MUST retain the enclosing function's label namespace and permit otherwise valid outward `break`, `continue`, and `goto` statements. Each child body MUST have an independent function label namespace: labels MUST be used within that child, and branches MUST NOT target labels in the parent or another child. [T-206]

A `?` statement in the block body, outside any child, MUST run where it stands while the children run, and MUST leave the group open for the statements after it; only when it fails MUST the join precede the return of its error, and a join error MUST replace the error of the statement. [T-206]

A `scope defer` written directly in the block body MUST run after the join at every exit of the block (the end, `return`, `break`, `continue`, labeled `break`, `goto`, a failed `?`, and a panic), and a scope defer of an enclosing block MUST run after the join of every group the exit leaves; a scope defer of a block nested in the body MUST run when that nested block exits, before the join. [T-206][T-104.7]

The context and limit operands of a `go group` statement MAY be any expressions, including calls that hold function literals, and the ejected program MUST evaluate them as the built program does. [T-206][T-0.3]

When built by `logos`, a child blocked in netpoll-backed network I/O MUST wake within 100 ms with a timeout error satisfying `os.IsTimeout`. [T-206]

In ejected code, cancellation MUST leave a child's blocked network I/O to complete on its own, and `Wait` MUST return once it completes. [T-206][T-0.3]

Cancellation MUST NOT wake a goroutine outside the group. [T-206]

Cgo calls MUST NOT wake this way. [T-206]

```lgo
func f(ctx context.Context) error {
    go group (ctx, limit = 8) {
        go { work()? }
    }
    return nil
}
```

A channel send, receive, or range directly in the `map` or `filter` lambda of a group stage segment, in any statement or expression position other than an operand of a select case, MUST wait on the cancellation of the stage like one directly in a group child, and `[correctness::unbounded_channel_op]` MUST NOT report it. [T-206]

The same operation in a lambda after the segment, in a plain chain, in a function literal inside a segment lambda, in an operand of a select case, or in a lambda after a method named `group` that is not a stage MUST still be reported. [T-206]

When the context argument of a group stage is a plain identifier, that identifier MUST denote the group context of the stage inside every `map` and `filter` lambda of the segment, and MUST denote the outer context in every lambda after the segment and in the rest of the function. [T-206]

When the context argument of a `go group` statement or of a group stage is not a plain identifier, no name MUST be rebound: the names in the block and in the lambdas keep denoting the outer context, a call on that name is not cancelled by the group, and only the channel operations written directly in a child or segment lambda wait on the cancellation. [T-206]

Results of a group stage MUST arrive in input order, and after `.unordered()` in completion order. [T-206]

Children in flight plus finished results not yet consumed MUST NOT exceed `limit`. [T-206]

`limit <= 0` MUST panic with `logos: group stage limit must be positive` when the range starts, before the source is touched. [T-206]

Each group stage MUST keep its own limit, including a stage inside a segment lambda. [T-206]

A `?` in a segment lambda MUST fail the stage. After `.unordered()` the first failure in completion order MUST be the returned error and MUST cancel the siblings at once. [T-206]

In input order the returned error MUST be the failure of the lowest input index, every element before it MUST be delivered first and none after it, and the siblings MUST be cancelled when that failure reaches the head of the window; a panic in a segment lambda counts as the failure of its index. [T-206]

A panic in a segment lambda MUST become `*logos/rt.PanicError`. [T-206]

When no child failed and the parent context is done after the join, the error MUST be the context's error. [T-206]

An ordered stage MUST fail exactly where the same chain without the stage fails: a failed element MUST NOT be passed over. [T-206]

A consumer that stops early MUST discard the failure of any element after the last delivered one. [T-206]

A `break`, a `return`, a panic in the loop body or in the source, and a terminal that has its answer MUST cancel and join every child before control leaves the statement. [T-206]

A receive-channel source MUST stop waiting when the stage is cancelled. [T-206]

The source MUST be pulled in the calling goroutine, and the loop body, the stages after the segment, and the terminal, with every lambda they hold, MUST run on that goroutine. [T-206]

A receive-channel source MUST deliver a finished result while the channel is quiet. [T-206]

A slice, array, or `iter.Seq` source MUST deliver finished results when the source yields its next element or ends, and an `iter.Seq` that blocks without yielding MUST delay the delivery of finished results and the end of the range after a cancel until it yields or returns, because the stage pulls the source on the calling goroutine and starts no goroutine for it. [T-206]

An empty source MUST yield no result and no error. [T-206]

The sequence that a group stage lowers to MUST do no work until it is ranged, and ranging it a second time MUST panic with `logos: group stage sequence ranged twice`. [T-206]

After a stage stopped, it MUST ignore the elements that the source still yields, MUST NOT start a child for them, and MUST NOT call the consumer again. [T-206]

A finished result that is never delivered MUST be closed when it implements `io.Closer` and the chain owns it, in reverse input order after the join. [T-206]

A result dropped by a `filter` of the segment, and the value that a failed `filter` predicate examined, MUST be closed once when the chain owns it, and a delivered result MUST NOT be closed by the stage. [T-206]

The chain owns a value that a `map` of the segment produced, by the rule of a fused chain. [T-104.6][T-206]

The stage MUST NOT close an element of the source, a value that a `map` returns unchanged or reads from elsewhere, or any other value that no `map` of the chain produced, whether the stage drops it, never delivers it, or stops before it hands the value to a lambda. [T-206]

A `Close` that the stage calls and that panics MUST propagate to the caller after the join, and the stage MUST still close every remaining result it owns. [T-206]

The ejected form of a group stage MUST build with upstream Go and print what the native build prints. [T-206][T-0.3]

`logos/sync.Locked[T]` MUST expose `WithLock(f func(*T))`. [T-206]

`WithLock` MUST hold `Lock`, defer `Unlock`, then call the function. [T-206]

It MUST unlock after a panic. [T-206]

A `WithLock` lambda MUST NOT return or store its argument. [T-206]

`logos/sync.Hot[T]` MUST expose `Load() const *T`, `Store(*T)`, and `Update(func(old const *T) *T)` with a CAS loop. [T-206]

Concurrent updates MUST never expose a partial value. [T-206]

No retire list MUST exist. [T-206]

Values MUST stay GC-managed. [T-206]

```lgo
var c Locked[Config]
c.WithLock(p => { p.Port = 8080 })
```
`Hot[T]` MUST hold its value through `atomic.Pointer[T]` only. [T-206]

## 3. Tier 3 specification

A rule that sits behind a `GOEXPERIMENT` flag MUST stay behind it [T-0.1][R-U1].

### 3.1 RFC-301: Tier-0 arena and placement regions

Substrate [T-301.1]:
Region memory MUST come from 64 KB chunks leased by the goroutine that holds region objects; each open region holds at most its own chain. [T-301.1]
A chunk MUST hold a noscan region that grows up and a scan region that grows down. [T-301.1]
An allocation that would cross the two cursors MUST link a fresh chunk and MUST still succeed. [T-301.1]
Scan-region objects MUST carry a type word, and the collector MUST scan them with the goroutine stack. [T-301.1]
Popped scan memory MUST be zeroed before reuse. [T-301.1]
After a release, a collection MUST find no stale pointers. [T-301.1]
Pointer-containing region objects MUST keep their heap referents alive across `runtime.GC()`. [T-301.1]
When a goroutine parks, it MUST release every chunk whose noscan and scan regions are both empty, including empty chunks behind a live chunk. [T-301.1]
`runtime.SetFinalizer` and `runtime.AddCleanup` on a region object MUST panic with `runtime: <API> on loGo/s arena object`, naming the called function. [T-301.1]
Cgo pointer checks MUST NOT panic on region addresses. [T-301.1]
Region placement MUST NOT change program meaning: every program MUST print the same output when every region allocation is a heap allocation. [T-301.1]

Implicit placement [T-301.1][R-U1]:
With implicit placement on, an object that escape analysis proves does not outlive its frame, but that cannot live on the stack, MUST allocate in the frame's region. [T-301.1]
Frame exit, including exit by panic, MUST release that region and restore the watermark. [T-301.1]
A returned value that the caller keeps MUST allocate in the caller's region only when escape analysis proves no pointer to it outlives the caller. [T-301.1]
A value stored into `sync.Pool` MUST stay on the heap. [T-301.1]
A loop back-edge MUST release iteration-local objects by restoring the watermark. [T-301.1]
Implicit placement MUST stay off unless the build sets `GOEXPERIMENT=logosarena`. [T-301.1]
Under `logos`, `[build] arena` MUST set that experiment, and its default MUST be true. [T-301.1][R-U1]
Under plain `go`, only `GOEXPERIMENT=logosarena` MUST enable it. [T-301.1][R-U1]

Placement regions [T-301.2]:
The std package `logos/arena` MUST provide `Open() Region`, `New[T any](r Region) *T`, `Make[T any](r Region, n int) []T`, and `Append[T any](r Region, s []T, v ...T) []T`. [T-301.2]
Without `GOEXPERIMENT=logosarena`, these MUST behave exactly as `Region{}`, `new(T)`, `make([]T, n)`, and `append(s, v...)`. [T-301.2]
Under `logos` with `[build] arena` true, each call MUST resolve at compile time, per type argument, to the noscan region when `comptime.HasPointers[T]()` would be false, and to the scan region otherwise. [T-301.2]
Constant-size noscan objects of at most 512 bytes MUST use the inline bump sequence on amd64 in non-race builds; other targets call the region slow path. [T-301.2]
`Open` MUST make a region owned by the innermost enclosing block. [T-301.2]
Exit from that block MUST release the region's objects, and inside a loop body each iteration MUST release them. [T-301.2]
An `Open` in an `if`, `switch`, or `for` init clause MUST belong to the whole statement, and its close MUST run after the statement. [T-301.2]
The compiler MUST place an object in `r` only when escape analysis proves that no pointer to it outlives the owner block of `r`. [T-301.2]
Otherwise the object MUST go on the heap, and `perf::arena_fallback` MUST name the call and the escaping flow where Sec 1.2 requires the report. [T-301.2]
A store of a region pointer into a region object whose owner block is the same as, or nested inside, the owner block of the stored pointer's region MUST NOT count as an escape. A store into an object of an enclosing region MUST count as an escape. [T-301.2]
In a LoGOs package, a `Region` that is stored outside a local variable, returned, sent on a channel, captured by a `go` statement, or captured by an escaping closure MUST fail with `correctness::region_escape`. [T-301.2]
This return restriction includes bare returns of named results containing a `Region`, directly or through a local holder binding. Returning ordinary values computed using a region does not itself return the region. [T-301.2]
In a Go package, passing an `r` to a callee is allowed. With `GOEXPERIMENT=logosarena` placement through that region follows the same escape rule as in a LoGOs package: the compiler places the object in the region only when escape analysis proves no pointer to it outlives the owner block of `r`, otherwise the object goes on the heap; a use that would fail with `correctness::region_escape` in a LoGOs package falls back to the heap instead. Under `logos lint` that fallback is reported as `perf::arena_fallback`, naming the call and the escaping flow; `logos build` and plain `go` report no rule ID. Without the experiment every `logos/arena` call behaves as its heap equivalent. [T-301.2][R-U1]
A `go group` child MAY read a parent-region object through a const view. [T-301.2]
A non-const capture MUST count as an escape of that object. [T-301.2]
`logos eject` MUST lower `arena.Open()` to `_logos_rt.Region{}`, `arena.New[T](r)` to `new(T)`, `arena.Make[T](r, n)` to `make([]T, n)`, and `arena.Append[T](r, s, v...)` to `append(s, v...)`. [T-301.2]
`logos/rt` MUST provide `type Region = arena.Region`. [T-301.2]
The shared support module MUST carry the `arena` package. [T-301.2]
The eject MUST append `_ = r` after each local Region declaration. [T-301.2]
The toolchain MUST NOT ship the upstream `arena` package, `reflect.ArenaNew`, or `GOEXPERIMENT=arenas`. [T-301.2]

Comptime and escape analysis:
Escape analysis MUST record a parameter that leaks only into the pointee of another parameter. [T-301.2][R-U1]
This precision MUST apply to LoGOs packages, and to Go packages under `GOEXPERIMENT=logosescape`, which is off by default. [T-301.2][R-U1]

Recursion MUST return region-allocated results to the caller intact. [T-301.1]
Idle parked goroutines MUST consume 0 bytes of region memory. [T-301.1]
A parked goroutine MUST retain every live chunk and preserve all live-object addresses. [T-301.1]
A parked goroutine with no live chunks MUST retain no region capacity; its ordinary stack and runtime metadata still exist. [T-301.1]
A loop with only iteration-local allocation MUST NOT grow arena use across iterations. [T-301.1]

### 3.2 RFC-302: Smart stack engine

The variable-size `make` stack buffer is 32 bytes at O0 and O1, and under plain `go` without `-O`. At O2 it is 256 bytes and at O3 it is 4096 bytes for non-standard packages; a `logos build` with no `-O` flag uses `logos.toml [build] opt_level`, which defaults to O2. Standard-library-provenance packages and functions in a recursive call-graph cycle keep the 32-byte buffer at every opt level. [T-302]
A slice `make` with variable effective capacity that flows into a cgo call argument or an `unsafe.Pointer` conversion MUST NOT use the stack buffer and MUST allocate its backing storage on the heap. Effective capacity is the explicit capacity when present, otherwise the length; it is fixed only when the compiler proves it constant with `ir.IsSmallIntConst` after literal rewriting. A `make` with fixed effective capacity MUST retain ordinary escape and size behavior; it is not a guarantee of stack allocation. [T-302]
Under `GOEXPERIMENT=logosstack`, the runtime MUST choose the initial goroutine stack size from the per-PC exponential moving average of final stack sizes. [T-302]
Without history for a PC, it MUST keep the current stack size. [T-302]

### 3.3 RFC-303: Hardware backend optimizations

Fault-suppressing masked tails MUST appear only in loops that the slice reductions of Sec 13 produce. [T-303]
Other code MUST carry no tail padding, and the build MUST NOT add guard padding for vector tails. [T-303]
BMI2 code MUST come from existing `GOAMD64=v3` rules. [T-303]
`CMOV` code MUST come from the existing baseline `CondSelect` rules at every `GOAMD64` level. [T-303]
Constant-count rotates MUST use BMI2 rotates at `GOAMD64=v3`. [T-303]
Variable-count rotates MUST stay on the existing shift and rotate path. [T-303]
In `.lgo`, non-constant `*` inside a `make` length or capacity MUST lower to `logos/rt.MulLen`. [T-303]
On overflow, `MulLen` MUST panic with `logos: make length overflow: w * h`. [T-303]
LSE atomics MUST come from the `GOARM64=v8.2` default. [T-303]
Cache-line spacing MUST keep the upstream per-GOARCH `CacheLinePadSize` values (64 on amd64, 128 on arm64). [T-303]
The runtime structs MUST carry no new padding. [T-303]
Preemption checks in long loops MUST fire every 4096 iterations, with the first check at iteration 4096, on GOARCH=wasm builds with the checks enabled. An explicit GOEXPERIMENT=preemptibleloops elsewhere yields the upstream per-backedge checks. [T-303]
The checks that fire every 4096 iterations MUST run only where async preemption is absent (`GOARCH=wasm`). [T-303]
A call-free loop MUST NOT starve peers. [T-303]
A peer goroutine MUST observe one quantum at 4096 iterations and two at 8192 iterations. [T-303]
`clear(b)` of a slice MUST reach a runtime memory clear and MUST persist before return. [T-303]
Dead-store elimination MUST NOT remove that clear. [T-303]
The rest of the WASM profile is defined in SPEC Sec 12. [T-Sec12]
Go 1.27 lowering already keeps `clear(b)` as a runtime memory clear. [T-303]

### 3.4 RFC-304: Fast-CGo engine

A LoGOs package MUST accept cgo files beside other `.lgo` files. [T-304][A-CODEGEN]
A file that imports `C` MUST use Go syntax only. [T-304][A-CODEGEN]
LoGo/s syntax in such a file MUST fail with `[correctness::cgo_file_logos_syntax] files that import "C" must use Go syntax only`. [T-304][A-CODEGEN]
Pointers that C returns MUST count as platform pointers. [T-202]
Storing one into a non-nil location MUST check it. [T-202]
When the fast path is on, a C function that carries both `#cgo noescape` and `#cgo nocallback` MUST take the fast call path. [T-304][R-U1]
This rule MUST hold in Go packages under `logos` and under plain `go`, and in LoGOs packages under `logos`. [T-304][R-U1]
Go code MUST need no new directive and no new syntax for the fast path. [T-304][R-U1]
A C function that carries only one of the two directives, or none, MUST keep the upstream call path. [T-304]
One fast path MUST serve Go packages and LoGOs packages: the single `runtime.logosCgocallFast` trampoline and the one cgo pipeline of this section. [T-304][R-U1]
The fast path MUST use the syscall-style `reentersyscall`/`asmcgocall`/`exitsyscall` sequence, so it may release the P like a syscall; fast-call diagnostics stay on the original M. [T-304]
It MUST NOT read a clock in the fast body: no per-call monotonic read takes part, because that read dominated the measured fast path and no consumer read the value. [T-304]
The fast path MUST stay behind `GOEXPERIMENT=logosfastcgo`, which is off by default under both commands. [T-304][R-U1]
Profiling with `SIGPROF` during fast calls MUST NOT crash. [T-304]
Fast calls MUST carry noescape and nocallback marks, spelled `//logos:cgo_fastcall` in `.lgo`. [T-304]
The `.lgo` marker MUST be recognized only as a two-field line in the C preamble selected by cgo, after comment bodies are joined and carriage returns removed. Ejection MUST map that marker to both upstream directives regardless of `GOEXPERIMENT=logosfastcgo`; detached, ordinary Go, and malformed comments MUST NOT acquire a cgo contract. [T-304][T-0.3]
A C function marked `nocallback` that calls into Go MUST still panic on the fast path. It MUST NOT hang and MUST NOT corrupt state. [T-304][R-U1]
A CPU profile MUST attribute time in a fast-called C function to its Go call site. [T-304][R-U1]
Go pointer arguments MUST stay valid during a fast call. [T-304]
The original M holds diagnostics only: `m.fastCgoFrame[0]` records the DI/arg address copy; record also zero-stores slots 1 to 3 so a reused M carries no stale addresses, and clear zeroes all four. The slots are plain `uintptr`, not pins or roots. [T-304]
The fast-call state MUST read zero after return without clearing the caller's argument values. [T-304]
POD and primitive-slice arguments MUST skip `_cgoCheckPointer`. [T-304]

Implementation note (informative): Fast calls preserve `runtime.NumCgoCall` accounting and reset the cgo traceback buffer.
The original M owns the diagnostic slots, which read zero after return, as the fast-call state rule above states, even when sysmon retakes its P.
Normal return and rejected callbacks clear this record before leaving syscall state.
The slots contain address copies, not GC roots or pins.
The ordinary cgo pointer-lifetime rules still apply.
With `GOEXPERIMENT=logosfastcgo`, eligible result wrappers omit stack-top lookups and result-frame relocation.
The trusted `nocallback` contract and the runtime's syscall stack rules keep that frame stable.
Outside fast-call eligibility, wrappers retain ordinary relocation unless
section 3.6's independent `logoscgoopt` experiment proves a stable frame.
Result stores, sanitizer operations, and callback guards are unchanged.
These implementation changes do not establish the speed or blocking gates above.

#### 3.4.1 Shared-library export usage (informative)

This subsection is informative and adds no requirements.
The [shared-library fixture](../src/cmd/go/testdata/script/logos_cgo_shared.txt) demonstrates C and Python calls on Linux.
Its `main` package contains a Logos core and a separate Go-syntax `.lgo` bridge.
The bridge imports `C` and declares the `//export` functions.

Enable cgo in `logos.toml`:

```toml
[build]
cgo = "allow"
```

Build the library with `logos build -buildmode=c-shared -o liblogos.so ./lib`.
The build also produces `liblogos.h`.
C callers include that generated header.
Python callers use the same ABI through `ctypes`, with explicit `argtypes` and `restype`.
The library includes the managed Go runtime; it is not runtime-free.

The fixture uses fixed-width integers, example status codes, and borrowed pointer-plus-length buffers copied before use.
Opaque integer handles backed by `runtime/cgo.Handle` retain Go objects without exposing Go pointers.
The caller supplies valid buffer ranges and aligned, writable output slots.
Selected exports recover Go panics and return status codes; this does not provide recovery from every fault.
Concurrent reads use an immutable object.
The caller joins all reader threads before deleting its handle.
The fixture's `-buildvcs=false` isolates the test from VCS metadata; it is not an FFI requirement.

### 3.5 RFC-305: GC scan work and optimization

GC improvements MUST use the shared runtime for Go and LoGOs packages under both commands. The region rules below MUST retain the `GOEXPERIMENT=logosarena` gate and the enablement defaults of Sec 3.1. [T-305][R-U1]
They MUST preserve typed pointer maps, external heap and stack roots, write barriers, safe points, and the lifetime and zeroing rules of Sec 3.1. [T-305]
They MUST preserve the existing collector selection and the semantics of `GOGC`, `GOMEMLIMIT`, `debug.SetGCPercent`, and `debug.SetMemoryLimit`. They MUST NOT add an alternate GC mode. [T-305]

#### 3.5.1 Region root work

The work that a goroutine root scan reports MUST include its live implicit arena allocations and its open placement regions. [T-305]
Each scanned allocation MUST contribute the byte extent from its base through its last typed pointer slot. This extent includes scalar gaps before that slot and excludes the scalar tail after it. For repeated element types, it MUST include the extent through the last element's last pointer slot. [T-305]
Nil and internal pointer values MUST contribute the same scan extent as external pointer values in those slots. Noscan allocations, scan headers, unused chunk capacity, and allocations removed by restore or close MUST NOT contribute. [T-305]
The runtime MUST sum that work across every live chunk and open region, and MUST report it through the existing stack-root scan credit. [T-305]
The controller's worst-case root-work estimate MUST include each leased region chunk once and exclude pooled chunks. Reuse, release, and failed acquisition MUST keep this bound consistent. This capacity bound MUST stay separate from completed scan work. [T-305]
The ordinary goroutine stack-size sample MUST include only the physical stack extent; region scan work MUST NOT enlarge that sample. [T-305]

Implementation note (informative): scan extent is the byte-based GC work estimate used by the collector. It is not a count of pointer words, retained capacity, or process memory. Correcting this estimate does not by itself establish a speed improvement. Optional speed changes require the fixed-work measurements and behavior checks of IMPLEMENTATION G6.

#### 3.5.2 Same-allocation pointer lookup

While scanning a region allocation, the runtime MAY omit heap metadata lookup for a pointer whose target lies within that same allocation. The check MUST use the complete allocation's half-open byte range; sharing a chunk alone MUST NOT qualify. [T-305]
The scanner MUST still visit every typed pointer slot, including repeated element types and slots beyond the first pointer-mask word. Internal pointer values MUST retain their scan-work contribution under Sec 3.5.1. [T-305]
External heap and stack pointers MUST retain their existing tracing behavior. Scalar words MUST NOT become roots, and restore or close MUST end the allocation's contribution under the existing lifetime rules. [T-305]

### 3.6 RFC-306: Cgo compiler optimization

The cgo optimizer MUST specialize the shared cgo pipeline independently of
fast-cgo trampoline selection. It MUST apply to Go packages built by `go` or
`logos` and to LoGOs packages built by `logos`. It MUST preserve Go semantics,
the C ABI, and the cgo pointer rules. [T-306]

#### 3.6.1 Stable return frames

`GOEXPERIMENT=logoscgoopt` MUST enable ordinary-cgo return-frame specialization
and MUST be off by default. `nologoscgoopt` MUST disable it. The experiment
MUST NOT require or enable `logosfastcgo`. [T-306]

With `logoscgoopt` enabled, a gc-generated result wrapper whose C function's
`#cgo nocallback` contract is known during generation MUST omit both
`_cgo_topofstack()` calls and the result-frame relocation.
`#cgo noescape` MUST NOT be required for this
specialization. Calls without `nocallback`, including `noescape`-only calls,
MUST retain relocation. The gccgo generator MUST remain unchanged. [T-306]

A contract that is unknown when the C wrapper is generated MUST retain the
conservative path. Section 3.6.3 makes parsed package contracts available
before emission when `logoscgoopt` is enabled. [T-306][T-306.1]

The ordinary wrapper MUST still call `runtime.cgocall` and enforce the
`nocallback` contract. Scheduler transitions, system-stack switching, GC
visibility, call accounting, tracing, profiling, and blocking behavior MUST
retain the ordinary cgo path. The specialization MUST preserve argument and
result layout, result stores, errno capture, and TSan/MSan operations. [T-306]

`nocallback` alone MUST NOT imply `noescape`: a function may retain a legally
pinned Go pointer. Its generated Go wrapper MUST retain the upstream escape
and liveness treatment. The existing `noescape` plus `nocallback` pair MUST
keep its upstream keepalive treatment. Pointer checks and pinning requirements
MUST remain unchanged. [T-306]

When `logosfastcgo` is also enabled, section 3.4 MUST continue to determine
trampoline eligibility. Disabling `logoscgoopt` MUST NOT disable the existing
fast-call frame specialization. A call that may enter Go MUST still relocate
its result frame after a callback that moves the goroutine stack. [T-306]

Correctness argument (informative): `runtime.cgocall` enters syscall state
before the C wrapper runs and leaves it after the wrapper returns. Without a
callback, the wrapper stores its result while that state is still active.
`runtime.copystack` rejects a nonzero `syscallsp`, and `isShrinkStackSafe`
rejects a syscall stack. The frame therefore cannot move in this interval.
The proof does not require a short call or a nonblocking function.

#### 3.6.2 Further compiler work (informative)

The following stages are planned work, not behavior enabled by `logoscgoopt`:

| Stage | Decision the compiler can improve | Evidence needed before implementation |
|---|---|---|
| C effect and capture summaries | Keep a non-retained argument on the Go stack; distinguish callback and memory effects | Analysis of the actual C implementation and its transitive callees, with per-parameter capture facts and proof that callbacks cannot invalidate stack addresses |
| ABI specialization | Remove redundant packing, loads, stores, and adapter work | Target-specific layout and calling-convention proofs, including aggregates, alignment, variadic calls, and instrumentation |
| Call-region optimization | Amortize boundary work across a proven sequence of calls | Preservation of visible side effects, errno, pointer lifetimes, callbacks, scheduling progress, and maximum interruption delay |
| Cross-language inlining | Remove the boundary for supported kernels | A shared analyzable representation with Go-compatible stack maps, GC roots, safe points, and exception behavior |

An effect summary needs the identity of the function body, transitive callees,
compiler and flags, target ABI, and linked definitions. Unknown indirect calls,
symbol interposition, inline assembly, and opaque libraries remain conservative.
An absent or invalidated summary cannot establish a safety property. A C
`const` parameter does not prove non-retention, `nocallback` does not prove
nonblocking behavior, and a profile cannot establish either contract.

LoGOs uses the Go gc backend. Enabling C LTO alone cannot inline C into a Go
caller. A common IR or a supported C-to-Go-IR translation would be a separate
compiler project. Arbitrary native C code cannot acquire Go safe points merely
by being fast in a benchmark.

Verification (informative): M20-S2 records generated-code checks, callback
stack-growth and pointer-lifetime regressions, and an identical-work A/B
benchmark with fast-cgo disabled. G2, G3, and G6 in IMPLEMENTATION.md govern
validation and promotion. The experiment's existence is not a performance
claim or approval to change its default.

#### 3.6.3 Package-wide cgo contracts

With `logoscgoopt` enabled for the gc generator, cgo MUST collect the
existing `nocallback` and `noescape` contracts from every parsed input
after parsing completes and before emitting any C wrapper. Directive-only
inputs and the contracts mapped from `//logos:cgo_fastcall` MUST participate.
An eligible call MUST NOT miss return-frame specialization solely because
its directive appears in a later input file. [T-306.1]

Collection MUST copy only contract metadata. It MUST preserve per-file C
preambles, macros, and declarations, the existing translation and recording
order, and the existing phase of unmatched-directive validation. Invalid
package names, incompatible C declarations, and unmatched directives MUST
retain their existing rejection. [T-306.1]

The final generated Go wrappers MUST remain unchanged: package collection
MUST NOT broaden fast-cgo eligibility, change callback guards, or change
escape and keepalive treatment. A noescape-only function MUST retain
result-frame relocation. Experiment-off, gccgo, and godefs generation MUST
remain unchanged. [T-306.1]

#### 3.6.4 Reuse validated heap spans

With `logoscgoopt` enabled, the unknown-pointer cgo checker MUST reuse the
heap span it has already validated when locating the containing allocation.
It MUST compute the same allocation base and scan the same pointer bitmap
and extent as the existing `findObject` path. [T-306.2]

An interior pointer MUST NOT restrict checking to bytes after that pointer.
Unpinned nested pointers MUST still be rejected in small objects, objects with
allocation headers, and large objects. Pinning the nested allocation MUST
retain the existing successful result. Nil, foreign, stack, and global
addresses MUST retain their existing treatment; unknown data/bss extents
remain conservative. [T-306.2]

Exported-result root and nested-pointer pinning, GC liveness, instrumentation,
ordinary runtime entry, and callback behavior MUST remain unchanged. The
disabled experiment MUST retain `findObject` and its returned base and span.
The private selection helper MUST inline in both settings. [T-306.2]

#### 3.6.5 Pointer-free typed addresses

With `logoscgoopt` enabled, a cgo argument whose original typed address
identifies a value containing no Go pointers MUST bypass redundant address
classification. The optimization MUST preserve cgo's existing address-expression
provenance and MUST NOT infer a contract from the C parameter type. [T-306.3]

Saved `unsafe.Pointer` values, pointer-bearing values, and slice or array
provenance MUST retain their existing checks. An arbitrary interior pointer
MUST still check the allocation required by the upstream cgo rules. A
pointer-free field MUST NOT make neighboring unpinned pointers safe to pass
through a saved pointer. [T-306.3]

The optimization MUST preserve argument evaluation count and order, deferred
calls, typed nil values, GC liveness, pinning requirements, callback and result
checking, and instrumentation. It MUST remain independent of `logosfastcgo`
and MUST NOT change the default of `logoscgoopt`. [T-306.3]

Correctness argument (informative): the existing address marker tells the
runtime to check the pointed-to Go type. When that type contains no pointers,
the checker returns without reading the address. Its earlier nil/non-Go
address returns have the same result. The new predicate performs only type
metadata reads and admits exactly this no-op case. The pointer-bearing and
unknown-provenance paths remain unchanged.

## 4. Lexer, parser, and syntax

### 4.1 Tokens and contextual words

The language MUST add no new global keywords [T-0.2]. The words `errdefer`, `yield`, `scope`, and `group` MUST act as contextual words valid only in `.lgo` files [T-0.2]. The word `errdefer` MUST start a statement only when an identifier or `func` follows it [T-0.2][T-104.2]. Inside a function whose results are `iter.Seq` or `iter.Seq2`, or an alias of either, `yield` followed directly by an operand start MUST start a statement; elsewhere a `yield e` that is not a valid Go statement as written still parses as a statement and the type checker rejects it with `correctness::yield_outside_generator` [T-0.2]. The word `scope` MUST appear only directly before `defer` [T-0.2]. The word `group` MUST appear only directly after `go` [T-0.2]. The word `const` MUST gain new positions before `for`, `if`, `switch`, and `func`, and those positions MUST remain syntax errors in plain Go [T-0.2]. The word `const` MUST also appear before an interface method declaration to mark a readonly obligation, before an interface property element, and before a type to form a const view, and in no other position outside the listed positions and ordinary Go constant declarations; each of those three positions MUST remain a syntax error in plain Go [T-0.2][T-201]. Outside its keyword position and outside `yield e` statements, each of `errdefer`, `yield`, `scope`, and `group` MUST keep its ordinary identifier meaning, so `errdefer := 1` MUST parse as an assignment [T-0.2][T-104.2]. The form `?` MUST scan as a postfix operator only in `.lgo` files [T-104.1]. The form `?.` MUST scan as one unit only when `?` joins `.` with no gap [T-104.3]. The form `??` MUST scan as one binary operator only when two `?` marks join with no gap [T-104.3]. The form `=>` MUST scan as one unit only when `=` joins `>` with no gap in `.lgo` files [T-104.6]. The form `$"..."` MUST scan as an interpolated string only when `$` joins an opening quote with no gap in `.lgo` files [T-104.4].

### 4.2 Grammar

New forms MUST follow this grammar [T-0.2][T-104.1][T-104.3][T-104.4][T-104.6][T-201][T-202][T-203.1][T-203.2][T-203.3]:

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
ReadonlyMethod ::= "const" Name Signature
ConstView    ::= "const" Type
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

The type `T??` MUST remain a syntax error [T-202]. The type `(T?)?` MUST nest as an option of an option [T-202]. The union in `UnionDecl` MUST list a finite set of terms with no `~` term when used as a value type; with a `~` term it is a constraint only [T-203.1]. A `ReadonlyMethod` MUST appear only as an interface method; `const Name() string` MUST be a readonly method obligation and MUST stay distinct from property `const Name string` [T-201][T-203.2]. Outside the listed positions, `const` keeps its ordinary Go meaning, and other positions remain syntax errors as in plain Go [T-201]. Operands inside `$"..."` are checked by `LogosInterpKindOf`; values outside its accepted kinds fail with `correctness::interp_type` [T-104.4]. The texts `{{` and `}}` MUST mean literal braces [T-104.4].

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

### 4.3 Disambiguation and semicolon insertion

An opening `(` MUST branch on the next token after its element list [T-104.6]. A `:` after the first element MUST mean a named tuple type [T-204.1]. A `,` list followed by `=>` MUST mean a lambda, and each element MUST be a plain name [T-104.6]. A lone name followed by `=>` MUST mean a one-parameter lambda [T-104.6]. A lone `(e)` with no `=>` after it MUST mean a parenthesized expression [T-104.6]. The token `??` MUST bind less tightly than `||`, so `a || b ?? c` MUST parse as `(a || b) ?? c` [T-104.3]. Chained `??` MUST group to the right, so `a ?? b ?? c` MUST parse as `a ?? (b ?? c)` [T-104.3]. A newline after `?` MUST end the statement [T-104.1]. A newline after `?.` MUST NOT end the statement [T-104.3]. A newline followed by spaces or tabs and then `.` plus a letter or `_` MUST NOT insert a semicolon [T-104.6]. The form `x` newline `.5` MUST still insert a semicolon [T-104.6].

## 5. Cross-package facts

Fact application and declaration overlays MUST resolve only source declaration identifiers when walking an imported package scope. Compiler-only symbols for hoisted local types and synthetic method entries MUST remain lazy; a generic local type requires its enclosing dictionary and MUST NOT be forced through the public type importer. Legal package-private and Unicode declarations, and methods reached through their declaring types, MUST retain their verified facts. [T-Sec5]

Cross-package facts MUST travel in the archive member `__.LOGOS` [T-Sec5]. Exported declaration signatures remain in `__.PKGDEF`; `__.LOGOS` carries fingerprint-bound typed fact records and no declaration header. [T-Sec5][T-201] The member MUST hold one fact record per function that the rules below give a record. A record carries `ReceiverMutated`, the `ParamMutated` mask, `FnFact.Readonly`, `MayYield`, `AlwaysNilError`, and generic assertion effects `AssertTP` and `AssertRTP` [T-Sec5][T-201]. A function carrying the `AlwaysNilError` flag MUST return a nil error on every return path [T-Sec5][T-102]. Facts MUST use format version 3; older members MUST be rejected rather than read through a compatibility path [T-Sec5][T-201]. Presence of assertion-effect flags MUST distinguish analyzed effects from unknown effects; explicit `AssertTP=0,AssertRTP=0` MUST denote proven-zero effects [T-Sec5][T-201]. Assertion and mutation masks MUST use arbitrary-arity hexadecimal masks without a 64-parameter limit [T-Sec5][T-201]. Encode and decode MUST preserve `FnFact.Readonly` [T-201][T-Sec5]. The facts MUST bind to the `__.PKGDEF` fingerprint, and a stale fingerprint MUST fail the dependent build [T-Sec5]. A Go importer of a LoGOs package MUST see the signatures that `logos eject` emits: a closed union value type as its methods-only interface with the terms dropped, `*T?` as the bare pointer, a non-nil `*T` as a plain `*T`, a property interface as its `Logos_GetF` and `Logos_SetF` accessor methods, a readonly interface method as its plain method, and a const view as its plain type [T-Sec5][R-U2][T-201]. Lowering and eject MUST erase only the `const` keyword of a readonly method for emitted Go, keeping source positions and qualification facts for LoGOs importers [T-201][T-0.3][T-Sec5]. A Go package member, when the writer emits it, MUST carry one `lang go` record and no declaration header [T-Sec5][R-U1][T-201]. Under the `logos` driver the member of a non-standard Go package MUST also carry one `fn` record per exported function or method with parameters and a body, except functions with type parameters, holding an explicit `ParamMutated` mask with `ParamMutated=0` denoting proven zero [T-Sec5][R-U1][T-201]. It MUST NOT carry `FnFact.Readonly` records [T-Sec5][R-U1]. `ReceiverMutated` appears only for methods whose receiver is written [T-Sec5][R-U1][T-201]. A package built without the `logos` driver MUST carry no `ParamMutated` records [T-Sec5][R-U1]. An importer MUST NOT read a package that carries the `lang go` record as a LoGOs package: no non-nil pointer check and no overlay applies to it, and its `ParamMutated` facts apply to const-view arguments as Sec 2.1 states [T-Sec5][R-U1]. Assertion effects are emitted for Go and LoGOs packages at every `-O` level and for every GOARCH, and a Go package with assertion effects writes `__.LOGOS` under `go build` and under `logos build` alike [T-Sec5][T-201]. Importers MUST consume assertion effects from `lang go` members [T-Sec5][R-U1]. Any written Go-package member carries one `lang go` record, and a Go package with no fact to record writes no member [T-Sec5][T-Sec12]. When this toolchain compiles a Go package that calls a LoGOs function or method, the compiler MUST insert a `logos/rt.MustNonNil` check for each non-nil pointer parameter of the callee in a direct call, in a `go` statement, and in a `defer` statement, using the callee facts from `__.LOGOS` [T-Sec5][R-U2]. A call through a function value, a method value, or an interface has no such check; this is a documented hole [T-Sec5][R-U2][A-GOBOUNDARY]. A nil receiver that a Go caller passes to a method with a non-nil pointer receiver has no such check either; this is a documented hole [T-Sec5][R-U2][A-GOBOUNDARY]. Code that an upstream toolchain builds from eject output has no such check either; this is a documented hole, like reflection and `unsafe` [T-Sec5][R-U2][A-GOBOUNDARY]. A Go mutation through a const view and a Go store of a non-member into a closed union are documented holes, and the compiler does not check them [T-Sec5][R-U2][A-GOBOUNDARY].
A package built without the `logos` driver MUST carry no `AlwaysNilError` records [T-Sec5][R-U1].
A package built for GOARCH=wasm MUST carry the `MayYield` flag in its function records, under `go build` and under `logos build` alike, and a package built for any other GOARCH MUST NOT carry it [T-Sec5][T-Sec12].
When this toolchain compiles the Go caller, it MUST insert `logos/rt.MustNonNil` and MUST panic with the call position. [T-Sec5]

## 6. Eject and error-handling semantics

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

## 7. Const maps

### 7.1 Const map semantics

A const map with up to 8 keys MUST lower to a `switch` lookup [T-Sec7]. A const map with more keys MUST lower to a static perfect-hash table in read-only data plus a generated lookup function [T-Sec7]. A lookup MUST take constant time: its cost MUST NOT grow with the number of keys of the const map [T-Sec7]. A key that matches a stored key only by hash MUST NOT hit: lookup MUST validate the full key [T-Sec7]. Construction MUST use at most 1,000,000 seed-candidate trials in total per const map [T-Sec7]. Exhaustion MUST fail with `comptime::constmap_seed_not_found` [T-Sec7]. A logarithmic fallback MUST NOT satisfy the contract [T-Sec7]. A lookup of an absent key MUST return none [T-Sec7]. Iteration with `range` MUST visit keys in ascending order: a string by its bytes, an integer or float by numeric value, `false` before `true`, a struct field by field in declaration order, and an array element by element in index order [T-Sec7]. The function `len` on a const map MUST be constant [T-Sec7]. Use of a const map as a `map[K]V` value MUST materialize a regular map once [T-Sec7]. The materialized map is a copy; writes to it do not change the const map's reads [T-Sec7]. A const map key MUST be a string, number, bool, or a struct or array of those; any other key type fails with `comptime::constmap_key_type` [T-Sec7]. A const map passes to a standard-library parameter only when that parameter is read-only: the compiler reads the ParamMutated mask from a fixed table of standard-library functions, because standard-library packages carry no facts, and any other imported function rejects the argument with `correctness::const_view_mutation` [T-201][T-Sec7]. A const map passes to a parameter of a function of the same package only when that function is a const func, which MUST receive arguments with Go semantics and MUST receive a const map materialized as a copy first, so the callee cannot change what the const map reads as; any other function of the package rejects the argument with `correctness::const_view_mutation` [T-201][T-Sec7]. A `-0.0` float key MUST canonicalize, and a `NaN` key MUST fail compilation [T-201][T-Sec7]. The table symbol MUST sit in read-only data, and package init MUST add no heap bytes for an unexported const map [T-Sec7].

A const map key of type complex64 or complex128 MUST be accepted. [T-201][T-Sec7]
A complex key with a NaN real or imaginary component MUST fail compilation with `[correctness::const_eval]`. [T-201][T-Sec7]
Const map range over complex keys MUST visit keys in sorted order by real part, then by imaginary part. [T-201][T-Sec7]
A const map literal MUST take an elided composite key or element type from the map's key or element type, exactly where Go allows eliding it in a map, slice, or array literal. [T-Sec7]

Range over a const view of a runtime map MUST visit each captured key-value entry exactly once, unless the loop exits early, including entries whose keys contain NaN. The compile-time NaN-key rejection above applies to const map construction, not runtime map views. Ordering between different dynamic key types, and for key kinds without an ordering specified above, is unspecified; such keys MUST NOT cause entries to be rejected or omitted. [T-201][T-Sec7]

```lgo
const mimeMap = map[string]string{
    "txt": "text/plain",
    "html": "text/html",
}
v := mimeMap[k] ?? "unknown"
```

## 8. Configuration schema

Earlier drafts named this file bettergo.toml. [T-Sec8]

### 8.1 File location and parsing

The file `logos.toml` MUST sit next to `go.mod` in the main module. [T-Sec8]
Plain `go` MUST read no `logos.toml`. [T-Sec8][R-U1]
The `[lints]` and `[capabilities]` tables MUST apply to modules of both kinds under `logos`; the `[build]` table does so except for the cgo key, which MUST have no effect in a Go main module. [T-Sec8][R-U1]
When the file is absent, the tool MUST use the default for every key. [T-Sec8]
The parser MUST accept tables. [T-Sec8]
The parser MUST accept key and value pairs, basic strings, literal strings, integers, booleans, string arrays, and hash comments. [T-Sec8]
Inputs larger than 1 MiB MUST fail parsing. [T-Sec8]
The parser MUST reject unknown tables and unknown keys. [T-Sec8]
Each error MUST report file, line, and column as logos.toml:<line>:<col>: <message>. [T-Sec8]

### 8.2 Project table

The project table holds strict_files. [T-Sec8]
strict_files is a string array. [T-Sec8]
strict_files lists the LoGOs extensions that the module enables, and its default is all four. [T-Sec8][T-101.1]
A source file whose LoGOs extension is not listed MUST fail with `[correctness::extension_not_enabled] <file>: extension <ext> is not enabled by logos.toml strict_files`. [T-Sec8][T-101.1]
Each entry MUST name one of the four source extensions: .lgo, U+1F4DC, U+26A1, or the gopher extension U+0295 U+25D4 U+03D6 U+25D4 U+0294. [T-Sec8][T-101.1]

### 8.3 Build table

cgo is a string with default "forbid" for a LoGOs main module. [T-Sec8][T-101.2]
It MUST have no effect in a Go main module. [T-101.2][R-U1]
It MUST be one of "forbid", "restricted", or "allow". [T-101.2]
With "forbid", a package that imports C fails with restriction::cgo_usage. [T-101.2]
With "restricted", logos enables cgo. [T-101.2]
With "restricted", each package that imports C MUST match cgo_allowlist. [T-101.2]
A package outside the list fails with restriction::cgo_usage. [T-101.2]
cgo_allowlist is a string array of path globs with default empty. [T-Sec8][T-101.2]
A double star matches any suffix. [T-Sec8][T-101.2]
An example entry is "example.com/c-sqlite/**". [A-EXAMPLES][T-101.2]
opt_level is a string with default "O2". [T-Sec8][T-105]
See Sec 1.5 for accepted `opt_level` values.
A `-O=<level>` flag on `logos build` MUST override the key. [T-105]
target_cpu is a string with default "deterministic". [T-Sec8][T-105]
It MUST be one of "deterministic", "baseline", or "native". [T-Sec8][T-105]
With "deterministic", logos sets GOAMD64 to v3 and GOARM64 to v8.2 unless the environment already sets them. [T-105]
With "native", logos uses the host level. [T-105]
See Sec 1.5 for the default binary CPU requirement.
arena is a boolean with default true. [T-Sec8][T-301.1]
A true value sets `GOEXPERIMENT=logosarena`, which enables implicit placement and the placement regions for Go packages and LoGOs packages under `logos`. [T-Sec8][T-301.1][R-U1]

### 8.4 Capabilities table

The capabilities table holds four severity keys and four allowlist keys. [T-Sec8]
transitive_reflect is a string with default "warn". [T-Sec8]
transitive_exec and transitive_net are strings with default "deny". [T-Sec8]
transitive_unsafe is a string with default "allow". An omitted file, table, or key MUST permit dependencies that import unsafe or hold assembly files or precompiled objects. [T-Sec8][T-Sec11]
transitive_unsafe and transitive_unsafe_allowlist also govern a package that holds assembly files or a precompiled object, see Sec 11.1. [T-Sec8][T-Sec11]
Each severity key MUST be one of "deny", "warn", or "allow". [T-Sec8]
reflect_allowlist, transitive_exec_allowlist, transitive_net_allowlist, and transitive_unsafe_allowlist are string arrays of path globs with default empty. [T-Sec8]
An entry MAY name a Go module or a LoGOs module. [T-Sec8][R-U2]
Example entries use module paths such as "example.com/proto-runtime/**". [A-EXAMPLES][T-Sec8]
logos enforces the policy for every module in the build. [T-Sec8][T-Sec11]

### 8.5 Lint tables

The lints table holds one severity key per group: correctness, concurrency, slop, complexity, perf, style, suspicious, portability, and pedantic. [T-Sec8][T-102]
Each value MUST be one of "forbid", "deny", "warn", or "allow". [T-Sec8][T-102]
correctness and concurrency default to forbid. [T-Sec8][T-102]
slop and complexity default to deny. [T-Sec8][T-102]
perf, style, suspicious, and portability default to warn. [T-Sec8][T-102]
pedantic defaults to allow. [T-Sec8][T-102]
forbid cannot be lowered and ignores allow directives in a LoGOs package. [T-102]
deny fails the build unless a logos allow directive for the rule precedes the statement or declaration. [T-102]
warn prints a diagnostic. [T-102]
allow stays silent. [T-102]
`--pedantic` and `--pedantic=true` override the pedantic key for logos build and logos lint, but they MUST NOT lower a stricter configured severity. [T-Sec8]
The lints.thresholds table of `logos.toml` holds six integer keys. [T-Sec8]
Defaults MUST be 3 for max_nesting_depth, 15 for max_cognitive_complexity, 60 for max_function_lines, 4 for max_function_params, 4096 for max_comptime_ast_nodes, and 100000 for max_ctfe_steps. [T-Sec8][T-102]
Third-party modules of either kind get no code lints. [T-102][R-U1]

## 9. End-to-end example

The normative example is test/logos/example/order_service.lgo. [T-Sec9]
It runs ProcessBatch against a fake database and ships an expected-output file. [T-Sec9]
The file uses only valid syntax, including leading-dot chains, block lambdas, and pointer assertion targets. [T-Sec9]
logos eject reproduces the same output from the ejected module. [T-Sec9]
The build steps are in IMPLEMENTATION.md, milestone M22. [T-Sec9]

## 10. Static serializer

### 10.1 Static marshaling

For each .lgo struct type with json tags, defaults, or non-nil pointer fields and no embedded fields, the lowering MUST generate a value-receiver MarshalJSON method. [T-Sec10]
Field kinds without static support MUST skip generation for that type. [T-Sec10]
These kinds are interfaces, maps with non-string keys, and embedded structs. [T-Sec10]
A struct with a field whose JSON tag name `encoding/json` v1 rejects as invalid, such as a name that holds U+2028 or U+2029, MUST skip generation for that type and take the reflective path under both settings of `goexperiment.jsonv2`. [T-Sec10]
Generated output MUST be byte-identical to encoding/json for the same value under both `goexperiment.jsonv2` and `!goexperiment.jsonv2`; `json.Marshal` on an eligible value MUST return the same bytes as the generated method. [T-Sec10]
The default build MUST use the fork baseline with `goexperiment.jsonv2` on. [T-Sec10]
The generator MUST NOT rewrite any call site. [T-Sec10]
The generator emits fixed key literals first, computes a size bound, then fills one buffer from a single make. [T-Sec10]
The buffer is returned to the caller, so it escapes to the heap. [T-Sec10]
No size threshold and no caller buffer apply. [T-Sec10]
The fast path MUST NOT call reflect. [T-Sec10]
Only unmodeled fields fall back to reflective json.Marshal per field. [T-Sec10]
Concurrent mutation during marshal is a data race under Go's memory model; the generated body adds no synchronization. [T-Sec10]
Escaping and float formatting MUST follow encoding/json under the active `goexperiment.jsonv2` setting with portable Go and no SIMD requirement. [T-Sec10]
Supported tag options are name, omitempty, omitzero, string, and -. [T-Sec10]
HTML characters, U+2028, U+2029, invalid UTF-8, and float exponents encode per encoding/json. [T-Sec10]
NaN and infinities report UnsupportedValueError. [T-Sec10]
A closed-union value marshals to its dynamic member static bytes. [T-Sec10][T-203.1]
A comptime generic that marshals over comptime fields expands once per instantiation. [T-205]
Each instantiation carries its own static helpers. [T-205]

### 10.2 Static unmarshaling

For each eligible struct, the lowering generates pointer-receiver UnmarshalJSON. [T-Sec10]
For a scalar field with the `,string` option, generated method dispatch MUST follow the active `encoding/json` profile: classic mode unwraps the outer string, while JSONv2 passes the original JSON token and gives `UnmarshalJSONFrom` precedence over `UnmarshalJSON` and `UnmarshalText`. [T-Sec10]
The compiler MUST select methods by their complete signatures. If a streaming method declines a value without consuming it, the generated decoder MUST call that method only once and then select the remaining method or scalar decoder at compile time; it MUST NOT construct a methodless type at run time. Non-scalar fields MUST use their ordinary fallback because `,string` does not apply to them. [T-Sec10]
Generated method helper type arguments MUST retain source-valid C selectors when a checked scalar receiver is instantiated with a C type; private cgo-generated type names MUST NOT appear in ejected Go. [T-Sec10]
These paths MUST preserve the active profile's error types, offsets, field context, partial updates, and updates through existing pointer aliases. JSONv2 fallback type errors name the root struct; classic fallback errors retain the inner struct context. A quoted null clears a scalar pointer according to that profile, including escaped spellings in classic mode and only the exact token `"null"` in JSONv2. [T-Sec10]
Generated quoted-text helper signatures MUST encode the destination pointer's element type without runtime kind inspection. A `reflect.Type` value MAY remain where a public JSON error API requires it. [T-Sec10]
Quoted text methods MUST receive the active profile's Unicode replacement behavior for malformed UTF-8 and unpaired UTF-16 surrogates. Appending to a callback's input MUST NOT alter the scanner token or change the subsequent quoted-null decision. [T-Sec10]
Decoding MUST fill defaults first, then parse. [T-Sec10]
A _logos_seen mask sized [(K+63)/64]uint64 tracks matched keys against a required-mask constant, where K is the number of required non-nil fields of the struct. [T-Sec10]
At the closing brace, a missing required non-nil field returns MissingFieldError naming type and field. [T-Sec10]
JSON null for a non-nil field returns the same error. [T-Sec10]
Keys MUST dispatch by switch up to 8 keys and by the Sec 7 const-map table above that. [T-Sec10][T-Sec7]
Keys whose 64-bit hashes are equal MUST each dispatch to their own field. [T-Sec10][T-Sec7]
The cost of decoding one key MUST NOT grow with the number of keys the type declares: dispatch MUST NOT scan the key set. [T-Sec10][T-Sec7]
A key that matches a declared key only by hash or prefix MUST NOT be accepted: dispatch MUST validate the full key. [T-Sec10][T-Sec7]
Unknown fields are ignored. [T-Sec10]
Duplicate keys MUST keep the last value. [T-Sec10]
Arrays MUST grow incrementally with no pre-allocation cap. [T-Sec10]
Decoding through json.Decoder uses the same method. [T-Sec10]
Decoding through a reflection-based decoder other than `encoding/json` lies outside this contract; this is a documented hole. [T-Sec10][A-GOBOUNDARY]
No validate tags are needed. [T-Sec10]
Unmarshal of a closed union MUST resolve members declared in another package, whatever the import name of that package in the file and whether or not the file imports it. [T-Sec10][T-203.1]
A member has a static key set when its type is a struct, or a pointer to a struct, that Sec 10.1 makes eligible. The set holds the JSON key of each visible field, and a field is visible when it is exported and has no `-` tag. A keyed member has a static key set with at least one key. A keyless member has none: its type is not an eligible struct, or the struct has no visible field. A keyed member trial MUST count a success only when at least one key in the object is in that member's static key set and its decode then succeeds. A key in the static key set of a sibling member but not in this member's disqualifies this member trial, while a key in no member's static key set is ignored. Keys MUST decide every keyed member trial, including a member that declares its own UnmarshalJSON. Such a decoder MUST run only after its keys match. A keyless member MUST be tried only when no key of a keyed member occurs in the input. Exactly one such member MUST succeed for the union to select it. A closed union member without an UnmarshalJSON method MUST fail with [correctness::union_member_no_json]. [T-Sec10][T-203.1]
A closed union member that its own package declares unexported, when that package is not the decoding package, MUST fail a field of that union with [correctness::union_member_no_json] at the field, because the decoding package cannot name the member or reach its UnmarshalJSON method. [T-Sec10][T-203.1]
It assigns the target only on a single success. [T-Sec10][T-203.1]
Ambiguity returns an error naming the union and the members. [T-Sec10][T-203.1]
Failed trials leave the target untouched. [T-Sec10][T-203.1]
A decode MAY reuse a trial outcome for equal bytes of the same union only inside one root value, the outermost union value whose trial is in flight, and MUST drop every outcome when that trial returns. [T-Sec10][T-203.1]
Decodes of separate buffers MUST NOT share trial outcomes; two goroutines that decode the same byte slice at the same time MAY share them. [T-Sec10][T-203.1]
A member UnmarshalJSON MAY therefore run fewer times than its bytes occur inside one root value. [T-Sec10][T-203.1]
Only top-level keys of the object MUST count as occurring. [T-Sec10][T-203.1]
Two keyed members sharing a key MUST both claim input with that key. [T-Sec10][T-203.1]
Non-object input MUST match only keyless members. [T-Sec10][T-203.1]
JSON null for a union field MUST leave the field nil with no trial. [T-Sec10][T-203.1]
Key matching MUST fold case as the default v1 encoding/json does. [T-Sec10][T-203.1]
UnionTrialError MUST name the union, the members in order, and the counts. [T-Sec10][T-203.1]

### 10.3 Binary encoding

For each JSON-eligible struct whose every JSON-visible field has a binary representation, the lowering generates MarshalBinary, UnmarshalBinary, and AppendBinary. [T-Sec10.1]
They implement encoding.BinaryMarshaler, encoding.BinaryUnmarshaler, and encoding.BinaryAppender. [T-Sec10.1]
Output MUST start with version byte 0x01, then fields in declaration order. [T-Sec10.1]
All integer kinds encode as 8 bytes little-endian of the 64-bit value. [T-Sec10.1]
float32 encodes as 4 bytes little-endian of Float32bits. [T-Sec10.1]
float64 encodes as 8 bytes little-endian of Float64bits. [T-Sec10.1]
bool MUST encode as 0x00 or 0x01. [T-Sec10.1]
string and a slice whose element type has underlying type uint8 encode as uvarint length plus raw bytes. [T-Sec10.1]
Slice and array encode as uvarint count plus elements. [T-Sec10.1]
Maps with string keys encode as uvarint count plus entries sorted by key. [T-Sec10.1]
Nested structs encode inline. [T-Sec10.1]
Optional values MUST encode as 0x00 when absent or 0x01 plus payload when present. [T-Sec10.1]
A field of type int or uint in a struct with a JSON-tagged field warns portability::serialized_unsized_int in LoGOs packages. [T-Sec10.1]
A tag `-` excludes the field from binary output; other tag options change nothing. [T-Sec10.1]
UnmarshalBinary rejects truncation, bad presence bytes, version mismatch, and trailing bytes with errors naming field and offset. [T-Sec10.1]
UnmarshalBinary MUST reject an integer that its field type cannot hold, with an error naming field and offset. [T-Sec10.1]
UnmarshalBinary MUST reject a missing value for a non-nil `*T` field of a LoGOs struct, with an error naming field and offset. [T-Sec10.1]
For elements with a nonzero wire payload, a count header MUST pre-allocate at most min(len(rest)/2, 1 MiB / sizeof(elem)) elements and MUST grow past that only as input bytes arrive. [T-Sec10.1]
Charge nothing against a total budget on the count header; a count whose element bytes are all present in the input is lawful and MUST decode, even when the total exceeds 1 MiB. [T-Sec10.1]
For slices whose elements have zero wire payload, the count itself MUST NOT exceed 1,048,576 (the 1 MiB prealloc bound at unit 1), checked first, and MUST NOT exceed 8 MiB / sizeof(elem), checked next; larger counts MUST be rejected with a count-out-of-range error, encoder symmetric, checked before allocation or element iteration, and a rejected count MUST leave no count header in the encoder output. One zero-wire count MUST NOT allocate more than 8 MiB of element storage in a single allocation. [T-Sec10.1]
Zero-wire slices MUST be handled in constant time with respect to count, without per-element iteration. [T-Sec10.1]
These resource bounds MUST NOT change the wire bytes of accepted values. [T-Sec10.1]
Binary decoding and encoding MUST reject a value nested deeper than 10,000 composite levels, counting the outermost struct as level 1. Entering a struct, slice, array, or map counts as one level, an empty one included; entering a tuple counts as entering a struct; entering a pointer or option payload adds none. The decoder error MUST name the field and the byte offset. [T-Sec10.1]

## 11. Capabilities auditor

### 11.1 Sink symbols

After deadcode, the linker MUST run a reachability audit from main.main and init functions. [T-Sec11]
The audit MUST run only in `logos` builds and MUST cover modules of both kinds. [T-Sec11][R-U1]
The audit MUST follow relocations. [T-Sec11]
It MUST add synthetic edges for interface calls and reflective method lookups. [T-Sec11]
Protected sinks are reflect.ValueOf, reflect.TypeOf, reflect.New, syscall.Exec, syscall.ForkExec, syscall.StartProcess, os.StartProcess, os.startProcess, net.sysSocket, syscall.Socket, syscall.Connect, runtime.cgocall, and runtime.logosCgocallFast. [T-Sec11]
The raw syscall entry points are protected sinks too. Every function of package syscall, on every GOOS and GOARCH, whose parameters are all uintptr and whose first two results are uintptr is one, when it is exported or carries a linkname push and the linker does not block it for dependencies, and so are the Call methods of syscall.Proc and syscall.LazyProc. These include syscall.Syscall, Syscall6, Syscall9, Syscall12, Syscall15, Syscall18, RawSyscall, RawSyscall6, SyscallN, AllThreadsSyscall, and AllThreadsSyscall6, and the libc trampolines that darwin, openbsd, solaris, and aix dependencies reach by linkname, such as syscall.syscall, syscall6, syscall10, rawSyscall6X, and sysvicall6. [T-Sec11]
Beyond package syscall the table covers golang.org/x/sys/unix.Syscall, Syscall6, Syscall9, RawSyscall, RawSyscall6, SyscallNoError, and RawSyscallNoError, the Call methods of golang.org/x/sys/windows.Proc and golang.org/x/sys/windows.LazyProc, and golang.org/x/sys/plan9.Syscall, Syscall6, RawSyscall, and RawSyscall6, and no other package: a raw entry point of a package outside that set is not a protected sink. [T-Sec11]
A typed wrapper of such a package that fixes its own procedure or trap number, such as windows.GetCurrentProcessId, is no entry point: the report names the package that implements the wrapper, not its callers, so the grant names that package. [T-Sec11]
The linker rejects a linkname reference from outside the standard library to the runtime's own assembly entry internal/runtime/syscall/linux.Syscall6 and to the darwin trampolines syscall.syscalln and syscall.rawsyscalln, so no dependency reaches them and they are no sinks. [T-Sec11]
A raw syscall can open a socket or start a process, so each raw entry point counts as both capabilities: a dependency that calls one is reported under both capabilities::unauthorized_exec and capabilities::unauthorized_net, and only a grant in both transitive_exec_allowlist and transitive_net_allowlist, or the severity allow on both keys, silences it. [T-Sec11]
A raw entry point blames only the dependency frame that calls it directly: a chain that reaches it through a standard-library wrapper such as os.ReadFile or syscall.Kill blames nobody, because the other sinks classify the standard library's own system calls, and the golang.org/x/sys/unix entry points blame their callers, not the package that defines them. [T-Sec11]
Taking the address of a raw entry point is a direct use: a function value of the entry, whether a local, a package variable, an argument, or a method value of a Call method, blames the nearest dependency frame that reaches the value, and only a standard-library frame between that frame and the entry stops the blame. [T-Sec11]
A direct reference to runtime.cgocall in a dependency, such as one made by //go:linkname, is reported with capabilities::unauthorized_cgo like a cgo wrapper, and the cgo policy authorizes it only for a package that uses cgo; a chain that reaches runtime.cgocall through a standard-library frame blames nobody. [T-Sec11]
When transitive_unsafe is explicitly "deny" or "warn", packages that import unsafe or hold assembly files are found by walking the cmd/go load graph and reading every file of each non-exempt package in it, including the files that the current configuration excludes. The default "allow" skips this audit. [T-Sec11]
A file excluded by a //go:build line, a custom tag, or a GOOS or GOARCH file-name suffix still counts: its imports are imports of the package, and an excluded assembly file marks the package. [T-Sec11]
Every build tag counts as both set and unset except ignore, and a file with a malformed constraint is excluded, which is the file set that go mod tidy reads. [T-Sec11]
The bound is one read of the import header of each excluded Go file of each non-exempt package in the current load graph. The audit does not follow an excluded file's import into a package that the current load graph lacks, and the sink reachability audit stays per linked configuration, so a program for another GOOS or GOARCH is audited by linking it for that target. [T-Sec11]
A package outside the exempt set that holds assembly files (`.s`, `.S`, or `.sx`) or a precompiled object (`.syso`) is an unclassified capability, because the audit cannot see a raw SYSCALL instruction inside it, and a constructor in a `.syso` object runs at process start under external linking. [T-Sec11]
cmd/go reports it with capabilities::unclassified_assembly and the shortest import chain, under the transitive_unsafe severity, and an entry in transitive_unsafe_allowlist authorizes it, since a grant of unsafe already names a package trusted to bypass the language's checks. [T-Sec11]
The standard library, first-party packages under a main module, and the program's own packages are exempt. [T-Sec11]
An external test package MUST retain the module ownership of its tested package. A `_test` suffix alone MUST NOT establish test provenance. [T-Sec11]
A dependency MUST NOT count as the standard library merely because its module path has no dot; only packages of the toolchain's own GOROOT are the standard library. [T-Sec11]
A dependency that imports a sink package but never calls it passes as dead code. [T-Sec11]
Reflection through reflect.Value.Method counts as reachable. [T-Sec11]
On diamond graphs the report MUST keep the shortest chain. [T-Sec11]
The audit MUST terminate on cyclic reachability graphs and report only chains from its entry or initialization roots; edge discovery order MUST NOT change the minimum chain length. [T-Sec11]
The prefix from an audit root to a blamed frame MUST remain a shortest rooted path when a relocation names an inlined owner before that owner's incoming edges have been discovered. Recorded call edges may contain cycles; discovery order MUST NOT create a cycle in the prefix's parent relation or manufacture a root for an unanchored predecessor. [T-Sec11]
The audit reports one chain per sink and dependency, so a chain through the standard library alone never hides another dependency's chain to the same sink. [T-Sec11]
A chain blames the dependency frame nearest the sink, and chains of standard-library or first-party frames blame nobody. [T-Sec11]
The call belongs to the package of the innermost inlined function that contains it, so a dependency function inlined into first-party or standard-library code is still the dependency's frame, and the interface calls and reflective lookups of an inlined function body count as the function's own. [T-Sec11]
Policy comes from logos.toml capabilities. [T-Sec8][T-Sec11]

### 11.2 Reachability report

A violation at severity `deny` aborts linking with ERROR capabilities::unauthorized_exec naming the dependency and the sink. A `warn` violation prints a warning. An `allow` violation stays silent. [T-Sec11]
The report MUST list the shortest call chain with file and line per frame. [T-Sec11]
For the allowlist keys that authorize each sink, see Sec 8.4. [T-Sec8][T-Sec11]
Unsafe violations use capabilities::unauthorized_unsafe with the shortest import chain. [T-Sec11]
cgo reachability reports capabilities::unauthorized_cgo unless the cgo policy allows the package. [T-Sec11][T-101.2]
Only a dependency that owns the cgo wrapper is blamed: the standard library's own cgo never blames the dependencies that call it. [T-Sec11]
The cgo sink is the per-call-site `pkg._Cfunc_*` or `pkg._C2func_*` wrapper of the package that owns it, reported as `runtime.cgocall`, because `runtime.cgocall` itself lies in the runtime and has no owning dependency to blame. [T-Sec11]

## 12. Wasm profile

### 12.1 Selective resume points

The WASM profile MUST activate automatically for GOOS=js GOARCH=wasm and GOOS=wasip1 GOARCH=wasm under `logos`. [T-Sec12]
It MUST apply to Go packages and to LoGOs packages; under plain `go` it needs the explicit `--wasm-features` flag. [T-Sec12][R-U1]
It uses the installed toolchain with no external tool. [T-Sec12]
A MayYield=0 function with straight-line code compiles without resume-point dispatch; one with inter-block jumps keeps the entry-loop dispatch that addresses those jumps. [T-Sec12]
A function keeps MayYield=0 only with no defer, no indirect call, no call to MayYield=1 or unknown callees, no transitive runtime.gopark, no channel, select, time.Sleep, Gosched, or blocking wasmimport call, no nil check, and no prologue stack check, because stack growth and preemption unwind through runtime.morestack. [T-Sec12]
Pure imports keep MayYield=0. [T-Sec12]
Unknown wasmimport targets default to MayYield=1. [T-Sec12]
A wrong MayYield=0 MUST trap instead of hanging. [T-Sec12]
A function with a call that keeps a resume point keeps resume-point dispatch. Without such a call it keeps dispatch only for inter-block jumps. MayYield=1 alone, through a prologue stack check or a nil check, needs none. Under --wasm-features=stack-switching, continuation calls replace dispatch, kept only for deferreturn, runtime and mcall-caller frames besides the entry-loop dispatch. [T-Sec12]
A caller of runtime.mcall MUST keep resume-point dispatch even when it is not in the runtime: the goroutine switch back re-enters that frame, which inlining (Gosched) can place in user code. [T-Sec12]
With `--wasm-features=stack-switching` on `logos build` or `go build` they MUST use stack switching instead. [T-Sec12][R-U1]
Pointers are 8 bytes and slice headers are 24 bytes. [T-Sec12]
alignof of int64, uint64, and float64 MUST equal 8, so 64-bit atomics stay naturally aligned with no forcing code. [T-Sec12]
Deadcode removes unreachable functions, but type and method tables remain so reflect and encoding/json keep resolving. [T-Sec12]

### 12.2 Host interface and loader

`logos build` MUST emit <name>.loader.mjs beside the wasm file when the output is a `.wasm` file, for a Go main package and for a LoGOs main package. [T-Sec12][R-U1]
Plain `go` MUST emit the same loader by default for such an output, and `--wasm-loader=false` MUST turn it off. The flag MUST stay absent from the `go help build` text, as Sec 0.1 states for flags that plain `go` gains. [T-Sec12][R-U1]
The loader MUST have identical bytes under both commands. [T-Sec12][R-U1]
Both commands MUST add the implicit `logos/rt` import for the loader helpers when they emit the loader for a `-o` output that names a `.wasm` file, unless packages of the main module or the package `github.com/gosuda/loGos/rt` already provide the helpers. [T-Sec12][R-U1]
A module graph that provides some but not all of the four loader helpers MUST fail the build with an error that names each missing helper export. [T-Sec12]
A package of the module graph provides a helper when it declares `//go:wasmexport` with exactly that helper name, however the directive is spaced; a longer name that starts with the helper name does not provide it. [T-Sec12]
Only a package of the main module or the package `github.com/gosuda/loGos/rt` MAY provide a loader helper, and a helper export in any other package of the module graph MUST fail the build with an error that names the helper and the package. [T-Sec12]
A build with no `-o` writes no loader, because the default output name of a wasm main package carries no `.wasm` suffix. [T-Sec12]
The loader MUST work under `GOOS=js` and, in a reactor module that the host initializes through WASI, under `GOOS=wasip1`. [T-Sec12]
`logos/rt` MUST export the wasm helpers `_logos_scratch_alloc`, `_logos_scratch_reset`, `_logos_errmsg_ptr`, and `_logos_errmsg_len` for the loader. [T-Sec12]
The loader exports instantiateLogos(wasmBytes, imports). [T-Sec12]
The loader MUST re-read memory.buffer after every call into wasm. [T-Sec12]
Calls after growth MUST see current memory. [T-Sec12]
The loader MUST accept an errCode of int32 or int64 width; the compiler accepts int32 and int64 results. The host wrapper throws on any nonzero (truthy) return. [T-Sec12]
String results are invalid. [T-Sec12]
On nonzero errCode the wrapper throws the message read from wasm memory. [T-Sec12]
`logos/rt.WasmError(err)` MUST return 0 for a nil error, and for a non-nil error it MUST record `err.Error()` as the last error message and return 1; the loader reads that message through `_logos_errmsg_ptr` and `_logos_errmsg_len`. [T-Sec12]
//logos:wasmexport f MUST behave as //go:wasmexport f. [T-Sec12]
//logos:wasmimport m f MUST behave as //go:wasmimport m f. [T-Sec12]

## 13. Backend peephole and SIMD passes

- Every pass in this section MUST apply to Go packages under `logos` and under plain `go`, and to LoGOs packages under `logos`, except a rule about a LoGOs language construct (array operators, reduction methods, and `make` length checks) and except the wasm loop-stride default, which applies under `logos` only. [T-303][R-U1]
- For a vector form selected by a runtime CPU feature check, every instruction of that form, including vector zero initialization, mask operations, and scalar-result operations with SIMD operands, MUST remain unreachable when the check is false. This rule MUST survive inlining and optimization for Go and LoGOs input under both commands. Zero-argument CSE MUST leave SIMD zeros in their original control-flow region. Branch elimination MUST NOT speculate operations with SIMD results or operands. Ordinary dominance-aware CSE MAY share equivalent values within a guarded region. [T-303][T-204.2][R-U1]

### 13.1 Arm64 load and store pairing

- Adjacent loads and stores that meet the safety predicates MAY fuse into one paired instruction. [T-303]
- Fusion MUST NOT change program meaning, fault behavior, or atomic order. [T-303]
- Candidates split by a call or an atomic MUST stay separate; loads MAY fuse across a block edge only when the source block has one successor and the target block has one predecessor. [T-303]
- A pair MUST NOT write back its base register. [T-303]
- The two destinations of a pair MUST differ. [T-303]
- A 16-byte pair offset outside the encodable range MUST stay separate loads or stores; 4-byte and 8-byte pairs MAY use the assembler fixup. [T-303]
- The 4-byte and 8-byte pairing above is upstream behavior. [T-303]
- Adjacent 16-byte vector loads MUST pair into FLDPQ when they meet the safety predicates with an encodable pair offset. [T-303]
- Adjacent 16-byte vector stores MUST pair into FSTPQ under the same conditions. [T-303]
- A 16-byte pair offset outside -1024..1008 or not a multiple of 16 MUST stay separate loads or stores. [T-303]
- With the default target_cpu the ARM64 target MUST use GOARM64=v8.2, so LSE atomics stay available. [T-303][T-105]

### 13.2 Amd64 masked tails, rotates, and checked lengths

- Masked tails MUST NOT read past the end of the slice. [T-303]
- In a native build with `GOEXPERIMENT=simd` on amd64 or arm64, `.sum()` on a bare slice of a fixed-width numeric element kind or an alias of one MUST load full vectors for the body and load the tail once through a masked load that enables only the remaining elements. [T-303]
- On amd64 that masked load MUST be VPMASKMOVD or VPMASKMOVQ for 32 and 64-bit elements, which need AVX2, and the zeroing K-masked VMOVDQU8 or VMOVDQU16 for 8 and 16-bit elements, which need AVX512. [T-303]
- The amd64 vector form MUST run only when the archsimd CPU feature check its instructions need is true and MUST take the scalar loop when it is false, and the check MUST be omitted when the build's `GOAMD64` level guarantees the feature (`v3` for AVX2, `v4` for AVX512). [T-303]
- On arm64, which has no masked load, the tail MUST load only the elements present. [T-303]
- An integer slice sum MUST accumulate in vector lanes. [T-303]
- A float slice sum MUST run the fixed pairwise order of Sec 1.4 over the loaded elements, so its bits equal the scalar form with and without `GOEXPERIMENT=simd`. [T-303][T-104.6]
- A slice sum with a fused operation, or over an `int`, `uint`, or `uintptr` element, MUST take the scalar form. [T-303]
- Eject output of slice reductions MUST use the scalar form, so it builds with an upstream toolchain. [T-303][T-0.3]
- At GOAMD64=v3/v4 an immediate 32/64-bit constant-count rotate MUST use RORXQ or RORXL besides SARX, SHLX, and SHRX. [T-303]
- Variable-count rotates MUST stay on the existing path. [T-303]
- A ternary-shaped branch at GOAMD64=v3 MAY compile to a conditional move. [T-303]
- In `.lgo` files a non-constant integer product in a `make` length or capacity MUST lower to `logos/rt.MulLen`; constant products stay constant. [T-303]
- An overflowing `make` length MUST panic with `logos: make length overflow: <expr>`. [T-303]
- A negative `make` length MUST panic. [T-303]
- The call below panics with `logos: make length overflow: w * h` on overflow. [T-303]

```lgo
buf := make([]Pixel, w * h)
```

### 13.3 Long-loop preemption on wasm

The long-loop preemption rules are stated in Sec 3.3 (RFC-303: Hardware backend optimizations).

### 13.4 Automatic loop vectorization

Automatic vectorization MUST apply to ordinary Go and LoGOs loops through the shared compiler backend, without a source annotation or an explicit SIMD import. [T-303.AV][R-U1]
The initial implementation MUST require `GOEXPERIMENT=simd`, enabled compiler optimization, and either amd64 with `GOAMD64=v3` or `v4`, or arm64. [T-303.AV]
Other targets and builds with SIMD disabled, `-N`, race detection, MSan, or ASan MUST keep the scalar loop. [T-303.AV]
Functions marked `//go:nosplit` MUST keep the scalar loop, because a fault-policy query requires a valid goroutine and an ordinary call budget. [T-303.AV]
The SSA control `-d=ssa/autovec/off` MUST disable the pass; `-d=ssa/autovec/debug=1` MUST report each accepted loop. [T-303.AV]

The compiler MAY vectorize a loop after it proves the following conditions. [T-303.AV]

- The loop has a signed index with an invariant initial value, a unit increment, and an invariant exclusive upper bound.
- The body is straight-line code with exactly one integer element store. Its only loop-carried values are the index and memory state.
- Every element access uses the same index and element width. All remaining scalar operands and pointer bases are available before the loop.
- Bounds and nil checks have already been proved unnecessary. The loop contains no call, atomic operation, allocation, write barrier, or other observable effect.
- The stored expression consists of integer element loads and invariant operands, optionally combined with addition, subtraction, negation, bitwise complement, AND, OR, or XOR on 8-, 16-, 32-, or 64-bit lanes, or multiplication on 16- or 32-bit lanes.
- A remaining shift operation uses a constant count smaller than the lane width. Left shifts and unsigned right shifts may use 16-, 32-, or 64-bit lanes; signed right shifts may use 16- or 32-bit lanes. An 8-bit left shift by one may use vector addition. Other remaining shifts keep the loop scalar.

For an accepted loop, the compiler MUST emit a 128-bit vector prefix and retain the original scalar loop for the remainder. [T-303.AV]
The vector prefix MUST begin at the original initial index. If the initial index is not less than the bound, the prefix MUST have zero iterations. If a guard fails, the retained scalar loop MUST begin at the original initial index. [T-303.AV]
A vector iteration MUST access only complete groups within the original iteration range; it MUST NOT read or write a padded tail. [T-303.AV]
Integer overflow and truncation MUST match the scalar operations. Defined integer types and aliases MUST retain their Go meaning. [T-303.AV]

A source and destination with identical base addresses MAY share a vector iteration. [T-303.AV]
For other addresses, the vector prefix MUST run only when its destination range is disjoint from every source range; a failed overlap guard MUST run the entire original scalar loop. [T-303.AV]
The guard MUST NOT overflow pointer-end or byte-count arithmetic. Source ranges MAY overlap each other. [T-303.AV]
The transformation MUST preserve nil and empty behavior, the final index, memory effects, and every panic and write that precedes it. [T-303.AV]
When the current goroutine enables `runtime/debug.SetPanicOnFault`, the original scalar loop MUST execute so widened memory accesses cannot suppress earlier scalar writes. The fault-policy query MUST run only on the loop-entry edge, and its memory state MUST precede the vector setup and scalar fallback. [T-303.AV]
Loops with unresolved checks, unsupported expressions, reductions, or additional loop-carried state MUST remain scalar. Floating-point arithmetic MUST remain scalar in this initial implementation. [T-303.AV]
The new vector loop MUST receive the existing loop-preemption checks when that experiment is enabled. [T-303.AV]
Eject MUST keep ordinary scalar source, so its output needs no SIMD package or experiment to compile with upstream Go. [T-303.AV][T-0.3]

## 14. JSON diagnostics schema

### 14.1 Emission

- `--json-diagnostics` on `logos lint`, on `logos build`, and on every other command that accepts it MUST write one JSON object per line to stderr for every compiler and lint diagnostic in `.lgo` files and in Go files. [T-Sec14][T-102][R-U1]
- Each line MUST be a complete object with schema version `1.0`. [T-Sec14]

### 14.2 Top-level fields

- Field `schema_version` is a string and its value is `1.0`. [T-Sec14]
- Field `severity` is a string with one of `forbid`, `deny`, or `warn`. [T-Sec14]
- Level `allow` emits no diagnostic. [T-Sec14]
- Field `rule_id` is a string of the form `group::name` for a lint diagnostic, and the empty string for a compiler diagnostic that carries no rule ID, including each Go compiler diagnostic. [T-Sec14][R-U1]
- Field `file` is a string with the path of the source file. [T-Sec14]
- Field `message` is a string with the human-readable diagnostic text. [T-Sec14]
- Field `span` is an object with the line, column, and byte range. [T-Sec14]
- Field `provenance` is an object with the source of the inferred type. [T-Sec14]
- Field `autofix_candidates` is an array of fix objects. [T-Sec14]

### 14.3 Span object

- Object `span` holds six integers called `start_line`, `start_col`, `end_line`, `end_col`, `byte_offset`, and `byte_len`. [T-Sec14]

### 14.4 Provenance object

- Object `provenance` holds `origin_expr` as a string, `origin_line` as an integer, and `inferred_type` as a string. [T-Sec14]

### 14.5 Autofix objects

- Each candidate holds `description` as a string and `replacement` as a string. [T-Sec14]
- Candidates that insert text also hold `insert_before_line` as an integer. [T-Sec14]

## 15. Scope defer and Locked

### 15.1 Scope defer

For the `scope` token rule, see Sec 4.1. [T-0.2]
`scope defer f()` MUST run `f` at block exit. [T-104.7]
Scope defers in one block MUST run last-in-first-out. [T-104.7]
The function, receiver, and arguments MUST evaluate at the statement. [T-104.7]
An indexed function value MUST be captured before control continues past the `scope defer` statement. [T-104.7]
A generic method MUST capture its receiver and bind its explicit type arguments there. Every exit MUST use those same type bindings, including an exit from a nested scope that shadows a type name. Remaining method type arguments MUST retain call-argument inference. [T-104.7]
`return`, `break`, `continue`, labeled `break`, and `goto` out of the block MUST run scope defers first. [T-104.7]
A panic MUST run scope defers during unwinding. [T-104.7]
A scope defer in a loop body MUST run every iteration. [T-104.7]
Cleanup on panic MUST use open-coded defers. [T-206][T-104.7]

```lgo
{
    mu.Lock()
    scope defer mu.Unlock()
    work()
}
```

### 15.2 Locked WithLock

For `WithLock` locking and unlock-after-panic, see Sec 2.6. [T-206]

This example imports `logos/sync` with the package name `sync`.

```lgo
var m sync.Locked[Config]

m.WithLock(func(c *Config) {
    c.Port = 8080
})
```

## 16. IDE protocol

### 16.1 Proxy model and position mapping

- `logos lsp` MUST keep one document store keyed by the file path derived from the source URI that holds the text, the client URI, the client version, and at most one lowered snapshot per package. [T-Sec16]
- The proxy MUST answer `logos/snapshotCounts` with the number of retained lowered snapshots per package. [T-Sec16]
- The `logos/snapshotCounts` answer MUST also carry `diagCache`, the number of cached downstream diagnostic payloads. [T-Sec16]
- A request on a `.lgo` document MUST use a snapshot whose versions equal the current client versions of every document of the package; the proxy MUST compute it on demand, once per version set, and concurrent requests MUST share one computation. [T-Sec16]
- The store MUST keep the newest client version of each open document; a change that carries an older version MUST be dropped and MUST NOT reach `gopls`. [T-Sec16]
- A frame that carries a duplicate Content-Length header MUST fail with a descriptive error. [T-Sec16]
- A frame header MUST contain at most 8192 bytes and 32 lines, including the line terminators and the blank separator. [T-Sec16]
- An incomplete header that exhausts either limit MUST fail with a descriptive error without waiting for more input. [T-Sec16]
- The proxy MUST NOT recompile without a request, a `didSave`, or the `didClose` of an unsaved `.lgo` document. [T-Sec16]
- The lowering MUST come from one `logos list` run with compiled JSON output and the overlay that returns, per `.lgo` file, the lowered text and the edit map; per package, the hints JSON and the compiler diagnostics. [T-Sec16]
- The metadata request MUST disable VCS build stamping so a repository-status failure does not suppress compiler results. [T-Sec16]
- Position translation MUST rest on the compiler-written origin segments of the eject map: a lowered offset inside a copied segment maps to its exact original offset, and an offset inside a compiler-generated run maps to the anchor position of that run; no heuristic or size-capped textual matching takes part. [T-Sec16]
- Closing an unsaved `.lgo` document MUST discard its unsaved overlay and MUST refresh the retained package's compiler results. [T-Sec16]
- Closing the last open document of a package MUST retire that package's lowered snapshot and its lowered documents in `gopls`. [T-Sec16]
- A `textDocument/formatting` request for a `.lgo` document MUST be refused instead of returning edits to lowered Go. Formatting requests for Go documents MUST pass through to `gopls`. [T-Sec16][R-U1]
- It MUST merge compiler JSON diagnostics into `publishDiagnostics`. [T-Sec16]
- It MUST report `logos lsp: gopls not found on PATH` when `gopls` is missing. [T-Sec16]
- The proxy MUST speak standard LSP over stdio. [T-Sec16]
- Stock LSP clients reach `.lgo` files through the proxy with no editor-specific code. [T-Sec16]

### 16.2 Inlay hints for lowering

- Inlay hints MUST show narrowed types such as `/* narrowed: *User */`. [T-Sec16]
- Inlay hints MUST show union members at switch statements. [T-Sec16]
- Inlay hints MUST show allocation badges `[stack]`, `[arena]`, `[heap]`, and `[rodata]`. [T-Sec16]
- Allocation badges MUST appear for Go packages too. [T-Sec16][R-U1]

### 16.3 Hover previews

- Hover over `?`, `const for`, `const switch T`, or `$"..."` MUST show the lowered code. [T-Sec16]
- Hover over an interpolation MUST render `Fixed: <n>B + dynamic` with the fixed byte count. [T-Sec16]

### 16.4 Code actions and cross-kind workspaces

- Code actions MUST apply the compiler autofix candidates. [T-Sec16]
- In a mixed workspace the overlay MUST hold the lowered `.lgo` files only, and Go packages MUST stay on disk for `gopls`. [T-Sec16][R-U3]
- A definition request across kinds MUST resolve to the declaration in its own file with the positions of that file. It MUST NOT write files to disk. [T-Sec16][R-U2]
- A Go package that imports a LoGOs package MUST resolve that import through the lowered overlay. [T-Sec16][R-U2]

## 17. Verification suite

### 17.1 Red-team completeness

- A release MUST have no `planned` row in `logos-doc/REDTEAM.md`: each row MUST be `tested` or `satisfied upstream` with its proving test, or `blocked` with its mechanism recorded and the release `-O` matrix gating it. [T-Sec17]
- The release `-O` matrix MUST pass `logos test` and `go test` at `-O=1`, `-O=2`, and `-O=3` over `std` and `cmd`, without `-short`, and MUST skip only the tests that `src/cmd/go/testdata/script/logos_redteam_80.txt` pins. [T-Sec17]

### 17.2 Zero-alloc and codegen checks

- A function marked `//logos:zeroalloc` that performs a heap allocation MUST fail with `[perf::zeroalloc_violation]` naming the function and the escape-analysis reason. [T-Sec17]
- The function below MUST fail with `[perf::zeroalloc_violation]`. [T-Sec17]

```lgo
//logos:zeroalloc
func pack(v int) []int {
    return []int{v}
}
```

- The diagnostic MUST NOT be lowered by a `logos.toml` severity and MUST NOT be silenced by a `//logos:allow` directive. [T-Sec17]
- A `//logos:zeroalloc` directive that is not on its own line directly above a function declaration MUST fail with `[perf::zeroalloc_violation]`. [T-Sec17]
- The check MUST cover the function literals inside the marked function and the bodies inlined into it. It MUST report the allocation sites that escape analysis puts on the heap and `make(chan T)`. It MUST NOT report a constant converted to an interface, a function literal that captures nothing, or a zero-size object. [T-Sec17]
- The check MUST NOT report an operation that falls back to the heap at run time: `append` past the capacity of its operand, a string concatenation or conversion longer than the stack buffer, or an insert that grows a map. [T-Sec17]
- Unless the directive reads `//logos:zeroalloc local`, the check MUST also follow calls and MUST fail with `[perf::zeroalloc_violation]` when the marked function calls, directly or through functions of its own package, a function whose body has a heap allocation site; when it calls through an interface method, or through a function value other than a local variable that is only assigned function literals; and when it calls a function of another package that the facts of that package do not record as allocation free (`AllocFree`). `//logos:zeroalloc local` checks only the function's own body, its function literals, and the bodies inlined into it. [T-Sec17]
- In a Go package, `logos lint` MUST apply the check, and `logos build` MUST apply it as well when the author wrote the directive; `go build` MUST ignore the directive. [T-Sec17][R-U1]

### 17.3 Reproducibility and sizes

- Two builds from clean caches MUST yield identical SHA-256 digests for linux/amd64, linux/arm64, and js/wasm. [T-Sec17]
- The stripped js/wasm size of the `repro` module built with `-trimpath -buildvcs=false -ldflags='-s -w'` MUST be tracked in `test/logos/size/wasm_size.txt`. Growth above 1% against the recorded value MUST fail, and a record more than 1% above the measured size MUST fail as stale. [T-Sec17]

### 17.4 End-to-end example exercise

- Ejected output MUST match native output for the example. [T-Sec17]

### 17.5 Interop suite

- Case 1: a LoGOs package that imports a Go package from another workspace module MUST build and run. [T-Sec17][R-U2]
- Case 2: a Go package that imports a LoGOs package from another workspace module MUST build and run under `logos`; Sec 0.1 states the result under plain `go`. [T-Sec17][R-U2][A-LGOSEL]
- Case 3: a LoGOs package that imports `fmt` and `context` MUST build and run. [T-Sec17][R-U2]
- Case 4: a module that holds both kinds; Sec 0.1 states the `correctness::mixed_module` rule. [T-Sec17][R-U3]
- Case 5: `logos build ./...` at a workspace root builds both modules as Sec 0.1 states, and each binary MUST run. [T-Sec17][R-U2]

## 18. Spec areas

### 18.1 Objective

LoGOs is a complete toolchain for Go and for the LoGOs language, which adds checked error propagation to Go. The command `logos eject` writes the pure-Go form of a LoGOs module.
Writers write `.lgo` files, the `logos` command checks and lowers them, and the pinned Go backend builds them.
The toolchain is one complete toolchain for Go and LoGOs: Go packages and LoGOs packages share one backend and one runtime, and each improvement serves both.

### 18.2 Compiler-made names

Compiler-made names use the `_logos_` prefix; comptime copies use the `_logosct` suffix. The only exceptions are the Go-fixed method names `MarshalJSON`, `UnmarshalJSON`, `MarshalBinary`, `AppendBinary`, and `UnmarshalBinary`, and the exported property accessors `Logos_GetF` and `Logos_SetF` of Sec 2.3, which Go importers can call.
Tuple temporary names MUST be distinct across all files and expansion sites of a package, including sites with identical source coordinates. Repeated compilation of the same ordered inputs MUST produce the same names. Source diagnostics MUST retain the original positions. [T-204.1]
Separately lowered inactive files MUST retain distinct tuple temporary names when combined in an ejected package. [T-204.1]
For example, the lowering of `v := parse(s)?` reads as follows (temporary names and site indices vary by position).

```go
v, _logos_e0 := parse(s)
if _logos_e0 != nil {
    _logos_rt.TraceErr(_logos_e0, &_logos_site0)
    return 0, _logos_e0
}
```
