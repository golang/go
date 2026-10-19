# Section 18: upstream file manifest (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 1159-1177 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 18: EXACT UPSTREAM GO COMPILER FILE MODIFICATION MANIFEST (`src/`):**

To keep upstream rebasing against Go `1.24`–`1.26` under **2 developer-days per 6-month release**, modifications to existing Go files are restricted to single-line hooks delegating to isolated `*_logos.go` files:

| Upstream Directory | Modified Upstream Files (Hook Lines Only) | Newly Added Isolated `loGo/s` Files |
| :--- | :--- | :--- |
| **`src/cmd/go/`** | `internal/load/pkg.go` (+8 lines: `.lgo`, `.📜`, `.⚡`, `.ʕ◔ϖ◔ʔ` NFC loader)<br>`internal/cfg/cfg.go` (+3 lines: `CGO_ENABLED=0` default, `zig cc` fallback)<br>`main.go` (+3 lines: register `lint`, `next`, `context`, `eject` subcommands) | `internal/lint/lint_logos.go`<br>`internal/next/next_logos.go`<br>`internal/context/context_logos.go`<br>`internal/toml/bettergo_toml.go` |
| **`src/cmd/compile/internal/syntax/`** | `tokens.go` (+5 token enums: `_Question`, `_QuestionDot`, `_QuestionQuestion`, `_Arrow`, `_InterpStringLit`)<br>`scanner.go` (+18 lines: `$`, `?`, `=>` dispatch)<br>`parser.go` (+25 lines: hook `stmtOrDecl`, `p.expr`, `p.type_`) | `nodes_logos.go`<br>`parser_logos.go` (LL(1) tuple/lambda, `errdefer`, `const for/if/switch`) |
| **`src/cmd/compile/internal/types2/`** | `interface.go` (-1 line: unseal union in value position; +6 lines: property contracts)<br>`typexpr.go` (+12 lines: `T?` option & tuple unlock)<br>`assignments.go` (+9 lines: `const` view & Non-Nil checks) | `const_logos.go`<br>`nilflow_logos.go`<br>`union_logos.go`<br>`subst_logos.go` (`const switch T` rebinding) |
| **`src/cmd/compile/internal/noder/`** | `irgen.go` (+6 lines: invoke `betterlint` & `eject` passes)<br>`stencil.go` (+14 lines: 2-Phase `const for` & interface auto-generification)<br>`linker.go` (+5 lines: write 1-byte `LogosFuncSummary`) | `comptime_logos.go`<br>`desugar_logos.go` (`?`, `errdefer`, `$"..."`, chains)<br>`izer_logos.go` (2-Pass JSON/String sizer)<br>`eject_logos.go` (`--emit-go` transpiler) |
| **`src/cmd/compile/internal/betterlint/`** | *(0 upstream files modified — 100% self-contained package)* | `runner.go`, `correctness.go`, `concurrency.go`, `slop.go`, `complexity.go`, `json_diag.go` |
| **`src/cmd/compile/internal/escape/`** | `escape.go` (+15 lines: Side-Stack `EscArena`, `EscReturn` merge, `sync.Pool` hard sink) | `arena_infer_logos.go` |
| **`src/cmd/compile/internal/ssa/`** | `compile.go` (+4 lines: register `loopvec`, `branchless`, `dse` volatile rules)<br>`prove.go` (+5 lines: `-O3` domain budget multiplier) | `loopvec_logos.go` (Fixed-array SIMD)<br>`branchless_logos.go` (`CMOVcc` / `CSEL`) |
| **`src/cmd/internal/obj/`** | `arm64/obj7.go` (+3 lines: invoke `coalesceLdpStp`)<br>`wasm/wasmobj.go` (+12 lines: Selective Asyncify color check) | `arm64/ldpstp_logos.go`<br>`wasm/asyncify_logos.go` |
| **`src/cmd/link/internal/ld/`** | `deadcode.go` (+14 lines: `[capabilities]` sink backtracer, WASM reflect strip)<br>`data.go` (+8 lines: sort cold symbols to `.text` tail)<br>`pcln.go` (+6 lines: strip `$spec$`/`$cold$` display names) | `capabilities_logos.go`<br>`rodata_map_logos.go` |
| **`src/runtime/`** | `runtime2.go` (+12 fields on `g` and `p`)<br>`malloc.go` (+8 lines: dispatch `-gc` mode)<br>`preempt.go` (+6 lines: RSEQ `.L_bump_start` check)<br>`netpoll.go` (+7 lines: `g.cancelChan` wake hook) | `arena_logos.go`, `fastcall_logos.s`<br>`gc_greentea.go`, `gc_zgc.go`, `gc_immix.go`, `gc_wasm.go`, `gc_batch.go` |

---

