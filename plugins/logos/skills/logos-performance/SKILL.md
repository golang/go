---
name: logos-performance
description: Optimize build and run-time performance with the logos toolchain, for .lgo code and for Go code built by it. Use when a logos.toml exists, the files are .lgo, or the user names the logos command, and the user says optimize, speed up, -O levels, binary size, allocation, arena, SIMD, or zero-alloc.
license: BSD-3-Clause
compatibility: Needs the logos command from github.com/gosuda/loGos on PATH.
metadata:
  version: "0.22.0"
  milestone: "M13"
---

# Performance optimization

The logos toolchain provides build-time and run-time optimization controls for .lgo packages and Go packages built by it.

## Procedure

1. Measure baseline performance first. Save benchmark output with `go test -bench=. -count=6 > baseline.txt`.
2. Change exactly one flag or configuration key in `logos.toml` or the build command.
3. Verify against the baseline with the same benchmark and inputs using `benchstat baseline.txt new.txt`.
4. Ensure tests and race checks pass under the selected mode before keeping the configuration.

## Optimization tiers (M13)

The `-O` build flag is accepted by both `logos build` and `go build`. In `logos.toml`, `[build] opt_level` sets the default for `logos` builds. Plain `go` reads no `logos.toml` and requires the explicit `-O` flag.

| Level | Inlining budget | Stack buffer threshold | Flags emitted |
|---|---|---|---|
| `O0` | none | none | `-N -l` |
| `O1` | upstream default | 32 B | upstream compiler defaults |
| `O2` | 120 | 256 B | `-d=inlbudget=120,variablemakethreshold=256` |
| `O3` | 400 | 4096 B | `-d=inlbudget=400,inlrecursivebudget=80,variablemakethreshold=4096` |
| `Os` | 15 | 32 B | `-d=inlbudget=15` |

Rules:

* `logos.toml [build] opt_level` defaults to `"O2"`. Command-line `-O` overrides the file setting (`src/cmd/go/testdata/script/logos_opt_levels.txt`).
* `target_cpu` in `logos.toml [build]` accepts `"deterministic"` (`GOAMD64=v3`, `GOARM64=v8.2`), `"baseline"` (`GOAMD64=v1`, `GOARM64=v8.0`), or `"native"` (selects host architecture features from `internal/cpu`).
* Improvements apply to Go packages built under `logos` and under plain `go` with `-O`.

## Fixed-array SIMD (M14)

loGo/s supports element-wise vector operators and reductions on qualifying fixed arrays.

Rules:

* Element-wise operators `+`, `-`, `*`, `/`, `&`, `|`, `^` and reductions `.sum()`, `.min()`, `.max()`, `.dot()` are supported on `[N]T` where `N` is a power of two, `T` is an integer or floating-point kind other than complex (`int`, `uint`, and `uintptr` use 64-bit lanes), and total array size is at most 64 bytes (`SPEC.md` Sec 2.4, `src/internal/types/testdata/logos/simd_types.lgo`, `test/logos/run/simd.lgo`).
* Array sizes exceeding 64 bytes fail with `correctness::simd_array_too_large`. Non-power-of-two lengths fail with `correctness::simd_array_shape`.
* Lowering emits vector instructions via `simd/archsimd` when built with `GOEXPERIMENT=simd` on amd64 or arm64 for fixed-width numeric kinds and aliases (`byte`, `rune`, `float32`, `type f32 = float32`), subject to CPU feature gates on amd64 (`AVX`, `AVX2`, `AVX512`) elided at matching `GOAMD64` levels.
* Defined element types, `int`, `uint`, and `uintptr` retain unrolled scalar lowering.
* When `GOEXPERIMENT=simd` is absent, the target is not amd64 or arm64, or CPU feature checks fail, lowering falls back to unrolled scalar code.
* Reductions use a fixed pairwise order across both vector and scalar forms, guaranteeing bit-identical floating-point output.
* Go packages have no array operators or reductions; they use `simd/archsimd` with `GOEXPERIMENT=simd` directly.

## WASM profile (M16)

The WASM profile activates automatically for `GOOS=js GOARCH=wasm` and `GOOS=wasip1 GOARCH=wasm` under `logos` only; plain `go` needs the explicit `--wasm-features` flag.

Rules:

* Pure non-yielding functions (`MayYield=0`) skip resume-point `br_table` dispatch in the function prologue and use native WASM calls with `local.get` and `local.set` (`test/logos/codegen/wasm_noyield.lgo`).
* Functions with `defer`, channel operations, select, or transitive calls to `runtime.gopark` retain `MayYield=1` and keep resume dispatch.
* The `--wasm-features=stack-switching` build flag compiles `MayYield=1` functions to `cont.new` and `resume` instructions.
* The loader `app.loader.mjs` is emitted beside the WASM binary by default under both `logos build` and plain `go build` when `-o` names a `.wasm` file; `--wasm-loader=false` turns it off (`logos_wasm_loader.txt`).
* Directives `//logos:wasmexport` and `//logos:wasmimport` lower to `//go:wasmexport` and `//go:wasmimport`.
* 64-bit integer and floating-point fields maintain natural 8-byte alignment in structs (`test/logos/codegen/wasm_align.lgo`).

## Stack sizing (M17)

Smart stack allocations keep dynamic allocations on the goroutine stack when escape analysis proves they do not escape.

Rules:

* Non-escaping dynamic slices created with `make([]byte, n)` use a variable-make stack buffer sized by tier for non-standard packages: default 32 bytes (O1 / plain), 256 bytes at O2, and 4096 bytes at O3. Standard-library-provenance packages keep the 32-byte buffer at every level (`base.Debug.VariableMakeThreshold`, `src/cmd/compile/internal/inline/stackbuf_logos_test.go`).
* Functions in recursive call-graph cycles keep the upstream variable-make threshold (capped at 32 bytes) at every level to prevent frame multiplication across recursion (row #134, `test/logos/codegen/recursive_stack.go`, `test/logos/codegen/recursive_stack.lgo`).
* Dynamic slice sizes exceeding the tier threshold fall back to heap allocation on the slow branch.
* Slices with variable effective capacity flowing into cgo calls or `unsafe.Pointer` conversions allocate backing storage on the heap (row #144, `src/cmd/cgo/internal/testlogos/fastcall_test.go`).
* Fixed-effective-capacity makes retain ordinary escape and size behavior, not a guarantee of stack allocation, and are not forced to the heap by foreign flow (`SPEC.md` Sec 3.2).
* Goroutine initial stack sizing with `GOEXPERIMENT=logosstack` chooses the initial stack size from the per-PC exponential moving average of final stack sizes.

## Tier-0 arena and bump allocation (M18)

Goroutine-local arena allocation is controlled by `GOEXPERIMENT=logosarena` and `logos.toml [build] arena`. Under `logos`, implicit placement is on by default: a missing key still enables the experiment, an explicit `arena = false` opts out, and plain `go` enables it only through the experiment (`logos_arena_default.txt`).

Rules:

* Non-escaping allocations exceeding stack buffer thresholds allocate into goroutine arena chunks instead of the heap when arena mode is active (`src/runtime/logos_arena_test.go`, `src/cmd/go/internal/cfg/logos_experiment_test.go`).
* Constant-size noscan objects of at most 512 bytes use the inline bump sequence on amd64 in non-race builds; other targets call the region slow path (`src/runtime/logos_arena_test.go`).
* Watermarks follow LIFO ordering and are restored at function returns and loop back-edges.
* Memory holding pointers is scanned by the garbage collector during stack scanning.
* Calling `runtime.SetFinalizer` on an arena object panics (`src/runtime/logos_arena_test.go`).

## Fast cgo (M20)

Fast-cgo bypasses standard cgo trampoline overhead for eligible non-blocking C functions.

| Language | Annotation |
|---|---|
| Go | `#cgo noescape F` and `#cgo nocallback F` |
| loGo/s | `//logos:cgo_fastcall F` in a `.lgo` file |

Rules:

* The fast path requires both `noescape` and `nocallback`. A function with only one directive stays on the plain cgo path (`src/cmd/cgo/internal/testlogos/fastcall_test.go`).
* The fast path requires `GOEXPERIMENT=logosfastcgo`, which is off by default under both `logos` and `go`.
* If a `nocallback` function calls back into Go, the runtime panics immediately instead of deadlocking (`TestFastcallNoCallbackViolation`).
* CPU profile samples during fast C calls attribute to the calling Go site (`TestFastcallProfileAttribution`).
* No deterministic timing gate is provided. The fast body must not read a clock. Do not claim a speedup. Blocking parity, callback parity, and profile attribution pass.

## Zero-allocation enforcement (M22)

The `//logos:zeroalloc` directive asserts that a function causes zero heap allocations.

Rules:

* Place `//logos:zeroalloc` directly above a function declaration in `.lgo` or `.go` files (`test/logos/codegen/zeroalloc.lgo`).
* If escape analysis detects any heap allocation, the build fails with `[perf::zeroalloc_violation] <func> allocates: <reason>`.
* Violations are the allocation sites that escape analysis puts on the heap, including the function literals and inlined bodies it covers (`test/logos/codegen/zeroalloc_sites.lgo`). Run-time heap fallbacks (`append` past capacity, long string ops, map growth) are not reported.
* Empty slice literals and zero-size interface values are exempt and allocate zero heap bytes.

## Diagnostics

| Rule | Fix |
|---|---|
| `correctness::simd_array_too_large` | Reduce array length or element type size so the fixed array does not exceed 64 bytes. |
| `correctness::simd_array_shape` | Use a power-of-two array length for element-wise SIMD operations. |
| `perf::zeroalloc_violation` | Eliminate heap allocations, use stack buffers, or remove the `//logos:zeroalloc` directive. |
| `correctness::cgo_file_logos_syntax` | Use standard Go syntax in files importing `"C"`. |
| `comptime::node_limit` | Reduce comptime AST unrolling or raise `max_comptime_ast_nodes` in `logos.toml`. |
