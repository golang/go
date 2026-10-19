# Section 22: verification suite (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 1270-1283 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 22: COMPILER VERIFICATION & FUZZING TEST SUITE (`test/logos/`):**

Before any `loGo/s` compiler binary passes CI, it SHALL pass the following 5 automated verification suites:

1. **Upstream Regression Gate (`src/run.bash`):**
   100% of upstream Go's standard library tests (`go test std cmd`) MUST pass unmodified when compiled with the `logos` toolchain at `-O1`, `-O2`, and `-O3`.
2. **`--emit-go` Round-Trip Equivalence Fuzzing (`test/logos/fuzz_emit_go/`):**
   For 10,000 generated `.lgo` programs exercising unions, `const switch T`, `errdefer`, `$"..."`, and `.filter().map()`, the test runner compiles the program natively (`logos build`) and via transpilation (`logos build --emit-go && go build`), asserting bit-identical stdout, stderr, exit codes, and panic stack traces.
3. **Adversarial Torture Suite (`test/logos/redteam_001_to_182/`):**
   182 dedicated regression tests—one for each Red-Team attack vector (`attack_001_split_union_gc.lgo` through `attack_182_wasm_int_size.lgo`)—verifying either exact compile-time `forbid`/`deny` rejection (`errorcheck`) or race-free runtime execution under `-race`, `-gc=greentea`, `-gc=zgc`, and `-gc=immix`.
4. **Zero-Alloc & Assembly Verification (`test/logos/codegen/`):**
   Uses Go's `codegen` harness (`// amd64:"VPADDD"`, `// arm64:"LDP"`, `// !"runtime.mallocgc"`) to assert that `//logos:zeroalloc` functions, SIMD arrays, and `cgo_fastcall` trampolines emit exact target instructions with zero regressions.
5. **Deterministic Cross-Build Hash Gate:**
   Compiles the reference `order-engine` benchmark across Linux/macOS hosts targeting `GOOS=linux GOARCH=amd64`, `GOARCH=arm64`, and `GOARCH=wasm`, asserting identical SHA-256 binary checksums and `< 45KB` stripped WASM output.
