# Section 1: tier 1 (toolchain, linter, desugaring) (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 22-107 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 1: TIER 1 SPECIFICATION (LOW BLAST RADIUS — TOOLCHAIN, LINTER, DESUGARING):**

*Target Subsystem: `src/cmd/go`, `src/cmd/compile/internal/syntax`, `src/cmd/compile/internal/betterlint`. Rebase conflict rate: $$< 1\%$$.*

**RFC-101: File Loader, Unicode Normalization & CGo Security Gate:**
1. **Extension Loader (`src/cmd/go/internal/load/pkg.go`):**
   * The file loader SHALL strip trailing Unicode Variation Selectors (`U+FE0E`, `U+FE0F`) and apply NFC normalization before matching extensions `.lgo`, `.📜`, `.⚡`, and `.ʕ◔ϖ◔ʔ`.
   * If two files in the same package share an identical base stem (e.g., `foo.lgo` and `foo.📜`), the compiler SHALL abort with `[correctness::duplicate_stem_filename]`.
2. **CGo Default-Off & Supply-Chain Allowlist (`src/internal/cfg/cfg.go`):**
   * `CGO_ENABLED` SHALL default to `0` across all architectures.
   * When a package contains `import "C"`, the compiler SHALL verify whether the package path matches `cgo_allowlist` in `bettergo.toml`. Unauthorized CGo imports SHALL fail with `[restriction::cgo_usage]`.
   * When CGo is permitted and no `CC` environment variable is set, `cmd/go` SHALL automatically bind `zig cc -target <arch>-<os>-musl` if `zig` is present in `PATH`.

**RFC-102: Embedded Compiler Linter (`go lint`) & Hard-Gate Error Matrix:**
1. **Architecture (`src/cmd/compile/internal/betterlint`):**
   * The linter SHALL execute in-memory inside `cmd/compile` over `syntax.File`, `types2.Info`, `escape` dataflow, and pre-codegen `ssa.Func`. Execution overhead SHALL NOT exceed $$2\%$$ of total compile time.
   * **Scoping Boundary:** Strict rules (`forbid` / `deny`) apply to first-party `.lgo`/`.📜`/`.⚡`/`.ʕ◔ϖ◔ʔ` files. First-party `.go` files downgrade style/migration rules to `warn`. Third-party modules (`pkg/mod/**`) bypass all lints except linker-enforced `[capabilities]` (`cgo`, `unsafe`, `reflect`, `os/exec`, `net`).
2. **Mandatory Compile-Error Rules (`forbid` — Non-Overridable even with comments):**
   * `correctness::nil_deref_unproven`: Dereferencing a nullable `T?` pointer, map index result, or unverified `(res, err)` tuple value prior to a nil/error guard.
   * `correctness::typed_nil_return`: Returning a concrete `(*T)(nil)` pointer as a non-nil `interface` or `error`.
   * `correctness::arena_pointer_escape`: Storing a Side-Stack arena or stack-first pointer into `sync.Pool`, global variables, or escaping heap structures.
   * `correctness::defer_erases_errdefer_state`: Mutating named return `err = nil` inside a standard `defer` when an `errdefer` is registered in the same function.
   * `correctness::infinite_recursion`: Unconditional self-recursive call with identical type and value arguments.
   * `concurrency::concurrent_multiword_race`: Sharing mutable multi-word values (`interface`, `[]T`, `string`, `map`) across goroutines without `const`, `sync.Locked[T]`, or `sync.Hot[T]`.
   * `concurrency::reentrant_withlock_deadlock`: Calling `.WithLock()` on the same `sync.Locked[T]` instance inside its own closure call graph.
   * `concurrency::unsupervised_goroutine`: Spawning raw `go fn()` outside a structured `go group` block in `.lgo` files.
3. **Strict Default Rules (`deny` — Overridable only via `//logos:allow(<rule>)`):**
   * `slop::err_chain_reset`: Calling `fmt.Errorf` with `err.Error()` or `%s`/`%v` instead of `%w`.
   * `slop::discarded_error`: Ignoring an `error` return value via `_ = fn()` or bare call (exempting proven nil-error functions and benign stdlib writers `fmt.Print*`, `bytes.Buffer.Write*`, `strings.Builder.Write*`).
   * `slop::todo_placeholder`: Leaving `// TODO`, `// FIXME`, or unimplemented panic stubs in non-test builds.
   * `correctness::unclosed_io_closer`: Failing to register `defer x.Close()?` on a local resource implementing `io.Closer`, or opening `io.Closer` inside `.map()` without rollback.
   * `correctness::unbounded_channel_op`: Performing channel send/receive without a `ctx.Done()` select branch, unless `escape`/SSA proves the channel is a locally allocated buffered channel (`make(chan T, k)`) where `k >= send_count`.
   * `complexity::cognitive_overflow`: Function nesting depth $$> 3$$, parameter count $$> 4$$, line count $$> 60$$, or Cognitive Complexity Score $$> 15$$:
     $$\text{Score} = \sum_{i} (1 + \text{NestingDepth}_i) + \text{Penalty}_{\text{mixed\_bool\_ops}}$$

**RFC-103: `go context` & `go next` Toolchain:**
1. **`go context <pkg>#<Symbol>`:** Extracts a minimal 100-line dependency and type signature skeleton (stripping private function bodies) in $$< 50\text{ms}$$ for LLM context injection.
2. **`go next [--fix | --ai-patch] ./...`:** Rewrites legacy Go patterns (`if err != nil`, `fmt.Sprintf`, manual loops) into canonical `loGo/s` syntax using byte-offset AST span splicing (preserving 100% of comments and formatting). Complex refactors emit `next_tasks.json`.

**RFC-104: Canonical Syntactic Sugar & AST Desugaring (`syntax` → `noder`):**
1. **Error Propagation (`expr?`):**
   * `x := fn()?` lowers to:
     ```go
     x, _err0 := fn()
     if _err0 != nil {
         runtime_recordErrPC(_err0, _currentPC)
         return _zero0, _err0
     }
     ```
2. **Error-Aware Defer (`defer fn()?`) & Error-Only Defer (`errdefer fn()`):**
   * `defer f.Close()?` executes unconditionally in LIFO order and merges errors without allocation when `_retErr == nil`:
     ```go
     defer func() {
         if _cErr := f.Close(); _cErr != nil {
             if _retErr == nil { _retErr = _cErr } else { _retErr = _logos_errors.Join(_retErr, _cErr) }
         }
     }()
     ```
   * `errdefer tx.Rollback()` executes ONLY when the enclosing function exits with `_retErr != nil && _retErr.data != nil` OR during panic/`runtime.Goexit()` unwinding (`!_errdefer_ok`). In `--emit-go`, unnamed error returns are promoted to `(_ret0 T, _retErr error)` and `_errdefer_ok = true` is injected prior to every non-error `return`.
3. **Optional Chaining (`?.`) & Coalescing (`??`):**
   * `u?.Profile?.Name ?? "Guest"` evaluates the receiver once into a temporary variable and short-circuits to `"Guest"` on `nil` / `none`.
   * Map default lookup `v := m[k] ?? defaultVal` lowers to `v, _ok := m[k]; if !_ok { v = defaultVal }`.
4. **2-Pass Exact-Allocation String Interpolation (`$"..."`):**
   * `$"User {getName()} has {count} items"` lowers in 3 strict steps:
     * **Step 0 (Single Evaluation):** Bind all expressions inside `{...}` to local temporaries (`_p0 := getName()`, `_p1 := count`) once in left-to-right order to prevent TOCTOU races and duplicate side effects.
     * **Step 1 (Exact Byte Sizing):** Compute `_size := 5 + len(_p0) + 5 + runtime.IntStringLen(_p1) + 6`.
     * **Step 2 (Zero-Waste Fill):** Allocate `_buf := make([]byte, _size)` (eligible for Stack-First allocation if $$\le 4\text{KB}$$) and fill via `copy` and `strconv.AppendInt`, followed by zero-copy `unsafe.String(&_buf[0], _size)`. Zero `fmt.Sprintf` reflection, zero format-string injection risk.
5. **Address of Literals (`&expr`), Struct Defaults & Strict Spread (`Target{..src}`):**
   * `&"hello"`, `&100`, `&User{...}` allocate an addressable temporary in `noder` and take its address.
   * Struct field defaults (`type Cfg struct { Port int = 8080 }`) are injected by `noder` during literal construction (`Cfg{}`) and prior to static JSON unmarshaling. Bare `var c Cfg` remains zero-initialized and triggers `[suspicious::unapplied_struct_default]`.
   * `Dst{..src}` copies identically named fields from `src` to `Dst`. If `Dst` and `src` share a field name with incompatible types, compilation fails immediately with `[correctness::spread_type_mismatch]`. In-place self-update `s = State{Tick: 1, ..s}` lowers to single-field mutation `s.Tick = 1` without copying the struct.
6. **Zero-Alloc Collection Method Chaining (UFCS Stream Fusion):**
   * Slices support `.filter(x => pred)`, `.map(x => expr)`, `.find(x => pred)`, `.any(x => pred)`, `.all(x => pred)`, `.sortBy(x => key)`, `.first()`, `.last()`, `.get(i)`, `.take(n)`, `.drop(n)`.
   * **Grammar Constraint:** Lambdas `x => expr` inside collection chains SHALL be single expressions; block lambdas `{ return ... }` are forbidden (early error return via `?` inside `expr` propagates out of the enclosing function). If `expr` returns an `io.Closer` and uses `?`, the compiler injects a reverse `.Close()` cleanup loop on partial failure.
   * **Pull-Based Lazy Fusion:** `.filter().map().take(k)` fuses into a single `for` loop that binds the root slice expression once (`_src := rootExpr`) and executes `break` immediately when `len(_out) == k`.
   * `.sortBy()` uses `cmp.Compare` total ordering (handling `NaN` and `-0.0` deterministically without panic). `.sum()` on floating-point slices uses pairwise summation without implicit FMA contraction to ensure bit-identical results across x86_64 and ARM64.

**RFC-105: Optimization Tiers (`-O0` to `-O3`, `-Os`, `-Oz`) & Synthetic PGO:**
* **`-O0`:** Debug mode (`inlineMaxBudget = 0`, no escape promotion).
* **`-O1`:** Upstream Go baseline (`inlineMaxBudget = 80`, stack slice limit `64KB`).
* **`-O2` (Default):** `inlineMaxBudget = 120`, stack slice limit `128KB`, `standard` Synthetic PGO (branches leading to `?` error return, `panic`, or `os.Exit` assigned $$0.05\%$$ probability; cold error blocks outlined and placed at the tail of `.text`), deterministic baseline `GOAMD64=v3` (AVX2+BMI2) / `GOARM64=v8.2` (LSE).
* **`-O3`:** `inlineMaxBudget = 400` (dampened to `80` inside recursive SCCs), full generic monomorphization, escape analysis fixed-point walk budget $$4\times$$, SSA `prove` pass domain limit $$5\times$$, `aggressive` Synthetic PGO (loop weights $$\times 10^d$$).
* **`-Os` / `-Oz`:** Binary size diet (`inlineMaxBudget = 15`, GCShape generic dictionary sharing, Identical Code Folding).

---

