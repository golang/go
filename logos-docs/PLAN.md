**DOCUMENT ID:** `RFC-LOGOS-SPEC-FINAL`
**PROJECT NAME:** `loGo/s` (Repository / CLI / Identifier: `logos`)
**FILE EXTENSIONS:** `.lgo` (Canonical Strict), `.📜` (Scroll Alias), `.⚡` (Lightning Alias), `.ʕ◔ϖ◔ʔ` (Go Issue `#59968` Gopher Alias), `.go` (Upstream Legacy Mode)
**STANDARD:** ASD-STE100 (Simplified Technical English) — Strict Normative Specification
**BASE UPSTREAM:** Go `https://github.com/golang/go/tree/release-branch.go1.27` / Version: `1.27.2` / Head: `022c863` (`src/cmd/compile`, `src/runtime`, `src/cmd/link`, `src/cmd/go`)
**STATUS:** Final Verified Master Specification (Hardened against 182 Red-Team Adversarial Vectors)

---

**SECTION 0: ARCHITECTURAL AXIOMS & TRANSPILATION CONTRACT:**

* **0.1. Upstream Compatibility:** The compiler SHALL compile all valid upstream Go source files (`.go`) and module dependencies (`go.mod` / `GOPATH/pkg/mod`) without syntax or runtime behavior modification.
* **0.2. Zero New Global Keywords:** The language SHALL NOT introduce any new globally reserved keywords that invalidate existing Go identifiers. All extensions SHALL reuse Go's 25 reserved keywords (`const`, `interface`, `struct`, `switch`, `defer`, `for`, `go`, `func`, `type`) or contextual statements (`errdefer`, `yield`).
* **0.3. Bidirectional Pure Go Transpilation (`--emit-go`):**
  * The command `logos build --emit-go ./...` (or `logos eject`) SHALL lower all Tier 1 and Tier 2 `.lgo`, `.📜`, `.⚡`, and `.ʕ◔ϖ◔ʔ` source files into 100% standard, `gofmt`-compliant `.gen.go` files.
  * Every emitted `.gen.go` file SHALL include `//go:build !logos` at line 1 so that standard `go build` compiles `.gen.go` while `logos build` compiles the native `.lgo` source without symbol redeclaration conflicts.
  * The transpiler SHALL inject collision-proof helper imports (`import _logos_strconv "strconv"`, `import _logos_errors "errors"`, `import _logos_unsafe "unsafe"`) and strip virtual `import "comptime"`.
* **0.4. AI-First Generation & 1-Second Human Supervision:** Every side effect, mutability contract, nilability state, and error path SHALL be explicitly visible in function signatures and call sites. Dangerous edge cases SHALL fail at compile time via `go lint` (`forbid` / `deny`).

---

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

**SECTION 2: TIER 2 SPECIFICATION (MEDIUM BLAST RADIUS — TYPE SYSTEM, UNIONS, INTERFACES, COMPTIME):**

*Target Subsystem: `src/cmd/compile/internal/types2`, `noder`, `walk`, `escape`, `cmd/link`. Rebase conflict rate: $$2\text{--}5\%$$.*

**RFC-201: Unified `const` Immutability & `.rodata` Placement:**
1. **Composite `.rodata` Constants & Deterministic CTFE:**
   * `const` declarations accept composite literals (structs, arrays, slices, maps) and calls to `const func` (Compile-Time Function Evaluation).
   * Pointer-free constants are emitted into ELF/Mach-O/PE `.rodata`; constants containing pointers/strings are emitted into `.data.rel.ro` (RELRO). Single constant size is capped at `1MB`.
   * **`const map` Lowering:** Maps with $$\le 8$$ keys lower to a length-switched `switch` statement. Maps with $$> 8$$ keys are compiled into a static `.rodata` SwissTable using a per-map collision-free seed found at compile time ($$O(1)$$ worst-case 1-probe lookup). `-0.0` float keys are canonicalized to `+0.0`; `NaN` keys fail compilation.
   * **`const func` Purity Rules:** A `const func` SHALL NOT perform I/O, spawn goroutines, read mutable globals, call `recover()`, cast pointers to `uintptr`, or iterate over a dynamic `map`. CTFE executes using target architecture integer widths (`types.PtrSize`) and aborts if step count exceeds `100,000`.
2. **Transitive Immutable Views (`const []T`, `const *T`, `const map[K]V`):**
   * Parameters and variables marked `const` forbid index mutation, field mutation, `append`, `delete`, or calling methods whose `.a` export summary header has `ReceiverMutated == 1`.
   * **Zero-Copy `string` ↔ `const []byte`:** Converting `string` to `const []byte` is always a `0ns` no-op (`ir.OCONVNOP`). Converting `const []byte` (or `[]byte`) to `string` is zero-copy ONLY when `escape.go` proves the resulting `string` does not escape the current function frame and the source slice has no active mutable aliases.

**RFC-202: Non-Nil Default (`*T`), Option (`T?`), and Interprocedural Nil-Flow Proof:**
1. **Strict Non-Nil Pointers (`.lgo`):**
   * `*T` is non-nullable by default. Assigning `nil` to `*T`, declaring uninitialized `var p *T`, `[N]*T`, or `make([]*T, len)` (where `len > 0`) without initialization is a compile error.
   * **`.go` Interop Boundary:** Pointers returned from legacy `.go` packages are typed as `*T?` unless the function's `.a` export summary bit `ReturnsNonNil == 1` is set by return-path verification.
2. **Unified Option Type (`T?` == `interface { T | nil }`):**
   * **Memory Layout:** If `T` has a zero-value niche (`*U`, `func`, `chan`), `T?` is represented as a single 8-byte pointer where `0x0` is `nil` (`0B` overhead). If `T` is a value type or nested option (`*U?`), `T?` lowers to `struct { val T; present bool }`. Direct `T??` syntax is forbidden by `betterlint`.
   * **Flow-Sensitive Promotion:** Checking `if x != nil` (or early-returning on `if x == nil { return ... }`) promotes `x` from `T?` to non-nil `T` in the subsequent lexical dominance frontier.
   * **Error-First Guard:** For any multi-return `res, err := fn()`, reading `res` before proving `err == nil` is blocked by `types2`.

**RFC-203: Unsealed Interfaces — First-Class Tagged Unions, Structural Fields & Auto-Generification:**
1. **Closed Union Interfaces (`interface { A | B }` as Value Types):**
   * The `types2` restriction blocking `interface { A | B }` from variable, parameter, and struct field positions is removed. Tilde (`~T`) approximations are forbidden in value unions.
   * **Two-Tier Deterministic Layout Law:**
     * **Pure-Scalar Non-Recursive Unions ($$\le 32\text{B}$$, `noscan`):** Lowered to a stack-allocated overlapped struct `[ _tag uint64 | _scalarPayload [24]byte ]`. Every write zero-extends the full payload slot (`MOVQ $0` / `STP ZR, ZR`) so that `==` and map hashing operate via single-instruction bitwise comparison without padding garbage.
     * **Pointer-Containing, Recursive (`Expr`, `List`), or Large ($$> 32\text{B}$$) Unions:** Lowered to the standard Go 16-byte interface layout `[ _tag *abi.Type (8B) | _data unsafe.Pointer (8B) ]` allocated on the Side-Stack Arena (or static `.rodata` for zero-sized/small-int variants).
   * **Exhaustive `switch` & Common Methods:** `switch` over a union checks variant exhaustiveness at compile time. Switch expressions (`res := switch x { case A => 1; case B => 2 }`) infer the Least Upper Bound type. Methods shared by all union variants can be declared in the union interface and called directly (`e.Pos()`) via a tag-switched jump.
2. **Interface Structural Field Contracts (Solving Go Issue `#51259` / Boilerplate Elimination):**
   * Interfaces MAY declare read-only properties (`const Name string`) and read-write properties (`HP int`):
     ```go
     type Creature interface {
         const Name string
         HP int
         Start() error
     }
     ```
   * **Satisfaction Rules:** A type `T` satisfies `const Name string` if `T` has either an exported field `Name string` OR a method `Name() string`. A type `T` satisfies mutable `HP int` ONLY if `T` is a pointer type (`*Struct`) with an exported field `HP int` OR implements both `HP() int` and `SetHP(int)`.
   * **Zero-Pollution Lowering:** For structs, the compiler populates synthetic accessor thunks directly into the `itab` slots without polluting `reflect.TypeOf(t).NumMethod()`. Compound mutations (`party[i()].HP += 10`) pin the receiver once (`_r := party[i()]`) and execute Write-Back (`_r._set_HP(_r._get_HP() + 10)`). Taking the address of an interface property (`&c.HP`) is a compile error.
3. **Inline Interface Assertion (`type MyStruct struct : io.Reader, Creature`):**
   * Replaces `var _ io.Reader = (*MyStruct)(nil)` with declaration-site verification in `types2`.
4. **Interface Lowering (Clone Specialization / Auto-Generification):**
   * For any function accepting an open interface parameter (`func Copy(w io.Writer, r io.Reader)` or `func Process(list const []Creature)`), the compiler retains the original `iface`-accepting symbol for function-pointer compatibility and generates monomorphized clones (`Copy$spec$os_File` with `obj.DUPOK` deduplication, stripped in `pclntab` for clean stack traces) for concrete call sites up to `max_type_depth = 4`.

**RFC-204: Unsealed First-Class Named Tuples & Fixed-Array SIMD:**
1. **First-Class Named Tuples (`(id int64, err error)`):**
   * `types2.Tuple` is unlocked in all type positions (`chan (int, error)`, `[] (x int, y int)`, map values) and lowers to an anonymous value struct `_Tuple2[T0, T1]`. Direct multi-return calls inside variadic arguments (`fmt.Println(pair())`) retain Go's unpacking semantics.
2. **Fixed-Array SIMD (`[4]float64`, `[8]float32`, `[16]uint8`, `[32]byte`, `[64]byte`):**
   * Element-wise operators (`+`, `-`, `*`, `/`, `&`, `|`, `^`) and reductions (`.sum()`, `.min()`, `.max()`, `.dot()`) on power-of-two numeric arrays lower directly to AVX2/AVX-512 and ARM64 NEON vector instructions.

**RFC-205: Type-Directed Metaprogramming (`const for`, `const if`, `const switch T`):**
1. **Compile-Time Reflection (`import "comptime"`):**
   * `const for _, f := range comptime.Fields[T]()` unrolls once per exported struct field (plus unexported fields if `T` is defined in the current package) during generic instantiation (`noder/stencil.go`, Two-Phase Deferred Stenciling).
   * Field projection `val.[f]` rewrites directly to `ir.OXDOT` (`val.FieldName`) preserving original DWARF source line numbers. `f.Type` substitutes directly into generic type arguments (`Resolve[f.Type](c)`).
   * **Guards:** Circular generic instantiation (`ServiceA` ↔ `ServiceB`) aborts immediately via `activeStencils` stack tracking. Total unrolled AST nodes per function are capped at `4,096`.
2. **Generic Type Switch (`const switch T`):**
   * Inside `func Fn[T any](v T, s []T)`, `const switch T { case int: ... case ~int: ... case []E: ... case io.Writer: ... }` evaluates at compile time in top-down first-match order (`betterlint` rejects shadowed unreachable cases).
   * For concrete cases (`case int:`, `case []E:`), `types2.Subst` rebind `v` (`int`) and all dependent composite variables (`s` becomes `[]int`) in that lexical block. For interface cases (`case io.Writer:`), `T` retains its concrete memory layout while gaining the interface's method set via constraint intersection.
   * **`--emit-go` Lowering:** Concrete cases emit zero-alloc `switch any((*T)(nil)).(type) { case *int: ... }` with `_logos_unsafe.Pointer` rebinding; interface cases emit `if _w, _ok := any(v).(io.Writer); _ok`.

**RFC-206: Fearless Concurrency (`go group`, `sync.Locked[T]`, `sync.Hot[T]`):**
1. **`go group (ctx, limit = N) { go { ... } }`:**
   * Guarantees all child goroutines exit before scope termination. On the first child error or panic (recovered into `runtime.PanicError`), the group cancels `_groupCtx`, automatically unblocks any child waiting on channels (via auto-injected `select { case <-_groupCtx.Done(): }`) or network sockets (via `runtime/netpoll.go` `g.cancelChan` interrupt), waits for all children to finish, and returns the error.
2. **`sync.Locked[T]` & `sync.Hot[T]`:**
   * `sync.Locked[T]` encapsulates mutex and value; `WithLock(v => ...)` registers `Unlock()` in the zero-cost `pclntab` unwind table (0ns happy path, 100% panic-safe unlock) and forbids returning inner pointers via `escape.go`.
   * `sync.Hot[T]` provides 0ns read-mostly RCU pointers using ARM64 `LDAPR`/`STLR` and Go's Yuasa deletion write barrier (zero retire-list memory overhead).

---

**SECTION 3: TIER 3 SPECIFICATION (HIGH BLAST RADIUS — PLUGGABLE GC, ALLOCATOR, SMART STACK, CGo, BACKENDS):**

*Target Subsystem: `src/runtime`, `src/cmd/compile/internal/ssa`, `src/cmd/internal/obj`. Rebase conflict rate: $$10\text{--}15\%$$.*

**RFC-301: Pluggable High-Performance GC Engine (`logos build -gc=<mode>`) & Tier-0 Arena:**
1. **Tier-0 Memory Subsystem (Active Across All GC Modes):**
   * **1.5ns Inline Noscan Bump Allocator:** For pointer-free objects (`noscan`), the compiler emits a 3-instruction bump allocation against `p.noscanBumpCur` protected by an Async-Preemption Restartable Sequence (`isAsyncSafePoint` PC range in `runtime/preempt.go` preventing `SIGURG` torn allocation).
   * **P-Leased Dual-Ended Side-Stack Arena (`spanAllocStack`):** Non-escaping dynamic allocations (`EscNone`, or `EscReturn` merged into caller's region) allocate on a 64KB chunk leased from `p.arenaPool` to `g.activeArena` while dirty (returned immediately on `gopark` when watermark == base).
     * *Bottom-Up Noscan Region:* Reclaimed in `0.5ns` by single register watermark pop (`loop-iteration` objects reset at every loop back-edge; function objects reset at epilogue and `runtime/panic.go` unwinder). Explicit `clear(b)` calls are marked `Volatile` in `ssa/dse.go` so cryptographic key wipes are never elided.
     * *Top-Down Scan Region:* Holds pointer-containing local structs; scanned as part of `scanstack(g)` and zeroed on watermark pop (`STP ZR, ZR`) to satisfy the Yuasa deletion barrier.
2. **Pluggable GC Modes:**
   * **`-gc=greentea` (Default — Enhanced Green Tea GC):** Eliminates mutator Mark Assist (zero tail-latency spikes on business goroutines). Uses Span-Centric sequential cache-local scanning, Sticky-Bit generational skipping for unmodified spans, and Elastic Background Worker Scaling (scaling dedicated GC worker threads when heap reaches 85% instead of halting request goroutines).
   * **`-gc=zgc` (Sub-50$$\mu\text{s}$$ Concurrent Compactor):** Uses 64-bit Colored Pointers (hardware ARM64 TBI / x86 LAM, falling back to 1-clock software `ANDQ` unmasking in load barriers on legacy x86 kernels—never using RSS-inflating multi-mapping) and untags pointers before syscalls/CGo. Pointer-keyed maps (`map[*T]V`) use immutable object-header identity hashes.
   * **`-gc=immix` (Maximum Throughput Evacuating GC):** Allocates into `32KB` blocks / `128B` lines with opportunistic defragmentation. Address-hashed objects set `blockPinned = 1`.
   * **`-gc=wasm` (Default on `GOARCH=wasm`):** Single-threaded incremental GC removing all atomic locks/barriers, pacing 1–2ms mark slices at host idle hooks, and splitting high-address scratch arenas from low-address permanent heap to prevent `memory.grow` bloat.
   * **`-gc=batch` / `-gc=epsilon`:** Completely omits SSA write barriers (`0ns` pointer store overhead). `batch` performs a parallel STW sweep only if `GOMEMLIMIT` reaches 90%; `epsilon` strips the GC runtime entirely for ultra-small serverless/WASM workers.

**RFC-302: Smart Stack Engine (`ssagen` & `runtime/stack.go`):**
* **Stack-First Dynamic Slices:** Non-escaping `make([]T, n)` where runtime `n <= 4KB` and `SP - n > g.stackguard0 + 1024` allocates directly on the hardware call stack (`SUBQ size, SP` in `0.5ns`). CGo arguments and `unsafe.Pointer` targets are excluded so `copystack` never invalidates external pointers.
* **Adaptive Goroutine Sizing:** Runtime tracks an 8-byte Exponential Moving Average (EMA) of stack high-water marks per `go fn()` PC and spawns subsequent goroutines at the pre-sized power-of-two frame (`newprocSized`), eliminating `morestack` copy overhead.

**RFC-303: Hardware Backend Extreme Optimizations (`x86_64`, `ARM64`, `WASM`):**
* **x86_64:** AVX-512 / AVX2 masked tail loops (`VMOVDQU8` with `BZHI` masks), 64-byte tail padding on all heap/arena/`.rodata` segments to prevent vector page-faults, BMI2 `SHLX`/`RORX`, branchless `CMOVcc` lowering, and `JO`-checked multiplication inside `make([]T, a * b)`.
* **ARM64:** Safe Peephole `LDP`/`STP` coalescing (restricted to 8-byte aligned offsets in `[-512, +504]`, distinct registers `Rt1 != Rt2 != Rn`, and no intervening labels/atomics), ARMv8.1 LSE atomics (`LDADDAL`, `CASAL`), and `128B` cache-line padding on runtime `P`/`M` structs for Apple Silicon.
* **Long-Loop Preemption:** Tight SIMD/scalar loops without calls inject a strip-mined preemption check (`CMPQ 16(R14), SP`) once every 4,096 iterations (`< 0.01%` overhead, zero goroutine starvation).
* **WASM Diet Profile:** Strips unreachable reflection/method tables, enforces 8-byte alignment on 64-bit atomics (`wasm32`), applies Selective Asyncify (only blocking functions get state machines), and emits a 1.5KB `.loader.mjs` that re-evaluates `wasmMemory.buffer` on every `wasmexport` access to prevent detached `ArrayBuffer` crashes.

**RFC-304: Fast-CGo Engine (`//logos:cgo_fastcall`):**
* Bypasses `entersyscall`/`exitsyscall` scheduler locks (~100ns → ~2ns). Switches `RSP` to the thread's 8MB `g0` system stack, swaps TLS `R14` to `g0` with `m.inFastCall = 1` (preventing false `split stack overflow` panics on `SIGPROF`/`SIGURG`), pins pointer arguments in `p.cgoPinnedPtrs[0..3]` (preventing `zgc`/`immix` relocation during the call), elides `_cgoCheckPointer` for proven POD/primitive slices, and enforces a 1ms `sysmon` P-retake watchdog.

---

**SECTION 4: CONCISE SUMMARY OF RED-TEAM FINDINGS & ARCHITECTURAL RESOLUTIONS (182 VECTORS):**

| Category | Key Adversarial Attack Vector (Red Team Finding) | Normative Architectural Resolution (Blue Team Fix) |
| :--- | :--- | :--- |
| **1. Memory & GC Safety** | • **Split-Union / Multi-Word Tearing:** Concurrent mutation of 16B `iface`/`slice` or union leaves torn pointers (UB).<br>• **Side-Stack Scan UAF:** Yuasa deletion barrier or concurrent `scanstack` reads stale/freed arena pointers.<br>• **Preemption Race in Bump Allocator:** `SIGURG` mid-sequence causes duplicate address allocation.<br>• **Moving GC vs Map Keys & CGo:** `zgc`/`immix` relocates `*T` map keys or active `cgo_fastcall` buffers.<br>• **ZGC Container OOMKilled:** Multi-mapping inflates Linux container RSS by $$3\times$$.<br>• **Mark Assist Removal OOM:** Mutators outpace background GC threads. | • Forbid unlocked multi-word sharing in `types2`; use 16B-aligned atomic slots in `sync.Hot` (#13, #148, #154).<br>• Split arena into bottom-up `Noscan` (0.5ns pop) and top-down `Scan` (zeroed on pop, scanned at safe-points); restore watermark on `panic` (#46, #63, #71, #81).<br>• Register 3-instruction bump allocator as Restartable Sequence in `isAsyncSafePoint` (#155).<br>• Pin `immix` blocks on address-hash; use identity hash in `zgc`; record CGo args in `p.cgoPinnedPtrs` (#159, #160).<br>• Replace ZGC multi-mapping with 1-clock software `ANDQ` unmasking on non-LAM x86 (#171).<br>• Scale dedicated background GC worker cores elastically at 85% heap (#161). |
| **2. Zero-Copy & `.rodata` Hazards** | • **Ghosted Slice Mutation:** `string(buf)` zero-copy mutated via surviving alias or `munmap` UAF.<br>• **PIE/ASLR `.rodata` Crash:** Constants with pointers segfault during dynamic relocation.<br>• **Crypto Key Wipe Elision:** `-O3` DSE deletes `clear(secretKey)` before function exit. | • Allow `[]byte`→`string` zero-copy only when `escape.go` proves `string` is non-escaping and source is unique; auto-clip subslices to `a[i:j:j]` (#2, #16, #83, #93, #152).<br>• Route pointer-containing `const` data to `.data.rel.ro` (RELRO); use per-map collision-free seeds (#3, #54, #73, #140).<br>• Mark explicit `clear()` as `Volatile` in `ssa/dse.go` (#149). |
| **3. Type System, Unions & Interfaces** | • **Value-Recursive Union Hang:** `type Expr interface { Add | Num }` causes infinite size calculation.<br>• **Interface Property Lost Update:** `c.Pos.Normalize()` or `party[i()].HP += 10` mutates temporary copy or double-evaluates receiver.<br>• **Spread Mass Assignment:** `Dto{..entity}` silently drops same-name fields with mismatched types. | • Represent pointer-containing, recursive, or $$>32\text{B}$$ unions as 16B `[ _tag | _ptr ]` interfaces on Side-Stack; zero-extend $$\le 32\text{B}$$ scalar unions (#82, #89, #97).<br>• Require pointer receivers for mutable interface fields; pin receiver once and emit Write-Back epilogue; ban `&c.HP` (#116, #117, #120, #132, #141).<br>• Emit immediate compile error on same-name type mismatch in `Target{..src}` (#119, #146). |
| **4. Comptime & `--emit-go` Fidelity** | • **Chicken-and-Egg `Resolve[f.Type]`:** `f.Type` fails typecheck before `const for` unrolling.<br>• **DI Circular Bomb:** `ServiceA` ↔ `ServiceB` hangs compiler stenciler.<br>• **`const switch T` Layout Crash:** Casting `[]*os.File` (8B) to `[]io.Writer` (16B) via `unsafe` segfaults.<br>• **`errdefer` vs `recover()`:** Using `recover()` corrupts panic stack traces; early `return` skips flag. | • Use 2-Phase Deferred Stenciling in `noder/stencil.go`; track `activeStencils` cycle stack and cap expansion at `4,096` AST nodes (#128, #129, #150).<br>• Rebind composite types (`[]T`) only on concrete type matches; use constraint intersection for interface cases (#45, #86, #91, #108).<br>• Inject `_errdefer_ok = true` before every non-error `return` AST node without `recover()`; forbid `defer` that zeroes `err` alongside `errdefer` (#77, #90, #96). |
| **5. Concurrency, Deadlock & Leaks** | • **`go group` Sibling Hang:** Sibling dies with error while peer blocks forever on `<-ch` or `conn.Read()`.<br>• **Half-Opened `.map()` FD Leak:** `paths.map(p => os.Open(p)?)` leaks FDs 0..2 when item 3 fails.<br>• **SIMD Tight-Loop Starvation:** Call-free vectorized loop blocks GC STW and preemption.<br>• **Timer Hoisting False Fire:** Reused `time.After` fires immediately from stale channel value. | • Auto-bind `<-_groupCtx.Done()` into group channel ops and interrupt blocked `netpoll` descriptors via `g.cancelChan`; support `go group (limit = N)` (#68, #136, #151, #178).<br>• Auto-inject reverse `.Close()` loop on error exit when `.map()` yields `io.Closer` (#142).<br>• Inject strip-mined preemption check every 4,096 loop iterations (#137).<br>• Use synchronous channel drain on hoisted `Timer.Reset` (#170, #172). |
| **6. Hardware, OS & Ecosystem** | • **ARM64 `LDP`/`STP` SIGILL:** Out-of-range offsets or aliased registers crash decoder.<br>• **Fast-CGo Signal Panic:** `SIGPROF` on `g0` stack with `_Grunning` TLS crashes unwinder.<br>• **JSON 10GB Prealloc Bomb:** Malicious length header triggers instant OOM.<br>• **Non-ASCII Extension Proxy Ban:** `proxy.golang.org` rejects `.ʕ◔ϖ◔ʔ` / `.📜` paths. | • Restrict `LDP`/`STP` fusion to `[-512, +504]` with strict register non-aliasing (#111).<br>• Swap `R14` TLS to `g0` and set `m.inFastCall = 1` during `cgo_fastcall` (#165, #173).<br>• Cap static deserializer pre-allocation at `min(remainingInputBytes, 1MB)` and check `make` math overflow via `JO` (#139, #169).<br>• Strip `U+FE0F` in loader; emit ASCII `.gen.go` files for public module publishing (#122, #133). |

---

**SECTION 5: LEXER, PARSER, AND AST NODE SPECIFICATION (`src/cmd/compile/internal/syntax`):**

**5.1. Token Table Additions (`syntax/tokens.go`):**
Zero new globally reserved keywords are added to the keyword hash table. The scanner (`syntax/scanner.go`) adds exactly 5 punctuation/literal tokens and 2 contextual statement identifiers:

| Token Constant | Lexical Form | Scanner State Transition (`scanner.next()`) |
| :--- | :--- | :--- |
| `_Question` | `?` | Triggered on `?` when next char is neither `.` nor `?`. Suppresses automatic semicolon insertion (`nlsemi = true` if postfix). |
| `_QuestionDot` | `?.` | Triggered on `?` followed immediately by `.`. Sets `nlsemi = false`. |
| `_QuestionQuestion` | `??` | Triggered on `?` followed immediately by `?`. Binary operator precedence `1` (below `||`). |
| `_Arrow` | `=>` | Triggered on `=` followed immediately by `>`. Used in single-expression lambdas and `switch` value cases. |
| `_InterpStringLit` | `$"..."` | Triggered on `$` followed immediately by `"`. Lexes balanced `{ expr }` sub-token streams inside string spans. |

**5.2. Concrete AST Struct Definitions (`syntax/nodes_logos.go`):**

```go
// 1. Postfix Error Propagation: expr?
type TryExpr struct {
    X expr
    expr
}

// 2. Optional Chaining & Coalescing: a?.b, a ?? b
type OptSelectorExpr struct {
    X   expr
    Sel *Name
    expr
}

// 3. Error-Returning & Error-Only Defer: defer fn()?, errdefer fn()
type DeferStmt struct {
    Call      expr
    Propagate bool // true for 'defer fn()?'
    ErrOnly   bool // true for 'errdefer fn()'
    ScopeOnly bool // true for 'scope defer fn()'
    stmt
}

// 4. Compile-Time Control Flow: const if, const for, const switch
type ConstIfStmt struct {
    Cond expr
    Then *BlockStmt
    Else stmt // *ConstIfStmt or *BlockStmt or nil
    stmt
}

type ConstForStmt struct {
    Index *Name // optional '_' or ident
    Field *Name // loop variable 'f'
    X     expr  // comptime.Fields[T]()
    Body  *BlockStmt
    stmt
}

type ConstSwitchStmt struct {
    Targ   expr // Generic type parameter identifier 'T'
    Cases  []*ConstCaseClause
    stmt
}

// 5. Compile-Time Field Projection: target.[f]
type FieldProjExpr struct {
    X     expr
    Field *Name
    expr
}

// 6. Interface Structural Property Declaration: const Name string; HP int
type FieldContractDecl struct {
    IsConst bool
    Name    *Name
    Type    expr
    decl
}
```

**5.3. LL(1) Non-Backtracking Disambiguation Algorithms (`syntax/parser.go`):**
* **Rule A (`ParenExpr` vs `TupleType` vs `ArrowLambda`):**
  When the parser encounters `(`, it calls `p.parseParenOrTupleOrLambda()` with zero token backtracking:
  1. Parse the first element using `p.exprOrType()`.
  2. If the next token is `:`, parse as a **Named Tuple Type** `(name1 Type1, name2 Type2)`.
  3. If the next token is `,`, parse the comma-separated list into `elems []Expr`. After `) `, inspect `p.tok`:
     * If `p.tok == _Arrow` (`=>`), validate that all `elems` are `*Name` identifiers and return `*ArrowFuncLit{Params: elems, Body: p.expr()}`.
     * Otherwise, return `*TupleLit{Elems: elems}` (or `*TupleType` in type context).
  4. If `len(elems) == 1` and `p.tok == _Arrow`, return `*ArrowFuncLit`. Otherwise return standard `*ParenExpr`.
* **Rule B (`errdefer` Contextual Parsing in `p.stmtOrDecl()`):**
  When `p.tok == _Name && p.lit == "errdefer"`, the parser peeks 1 token ahead (`p.peek()`). If the next token starts an expression (`_Name`, `_Lparen`, etc.) rather than `:=`, `=`, `,`, or `:`, it consumes `errdefer` and returns `&DeferStmt{Call: p.expr(), ErrOnly: true}`. Thus existing Go code `errdefer := 1` parses as a standard assignment.

---

**SECTION 6: TYPE CHECKER, UNIFIED `.a` EXPORT HEADER, AND ABI LAYOUT (`types2` & `types`):**

**6.1. Unified IR `.a` Export Summary Bitfield (`src/cmd/compile/internal/noder/linker.go`):**
To maintain $$O(1)$$ separate compilation without re-parsing imported function bodies (Resolution 44), each exported function/method in the `.a` package export header appends a 1-byte `LogosFuncSummary` bitmask:

| Bit Index | Flag Constant | Meaning & Verification Rule |
| :--- | :--- | :--- |
| `Bit 0` | `SumReturnsNonNil` | `1` if every `return` path in CFG returns a provably non-nil pointer (`&T{}`, `new(T)`). Allows `.lgo` callers to treat `.go` return type as `*T` instead of `*T?`. |
| `Bit 1` | `SumAlwaysNilError` | `1` if the trailing `error` return slot is constant `nil` on all return paths (e.g., `bytes.Buffer.Write`). Exempts callers from `slop::discarded_error`. |
| `Bit 2` | `SumIsPureConst` | `1` if function satisfies `const func` purity constraints (no I/O, no global reads, deterministic). Enables cross-package CTFE. |
| `Bit 3` | `SumReceiverMutated` | `1` if method modifies any field through its receiver pointer or passes receiver to a non-`const` parameter. Blocks calls on `const *T` receivers. |
| `Bit 4..7` | `SumParamMutatedMask` | 4-bit mask for parameters `0..3`: `1` if parameter slice/pointer/map is mutated. Allows passing `.rodata` slices to read-only legacy `.go` functions without `.clone()`. |

**6.2. Exact Byte-Layout Algorithms for `T?` and Union Interfaces (`src/cmd/compile/internal/types/size.go`):**

1. **Option Type `T?` (`interface { T | nil }`):**
   $$\text{Layout}(T?) = \begin{cases} \text{sizeof}(T) = 8\text{B}, \text{Niche} = \text{0x0} & \text{if } T \in \{*U, \text{func}, \text{chan}, \text{unsafe.Pointer}\} \land \neg\text{IsOption}(U) \\ \text{struct}\{ \text{val } T; \text{present bool} \} & \text{otherwise (scalars, structs, slices, nested } *U?\text{)} \end{cases}$$

2. **Closed Union Interface `type U interface { V_1 | V_2 | ... | V_k }`:**
   * **Tarjan SCC Recursion Detection:** Build directed type dependency graph $$G = (V, E)$$ where edge $$A \to B$$ exists if $$A$$ contains $$B$$ by value (not behind `*B`, `[]B`, `map`).
   * **Case 1 (Pure-Scalar Non-Recursive Small Union):** If ` !inCycle(U)` AND $$\forall i, \text{PtrBytes}(V_i) == 0$$ AND $$\max_i \text{sizeof}(V_i) \le 24\text{B}$$:
     ```
     +------------------------+------------------------------------------------+
     | _tag: uint64 (8 Bytes) | _payload: [max_align_pad]byte (8..24 Bytes)    |
     +------------------------+------------------------------------------------+
     ```
     * **Canonical Zero-Extension Invariant (Resolutions 89, 97):** Whenever variant $$V_i$$ is written to `U`, `ssa` emits a full 8-byte zero-initializing store to all payload words (`MOVQ $0, 8(SP)` / ARM64 `STP ZR, ZR`) or constructs the payload in a zeroed register prior to storing. Consequently, `u1 == u2` compiles to a branchless 2-word `CMPQ` without tag switching.
   * **Case 2 (Pointer-Containing, Recursive, or Large Union `> 24B`):**
     ```
     +-----------------------------+-------------------------------------------+
     | _type: *abi.Type (8 Bytes)  | _data: unsafe.Pointer (8 Bytes)           |
     +-----------------------------+-------------------------------------------+
     ```
     * Identical to standard Go `eface` (16 bytes, `PtrBytes = 16`).
     * `nil` variant is canonicalized to `(_type = 0x0, _data = 0x0)`. Because `*T` variants are strictly Non-Nil (Resolution 115), `(_type = *T, _data = 0x0)` is unreachable at runtime.

**6.3. Interface Structural Property Lowering (`noder/Properties`):**
Given `type Creature interface { const Name string; HP int; Start() error }`:
1. **Internal Method Set Expansion:** `types2` expands `Creature`'s canonical method set in alphabetical order:
   * `Start() error`
   * `_logos_get_HP() int`
   * `_logos_get_Name() string`
   * `_logos_set_HP(int)`
2. **Satisfaction & `itab` Generation (`reflectdata/alg.go`):**
   When `*Monster` (`type Monster struct { Name string; HP int }`) is converted to `Creature`:
   * `Monster`'s exported method table (`abi.Type`) is **NOT** modified (preserving `reflect.NumMethod()`).
   * Instead, `itab` construction synthesizes two 3-instruction field-offset thunks:
     ```nasm
     ; _logos_get_Monster_HP(rcvr *Monster) int  (Assuming HP is at offset +16)
     MOVQ 16(AX), AX
     RET

     ; _logos_set_Monster_HP(rcvr *Monster, val int)
     MOVQ BX, 16(AX)
     RET
     ```
   * When `Attack(c Creature)` is auto-generified (`-O2`/`-O3`), `c.HP -= 10` bypasses the `itab` thunk entirely and emits `SUBQ $10, 16(AX)` (`0ns`).

---

**SECTION 7: EXACT `--emit-go` TRANSPILATION LOWERING TEMPLATES (`src/cmd/compile/internal/eject`):**

Every `loGo/s` construct has a deterministic 1:1 pure-Go AST lowering rule:

**7.1. `errdefer` + `defer fn()?` + `?` Propagation Lowering:**
* **Input (`transaction.lgo`):**
  ```go
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
* **Emitted Pure Go (`transaction.gen.go`):**
  ```go
  //go:build !logos
  package bank

  import _logos_errors "errors"

  func Transfer(db *DB, from int64, to int64, amt int64) (_ret0 int64, _retErr error) {
      _errdefer_ok := false
      tx, _err0 := db.Begin()
      if _err0 != nil {
          return _ret0, _err0
      }
      defer func() {
          if !_errdefer_ok || (_retErr != nil) {
              if _rbErr := tx.Rollback(); _rbErr != nil {
                  if _retErr == nil {
                      _retErr = _rbErr
                  } else {
                      _retErr = _logos_errors.Join(_retErr, _rbErr)
                  }
              }
          }
      }()
      defer func() {
          if _clErr := tx.Close(); _clErr != nil {
              if _retErr == nil {
                  _retErr = _clErr
              } else {
                  _retErr = _logos_errors.Join(_retErr, _clErr)
              }
          }
      }()
      bal, _err1 := tx.AddBalance(from, -amt)
      if _err1 != nil {
          return _ret0, _err1
      }
      _, _err2 := tx.AddBalance(to, amt)
      if _err2 != nil {
          return _ret0, _err2
      }
      if _err3 := tx.Commit(); _err3 != nil {
          return _ret0, _err3
      }
      _errdefer_ok = true
      return bal, nil
  }
  ```

**7.2. `const switch T` Generic Type Switch Lowering:**
* **Input (`codec.lgo`):**
  ```go
  func EncodeSlice[T any](buf []byte, items const []T) ([]byte, error) {
      const switch T {
      case int64:
          for _, v := range items {
              buf = strconv.AppendInt(buf, v, 10)
          }
          return buf, nil
      case io.WriterTo:
          // Constraint intersection: T keeps concrete layout, gains WriteTo
          var b bytes.Buffer
          for _, v := range items {
              v.WriteTo(&b)?
          }
          return append(buf, b.Bytes()...), nil
      }
  }
  ```
* **Emitted Pure Go (`codec.gen.go`):**
  ```go
  //go:build !logos
  package codec

  import (
      "bytes"
      "io"
      "strconv"
      _logos_unsafe "unsafe"
  )

  func EncodeSlice[T any](buf []byte, items []T) ([]byte, error) {
      switch any((*T)(nil)).(type) {
      case *int64:
          _items := *(*[]int64)(_logos_unsafe.Pointer(&items))
          for _, v := range _items {
              buf = strconv.AppendInt(buf, v, 10)
          }
          return buf, nil
      default:
          if _, _ok := any(*new(T)).(io.WriterTo); _ok {
              var b bytes.Buffer
              for _, v := range items {
                  _, _err0 := any(v).(io.WriterTo).WriteTo(&b)
                  if _err0 != nil {
                      return nil, _err0
                  }
              }
              return append(buf, b.Bytes()...), nil
          }
          return buf, nil
      }
  }
  ```

---

**SECTION 8: TIER-3 RUNTIME MEMORY SUBSYSTEM & GC INTERNALS (`src/runtime`):**

**8.1. `g` and `p` Struct Additions (`src/runtime/runtime2.go`):**

```go
type g struct {
    // ... existing upstream fields ...
    activeArena   *sideStackChunk // Leased 64KB chunk while dirty; nil when parked at base
    cancelChan    uintptr         // Pointer to enclosing 'go group' cancellation channel
    lastErrData    unsafe.Pointer  // Tracks error identity for zero-alloc ring-buffer trace
    errTracePC    [8]uintptr      // 64B inline error propagation PC ring buffer
    errTraceLen   uint8
}

type p struct {
    // ... existing upstream fields ...
    noscanBumpCur uintptr         // Current cursor in 64KB P-local noscan span
    noscanBumpEnd uintptr         // End limit of P-local noscan span
    arenaPool     *sideStackChunk // Lock-free freelist of clean 64KB side-stack chunks
    cgoPinnedPtrs [4]uintptr      // Pages pinned during active //logos:cgo_fastcall
}

type sideStackChunk struct {
    next       *sideStackChunk
    noscanBase uintptr // Bottom of 64KB chunk (grows UP: noscanCur += size)
    noscanCur  uintptr
    scanCur    uintptr // Top of 64KB chunk (grows DOWN: scanCur -= size)
    scanBase   uintptr // Base + 65536
}
```

**8.2. 1.5ns Inline Noscan Bump Allocator Assembly & `SIGURG` RSEQ Protection (`ssa/gen/AMD64.rules`):**
For `new(T)` or `make([]byte, K)` where `T` is `noscan` and `size <= 512`:
```nasm
; Start of Async-Preemption Restartable Sequence (Registered in pclntab)
.L_bump_start:
    MOVQ    g_m(R14), R12                ; R12 = g.m
    MOVQ    m_p(R12), R12                ; R12 = g.m.p
    MOVQ    p_noscanBumpCur(R12), AX     ; AX = old cursor (allocated object address)
    LEAQ    <size>(AX), BX               ; BX = new cursor
    CMPQ    BX, p_noscanBumpEnd(R12)     ; Check span exhaustion
    JA      .L_bump_slowpath
    MOVQ    BX, p_noscanBumpCur(R12)     ; Commit cursor (Single instruction commit point)
.L_bump_end:
```
* **`SIGURG` Handler Rule (`runtime/preempt.go`):** If a thread receives `SIGURG` with `RIP` in `[.L_bump_start, .L_bump_end)`, the signal handler resets `ctxt.Set_rip(.L_bump_start)` before injecting `asyncPreempt`. Because no memory has been modified prior to the final `MOVQ BX, p_noscanBumpCur(R12)`, rewinding `RIP` to `.L_bump_start` is 100% idempotent and eliminates double-allocation races without atomic `LOCK` instructions.

**8.3. Dual-Ended Side-Stack Arena Lifecycle (`runtime/arena_logos.go`):**
1. **Invariant:** `noscanCur <= scanCur`. If an allocation causes `noscanCur > scanCur`, the runtime links an overflow chunk or falls back to `mallocgc`.
2. **Loop-Back-Edge Reset (Resolution 181):** For allocations whose SSA liveness does not cross a loop back-edge, the compiler records `_loopNoscanMark := g.activeArena.noscanCur` at the loop header and restores `g.activeArena.noscanCur = _loopNoscanMark` (`MOVQ` 1 instruction) at the end of every iteration.
3. **Function Exit & Scan Zeroing (Resolution 71, 81):**
   ```nasm
   ; Epilogue for functions using Side-Stack Scan region (e.g., 32 bytes used):
   ADDQ    $32, g_activeArena_scanCur(R14)  ; 1. Raise scanCur watermark FIRST (atomic wrt suspendG)
   VMOVDQU Y0, -32(RBX)                     ; 2. Zero popped 32B region (Y0 = 0)
   ```
4. **Parking Release (Resolution 4, 11):** Inside `runtime.gopark()`, if `gp.activeArena != nil && gp.activeArena.noscanCur == gp.activeArena.noscanBase && gp.activeArena.scanCur == gp.activeArena.scanBase`, the chunk is immediately pushed back to `gp.m.p.ptr().arenaPool` and `gp.activeArena = nil`. Idle parked goroutines consume **0 bytes** of arena memory.

**8.4. Pluggable GC Write/Load Barrier Dispatch (`src/cmd/compile/internal/ssagen/ssa.go`):**
At compile time, `-gc=<mode>` selects the SSA barrier emitter:

| GC Mode | Pointer Store (`*slot = ptr`) | Pointer Load (`ptr = *slot`) | Map `map[*T]V` Hashing |
| :--- | :--- | :--- | :--- |
| **`greentea` (Default)** | Hybrid Yuasa + 1-byte Card Table (`MOVB $1, (addr>>9)+cardBase`) | Plain `MOVQ (AX), BX` (`0ns`) | Standard `memhash` (Non-moving) |
| **`zgc`** | Plain `MOVQ BX, (AX)` | `MOVQ (AX), BX` + `TESTQ BX, R15_BadMask` + `JNZ zgcLoadSlow` | Object Header Identity Hash (`32-bit`) |
| **`immix`** | Coalesced Line-Mark Barrier | Plain `MOVQ (AX), BX` (`0ns`) | Sets `blockPinned = 1` on first hash |
| **`wasm`** | Plain `i32.store` (Zero atomics) | Plain `i32.load` (`0ns`) | Standard `memhash32` |
| **`batch` / `epsilon`** | Plain `MOVQ BX, (AX)` (**0 barriers**) | Plain `MOVQ (AX), BX` (`0ns`) | Standard `memhash` |

**8.5. `//logos:cgo_fastcall` Exact Assembly Trampoline (`src/runtime/sys_linux_amd64.s`):**
```nasm
TEXT runtime·logosFastCgoCall(SB), NOSPLIT, $0-24
    MOVQ    R14, R13                      ; R13 = current g
    MOVQ    g_m(R13), R12                 ; R12 = g.m
    MOVQ    m_p(R12), R11                 ; R11 = g.m.p

    ; 1. Pin pointer arg0 in P-local slot to block ZGC/Immix evacuation (0.3ns)
    MOVQ    DI, p_cgoPinnedPtrs+0(R11)

    ; 2. Record fastcall timestamp for sysmon 1ms watchdog
    MOVQ    runtime·nanotime1(SB), AX     ; (or cached coarse tick)
    MOVQ    AX, p_fastCallEnterNano(R11)

    ; 3. Switch RSP and R14 TLS to g0 system stack (prevents SIGPROF/SIGURG crash)
    MOVQ    SP, (g_sched+gobuf_sp)(R13)   ; Save g stack pointer
    MOVQ    m_g0(R12), R14                ; Swap TLS goroutine register to g0!
    MOVQ    (g_sched+gobuf_sp)(R14), SP   ; Switch to 8MB OS thread stack
    MOVL    $1, m_inFastCall(R12)

    ; 4. Direct C function call (System V AMD64 ABI args already in DI, SI, DX, CX)
    CALL    AX

    ; 5. Restore g stack, TLS R14, and unpin pointers
    MOVL    $0, m_inFastCall(R12)
    MOVQ    $0, p_fastCallEnterNano(R11)
    MOVQ    $0, p_cgoPinnedPtrs+0(R11)
    MOVQ    R13, R14                      ; Restore TLS g
    MOVQ    (g_sched+gobuf_sp)(R13), SP   ; Restore g stack pointer
    RET
```

---

**SECTION 9: `.rodata` COMPILER SERIALIZATION & PERFECT HASH TABLE (`reflectdata/rodata_logos.go`):**

**9.1. Collision-Free `const map[K]V` Layout in `.rodata` (Resolution 73, 140):**
When compiling `const mimeMap = map[string]string{ ... }` with $$N > 8$$ entries:
1. The compiler allocates a bucket array of size $$M = 2^{\lceil \log_2(1.25 N) \rceil}$$.
2. During compilation, a deterministic PRNG searches for a 64-bit seed $$S$$ (or a 2-level displacement table `disp [M/4]uint8`) such that every key $$k_i$$ maps to a bucket with probe distance $$\le 1$$ (16-way SIMD SwissTable group).
3. The entire bucket structure, key string headers, and value array are written as an immutable `obj.RODATA` (or `.data.rel.ro` if values contain pointers) symbol with `h.flags = flagReadOnlyMap`.
4. At call sites `v := mimeMap[reqExt]`, the compiler inlines a 6-instruction lookup without calling `runtime.mapaccess1_faststr` (`~2.5ns` vs `~14ns`).

---

**SECTION 10: `bettergo.toml` CANONICAL CONFIGURATION SCHEMA:**

```toml
[project]
name = "order-engine"
edition = "2026"
strict_files = [".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"]

[build]
opt_level = "O2"                     # "O0" | "O1" | "O2" | "O3" | "Os" | "Oz"
gc = "greentea"                      # "greentea" | "zgc" | "immix" | "wasm" | "batch" | "epsilon"
synthetic_pgo = "standard"           # "off" | "standard" | "aggressive"
target_cpu = "deterministic"         # "deterministic" (v3/v8.2) | "native"
cgo = "restricted"                   # "forbid" | "restricted" | "allow"
cgo_allowlist = [
    "github.com/duckdb/duckdb-go/**",
]

[capabilities]
# Enforced across transitive third-party dependencies via cmd/link reachability audit
direct_reflect = "deny"
transitive_reflect = "warn"
reflect_allowlist = ["google.golang.org/protobuf/**"]
transitive_unsafe_allowlist = ["golang.org/x/sys/**", "github.com/klauspost/compress/**"]
transitive_exec_allowlist = []

[lints]
correctness = "forbid"               # Cannot be bypassed with //logos:allow
concurrency = "forbid"
slop = "deny"
complexity = "deny"
perf = "warn"
style = "warn"
pedantic = "allow"                   # Opt-in via --pedantic

[lints.thresholds]
max_nesting_depth = 3
max_cognitive_complexity = 15
max_function_lines = 60
max_function_params = 4
max_comptime_ast_nodes = 4096
max_ctfe_steps = 100000
```

---

**SECTION 11: END-TO-END SHOWCASE ("GO-SPRING" + EXTREME PERFORMANCE IN `.lgo`):**

```go
// File: order_service.lgo (or order_service.⚡)
package order

import (
    "comptime"
    "context"
    "log/slog"
    "sync"
)

// 1. Interface with Structural Field Contract (Zero Getter/Setter Boilerplate)
type Auditable interface {
    const ID int64
    const Author string
    Validate() error
}

// 2. Closed Union Interface (16B Zero-Heap Side-Stack Sum Type)
type OrderResult interface {
    Filled   struct { TxID string; Price int64 } |
    Rejected struct { Reason string; Code int } |
    Pending
}
type Pending struct{}

// 3. Compile-Time .rodata Map (0ns init, 0% GC scan, O(1) Perfect Hash)
const feeTierBps = map[string]int64{
    "VIP":      2,
    "STANDARD": 10,
    "GUEST":    25,
}

// 4. Struct with Default Values & Inline Interface Assertion
type CreateOrderReq struct : Auditable {
    ID       int64
    Author   string
    Symbol   string
    Qty      int64  = 1
    Tier     string = "STANDARD"
    PromoTag string? // Option type (string | nil)
}

func (r const *CreateOrderReq) Validate() error {
    if r.Qty <= 0 {
        return errors.New($"invalid quantity {r.Qty} for order {r.ID}")
    }
    return nil
}

// 5.Compile-Time DI Autowiring ("Go-Spring" in 8 lines, 0 runtime reflection)
func Autowire[T any](c *Container) *T {
    svc := new(T)
    const for _, f := range comptime.Fields[T]() {
        svc.[f] = Resolve[f.Type](c)
    }
    return svc
}

type OrderService struct {
    db     *Database
    logger *slog.Logger
    ledger sync.Locked[map[int64]OrderResult]
    config sync.Hot[RiskConfig] // 0ns RCU read pointer
}

// 6. Fearless Concurrency, errdefer, ?, and Zero-Alloc Stream Fusion
func (s *OrderService) ProcessBatch(ctx context.Context, batch const []CreateOrderReq) ([]string, error) {
    // Validate all requests using interface lowering (monomorphized, 0ns virtual call)
    for i := range batch {
        batch[i].Validate()?
    }

    tx := s.db.Begin(ctx)?
    errdefer tx.Rollback()? // Executes ONLY on error return or panic unwinding
    defer tx.Close()?       // Executes always, merges Close error via errors.Join

    // Concurrent bounded processing (cancels siblings & unblocks netpoll on first error)
    go group (ctx, limit = 8) {
        for _, req := range batch {
            go {
                bps := feeTierBps[req.Tier] ?? 25
                price := tx.ExecuteTrade(req.Symbol, req.Qty, bps)?

                // Thread-safe locked map update (panic-safe zero-cost unlock)
                s.ledger.WithLock(m => {
                    m[req.ID] = Filled{TxID: $"TX-{req.ID}", Price: price}
                })
            }
        }
    }

    tx.Commit()?

    // Zero-alloc fused single loop: filter -> map -> 2-Pass exact string interpolation
    receipts := batch
        .filter(r => r.Qty >= 10)
        .map(r => $"Order #{r.ID} by {r.Author} ({r.PromoTag ?? "NO-PROMO"})")

    return receipts, nil
}
```

---

**SECTION 12: COMPILE-TIME 2-PASS JSON / BINARY SERIALIZER ENGINE (`src/cmd/compile/internal/noder/izer.go`):**

When `json.Marshal(v)` or `json.Unmarshal(data, &v)` is called with a concrete struct or union type `T` (or inside a generic function instantiated via `const for`), the compiler intercepts the standard `encoding/json` reflection path and lowers the call to a statically unrolled 2-pass function `runtime_logos.MarshalStatic[T](v)`.

**12.1. Exact 2-Pass Static Serialization Algorithm (`MarshalStatic[T]`):**

Given `type User struct { ID int64 `json:"id"`; Name string `json:"name"`; Role string? `json:"role,omitempty"` }`:

1. **Compile-Time Constant Folding of JSON Keys:**
   During `const for` unrolling over `comptime.Fields[User]()`, the compiler pre-computes the exact byte literals for keys and punctuation:
   * Prefix 0: `{"id":` (6 bytes)
   * Prefix 1: `,"name":"` (9 bytes) + Suffix `"` (1 byte)
   * Prefix 2 (Optional): `,"role":"` (9 bytes) + Suffix `"` (1 byte)
   * Closing brace: `}` (1 byte)
   * Sum of fixed bytes = $$6 + 9 + 1 + 1 = 17\text{ bytes}$$.

2. **Generated IR Lowering (Zero `growslice`, Zero `reflect`):**
   ```go
   func MarshalStatic_User(u const *User) ([]byte, error) {
       // PASS 0: Pin string headers into local registers once (prevents TOCTOU race #64)
       _name := u.Name
       _role := u.Role

       // PASS 1: Exact Upper-Bound / Exact Byte Length Calculation (4 CPU clocks)
       // runtime.Int64StringLen(x) uses BSR/LZCNT + lookup table (1.2ns)
       _exactLen := 17 + runtime.Int64StringLen(u.ID) + runtime.JsonEscapedLen(_name)
       if _role != nil {
           _exactLen += 10 + runtime.JsonEscapedLen(*_role)
       }

       // PASS 2: Single Allocation (Stack-First if <= 4KB, or Caller RVO Buffer)
       _buf := make([]byte, 0, _exactLen)
       _buf = append(_buf, `{"id":`...)
       _buf = strconv.AppendInt(_buf, u.ID, 10)
       _buf = append(_buf, `,"name":"`...)
       _buf = runtime.AppendJsonEscaped(_buf, _name)
       _buf = append(_buf, '"')
       if _role != nil {
           _buf = append(_buf, `,"role":"`...)
           _buf = runtime.AppendJsonEscaped(_buf, *_role)
           _buf = append(_buf, '"')
       }
       _buf = append(_buf, '}')
       return _buf, nil
   }
   ```
   * **SIMD Escape Scanning (`runtime.JsonEscapedLen`):** Uses AVX2 `VPCMPEQB` / ARM64 NEON `CMEQ` to test 32 bytes at a time for `"`, `\`, or control chars `< 0x20`. If the bitmask is `0` (99.9% of ASCII strings), `AppendJsonEscaped` executes a raw `memmove` without byte-by-byte branching.

**12.2. Static Deserialization Security & Non-Nil Enforcement (`UnmarshalStatic[T]`):**
1. **Default Initialization:** Prior to parsing, the target struct `dst` is initialized with its declared field defaults (e.g., `dst.Port = 8080`, `dst.TLS = true`, Resolution 123).
2. **Required Bitmask Tracking (Resolution 7):**
   Every struct with $K$ Non-Nil fields (`*T` or non-optional fields without default values) allocates a local bitmask `var _seenMask uint64 = 0`.
   * As each JSON key is matched via a compile-time Perfect Hash / prefix trie, its corresponding bit `_seenMask |= (1 << fieldIdx)` is set.
   * At the closing `}`, if `(_seenMask & _requiredMask) != _requiredMask`, the parser returns `json.ErrMissingRequiredField(fieldName)` without requiring third-party `validate:"required"` reflection tags.
3. **Allocation Bomb Guard (Resolution 139):**
   When decoding a JSON array `[...]` into `[]Elem`, initial slice capacity is strictly clamped to:
   $$\text{cap}_{\text{init}} = \min\left(\left\lfloor\frac{\text{len(remainingInput)}}{2}\right\rfloor, \left\lfloor\frac{1,048,576}{\text{sizeof(Elem)}}\right\rfloor\right)$$

---

**SECTION 13: LINKER REACHABILITY & CAPABILITIES AUDITOR (`src/cmd/link/internal/ld/deadcode.go`):**

To prevent malicious or bloated transitive dependencies from secretly invoking `reflect`, `unsafe`, `cgo`, `os/exec`, or `net` (Resolution 25, 162), the linker integrates a zero-cost call-graph audit directly into the Dead-Code Elimination (`deadcode`) mark phase.

**13.1. Protected Runtime / Stdlib Sink Symbols:**
```go
var capabilitySinks = map[string]CapabilityKind{
    // Reflection Sinks
    "reflect.ValueOf":               CapReflect,
    "reflect.TypeOf":                CapReflect,
    "reflect.New":                   CapReflect,
    // Subprocess / Execution Sinks
    "syscall.Exec":                  CapExec,
    "syscall.ForkExec":              CapExec,
    "os.startProcess":               CapExec,
    // Network Sinks
    "net.sysSocket":                 CapNet,
    // CGo Sinks
    "runtime.cgocall":               CapCgo,
    "runtime.logosFastCgoCall":      CapCgo,
}
```

**13.2. Reachability Backtrace Algorithm:**
1. During `deadcode(ctxt)`, the linker traverses symbol relocations starting from `main.main` and ` main.init`, recording `parentSym[childIdx] = callerIdx`.
2. Whenever a `childIdx` matches a symbol in `capabilitySinks`, the linker walks `parentSym` backward to find the first package outside `GOROOT/src/` (the originating package `P`).
3. If `P` is not permitted by `[capabilities]` in `bettergo.toml`, linking aborts with the exact shortest call chain:
   ```
   ERROR [capabilities::unauthorized_exec]: Transitive dependency 'github.com/evil/logger' reaches 'os.startProcess'
     --> Reachability Call Chain (cmd/link deadcode audit):
      1. main.main()                              at cmd/server/main.lgo:18
      2.   -> github.com/evil/logger.Init()       at pkg/mod/github.com/evil/logger@v1.0.2/init.go:44
      3.   -> os/exec.(*Cmd).Start()              at GOROOT/src/os/exec/exec.go:692
      4.   -> os.startProcess()                   at GOROOT/src/os/exec_posix.go:54
     = Help: Remove the dependency or explicitly authorize "github.com/evil/logger/**" under 'transitive_exec_allowlist' in bettergo.toml.
   ```

---

**SECTION 14: WASM NATIVE PROFILE & SELECTIVE ASYNCIFY (`GOOS=js` / `GOOS=wasip1`, `GOARCH=wasm`):**

When targeting WebAssembly, the compiler and linker automatically activate the WASM Native Profile without requiring external tools like TinyGo.

**14.1. Selective Asyncify Call-Graph Partitioning (`src/cmd/internal/obj/wasm/wasmobj.go`):**
Standard Go compiles **every** function into a giant `loop` + `br_table` state machine so any function can yield its goroutine stack, inflating binary size by $$3\times$$ and slowing execution by $$4\times$$.
`loGo/s` partitions all functions into two sets on the linker call graph:
* **Color 0 (`SyncPure` — ~90% of functions):** Functions that never transitively call `runtime.gopark`, channel operations, `time.Sleep`, or blocking `wasmimport` functions.
  * *Codegen:* Emitted as pure, native WASM functions with zero `br_table` loop wrappers, using hardware WASM registers (`local.get` / `local.set`) and native WASM `call`.
* **Color 1 (`MayYield` — ~10% of functions):** Functions that can block or yield.
  * *Codegen:* Emitted with lightweight stack-saving state machines (or WASM Stack-Switching `cont.new` / `resume` when `--wasm-features=stack-switching` is enabled).

**14.2. Zero-Copy Host FFI (`//logos:wasmexport` & `//logos:wasmimport`):**
1. **`wasm32` Pointer Model & 64-Bit Alignment (Resolution 78):**
   Pointers and slice fields (`ptr`, `len`, `cap`) are 32-bit (`12B` slice header, `8B` string header). However, `types/size.go` forces `alignof(int64) = 8`, `alignof(uint64) = 8`, and `alignof(float64) = 8` so 64-bit atomic instructions (`i64.atomic.rmw.add`) never trap on unaligned addresses.
2. **Detached `ArrayBuffer` Protection in Auto-Generated `.loader.mjs` (Resolution 27):**
   ```javascript
   // Auto-generated app.loader.mjs (1.4 KB)
   export async function instantiateLogos(wasmBytes, imports = {}) {
     let wasmExports;
     const getMem = () => new Uint8Array(wasmExports.memory.buffer); // Always fresh after memory.grow!
     const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
     wasmExports = instance.exports;
     return {
       processImage(pixelsUint8, brightness) {
         const ptr = wasmExports._logos_scratch_alloc(pixelsUint8.byteLength);
         getMem().set(pixelsUint8, ptr); // Re-fetches buffer AFTER potential memory.grow inside alloc!
         const errCode = wasmExports.processImage(ptr, pixelsUint8.byteLength, brightness);
         wasmExports._logos_scratch_reset(ptr);
         if (errCode !== 0) throw new Error(readErrorString(wasmExports, getMem));
       }
     };
   }
   ```

---

**SECTION 15: HARDWARE BACKEND PEEPHOLE & SIMD PASSES (`src/cmd/compile/internal/ssa`):**

**15.1. ARM64 Safe `LDP` / `STP` Coalescing Pass (`src/cmd/internal/obj/arm64/ldpstp_logos.go`):**
Runs after register allocation (`regalloc`). Scans basic blocks for adjacent 64-bit or 128-bit memory instructions `I1` and `I2` sharing base register `Rn`:
* **Safety Predicates (Resolution 111):**
  1. `I1.Offset` and `I2.Offset` are exact multiples of `8` (for `X` registers) or `16` (for `Q` vector registers).
  2. `I2.Offset == I1.Offset + regSize`.
  3. Signed scaled offset check: $$-512 \le I1.\text{Offset} \le +504$$ (for 64-bit) or $$-1024 \le I1.\text{Offset} \le +1008$$ (for 128-bit).
  4. **Anti-Aliasing Check:** For loads (`MOVD (Rn), Rt1` + `MOVD 8(Rn), Rt2`), require `Rt1 != Rt2`, `Rt1 != Rn`, and `Rt2 != Rn` (preventing ARMv8 `SIGILL` Constrained Unpredictable traps and base-clobber bugs on fault replay).
  5. **Barrier Check:** Zero intervening branch labels (`p.Mark & LABEL`), `CALL`, or atomic LSE instructions between `I1` and `I2`.
* **Transformation:** Replaces `MOVD offset(Rn), Rt1` + `MOVD offset+8(Rn), Rt2` with `LDP offset(Rn), (Rt1, Rt2)`. Consecutive zero stores (`MOVD ZR, offset(Rn)`) fuse into `STP (ZR, ZR), offset(Rn)`.

**15.2. x86_64 AVX-512 / AVX2 Fault-Safe Tail Masking & Checked Allocation (`ssa/gen/AMD64.rules`):**
1. **64-Byte Allocator Tail Padding (Resolution 22):**
   All `mheap` spans, Side-Stack Arena chunks, and `.rodata` sections reserve an unallocated 64-byte guard pad at the end of every mapped OS page before a `PROT_NONE` boundary, ensuring 32-byte (`YMM`) and 64-byte (`ZMM`) unaligned vector loads never trigger spurious page faults.
2. **Checked Multiplication in `make([]T, a * b)` (Resolution 169):**
   ```nasm
   MOVQ    a_reg, AX
   IMULQ   b_reg, AX                     ; Signed 64-bit multiply
   JO      .L_panic_makeslicelen         ; Jump on hardware overflow flag (OF == 1)
   CMPQ    AX, $0
   JL      .L_panic_makeslicelen
   ```

---

**SECTION 16: `--json-diagnostics` SCHEMA FOR AI AGENTS & SUPERVISORS (`betterlint/json.go`):**

When invoked with `logos lint --json-diagnostics ./...` or `logos build --json-diagnostics`, all compiler and linter errors emit deterministic single-line JSON records for zero-shot LLM self-healing:

```json
{
  "schema_version": "1.0",
  "severity": "forbid",
  "rule_id": "correctness::nil_deref_unproven",
  "file": "pkg/auth/session.lgo",
  "span": { "start_line": 42, "start_col": 14, "end_line": 42, "end_col": 26, "byte_offset": 1184, "byte_len": 12 },
  "message": "cannot dereference nullable type '*User?' without proving '!= nil'",
  "provenance": {
    "origin_expr": "m[sessionID]",
    "origin_line": 40,
    "inferred_type": "*User? (interface { *User | nil })"
  },
  "autofix_candidates": [
    {
      "description": "Use optional chaining with default fallback",
      "replacement": "user?.Name ?? \"anonymous\""
    },
    {
      "description": "Insert early-return nil guard",
      "insert_before_line": 42,
      "replacement": "if user == nil {\n\treturn \"\", ErrSessionNotFound\n}\n"
    }
  ]
}
```

---

**SECTION 17: COMPLETE LEDGER OF ALL 182 RED-TEAM ATTACKS & NORMATIVE RESOLUTIONS:**

For total traceability across all 18 adversarial review rounds, every attack vector and its exact compiler/runtime resolution is indexed below:

* **Round 1–2 (Core Design & Safety — `#1` to `#18`):**
  * `#1` Split-union pointer/scalar overlap GC confusion → **Res `#1`, `#82`:** Pointer-containing unions always use 16B `[_type | _data]` layout; pure-scalar unions (`<=32B`) overlap in `noscan` memory.
  * `#2` `string(constBytes)` mutated via surviving mutable alias → **Res `#2`, `#83`:** Zero-copy `[]byte`→`string` allowed only when `string` does not escape current frame and slice has no aliases.
  * `#3` `.rodata` map breaks due to runtime random `hashkey` seed → **Res `#3`, `#73`, `#140`:** `const map` uses compile-time collision-free static SwissTable in `.rodata` (or `switch` for $$\le 8$$ keys).
  * `#4` Side-Stack Arena OOM across 100K parked goroutines → **Res `#4`, `#11`:** Lease-On-Use from `p.arenaPool`; returned to `P` immediately in `gopark` when watermark == base (`0B` held while parked).
  * `#5` `//logos:cgo_fastcall` blocks OS thread forever → **Res `#5`:** Record `p.fastCallEnterNano`; `sysmon` retakes `P` if blocked $$> 1\text{ms}$$.
  * `#6` Block lambda `return` ambiguity inside `.map()` → **Res `#6`:** Collection methods permit single-expression lambdas `x => expr` only; `?` propagates out of enclosing function.
  * `#7` Non-Nil `*T` bypassed via `[10]*T`, missing map key, or `{}` JSON → **Res `#7`:** Uninitialized `[N]*T` banned; map index returns `V?`; static JSON decoder tracks `_seenMask` for required Non-Nil fields.
  * `#8` Legacy `.go` functions returning `nil` into `.lgo` `*T` → **Res `#8`, `#44`:** `.go` pointers imported as `*T?` unless `.a` export summary bit `SumReturnsNonNil == 1`.
  * `#9` `.text.cold` & RVO `_outPtr` corrupting DWARF / Delve debugger → **Res `#9`, `#105`:** Keep cold blocks at tail of standard `.text` section; emit `DW_TAG_inlined_subroutine` and strip `$cold`/`$spec` in `pclntab`.
  * `#10` Exhaustive union `switch` bypassed via `default:` hiding new variants → **Res `#10`:** `pedantic::exhaustive_union_switch` warns on `default:` over closed unions.
  * `#11` Async preemption (`SIGURG`) migrating `G` across `P`s mid-arena use → **Res `#11`:** Dirty chunk is pinned to `g.activeArena` until watermark returns to base.
  * `#12` Inline bump allocator corrupting `gcmarkBits` across size classes → **Res `#12`:** Restrict inline 1.5ns bump allocator strictly to `noscan` objects (`p.noscanBumpCur`).
  * `#13` Concurrent multi-word union write causing type-confusion UB → **Res `#13`, `#148`:** `concurrency::concurrent_multiword_race` (`forbid`) bans unlocked sharing of unions/interfaces/slices across goroutines.
  * `#14` `if x := fn(); x == nil` outer-scope promotion breaking redeclarations → **Res `#14`:** Preserve standard Go block scoping; promote existing outer `x` from `T?` to `T` after `if x == nil { return }`.
  * `#15` Auto-Generification exploding binary size on megamorphic interfaces → **Res `#15`, `#48`, `#92`:** Clone Specialization keeps original `iface` symbol, caps type nesting at `max_type_depth = 4`, and respects `-Os`/`-Oz`.
  * `#16` Auto-Clip `s[i:j:j]` breaking `s = s[:0]` buffer reuse loop → **Res `#16`, `#93`:** Exempt in-place self-reslicing `s = s[...]` when no aliases exist; clip all other subslicing to `cap = len`.
  * `#17` Tuple `(int, error)` colliding with multi-return unpacking → **Res `#17`, `#39`:** Direct calls in multi-assignment or variadic args unpack; single-variable binding packs into `_Tuple2`.
  * `#18` Named return shadowing corrupting `defer f.Close()?` → **Res `#18`:** Bind `?` and `defer?` directly to compiler-synthesized hidden return slot `_retErr` and ban named-return shadowing.

* **Round 3–4 (Kernel, Hardware & Microarchitecture — `#19` to `#35`):**
  * `#19` `comptime.ViewAs[T]` unaligned trap (`SIGBUS`) and padding garbage → **Res `#19`, `#72`:** Require zero-padding POD structs (`bool`/pointers banned), emit unaligned-safe loads, and tie escape lifetime to source slice.
  * `#20` Side-Stack object pointing to Call-Stack var invalidated by `copystack` → **Res `#20`:** Unidirectional pointer rule: any local whose address is stored into Side-Stack is promoted to Side-Stack.
  * `#21` Slowloris attack exhausting pinned `io_uring` buffers → **Res `#21`:** Use kernel `IORING_REGISTER_PBUF_RING` (allocate only on packet arrival) or standard netpoll.
  * `#22` AVX-512/AVX2 vector tail read crossing into `PROT_NONE` page → **Res `#22`:** Enforce 64B tail padding on all spans/arenas/`.rodata` and use hardware mask registers (`k1`).
  * `#23` Accessing valid `res` when `defer f.Close()?` fails on return → **Res `#23`, `#156`:** `types2` forbids reading `res` from `(res, err)` before `err == nil` is proven.
  * `#24` Non-deterministic `map` iteration inside `const func` → **Res `#24`:** Iterating dynamic maps inside `const func` is a compile error; `const map` keys are sorted lexicographically.
  * `#25` Transitive dependencies bypassing `reflect` ban via `unsafe` / `os/exec` → **Res `#25`:** Linker `deadcode.go` `[capabilities]` auditor checks `reflect`, `unsafe`, `cgo`, `exec`, `net`.
  * `#26` ARM64 weak memory ordering exposing uninitialized `sync.Hot[T]` fields → **Res `#26`:** Emit `STLR` (Store-Release) on `Hot.Store` and `LDAPR` / address-dependency on `Hot.Load`.
  * `#27` WASM `memory.grow` detaching JS `Uint8Array` views mid-call → **Res `#27`:** `.loader.mjs` re-evaluates `wasmExports.memory.buffer` dynamically after every allocation.
  * `#28` RVO destination-passing aliasing (`arr = Reverse(arr)`) corrupting input → **Res `#28`:** Pass `_outPtr` only for fresh, unaliased destination slots.
  * `#29` ZGC colored pointers rejected by Linux kernel `write()` / `io_uring` (`-EFAULT`) → **Res `#29`, `#168`, `#171`:** Enable ARM64 TBI / x86 LAM; untag pointers before syscalls/CGo; use 1-clock `ANDQ` unmasking on non-LAM x86.
  * `#30` Self-modifying Polymorphic Inline Cache (PIC) causing 64-core cache thrashing → **Res `#30`:** Purely static compile-time devirtualization (`if r.tab == &ITab`), zero runtime memory writes.
  * `#31` Stream fusion `.filter().map(sideEffect).take(5)` over-executing side effects → **Res `#31`:** Pull-based single-loop fusion breaks immediately at `len == 5`; lint warns on side effects before `.sortBy()`.
  * `#32` Execution order chaos between `defer`, `scope defer`, and `errdefer` → **Res `#32`, `#135`:** `scope defer` runs at block exit (`}`); `defer`, `defer?`, and `errdefer` share one strict LIFO stack at function exit.
  * `#33` Stack-First `4KB` slice allocation overflowing 2KB goroutine frame → **Res `#33`:** Runtime fast-path checks `SP - size >= g.stackguard0 + 1024`; otherwise falls back to Side-Stack Arena.
  * `#34` String interpolation `$"..."` allocating heap intermediate strings → **Res `#34`:** Format integer/float segments directly into target buffer via `strconv.Append*`.
  * `#35` `go test` / race detector incompatibility with custom memory passes → **Res `#35`, `#101`:** Emit `runtime.racemalloc` / `racefree` hooks on arena watermark resets when `-race` is enabled.

* **Round 5–7 (Unsealed Types, `--emit-go`, and Deep Runtime — `#36` to `#60`):**
  * `#36` Overlapping union tags & `~int` infinite sets → **Res `#36`:** Ban `~` in value unions; use global `*abi.Type` pointer as `_tag` for `0ns` subset coercion.
  * `#37` `--emit-go` failing on exported `const switch T` functions → **Res `#37`, `#108`:** Transpile concrete cases to `switch any((*T)(nil)).(type)` + `unsafe.Pointer` cast; interface cases to `any(v).(Iface)`.
  * `#38` Parser backtracking explosion on `(a, b) => expr` vs tuples → **Res `#38`:** LL(1) single-pass `parseParenOrTupleOrLambda` with 1-token lookahead after `)`.
  * `#39` `fmt.Println(pair())` ambiguity between 1 tuple and 2 args → **Res `#39`:** Direct calls in variadic/multi-arg positions unpack; variable assignment packs.
  * `#40` Comparing unions/tuples containing slices (`==` / map keys) panicking at runtime → **Res `#40`:** Mark unions/tuples containing `[]T`, `map`, or `func` as incomparable in `types2`.
  * `#41` Mutating source slice inside `.filter()` / `.map()` invalidating loop bounds → **Res `#41`:** Bind collection receiver and lambda parameter as `const` inside chains.
  * `#42` `defer f.Close()?` allocating `errors.Join` slice on every error → **Res `#42`:** When `_retErr == nil`, assign `_retErr = _closeErr` directly (`0` allocations).
  * `#43` Zero-sized struct `Pending{}` inside 16B union causing heap allocation → **Res `#43`:** Point zero-sized union variants to static `runtime.zerobase` in `.rodata`.
  * `#44` Cross-package nil/const inference breaking separate compilation speed → **Res `#44`:** Store 1-byte `LogosFuncSummary` bitmask in `.a` package export headers.
  * `#45` `const switch T` casting `[]*os.File` (8B) to `[]io.Writer` (16B) causing `SIGSEGV` → **Res `#45`, `#86`:** Rebind composite types (`[]T`) only on concrete matches; intersect constraints on interface matches.
  * `#46` Side-Stack Arena holding heap pointers missed by GC root scan → **Res `#46`:** Dual-ended chunk (`Noscan` bottom-up, `Scan` top-down scanned by `scanstack`).
  * `#47` `const func` purity broken by `recover()` or `uintptr` ASLR address checks → **Res `#47`:** Forbid `recover()`, `unsafe.Pointer`→`uintptr`, and time/rand in `const func`.
  * `#48` Auto-Generification breaking `var fn func(io.Reader) = Process` → **Res `#48`:** Keep original `Process(io.Reader)` entry point and emit specialized clones alongside it.
  * `#49` `T?` niche optimization breaking when `T` is a value union → **Res `#49`:** Value unions include `nil` as tag `0`; `hasZeroNiche` applies only to pointers/funcs/chans.
  * `#50` `const` `.rodata` slices rejected by `reflect.DeepEqual` or `fmt.Println` → **Res `#50`, `#98`:** Go runtime already supports non-heap `.rodata` pointers in reflection and `fmt`.
  * `#51` Infinite CTFE loop freezing IDE (`gopls`) → **Res `#51`:** Enforce `max_ctfe_steps = 100,000` and `1MB` `.rodata` blob limit.
  * `#52` Inlined `defer` inside loop exhausting stack slots → **Res `#52`, `#113`:** Lower inlined callee `defer` to block-exit `scope defer` with `pclntab` unwind mapping.
  * `#53` Returning recursive `Expr` AST from helper function escaping local arena → **Res `#53`:** `EscReturn` region merging allocates into the root caller's Side-Stack watermark.
  * `#54` `.rodata` constants with string/pointer fields crashing under PIE/ASLR relocation → **Res `#54`:** Place pointer-containing constants in `.data.rel.ro` (RELRO).
  * `#55` Ignored error leaving stale PC frames in `g.errTracePC` ring buffer → **Res `#55`:** Compare `err.data == g.lastErrData`; reset `g.errTraceLen = 0` on mismatch.
  * `#56` Inlined `scope defer fn(x)` evaluating `x` at block end instead of defer site → **Res `#56`:** Eagerly bind defer arguments to temporaries `_deferArg0 := x` at registration site.
  * `#57` Overlapping cases in `const switch T` (`*bytes.Buffer` vs `io.Writer`) → **Res `#57`:** Top-down first-match semantics; `betterlint` rejects shadowed unreachable cases.
  * `#58` Duplicate specialized clones across packages causing linker symbol collision → **Res `#58`:** Mark specialized clone symbols with `obj.DUPOK`.
  * `#59` CGo `_cgoCheckPointer` panicking on Side-Stack addresses → **Res `#59`:** Allocate Side-Stack chunks from `mheap_.allocManual(spanAllocStack)`.
  * `#60` Value-passed struct with Non-Nil `*T` mutated to zero inside helper → **Res `#60`:** All structs containing `*T` forbid zero-clearing (`*p = Struct{}`) without full field initialization.

* **Round 8–12 (Live-Fire Red/Blue & Secondary Patch Interactions — `#61` to `#118`):**
  * `#61` Nested option `Get[*User?]()` niche collision (`None` vs `Some(nil)`) → **Res `#61`:** When `T` is already optional (`*U?`), outer `T?` uses `struct { val T; present bool }`.
  * `#62` Calling mutating method on `const *User` bypassing immutability → **Res `#62`:** Block methods with `SumReceiverMutated == 1` on `const` receivers.
  * `#63` `panic` + `recover()` leaking Side-Stack Arena watermarks → **Res `#63`:** Record `d.arenaMark` in `runtime/panic.go` and restore watermark on unwind.
  * `#64` Concurrent string mutation between Pass 1 and Pass 2 of `$"..."` → **Res `#64`, `#126`:** Pin all interpolated expressions into local temporaries once before Pass 1.
  * `#65` `comptime.Fields[T]()` exposing private fields of external packages → **Res `#65`:** Yield exported fields only (plus unexported fields when `T` is in current package).
  * `#66` `errdefer` in function with unnamed `(int, error)` return failing in `--emit-go` → **Res `#66`:** Transpiler promotes unnamed returns to `(_ret0 int, _retErr error)`.
  * `#67` FMA float contraction causing x86 vs ARM64 drift in `-O3` → **Res `#67`:** Disable implicit FMA unless `math.FMA` is called; use pairwise tree summation for `.sum()`.
  * `#68` `go group` returning on first error while sibling still reads stack vars (UAF) → **Res `#68`:** Cancel group context on first error, then unconditionally `Wait()` for all siblings before scope exit.
  * `#69` `go next` AST printer stripping comments and `//go:build` tags → **Res `#69`:** Use byte-offset source span splicing instead of full AST re-printing.
  * `#70` `const []int{1,2,3}` literal allocating on heap when passed to `const []int` param → **Res `#70`:** Promote constant slice literals passed to `const` params into `.rodata`.
  * `#71` Side-Stack Scan region holding dead pointers breaking Yuasa barrier on reuse → **Res `#71`, `#81`:** Raise `scanCur` watermark first, then zero popped scan slots (`STP ZR, ZR`).
  * `#72` `comptime.ViewAs` interior pointer confusing GC `findObject` → **Res `#72`:** Go `findObject` resolves interior pointers via `span.base()`; `escape` links view to backing slice.
  * `#73` MPHF solver hanging compile time on 10,000-key `const map` → **Res `#73`:** Use static SwissTable with collision-free seed search instead of exponential graph solver.
  * `#74` `runtime.FuncForPC` returning mangled `Process$spec$os_File` breaking frameworks → **Res `#74`:** Strip `$spec$` and `$cold$` suffixes in `pclntab` display names.
  * `#75` Order-dependent type inference in `switch` expressions with `nil` branch → **Res `#75`:** Compute symmetric Least Upper Bound (`*User` $$\sqcup$$ `nil` $$=$$ `*User?`).
  * `#76` `const for` skipping `types2` check on unrolled body statements → **Res `#76`, `#128`:** Run full per-iteration `types2` assignment and call verification after `const if` pruning.
  * `#77` Standard `defer` setting `err = nil` after `errdefer` was registered → **Res `#77`:** `correctness::defer_erases_errdefer_state` (`forbid`) rejects `err = nil` inside `defer` when `errdefer` is present.
  * `#78` `wasm32` 4-byte alignment trapping on 64-bit `atomic.AddInt64` → **Res `#78`:** Force 8-byte alignment for `int64`, `uint64`, `float64` on `wasm32`.
  * `#79` `--emit-go` helper imports colliding with user variable `strconv := 1` → **Res `#79`:** Use reserved `_logos_strconv`, `_logos_errors`, `_logos_unsafe` aliases.
  * `#80` Cross-layer regression matrix verification → **Res `#80`:** 6-layer compiler invariant gate established.
  * `#81` Concurrent GC `suspendG` racing with Side-Stack Scan zeroing → **Res `#81`:** Increment `scanCur` before zeroing in epilogue; `scanstack` reads `[scanCur, scanBase)`.
  * `#82` 24B/32B Split-Union requiring custom GC bitmap compiler changes → **Res `#82`:** Eliminate Split-Union; pointer unions are always 16B `eface`, scalar unions are `noscan`.
  * `#83` Escaping `const []byte`→`string` zero-copy mutating when caller reuses slice → **Res `#83`:** Zero-copy `[]byte`→`string` restricted strictly to non-escaping strings.
  * `#84` `--emit-go` evaluating `getUsers().filter().map()` root expression twice → **Res `#84`:** Bind `_src := getUsers()` once before emitted loop.
  * `#85` Typed nil error triggering `errdefer` unexpectedly → **Res `#85`:** `typed_nil_return` is `forbid`; `errdefer` checks `_retErr != nil && _retErr.data != nil`.
  * `#86` `const switch T` losing outer generic constraint methods inside `case io.Writer:` → **Res `#86`:** Intersect outer constraint with case interface (`io.Closer` $$\cap$$ `io.Writer`).
  * `#87` `errdefer` skipped when function exits via `panic` instead of `return err` → **Res `#87`, `#90`:** Use `_errdefer_ok` boolean flag (remains `false` on panic/Goexit/error return).
  * `#88` `cgo_fastcall` corrupting callee-saved registers between Go Plan9 ABI and C ABI → **Res `#88`:** Emit standard System V / ARM64 ABI register mapping in `logosFastCgoCall`.
  * `#89` Overlapped scalar union leaving tail garbage from larger variant (`==` fails) → **Res `#89`, `#97`:** Zero-extend full union slot on every variant write.
  * `#90` `recover()` inside emitted `errdefer` corrupting panic stack frames → **Res `#90`, `#96`:** Never use `recover()`; set `_errdefer_ok = true` before every non-error `return`.
  * `#91` `const switch T` ambiguity on defined types (`type UserID int` vs `case int:`) → **Res `#91`:** `case int:` matches exact `int`; `case ~int:` matches underlying `int` types.
  * `#92` Cross-package polymorphic recursion (`Wrap[Wrap[T]]`) causing infinite specialization → **Res `#92`:** Cap specialization type nesting depth at `max_type_depth = 4`.
  * `#93` Malicious library reading `b[:cap(b)]` on `const []byte` (Heartbleed leak) → **Res `#93`:** Auto-Clip (`cap = len`) plus `types2` ban on `b[:cap]` expansion of `const` slices.
  * `#94` `go context` taking 2 seconds on monorepos by typechecking whole graph → **Res `#94`:** Use syntax-only shallow symbol indexing (`< 50ms`).
  * `#95` `.take(-5)` or `.drop(99999)` panicking on slice bounds → **Res `#95`:** Clamp collection method indices to `[0, len(s)]` via branchless `CMOV`.
  * `#96` Early `return nil` mid-function skipping `_errdefer_ok = true` → **Res `#96`:** AST walker injects `_errdefer_ok = true` before every non-error `return` statement.
  * `#97` Struct variant internal alignment hole (`uint8` + `uint32`) holding stale bytes in union → **Res `#97`:** Zero full 8B/16B slot before storing struct variant fields.
  * `#98` `reflect.DeepEqual` on `.data.rel.ro` composite constants → **Res `#98`:** Verified 100% compatible with upstream Go runtime.
  * `#99` Calling `u.Method()` directly on `User?` option without nil check → **Res `#99`:** `T?` does not inherit `T`'s methods; requires `u?.Method()` or nil narrowing.
  * `#100` `c := a + b` on `[1024]float64` blowing register file → **Res `#100`:** Restrict fixed-array SIMD operators to hardware register widths (`<= 64 bytes`).
  * `#101` `go test -race` false positives on reused Side-Stack chunks → **Res `#101`:** Emit `racemalloc`/`racefree` shadow resets when `cfg.BuildRace` is active.
  * `#102` CTFE `1 << 32` producing different results when cross-compiling to 32-bit WASM → **Res `#102`:** Evaluate CTFE `int`/`uint` using target `types.PtrSize` bit-widths.
  * `#103` `-0.0` vs `+0.0` and `NaN` keys in `const map[float64]V` → **Res `#103`:** Canonicalize `-0.0` to `+0.0`; forbid `NaN` keys at compile time.
  * `#104` eBPF / Datadog uprobes breaking on frame-pointer omission or cold outlining → **Res `#104`:** Preserve `BP`/`R29` frame pointers on all non-leaf functions and cold blocks.
  * `#105` macOS `codesign` rejecting custom `.text.cold` Mach-O sections → **Res `#105`:** Place cold symbols at the high-address tail of standard `.text` (`__TEXT,__text`).
  * `#106` `runtime.SetFinalizer` / `AddCleanup` registered on Side-Stack object → **Res `#106`:** `escape.go` forces finalizer targets to `EscHeap`; runtime panics if span is `spanAllocStack`.
  * `#107` `-march=native` breaking Bazel SHA-256 reproducible builds → **Res `#107`:** Default to deterministic `GOAMD64=v3` / `GOARM64=v8.2`; require explicit `target_cpu = "native"`.
  * `#108` `--emit-go` type-assertion fallback for `const switch T` interface cases → **Res `#108`:** Emit `if _w, _ok := any(v).(Iface); _ok` (plus pointer receiver check).
  * `#109` `go fmt` rewriting legacy `.go` files into `.lgo` syntax causing massive git diffs → **Res `#109`:** `go fmt` is strictly idempotent and never alters AST semantics (`go next` handles migration).
  * `#110` `runtime.Freeze(obj)` pointing to mutable heap child causing GC UAF → **Res `#110`:** Transitive Deep Freeze promotes entire subgraph to `spanAllocImmortal` and returns `const *T`.
  * `#111` ARM64 `LDP`/`STP` fusion causing `SIGILL` on `Rt1 == Rt2` or offset `> 504` → **Res `#111`:** Enforce strict offset range `[-512, +504]` and register non-aliasing in `ldpstp_logos.go`.
  * `#112` `.lgo` and `.gen.go` coexisting in repo causing `redeclared in this block` → **Res `#112`:** Prepend `//go:build !logos` to all emitted `.gen.go` files.
  * `#113` Inlining function containing `defer` into loop leaking resources until outer exit → **Res `#113`:** Convert inlined callee `defer` into block-scoped `scope defer`.
  * `#114` `--pedantic` struct field reordering breaking `const for` JSON/SQL field order → **Res `#114`:** Preserve declaration index in `comptime.Fields[T]()`; only optimize physical `field.Offset`.
  * `#115` Typed nil `(*User)(nil)` inside union `interface { *User | Error }` → **Res `#115`:** `*User` variant is Non-Nil; `nil` variant always normalizes to `(_type = 0, _data = 0)`.
  * `#116` Mutating `c.HP -= 10` on value-boxed struct inside `Creature` interface → **Res `#116`:** Interfaces with mutable fields (`HP int`) can only be satisfied by pointer types (`*Struct`).
  * `#117` Taking address `&c.HP` of interface property crashing on method-backed types → **Res `#117`:** Forbid `&` address operator on interface properties.
  * `#118` Embedded struct fields failing interface property contract satisfaction → **Res `#118`, `#145`:** Use standard Go promoted field selector depth rules.

* **Round 13–18 (Boilerplate, Go-Spring DI, Pluggable GC, Concurrency & Extreme Edge Cases — `#119` to `#182`):**
  * `#119` `Dto{..entity}` spread silently dropping same-name fields with different types → **Res `#119`:** `spread_type_mismatch` (`forbid`) aborts compile on incompatible same-name fields.
  * `#120` Synthetic interface field getters polluting `reflect.TypeOf(x).NumMethod()` → **Res `#120`:** Place synthetic field thunks exclusively in `itab` slots, never in `abi.Type` method tables.
  * `#121` Passing `[]*Monster` to `[]Creature` requiring manual copy loop → **Res `#121`:** Auto-generify `const []Interface` parameters to `const []_T` (`0` allocations).
  * `#122` Emoji extensions `.⚡` / `.📜` / `.ʕ◔ϖ◔ʔ` failing on `U+FE0F` selector drift → **Res `#122`:** Strip `U+FE0E`/`U+FE0F` and NFC-normalize filenames in `cmd/go/internal/load`.
  * `#123` Struct default `TLS bool = true` overriding explicit JSON `{"tls": false}` → **Res `#123`:** Apply defaults before unmarshaling so explicit JSON keys cleanly overwrite defaults.
  * `#124` Auto-generified `io.Writer` function breaking dynamic `w.(http.Flusher)` check → **Res `#124`:** Fold type assertion to compile-time constant `true`/`false` per concrete `_T`.
  * `#125` `errdefer tx.Rollback()?` overwriting primary function return error → **Res `#125`:** Join rollback error with primary error via `errors.Join(_retErr, _rbErr)`.
  * `#126` `$"ID: {nextID()}"` calling `nextID()` twice in 2-Pass sizing → **Res `#126`:** Bind all `{expr}` to local variables prior to Pass 1.
  * `#127` Interface Default Methods causing C++ diamond inheritance ambiguity → **Res `#127`:** Default method bodies inside interfaces are rejected from the spec.
  * `#128` `Resolve[f.Type](c)` failing `types2` before `const for` unrolls → **Res `#128`:** Two-Phase Deferred Stenciling in `noder/stencil.go`.
  * `#129` Circular DI `ServiceA` ↔ `ServiceB` crashing compiler in `Autowire[T]` → **Res `#129`:** Track `activeStencils` stack and emit `[comptime::circular_instantiation]`.
  * `#130` `MapTo[Dst, Src]` failing when same-named fields have distinct types → **Res `#130`:** Provide `comptime.FieldType[Src](df.Name) == df.Type` compile-time type comparison.
  * `#131` Embedding `interface { const Name string }` and `interface { Name string }` → **Res `#131`:** Subsume read-only + read-write property of identical type into read-write contract.
  * `#132` `party[nextIdx()].HP += 10` evaluating `nextIdx()` twice → **Res `#132`:** Pin LHS receiver expression to temporary `_recv` once before getter/setter call.
  * `#133` `proxy.golang.org` rejecting `.ʕ◔ϖ◔ʔ` non-ASCII paths on `go get` → **Res `#133`:** Warn on public modules and emit ASCII `.gen.go` files for module proxy publishing.
  * `#134` `-O3` aggressive inlining inflating recursive function stack frames → **Res `#134`:** Dampen inline budget to `80` inside recursive call-graph SCCs.
  * `#135` `defer f.Close()?` violating LIFO order with subsequent `defer` blocks → **Res `#135`:** Desugar `defer fn()?` in-place on the standard LIFO defer stack.
  * `#136` `go group` sibling dying while peer hangs forever on unbuffered `<-ch` → **Res `#136`:** Auto-inject `case <-_groupCtx.Done():` into channel operations inside `go group`.
  * `#137` Call-free `-O3` SIMD loop starving scheduler and blocking GC STW → **Res `#137`:** Inject strip-mined preemption check (`CMPQ 16(R14), SP`) every 4,096 iterations.
  * `#138` Side-Stack pointer stored into `sync.Pool` causing mystery UAF panic later → **Res `#138`:** `escape.go` treats `sync.Pool.Put` as hard `EscHeap` sink; `forbid` arena escape.
  * `#139` Malicious JSON `[...]` header triggering 10GB slice pre-allocation OOM → **Res `#139`:** Clamp static decoder pre-allocation to `min(remainingBytes/2, 1MB)`.
  * `#140` Known `.rodata` hash seed enabling hash-flooding DoS on `const map` → **Res `#140`:** Compute per-map collision-free seed at compile time (max probe length = 1).
  * `#141` `c.Pos.Normalize()` on interface struct property losing mutation on temporary → **Res `#141`:** Emit `_tmp := c._get_Pos(); (&_tmp).Normalize(); c._set_Pos(_tmp)` Write-Back.
  * `#142` `paths.map(p => os.Open(p)?)` leaking FDs `0..2` when element `3` fails → **Res `#142`:** Auto-inject reverse `.Close()` rollback loop on error exit when element implements `io.Closer`.
  * `#143` Buffered channel `<-ch` retaining dead pointer in ring buffer slot → **Res `#143`:** Zero pointer-containing channel buffer slot immediately upon receive.
  * `#144` Stack-First `make([]byte, n)` passed to CGo invalidated by `copystack` → **Res `#144`:** Exclude CGo and `unsafe.Pointer` arguments from Stack-First `SP` allocation.
  * `#145` Struct field `Name` vs embedded method `Base.Name()` ambiguity in interface → **Res `#145`:** Shallowest selector depth wins; equal depth fails with `ambiguous selector`.
  * `#146` `state = GameState{Tick: i, ..state}` copying 4KB struct on every loop tick → **Res `#146`:** Lower self-reassigned spread to in-place field mutation `state.Tick = i`.
  * `#147` Unhandled panic in background goroutine crashing entire server process → **Res `#147`:** `go group` recovers child panics into `runtime.PanicError` and cancels group cleanly.
  * `#148` 16B interface/slice concurrent read/write tearing → **Res `#148`:** Blocked at compile time by `concurrency::concurrent_multiword_race` (`forbid`).
  * `#149` `-O3` Dead-Store Elimination deleting `clear(secretKey)` before return → **Res `#149`:** Mark explicit `clear()` calls as non-elidable `Volatile` stores in `ssa/dse.go`.
  * `#150` Triple-nested `const for` exploding AST to 32,768 nodes and hanging compiler → **Res `#150`:** Enforce `max_comptime_ast_nodes = 4,096` per function.
  * `#151` `go group` sibling stuck in blocking `conn.Read()` after group context cancels → **Res `#151`:** Link `g.cancelChan` into `runtime/netpoll.go` to immediately unblock socket I/O on group cancel.
  * `#152` `string(mmapSlice)` zero-copy crashing after `syscall.Munmap` → **Res `#152`:** Escaping strings always copy to Go heap (`slicebytetostring`).
  * `#153` `sync.Hot[T].Store()` at 1M ops/sec leaking retire-list memory → **Res `#153`:** Zero-retire-list design relying directly on Go's Yuasa deletion barrier.
  * `#154` Global 16B alignment inflating struct padding by 33% and breaking Go ABI → **Res `#154`:** Keep standard 8B alignment globally; apply 16B alignment only inside `sync.Hot[T]`.
  * `#155` `SIGURG` arriving mid-bump-allocation causing double allocation of same address → **Res `#155`:** Rewind `RIP` to `.L_bump_start` if preempted before final cursor store.
  * `#156` Caller reading valid `hash` when `defer f.Close()?` returns disk-full error → **Res `#156`:** Error-First Unpacking Guard blocks reading `hash` until `err == nil` is checked.
  * `#157` Unconditional self-recursion inside `const switch T` branch → **Res `#157`:** Blocked at compile time by `correctness::infinite_recursion` (`forbid`).
  * `#158` Substring `hugeStr[0:10]` pinning 10MB string in heap forever → **Res `#158`:** `escape` warns on escaping substrings of large buffers (`perf::large_backing_array_pin`).
  * `#159` `zgc`/`immix` moving `*Node` pointer and breaking `map[*Node]bool` hash lookup → **Res `#159`:** Pin hashed blocks in `immix`; use immutable object-header identity hash in `zgc`.
  * `#160` `zgc`/`immix` moving slice backing array during active `cgo_fastcall` → **Res `#160`:** Pin active CGo pointer args in `p.cgoPinnedPtrs[0..3]` (`0.3ns` store).
  * `#161` Removing Mark Assist causing OOM when allocation outpaces background GC → **Res `#161`:** Elastic Background Worker Scaling (boost dedicated GC threads at 85% heap).
  * `#162` Strict `go lint` (`forbid`) breaking third-party `go.mod` libraries → **Res `#162`:** Exempt `pkg/mod/**` from code lints; enforce only linker `[capabilities]` on dependencies.
  * `#163` `unbounded_channel_op` false-positiving on non-blocking `make(chan T, 1)` send → **Res `#163`:** Exempt locally proven buffered channel sends where `cap >= send_count`.
  * `#164` `discarded_error` false-positiving on `fmt.Println` and `bytes.Buffer.Write` → **Res `#164`:** Exempt functions with `SumAlwaysNilError == 1` and stdlib console/memory writers.
  * `#165` `cgo_fastcall` overflowing 2KB goroutine stack or crashing `SIGSEGV` handler → **Res `#165`:** Switch `RSP` to 8MB `g0` system stack in `logosFastCgoCall`.
  * `#166` `-gc=epsilon` / `batch` dying on unexpected memory spike → **Res `#166`:** `-gc=batch` triggers emergency parallel STW sweep at 90% `GOMEMLIMIT` while keeping `0` write barriers.
  * `#167` Nested `account.WithLock()` inside `account.WithLock()` causing self-deadlock → **Res `#167`:** Blocked at compile time by `concurrency::reentrant_withlock_deadlock` (`forbid`).
  * `#168` `-gc=zgc` crashing with `SIGSEGV` on legacy x86 CPUs lacking Intel LAM → **Res `#168`, `#171`:** Probe CPU/kernel at `osinit`; fallback to software `ANDQ` load-barrier unmasking.
  * `#169` `make([]Pixel, width * height)` integer overflow allocating tiny buffer → **Res `#169`:** Emit hardware overflow check (`IMULQ` + `JO`) on arithmetic inside `make()` lengths.
  * `#170` `case <-time.After(5*time.Second):` inside loop leaking 100K timers/sec → **Res `#170`:** Hoist and reuse a single `time.Timer` across loop iterations at `-O2`/`-O3`.
  * `#171` ZGC multi-mapping causing 3x RSS reporting and Kubernetes `OOMKilled` → **Res `#171`:** Ban multi-mapping; use single mapping + 1-clock software unmasking on non-LAM hosts.
  * `#172` Hoisted `time.After` timer firing immediately on iteration 2 due to undrained channel → **Res `#172`:** Use Go 1.23+ synchronous channel-draining `resetHoistedTimer`.
  * `#173` `SIGPROF` arriving during `cgo_fastcall` panicking with `split stack overflow` → **Res `#173`:** Swap TLS `R14` register to `g0` and set `m.inFastCall = 1` before switching `SP`.
  * `#174` Cross-OS `const if runtime.GOOS == "linux"` triggering `unused function` lint on macOS → **Res `#174`:** Mark symbol references across all `const if` branches prior to dead-branch pruning.
  * `#175` `SetFinalizer` resurrecting dead object in `immix`/`zgc` causing UAF → **Res `#175`:** Two-cycle finalizer liveness marking plus `forbid` lint on finalizer pointer escape.
  * `#176` `panic()` inside `sync.Locked.WithLock` leaving mutex permanently locked → **Res `#176`:** Register `Unlock()` in zero-cost `pclntab` unwind table (`0ns` normal path, 100% unwind-safe).
  * `#177` User input containing `{secret}` inside `$"Hello {input}"` causing format injection → **Res `#177`:** `$"..."` is pure compile-time lexical concatenation with zero runtime format parsing.
  * `#178` Spawning 1M goroutines in a loop inside `go group` exhausting RAM → **Res `#178`:** Warn on unbounded loop spawn and provide `go group (ctx, limit = N)` bounded concurrency.
  * `#179` `NaN` in float slice causing `.sortBy()` to panic or loop forever → **Res `#179`:** Compile `.sortBy()` using `cmp.Compare` total ordering (`NaN` ordered deterministically).
  * `#180` Cross-package `init()` order affecting `const` composite tables → **Res `#180`:** All `const` composites are evaluated at compile time into `.rodata` before any `init()` runs.
  * `#181` 1M-iteration loop exhausting Side-Stack Arena before function returns → **Res `#181`:** Reset Side-Stack watermark at every loop back-edge for iteration-local allocations.
  * `#182` `int` field in shared `.lgo` JSON struct overflowing on 32-bit WASM client → **Res `#182`:** `portability::serialized_unsized_int` warns on bare `int`/`uint` in serialized structs.

---

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

**SECTION 19: PLUGGABLE GC STATE MACHINES & MEMORY LAYOUTS (`src/runtime/gc_*.go`):**

**19.1. Engine 1: Enhanced Green Tea GC (`-gc=greentea` — Default):**
1. **Span-Centric Page Scanning (`gc_greentea.go`):**
   Standard Go pushes individual object pointers onto a LIFO workbuf, causing random L3/DRAM cache misses when traversing pointer graphs across spans.
   `greentea` replaces pointer workbufs with a **Span Deck (`spanWorkBitmap`)**:
   * When marking encounters a pointer `p` into span `S`, it sets the object's mark bit in `S.gcmarkBits` and sets `S.hasGreyObjects = 1` in the page summary bitmap without dereferencing `p` immediately.
   * Worker threads pop dirty spans in monotonically increasing virtual address order and scan all grey objects within that `8KB` span sequentially using hardware L1 prefetching (`PREFETCHT0`).
2. **Sticky-Bit Generational Card Table:**
   * A 1-byte-per-512B card table (`heapCards`) tracks pointer mutations.
   * During minor GC cycles (triggered when heap grows by 50%), spans marked `Mature` (survived $$\ge 2$$ cycles) whose cards are `0x00` are skipped entirely without scanning their `gcmarkBits`.
3. **Elastic Worker Scaling (Replacing Mutator Mark Assist — Resolution 161):**
   $$\text{GCWorkerCPUFraction}(u) = \begin{cases} 0.25 & \text{if } u < 0.80 \\ 0.25 + 0.50 \times \left(\frac{u - 0.80}{0.15}\right) & \text{if } 0.80 \le u < 0.95 \\ 0.75 & \text{if } u \ge 0.95 \end{cases}$$
   where $$u = \frac{\text{HeapLive}}{\text{HeapGoal}}$$. User request goroutines never execute `gcAssistAlloc`.

**19.2. Engine 2: Sub-50$$\mu\text{s}$$ Colored-Pointer Compactor (`-gc=zgc`):**
1. **64-Bit Colored Pointer Bit Layout:**
   ```
   +----------+-------+-------+----------+-------+------------------------------------------+
   | Bits 63..48 | 47 | 46    | 45       | 44    | Bits 43..0 (44-Bit Canonical Heap Addr)  |
   +----------+-------+-------+----------+-------+------------------------------------------+
   | Unused/TBI  | 0  | Final | Remapped | Mark1 | Mark0 (Bit 42) + 4TB Virtual Address     |
   +----------+-------+-------+----------+-------+------------------------------------------+
   ```
   * **Hardware Acceleration:** On ARM64 (`TBI` enabled by default in Linux/macOS) and x86_64 with Intel LAM / AMD UAI, the color bits reside in bits `59..56` (ignored by MMU).
   * **Legacy x86 Software Unmasking Fallback (Resolution 171):** When `runtime.hasHardwarePointerTagging == false`, the load barrier emits:
     ```nasm
     MOVQ    (AX), BX                      ; Load colored pointer
     TESTQ   BX, R15                       ; R15 holds current 'BadColorMask'
     JNZ     .L_zgc_self_heal              ; Slowpath: relocate or remap pointer via CAS
     ANDQ    $0x00000FFFFFFFFFFF, BX       ; 1-clock unmask (Zero container RSS inflation!)
     ```
2. **Concurrent Relocation & Forwarding Table:**
   * Evacuation targets pages with fragmentation $$> 35\%$$.
   * Forwarding tables (`srcOffset -> dstAddr`) are stored **off-heap** in page metadata headers so evacuated source pages can be immediately returned to the OS (`madvise(MADV_DONTNEED)`) the instant relocation completes, even before all pointers to the old page have been self-healed.

**19.3. Engine 3: Immix Line & Block Throughput Engine (`-gc=immix`):**
1. **Block (`32KB`) and Line (`128B`) Hierarchy:**
   * Each `32KB` block contains `256` lines of `128B` each, prefixed by a 256-byte metadata header (`lineMark [256]uint8`).
   * Thread-local allocators bump-allocate rapidly across contiguous free lines (`hole` hunting):
     ```go
     // Fast-path bump within current free hole [cursor, limit):
     if cursor + size <= holeLimit {
         obj := cursor
         cursor += size
         return obj
     }
     // Medium-path: scan 256-byte lineMark array via AVX2 VPCMPEQB to find next 0x00 hole in 3ns!
     ```
2. **Opportunistic Evacuation & Address-Hash Pinning (Resolution 159):**
   * If `map[*T]V` hashes an object's address, `blockHeader.pinned = 1` is set. Pinned blocks are never evacuated; unpinned blocks with $$> 50\%$$ dead lines are evacuated into fresh blocks during mark phase.

---

**SECTION 20: ZERO-COST `pclntab` UNWIND TABLE FOR `scope defer` & `sync.Locked` (`src/cmd/internal/obj` & `runtime/panic.go`):**

Standard Go's open-coded `defer` stores a 1-byte bitmask (`deferBits`) on the stack frame and executes conditional `TESTB` branches at every function exit.
For `scope defer` and `sync.Locked.WithLock()` (Resolutions 113, 176), `loGo/s` eliminates even the `deferBits` stack byte by emitting a **Zero-Cost PC-Range Unwind Table (`FUNC_UNWIND_TABLE`)** inside `pclntab`:

1. **Normal Execution Path (Happy Path):**
   * Entry: `runtime.lock2(&mu)`
   * Body: `<inlined user statements>`
   * Exit: `runtime.unlock2(&mu)`
   * Overhead compared to raw manual `Lock()`/`Unlock()`: **0 instructions, 0 stack bytes, 0ns**.
2. **Static Unwind Entry (`funcdata` Index `FUNCDATA_LogosScopeUnwind`):**
   ```
   +-----------------------+-----------------------+------------------------------+
   | StartPC Offset (u32)  | EndPC Offset (u32)    | LandingPad / CleanupPC (u32) |
   +-----------------------+-----------------------+------------------------------+
   | 0x0040 (After Lock)   | 0x00A8 (Before Unlock)| 0x01F0 (Cold Unlock Stub)    |
   +-----------------------+-----------------------+------------------------------+
   ```
3. **Panic Unwinder Execution (`runtime/panic.go`):**
   When `gopanic` unwinds a stack frame at instruction pointer `frame.pc`, it binary-searches `FUNCDATA_LogosScopeUnwind` (cost incurred **only** when a panic actually happens). If `StartPC <= frame.pc < EndPC`, the unwinder invokes the cold cleanup stub (`runtime.unlock2(&mu)` or `g.activeArena.cur = savedMark`) before continuing stack unwinding.

---

**SECTION 21: IDE & LANGUAGE SERVER (`gopls`) PROTOCOL EXTENSIONS:**

To ensure VSCode, Cursor, Zed, and Neovim work out-of-the-box with `.lgo`, `.📜`, `.⚡`, and `.ʕ◔ϖ◔ʔ` files, `loGo/s` extends `gopls` (`golang.org/x/tools/gopls`) with 4 features:

1. **Inlay Hints for Compiler Lowering:**
   * Displays inferred union tags, `T?` flow-narrowed non-nil types (`/* narrowed: *User */`), and `escape` allocation tier badges (`[stack]`, `[side-arena]`, `[heap]`, `[.rodata]`) inline in the editor.
2. **Hover Comptime Expansion Preview:**
   * Hovering over `const for`, `const switch T`, or `$"..."` renders the exact unrolled Go code and pre-computed byte buffer size (`Fixed: 17B + dynamic`) in the hover tooltip.
3. **Instant Code Actions (`QuickFix`):**
   * Binds `betterlint`'s `autofix_candidates` directly to LSP `textDocument/codeAction`, allowing 1-click or save-time conversion of `if err != nil` to `?` and nil-guards to `?.` / `??`.
4. **Virtual `.gen.go` Synchronization:**
   * When a workspace contains mixed `.lgo` and `.go` packages, `gopls` resolves type definitions directly across both file types in memory without requiring disk emission of `.gen.go` files.

---

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
