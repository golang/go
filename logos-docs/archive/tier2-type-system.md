# Section 2: tier 2 (type system, unions, interfaces, comptime) (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 108-178 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

