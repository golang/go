# Section 17: red-team ledger (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 961-1158 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

