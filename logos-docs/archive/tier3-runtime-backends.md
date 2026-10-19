# Section 3: tier 3 (GC, allocator, stack, cgo, backends) (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 179-210 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

