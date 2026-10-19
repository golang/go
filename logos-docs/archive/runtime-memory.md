# Section 8: runtime memory and GC internals (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 499-601 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

