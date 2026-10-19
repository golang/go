# Section 20: unwind tables (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 1232-1254 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

