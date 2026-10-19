# Section 15: backend peephole and SIMD passes (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 902-927 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

