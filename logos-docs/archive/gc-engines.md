# Section 19: GC engines (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 1178-1231 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

