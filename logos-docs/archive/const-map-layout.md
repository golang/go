# Section 9: rodata const map layout (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 602-612 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 9: `.rodata` COMPILER SERIALIZATION & PERFECT HASH TABLE (`reflectdata/rodata_logos.go`):**

**9.1. Collision-Free `const map[K]V` Layout in `.rodata` (Resolution 73, 140):**
When compiling `const mimeMap = map[string]string{ ... }` with $$N > 8$$ entries:
1. The compiler allocates a bucket array of size $$M = 2^{\lceil \log_2(1.25 N) \rceil}$$.
2. During compilation, a deterministic PRNG searches for a 64-bit seed $$S$$ (or a 2-level displacement table `disp [M/4]uint8`) such that every key $$k_i$$ maps to a bucket with probe distance $$\le 1$$ (16-way SIMD SwissTable group).
3. The entire bucket structure, key string headers, and value array are written as an immutable `obj.RODATA` (or `.data.rel.ro` if values contain pointers) symbol with `h.flags = flagReadOnlyMap`.
4. At call sites `v := mimeMap[reqExt]`, the compiler inlines a 6-instruction lookup without calling `runtime.mapaccess1_faststr` (`~2.5ns` vs `~14ns`).

---

