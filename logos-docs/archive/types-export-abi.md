# Section 6: type checker, export data, ABI layout (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 314-373 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

