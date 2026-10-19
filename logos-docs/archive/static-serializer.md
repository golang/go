# Section 12: static serializer (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 766-825 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

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

