# Section 14: WASM profile (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 865-901 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 14: WASM NATIVE PROFILE & SELECTIVE ASYNCIFY (`GOOS=js` / `GOOS=wasip1`, `GOARCH=wasm`):**

When targeting WebAssembly, the compiler and linker automatically activate the WASM Native Profile without requiring external tools like TinyGo.

**14.1. Selective Asyncify Call-Graph Partitioning (`src/cmd/internal/obj/wasm/wasmobj.go`):**
Standard Go compiles **every** function into a giant `loop` + `br_table` state machine so any function can yield its goroutine stack, inflating binary size by $$3\times$$ and slowing execution by $$4\times$$.
`loGo/s` partitions all functions into two sets on the linker call graph:
* **Color 0 (`SyncPure` — ~90% of functions):** Functions that never transitively call `runtime.gopark`, channel operations, `time.Sleep`, or blocking `wasmimport` functions.
  * *Codegen:* Emitted as pure, native WASM functions with zero `br_table` loop wrappers, using hardware WASM registers (`local.get` / `local.set`) and native WASM `call`.
* **Color 1 (`MayYield` — ~10% of functions):** Functions that can block or yield.
  * *Codegen:* Emitted with lightweight stack-saving state machines (or WASM Stack-Switching `cont.new` / `resume` when `--wasm-features=stack-switching` is enabled).

**14.2. Zero-Copy Host FFI (`//logos:wasmexport` & `//logos:wasmimport`):**
1. **`wasm32` Pointer Model & 64-Bit Alignment (Resolution 78):**
   Pointers and slice fields (`ptr`, `len`, `cap`) are 32-bit (`12B` slice header, `8B` string header). However, `types/size.go` forces `alignof(int64) = 8`, `alignof(uint64) = 8`, and `alignof(float64) = 8` so 64-bit atomic instructions (`i64.atomic.rmw.add`) never trap on unaligned addresses.
2. **Detached `ArrayBuffer` Protection in Auto-Generated `.loader.mjs` (Resolution 27):**
   ```javascript
   // Auto-generated app.loader.mjs (1.4 KB)
   export async function instantiateLogos(wasmBytes, imports = {}) {
     let wasmExports;
     const getMem = () => new Uint8Array(wasmExports.memory.buffer); // Always fresh after memory.grow!
     const { instance } = await WebAssembly.instantiate(wasmBytes, imports);
     wasmExports = instance.exports;
     return {
       processImage(pixelsUint8, brightness) {
         const ptr = wasmExports._logos_scratch_alloc(pixelsUint8.byteLength);
         getMem().set(pixelsUint8, ptr); // Re-fetches buffer AFTER potential memory.grow inside alloc!
         const errCode = wasmExports.processImage(ptr, pixelsUint8.byteLength, brightness);
         wasmExports._logos_scratch_reset(ptr);
         if (errCode !== 0) throw new Error(readErrorString(wasmExports, getMem));
       }
     };
   }
   ```

---

