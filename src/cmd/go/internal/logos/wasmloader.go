// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import "cmd/go/internal/cfg"

// wasmLoaderText is the host loader that a wasm build writes beside its
// .wasm output (SPEC Sec 14.2): the text of wasmloader.mjs.tmpl, held as a
// constant because the bootstrap build of cmd/go has no go:embed support.
// It is static, so the template carries no actions and every build writes
// the same bytes. Edit wasmloader.mjs.tmpl first and mirror the change
// here; TestWasmLoaderTextMatchesTemplate fails on any drift.
const wasmLoaderText = `import "./wasm_exec.js";
function readErrorString(wasmExports, getMem) {
 const at = wasmExports._logos_errmsg_ptr() >>> 0;
 return new TextDecoder().decode(getMem().subarray(at, at + wasmExports._logos_errmsg_len()));
}
export async function instantiateLogos(wasmBytes, imports) {
 const go = new Go();
 const { instance } = await WebAssembly.instantiate(wasmBytes, { ...go.importObject, ...imports });
 const wasmExports = instance.exports;
 const getMem = () => new Uint8Array((wasmExports.mem ?? wasmExports.memory).buffer);
 if (wasmExports.run) await Promise.race([go.run(instance), 0]);
 const exports = {};
 for (const name in wasmExports) {
  const fn = wasmExports[name];
  if (typeof fn != "function" || /^(_|run$|resume$|getsp$)/.test(name)) continue;
  exports[name] = (...args) => {
   const held = [];
   try {
    const flat = args.flatMap((a) => {
     if (!ArrayBuffer.isView(a)) return [a];
     const v = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
     const p = wasmExports._logos_scratch_alloc(v.length) >>> 0;
     held.push([v, p]);
     getMem().set(v, p);
     return [p, v.length];
    });
    if (fn(...flat)) throw new Error(readErrorString(wasmExports, getMem));
   } finally {
    for (const [v, p] of held) {
     v.set(getMem().subarray(p, p + v.length));
     wasmExports._logos_scratch_reset(p);
    }
   }
  };
 }
 return { instance, exports };
}
`

func init() {
	cfg.WasmLoader = wasmLoaderText
}
