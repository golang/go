// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logos/wasmyield"
	"cmd/compile/internal/objw"
	"cmd/internal/obj"
)

// logosWasmHold is the M16-S1 hook of Compile. On GOARCH=wasm it records
// the MayYield summary of fn; with the WASM profile on it also takes
// ownership of the assembly, which wasmyield.Finish performs once every
// callee fact of the package is known, and Compile returns early.
func logosWasmHold(fn *ir.Func, pp *objw.Progs) bool {
	if !wasmyield.Active() {
		return false
	}
	text := pp.Text
	return wasmyield.Record(fn, text, func(newprog obj.ProgAlloc) {
		obj.Flushplist(base.Ctxt, &obj.Plist{Firstpc: text, Curfn: fn}, newprog)
		if fn.IsPackageInit() && base.Debug.WrapGlobalMapCtl != 1 {
			weakenGlobalMapInitRelocs(fn)
		}
		fieldtrack(text.From.Sym, fn.FieldTrack)
	})
}

// logosWasmImport is the M16-S1 hook of CreateWasmImportWrapper: it records
// the import wrapper so callers see the yield fact of the import.
func logosWasmImport(fn *ir.Func) {
	wasmyield.RecordImport(fn)
}
