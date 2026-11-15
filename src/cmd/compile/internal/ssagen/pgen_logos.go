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

func logosWasmImport(fn *ir.Func) {
	wasmyield.RecordImport(fn)
}
