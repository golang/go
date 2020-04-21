// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import "cmd/link/internal/sym"

// Temporary dumping around for sym.Symbol version of helper
// functions in xcoff.go, still being used for some archs/oses.
// FIXME: get rid of this file when dodata() is completely
// converted.

// xcoffUpdateOuterSize stores the size of outer symbols in order to have it
// in the symbol table.
func xcoffUpdateOuterSize(ctxt *Link, size int64, stype sym.SymKind) {
	if size == 0 {
		return
	}

	switch stype {
	default:
		Errorf(nil, "unknown XCOFF outer symbol for type %s", stype.String())
	case sym.SRODATA, sym.SRODATARELRO, sym.SFUNCTAB, sym.SSTRING:
		// Nothing to do
	case sym.STYPERELRO:
		if ctxt.UseRelro() && (ctxt.BuildMode == BuildModeCArchive || ctxt.BuildMode == BuildModeCShared || ctxt.BuildMode == BuildModePIE) {
			// runtime.types size must be removed, as it's a real symbol.
			outerSymSize["typerel.*"] = size - ctxt.Syms.ROLookup("runtime.types", 0).Size
			return
		}
		fallthrough
	case sym.STYPE:
		if !ctxt.DynlinkingGo() {
			// runtime.types size must be removed, as it's a real symbol.
			outerSymSize["type.*"] = size - ctxt.Syms.ROLookup("runtime.types", 0).Size
		}
	case sym.SGOSTRING:
		outerSymSize["go.string.*"] = size
	case sym.SGOFUNC:
		if !ctxt.DynlinkingGo() {
			outerSymSize["go.func.*"] = size
		}
	case sym.SGOFUNCRELRO:
		outerSymSize["go.funcrel.*"] = size
	case sym.SGCBITS:
		outerSymSize["runtime.gcbits.*"] = size
	case sym.SITABLINK:
		outerSymSize["runtime.itablink"] = size

	}
}
