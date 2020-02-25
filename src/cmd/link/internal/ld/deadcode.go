// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/sym"
)

// deadcode marks all reachable symbols.
//
// The basis of the dead code elimination is a flood fill of symbols,
// following their relocations, beginning at *flagEntrySymbol.
//
// This flood fill is wrapped in logic for pruning unused methods.
// All methods are mentioned by relocations on their receiver's *rtype.
// These relocations are specially defined as R_METHODOFF by the compiler
// so we can detect and manipulated them here.
//
// There are three ways a method of a reachable type can be invoked:
//
//	1. direct call
//	2. through a reachable interface type
//	3. reflect.Value.Call, .Method, or reflect.Method.Func
//
// The first case is handled by the flood fill, a directly called method
// is marked as reachable.
//
// The second case is handled by decomposing all reachable interface
// types into method signatures. Each encountered method is compared
// against the interface method signatures, if it matches it is marked
// as reachable. This is extremely conservative, but easy and correct.
//
// The third case is handled by looking to see if any of:
//	- reflect.Value.Call is reachable
//	- reflect.Value.Method is reachable
// 	- reflect.Type.Method or MethodByName is called.
// If any of these happen, all bets are off and all exported methods
// of reachable types are marked reachable.
//
// Any unreached text symbols are removed from ctxt.Textp.
func deadcode(ctxt *Link) {
	deadcode2(ctxt)
}

// addToTextp populates the context Textp slice (needed in various places
// in the linker) and also the unit Textp slices (needed by the "old"
// phase 2 DWARF generation).
func addToTextp(ctxt *Link) {

	// First set up ctxt.Textp, based on ctxt.Textp2.
	textp := make([]*sym.Symbol, 0, len(ctxt.Textp2))
	haveshlibs := len(ctxt.Shlibs) > 0
	for _, tsym := range ctxt.Textp2 {
		sp := ctxt.loader.Syms[tsym]
		if sp == nil || !ctxt.loader.AttrReachable(tsym) {
			panic("should never happen")
		}
		if haveshlibs && sp.Type == sym.SDYNIMPORT {
			continue
		}
		textp = append(textp, sp)
	}
	ctxt.Textp = textp

	// Dupok symbols may be defined in multiple packages; the
	// associated package for a dupok sym is chosen sort of
	// arbitrarily (the first containing package that the linker
	// loads). The loop below canonicalizes the File to the package
	// with which it will be laid down in text. Assumes that
	// ctxt.Library is already in postorder.
	for _, doInternal := range [2]bool{true, false} {
		for _, lib := range ctxt.Library {
			if isRuntimeDepPkg(lib.Pkg) != doInternal {
				continue
			}
			for _, dsym := range lib.DupTextSyms2 {
				tsp := ctxt.loader.Syms[dsym]
				if !tsp.Attr.OnList() {
					tsp.Attr |= sym.AttrOnList
					tsp.File = objabi.PathToPrefix(lib.Pkg)
				}
			}
		}
	}

	// Finally, set up compilation unit Textp slices. Can be removed
	// once loader-Sym DWARF-gen phase 2 is always enabled.
	for _, lib := range ctxt.Library {
		for _, unit := range lib.Units {
			for _, usym := range unit.Textp2 {
				usp := ctxt.loader.Syms[usym]
				usp.Attr |= sym.AttrOnList
				unit.Textp = append(unit.Textp, usp)
			}
		}
	}
}
