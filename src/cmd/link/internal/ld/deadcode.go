// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
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
//	3. reflect.Value.Method (or MethodByName), or reflect.Type.Method
//	   (or MethodByName)
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
//	- reflect.Value.Method or MethodByName is reachable
// 	- reflect.Type.Method or MethodByName is called (through the
// 	  REFLECTMETHOD attribute marked by the compiler).
// If any of these happen, all bets are off and all exported methods
// of reachable types are marked reachable.
//
// Any unreached text symbols are removed from ctxt.Textp.
func deadcode(ctxt *Link) {
	deadcode2(ctxt)
}

// addToTextp populates the context Textp slice (needed in various places
// in the linker).
func addToTextp(ctxt *Link) {
	// Set up ctxt.Textp, based on ctxt.Textp2.
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
}
