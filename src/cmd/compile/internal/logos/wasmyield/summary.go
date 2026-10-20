// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasmyield

import (
	"slices"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/obj"
	"cmd/internal/obj/wasm"
)

// pinned names the scheduler entry points that always yield, whatever the
// analysis of their bodies says. runtime.gopark is the anchor of SPEC
// Sec 14.1: no function that can reach it keeps MayYield=0.
var pinned = map[string]bool{
	"runtime.gopark":       true,
	"runtime.goparkunlock": true,
	"runtime.Gosched":      true,
	"time.Sleep":           true,
}

// summarize reads the Prog list the SSA code generator produced for sym,
// before preprocess expands it, and lists the ways the function can unwind.
func summarize(sym *obj.LSym, text *obj.Prog) *unit {
	u := &unit{sym: sym, object: objectName(sym)}
	u.hard = pinned[sym.Name] || wasm.NeedMoreStack(base.Ctxt, text)
	for p := text; p != nil; p = p.Link {
		switch p.As {
		case obj.ACALL:
			u.addCall(p.To)
		case wasm.ACALLNORESUME:
			u.hard = u.hard || p.To.Sym == ir.Syms.SigPanic
		case wasm.ARETUNWIND:
			u.hard = true
		case obj.ARET:
			if p.To.Type == obj.TYPE_MEM && p.To.Sym != nil {
				u.addSym(p.To.Sym)
			}
		}
	}
	return u
}

// addCall records the callee of a resume-point call. An indirect call and a
// call to deferreturn are yield sources of their own.
func (u *unit) addCall(to obj.Addr) {
	if to.Type != obj.TYPE_MEM || to.Sym == nil || to.Sym == ir.Syms.Deferreturn {
		u.hard = true
		return
	}
	u.addSym(to.Sym)
}

func (u *unit) addSym(sym *obj.LSym) {
	if !slices.Contains(u.calls, sym) {
		u.calls = append(u.calls, sym)
	}
}
