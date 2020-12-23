// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// evalunsafe evaluates a package unsafe operation and returns the result.
func evalunsafe(n ir.Node) int64 {
	switch n.Op() {
	case ir.OALIGNOF, ir.OSIZEOF:
		n := n.(*ir.UnaryExpr)
		n.X = typecheck(n.X, ctxExpr)
		n.X = defaultlit(n.X, nil)
		tr := n.X.Type()
		if tr == nil {
			return 0
		}
		types.CalcSize(tr)
		if n.Op() == ir.OALIGNOF {
			return int64(tr.Align)
		}
		return tr.Width

	case ir.OOFFSETOF:
		// must be a selector.
		n := n.(*ir.UnaryExpr)
		if n.X.Op() != ir.OXDOT {
			base.Errorf("invalid expression %v", n)
			return 0
		}
		sel := n.X.(*ir.SelectorExpr)

		// Remember base of selector to find it back after dot insertion.
		// Since r->left may be mutated by typechecking, check it explicitly
		// first to track it correctly.
		sel.X = typecheck(sel.X, ctxExpr)
		sbase := sel.X

		tsel := typecheck(sel, ctxExpr)
		n.X = tsel
		if tsel.Type() == nil {
			return 0
		}
		switch tsel.Op() {
		case ir.ODOT, ir.ODOTPTR:
			break
		case ir.OCALLPART:
			base.Errorf("invalid expression %v: argument is a method value", n)
			return 0
		default:
			base.Errorf("invalid expression %v", n)
			return 0
		}

		// Sum offsets for dots until we reach sbase.
		var v int64
		var next ir.Node
		for r := tsel; r != sbase; r = next {
			switch r.Op() {
			case ir.ODOTPTR:
				// For Offsetof(s.f), s may itself be a pointer,
				// but accessing f must not otherwise involve
				// indirection via embedded pointer types.
				r := r.(*ir.SelectorExpr)
				if r.X != sbase {
					base.Errorf("invalid expression %v: selector implies indirection of embedded %v", n, r.X)
					return 0
				}
				fallthrough
			case ir.ODOT:
				r := r.(*ir.SelectorExpr)
				v += r.Offset
				next = r.X
			default:
				ir.Dump("unsafenmagic", tsel)
				base.Fatalf("impossible %v node after dot insertion", r.Op())
			}
		}
		return v
	}

	base.Fatalf("unexpected op %v", n.Op())
	return 0
}
