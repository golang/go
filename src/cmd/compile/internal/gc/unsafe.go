// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

// evalunsafe evaluates a package unsafe operation and returns the result.
func evalunsafe(n *ir.Node) int64 {
	switch n.Op {
	case ir.OALIGNOF, ir.OSIZEOF:
		n.Left = typecheck(n.Left, ctxExpr)
		n.Left = defaultlit(n.Left, nil)
		tr := n.Left.Type
		if tr == nil {
			return 0
		}
		dowidth(tr)
		if n.Op == ir.OALIGNOF {
			return int64(tr.Align)
		}
		return tr.Width

	case ir.OOFFSETOF:
		// must be a selector.
		if n.Left.Op != ir.OXDOT {
			base.Errorf("invalid expression %v", n)
			return 0
		}

		// Remember base of selector to find it back after dot insertion.
		// Since r->left may be mutated by typechecking, check it explicitly
		// first to track it correctly.
		n.Left.Left = typecheck(n.Left.Left, ctxExpr)
		sbase := n.Left.Left

		n.Left = typecheck(n.Left, ctxExpr)
		if n.Left.Type == nil {
			return 0
		}
		switch n.Left.Op {
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
		for r := n.Left; r != sbase; r = r.Left {
			switch r.Op {
			case ir.ODOTPTR:
				// For Offsetof(s.f), s may itself be a pointer,
				// but accessing f must not otherwise involve
				// indirection via embedded pointer types.
				if r.Left != sbase {
					base.Errorf("invalid expression %v: selector implies indirection of embedded %v", n, r.Left)
					return 0
				}
				fallthrough
			case ir.ODOT:
				v += r.Xoffset
			default:
				ir.Dump("unsafenmagic", n.Left)
				base.Fatalf("impossible %#v node after dot insertion", r.Op)
			}
		}
		return v
	}

	base.Fatalf("unexpected op %v", n.Op)
	return 0
}
