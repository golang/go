// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package devirtualize implements a simple "devirtualization"
// optimization pass, which replaces interface method calls with
// direct concrete-type method calls where possible.
package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// Func devirtualizes calls within fn where possible.
func Func(fn *ir.Func) {
	ir.CurFunc = fn
	ir.VisitList(fn.Body, func(n ir.Node) {
		if call, ok := n.(*ir.CallExpr); ok {
			Call(call)
		}
	})
}

// Call devirtualizes the given call if possible.
func Call(call *ir.CallExpr) {
	if call.Op() != ir.OCALLINTER {
		return
	}
	sel := call.X.(*ir.SelectorExpr)
	r := ir.StaticValue(sel.X)
	if r.Op() != ir.OCONVIFACE {
		return
	}
	recv := r.(*ir.ConvExpr)

	typ := recv.X.Type()
	if typ.IsInterface() {
		return
	}

	dt := ir.NewTypeAssertExpr(sel.Pos(), sel.X, nil)
	dt.SetType(typ)
	x := typecheck.Callee(ir.NewSelectorExpr(sel.Pos(), ir.OXDOT, dt, sel.Sel))
	switch x.Op() {
	case ir.ODOTMETH:
		x := x.(*ir.SelectorExpr)
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "devirtualizing %v to %v", sel, typ)
		}
		call.SetOp(ir.OCALLMETH)
		call.X = x
	case ir.ODOTINTER:
		// Promoted method from embedded interface-typed field (#42279).
		x := x.(*ir.SelectorExpr)
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "partially devirtualizing %v to %v", sel, typ)
		}
		call.SetOp(ir.OCALLINTER)
		call.X = x
	default:
		// TODO(mdempsky): Turn back into Fatalf after more testing.
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "failed to devirtualize %v (%v)", x, x.Op())
		}
		return
	}

	// Duplicated logic from typecheck for function call return
	// value types.
	//
	// Receiver parameter size may have changed; need to update
	// call.Type to get correct stack offsets for result
	// parameters.
	types.CheckSize(x.Type())
	switch ft := x.Type(); ft.NumResults() {
	case 0:
	case 1:
		call.SetType(ft.Results().Field(0).Type)
	default:
		call.SetType(ft.Results())
	}
}
