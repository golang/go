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

	// For promoted methods (including value-receiver methods promoted to pointer-receivers),
	// the interface method wrapper may contain expressions that can panic (e.g., ODEREF, ODOTPTR, ODOTINTER).
	// Devirtualization involves inlining these expressions (and possible panics) to the call site.
	// This normally isn't a problem, but for go/defer statements it can move the panic from when/where
	// the call executes to the go/defer statement itself, which is a visible change in semantics (e.g., #52072).
	// To prevent this, we skip devirtualizing calls within go/defer statements altogether.
	goDeferCall := make(map[*ir.CallExpr]bool)
	ir.VisitList(fn.Body, func(n ir.Node) {
		switch n := n.(type) {
		case *ir.GoDeferStmt:
			if call, ok := n.Call.(*ir.CallExpr); ok {
				goDeferCall[call] = true
			}
			return
		case *ir.CallExpr:
			if !goDeferCall[n] {
				Call(n)
			}
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

	if base.Debug.Unified != 0 {
		// N.B., stencil.go converts shape-typed values to interface type
		// using OEFACE instead of OCONVIFACE, so devirtualization fails
		// above instead. That's why this code is specific to unified IR.

		// If typ is a shape type, then it was a type argument originally
		// and we'd need an indirect call through the dictionary anyway.
		// We're unable to devirtualize this call.
		if typ.IsShape() {
			return
		}

		// If typ *has* a shape type, then it's an shaped, instantiated
		// type like T[go.shape.int], and its methods (may) have an extra
		// dictionary parameter. We could devirtualize this call if we
		// could derive an appropriate dictionary argument.
		//
		// TODO(mdempsky): If typ has has a promoted non-generic method,
		// then that method won't require a dictionary argument. We could
		// still devirtualize those calls.
		//
		// TODO(mdempsky): We have the *runtime.itab in recv.TypeWord. It
		// should be possible to compute the represented type's runtime
		// dictionary from this (e.g., by adding a pointer from T[int]'s
		// *runtime._type to .dict.T[int]; or by recognizing static
		// references to go:itab.T[int],iface and constructing a direct
		// reference to .dict.T[int]).
		if typ.HasShape() {
			if base.Flag.LowerM != 0 {
				base.WarnfAt(call.Pos(), "cannot devirtualize %v: shaped receiver %v", call, typ)
			}
			return
		}

		// Further, if sel.X's type has a shape type, then it's a shaped
		// interface type. In this case, the (non-dynamic) TypeAssertExpr
		// we construct below would attempt to create an itab
		// corresponding to this shaped interface type; but the actual
		// itab pointer in the interface value will correspond to the
		// original (non-shaped) interface type instead. These are
		// functionally equivalent, but they have distinct pointer
		// identities, which leads to the type assertion failing.
		//
		// TODO(mdempsky): We know the type assertion here is safe, so we
		// could instead set a flag so that walk skips the itab check. For
		// now, punting is easy and safe.
		if sel.X.Type().HasShape() {
			if base.Flag.LowerM != 0 {
				base.WarnfAt(call.Pos(), "cannot devirtualize %v: shaped interface %v", call, sel.X.Type())
			}
			return
		}
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
