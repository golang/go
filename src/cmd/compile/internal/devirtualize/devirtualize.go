// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package devirtualize implements two "devirtualization" optimization passes:
//
//   - "Static" devirtualization which replaces interface method calls with
//     direct concrete-type method calls where possible.
//   - "Profile-guided" devirtualization which replaces indirect calls with a
//     conditional direct call to the hottest concrete callee from a profile, as
//     well as a fallback using the original indirect call.
package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

const go125ImprovedConcreteTypeAnalysis = true

// StaticCall devirtualizes the given call if possible when the concrete callee
// is available statically.
func StaticCall(call *ir.CallExpr) {
	// For promoted methods (including value-receiver methods promoted
	// to pointer-receivers), the interface method wrapper may contain
	// expressions that can panic (e.g., ODEREF, ODOTPTR,
	// ODOTINTER). Devirtualization involves inlining these expressions
	// (and possible panics) to the call site. This normally isn't a
	// problem, but for go/defer statements it can move the panic from
	// when/where the call executes to the go/defer statement itself,
	// which is a visible change in semantics (e.g., #52072). To prevent
	// this, we skip devirtualizing calls within go/defer statements
	// altogether.
	if call.GoDefer {
		return
	}

	if call.Op() != ir.OCALLINTER {
		return
	}

	sel := call.Fun.(*ir.SelectorExpr)
	var typ *types.Type
	if go125ImprovedConcreteTypeAnalysis {
		typ = concreteType(sel.X)
		if typ == nil {
			return
		}

		// Don't try to devirtualize calls that we statically know that would have failed at runtime.
		// This can happen in such case: any(0).(interface {A()}).A(), this typechecks without
		// any errors, but will cause a runtime panic. We statically know that int(0) does not
		// implement that interface, thus we skip the devirtualization, as it is not possible
		// to make a type assertion from interface{A()} to int (int does not implement interface{A()}).
		if !typecheck.Implements(typ, sel.X.Type()) {
			return
		}
	} else {
		r := ir.StaticValue(sel.X)
		if r.Op() != ir.OCONVIFACE {
			return
		}
		recv := r.(*ir.ConvExpr)
		typ = recv.X.Type()
		if typ.IsInterface() {
			return
		}
	}

	// If typ is a shape type, then it was a type argument originally
	// and we'd need an indirect call through the dictionary anyway.
	// We're unable to devirtualize this call.
	if typ.IsShape() {
		return
	}

	// If typ *has* a shape type, then it's a shaped, instantiated
	// type like T[go.shape.int], and its methods (may) have an extra
	// dictionary parameter. We could devirtualize this call if we
	// could derive an appropriate dictionary argument.
	//
	// TODO(mdempsky): If typ has a promoted non-generic method,
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

	dt := ir.NewTypeAssertExpr(sel.Pos(), sel.X, nil)
	dt.SetType(typ)
	x := typecheck.XDotMethod(sel.Pos(), dt, sel.Sel, true)
	switch x.Op() {
	case ir.ODOTMETH:
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "devirtualizing %v to %v", sel, typ)
		}
		call.SetOp(ir.OCALLMETH)
		call.Fun = x
	case ir.ODOTINTER:
		// Promoted method from embedded interface-typed field (#42279).
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "partially devirtualizing %v to %v", sel, typ)
		}
		call.SetOp(ir.OCALLINTER)
		call.Fun = x
	default:
		base.FatalfAt(call.Pos(), "failed to devirtualize %v (%v)", x, x.Op())
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
		call.SetType(ft.Result(0).Type)
	default:
		call.SetType(ft.ResultsTuple())
	}

	// Desugar OCALLMETH, if we created one (#57309).
	typecheck.FixMethodCall(call)
}

const concreteTypeDebug = false

// concreteType determines the concrete type of n, following OCONVIFACEs and type asserts.
// Returns nil when the concrete type could not be determined, or when there are multiple
// (different) types assigned to an interface.
func concreteType(n ir.Node) (typ *types.Type) {
	return concreteType1(n, make(map[*ir.Name]*types.Type))
}

func concreteType1(n ir.Node, analyzed map[*ir.Name]*types.Type) (typ *types.Type) {
	nn := n // copy for debug messages

	if concreteTypeDebug {
		defer func() {
			if typ == nil {
				base.WarnfAt(n.Pos(), "%v concrete type not found", nn)
			} else {
				base.WarnfAt(n.Pos(), "%v found concrete type %v", nn, typ)
			}
		}()
	}

	for {
		if concreteTypeDebug {
			base.WarnfAt(n.Pos(), "%v analyzing concrete type of %v", nn, n)
		}

		switch n1 := n.(type) {
		case *ir.ConvExpr:
			if n1.Op() == ir.OCONVNOP && types.Identical(n1.Type(), n1.X.Type()) {
				n = n1.X
				continue
			}
			if n1.Op() == ir.OCONVIFACE {
				n = n1.X
				continue
			}
		case *ir.InlinedCallExpr:
			if n1.Op() == ir.OINLCALL {
				n = n1.SingleResult()
				continue
			}
		case *ir.ParenExpr:
			n = n1.X
			continue
		case *ir.TypeAssertExpr:
			if !n.Type().IsInterface() {
				// Asserting to a static type, take use of that as this will
				// cause a runtime panic, if not satisfied.
				return n.Type()
			}
			n = n1.X
			continue
		case *ir.CallExpr:
			if n1.Fun != nil {
				results := n1.Fun.Type().Results()
				if len(results) == 1 {
					retTyp := results[0].Type
					if !retTyp.IsInterface() {
						return retTyp
					}
				}
			}
			return nil
		}

		if !n.Type().IsInterface() {
			return n.Type()
		}

		return concreteType2(n, analyzed)
	}
}

func concreteType2(n ir.Node, analyzed map[*ir.Name]*types.Type) *types.Type {
	if n.Op() != ir.ONAME {
		return nil
	}

	name := n.(*ir.Name).Canonical()
	if name.Class != ir.PAUTO {
		return nil
	}

	if name.Op() != ir.ONAME {
		base.Fatalf("reassigned %v", name)
	}

	if name.Addrtaken() {
		return nil // conservatively assume it's reassigned with a different type indirectly
	}

	if typ, ok := analyzed[name]; ok {
		return typ
	}

	// For now set the Type to nil, as we don't know it yet, we will update
	// it at the end of this function, if we find a concrete type.
	// This is not ideal, as in-process concreteType1 calls (that this function also
	// executes) will get a nil (from the map lookup above), where we could determine the type.
	analyzed[name] = nil

	if concreteTypeDebug {
		base.WarnfAt(name.Pos(), "analyzing assignments to %v", name)
	}

	// isName reports whether n is a reference to name.
	isName := func(x ir.Node) bool {
		if x == nil {
			return false
		}
		n, ok := ir.OuterValue(x).(*ir.Name)
		return ok && n.Canonical() == name
	}

	var typ *types.Type

	handleType := func(pos src.XPos, t *types.Type) bool {
		if t == nil || t.IsInterface() {
			if concreteTypeDebug {
				base.WarnfAt(pos, "%v assigned with a non concrete type", name)
			}
			typ = nil
			return true
		}

		if concreteTypeDebug {
			base.WarnfAt(pos, "%v assigned with a concrete type %v", name, t)
		}

		if typ == nil || types.Identical(typ, t) {
			typ = t
			return false
		}

		// Different type.
		typ = nil
		return true
	}

	handleNode := func(n ir.Node) bool {
		if n == nil {
			return false
		}
		if concreteTypeDebug {
			base.WarnfAt(n.Pos(), "%v found assignment %v = %v, analyzing the RHS node", name, name, n)
		}
		return handleType(n.Pos(), concreteType1(n, analyzed))
	}

	var do func(n ir.Node) bool
	do = func(n ir.Node) bool {
		switch n.Op() {
		case ir.OAS:
			n := n.(*ir.AssignStmt)
			if isName(n.X) {
				return handleNode(n.Y)
			}
		case ir.OAS2:
			n := n.(*ir.AssignListStmt)
			for i, p := range n.Lhs {
				if isName(p) {
					return handleNode(n.Rhs[i])
				}
			}
		case ir.OAS2DOTTYPE:
			n := n.(*ir.AssignListStmt)
			for _, p := range n.Lhs {
				if isName(p) {
					return handleNode(n.Rhs[0])
				}
			}
		case ir.OAS2FUNC:
			n := n.(*ir.AssignListStmt)
			for i, p := range n.Lhs {
				if isName(p) {
					rhs := n.Rhs[0]
					for {
						if r, ok := rhs.(*ir.ParenExpr); ok {
							rhs = r.X
							continue
						}
						break
					}
					if call, ok := rhs.(*ir.CallExpr); ok {
						retTyp := call.Fun.Type().Results()[i].Type
						if !retTyp.IsInterface() {
							return handleType(n.Pos(), retTyp)
						}
					}
					typ = nil
					return true
				}
			}
		case ir.OAS2MAPR, ir.OAS2RECV, ir.OSELRECV2:
			n := n.(*ir.AssignListStmt)
			for _, p := range n.Lhs {
				if isName(p) {
					return handleType(n.Pos(), n.Rhs[0].Type())
				}
			}
		case ir.OADDR:
			n := n.(*ir.AddrExpr)
			if isName(n.X) {
				base.FatalfAt(n.Pos(), "%v not marked addrtaken", name)
			}
		case ir.ORANGE:
			n := n.(*ir.RangeStmt)
			xTyp := n.X.Type()

			// range over an array pointer
			if xTyp.IsPtr() && xTyp.Elem().IsArray() {
				xTyp = xTyp.Elem()
			}

			if xTyp.IsArray() || xTyp.IsSlice() {
				if isName(n.Key) {
					// This is an index, int has no methods, so nothing to devirtualize.
					typ = nil
					return true
				}
				if isName(n.Value) {
					return handleType(n.Pos(), xTyp.Elem())
				}
			} else if xTyp.IsChan() {
				if isName(n.Key) {
					return handleType(n.Pos(), xTyp.Elem())
				}
				base.Assertf(n.Value == nil, "n.Value != nil in range over chan")
			} else if xTyp.IsMap() {
				if isName(n.Key) {
					return handleType(n.Pos(), xTyp.Key())
				}
				if isName(n.Value) {
					return handleType(n.Pos(), xTyp.Elem())
				}
			} else {
				// unknown type
				typ = nil
				return true
			}
		case ir.OCLOSURE:
			n := n.(*ir.ClosureExpr)
			if ir.Any(n.Func, do) {
				return true
			}
		}
		return false
	}

	ir.Any(name.Curfn, do)
	analyzed[name] = typ
	return typ
}
