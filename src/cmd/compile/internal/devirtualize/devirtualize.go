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

// TODO: update tests to include full errors.

// const go125ImprovedConcreteTypeAnalysis = true
var go125ImprovedConcreteTypeAnalysis = true

// StaticCall devirtualizes the given call if possible when the concrete callee
// is available statically.
func StaticCall(call *ir.CallExpr) {
	//go125ImprovedConcreteTypeAnalysis = base.Debug.Testing != 0
	//concreteTypeDebug = base.Debug.Testing != 0

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
		// to make an assertion: any(0).(interface{A()}).(int) (int does not implement interface{A()}).
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

	dt := ir.NewTypeAssertExpr(sel.Pos(), sel.X, typ)

	if go125ImprovedConcreteTypeAnalysis {
		// Consider:
		//
		//	var v Iface
		// 	v.A()
		// 	v = &Impl{}
		//
		// Here in the devirtualizer, we determine the concrete type of v as beeing an *Impl,
		// but in can still be a nil interface, we have not detected that. The v.(*Impl)
		// type assertion that we make here would also have failed, but with a different
		// panic "pkg.Iface is nil, not *pkg.Impl", where previously we would get a nil panic.
		// We fix this, by introducing an additional nilcheck on the itab.
		dt.EmitItabNilCheck = true
		dt.SetPos(call.Pos())
	}

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
		// TODO: with interleaved inlining and devirtualization we migth get here multiple times for the same call.
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

var concreteTypeDebug = false

// concreteType determines the concrete type of n, following OCONVIFACEs and type asserts.
// Returns nil when the concrete type could not be determined, or when there are multiple
// (different) types assigned to an interface.
func concreteType(n ir.Node) (typ *types.Type) {
	var assignements map[*ir.Name][]valOrTyp
	var baseFunc *ir.Func
	typ, isNil := concreteType1(n, make(map[*ir.Name]*types.Type), func(n *ir.Name) []valOrTyp {
		if assignements == nil {
			assignements = make(map[*ir.Name][]valOrTyp)
			if n.Curfn == nil {
				base.Fatalf("n.Curfn == nil: %v", n)
			}
			assignements = ifaceAssignments(n.Curfn)
			baseFunc = n.Curfn
		}
		if !n.Type().IsInterface() {
			base.Fatalf("name passed to getAssignements is not of an interface type: %v", n.Type())
		}
		if n.Curfn != baseFunc {
			base.Fatalf("unexpected n.Curfn = %v; want = %v", n.Curfn, baseFunc)
		}
		return assignements[n]
	})
	if isNil && typ != nil {
		base.Fatalf("typ = %v; want = <nil>", typ)
	}
	if typ != nil && typ.IsInterface() {
		base.Fatalf("typ.IsInterface() = true; want = false; typ = %v", typ)
	}
	return typ
}

func concreteType1(n ir.Node, analyzed map[*ir.Name]*types.Type, getAssignements func(*ir.Name) []valOrTyp) (out *types.Type, isNil bool) {
	nn := n

	if go125ImprovedConcreteTypeAnalysis {
		defer func() {
			if concreteTypeDebug {
				base.WarnfAt(nn.Pos(), "concreteType1(%v) -> (typ: %v; isNil: %v)", nn, out, isNil)
			}
		}()
	}

	for {
		if concreteTypeDebug {
			base.WarnfAt(n.Pos(), "concreteType1(%v) current iteration node: %v", nn, n)
		}

		if !n.Type().IsInterface() {
			if n, ok := n.(*ir.CallExpr); ok {
				if n.Fun != nil {
					results := n.Fun.Type().Results()
					base.Assert(len(results) == 1)
					retTyp := results[0].Type
					if !retTyp.IsInterface() {
						// TODO: isnt n.Type going to return that?
						return retTyp, false
					}
				}
			}
			return n.Type(), false
		}

		switch n1 := n.(type) {
		case *ir.ConvExpr:
			// OCONVNOP might change the type, thus check whether they are identical.
			// TODO: can this happen for iface?
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
				// TODO: single is fine?
				n = n1.SingleResult()
				continue
			}
		case *ir.ParenExpr:
			n = n1.X
			continue
		case *ir.TypeAssertExpr:
			n = n1.X
			continue
		}

		break
	}

	if n.Op() != ir.ONAME {
		return nil, false
	}

	name := n.(*ir.Name).Canonical()
	if name.Class != ir.PAUTO {
		return nil, false
	}

	if name.Op() != ir.ONAME {
		base.Fatalf("reassigned %v", name)
	}

	if name.Addrtaken() {
		return nil, false // conservatively assume it's reassigned with a different type indirectly
	}

	if name.Curfn == nil {
		// TODO: think, we should hit here with an iface global?
		base.Fatalf("nil Func %v", name)
	}

	if concreteTypeDebug {
		base.WarnfAt(src.NoXPos, "concreteType1(%v) name: %v, analyzing assignements", nn, name)
	}

	if typ, ok := analyzed[name]; ok {
		return typ, false
	}

	// For now set the Type to nil, as we don't know it yet, we will update
	// it at the end of this function, if we find a concrete type.
	// This is not ideal, as in-process concreteType1 calls (that this function also
	// executes) will get a nil (from the map lookup above), where we could determine the type.
	analyzed[name] = nil

	assignements := getAssignements(name)
	if len(assignements) == 0 {
		// Variable either declared with zero value, or only assigned
		// with nil (getAssignements does not return such assignements).
		if concreteTypeDebug {
			base.WarnfAt(src.NoXPos, "concreteType1(%v) name: %v assigned with only nil values", nn, name)
		}
		return nil, true
	}

	var typ *types.Type
	for _, v := range assignements {
		t := v.typ
		if v.node != nil {
			var isNil bool
			if concreteTypeDebug {
				base.WarnfAt(src.NoXPos, "concreteType1(%v) name: %v assigned with node %v", nn, name, v.node)
			}
			t, isNil = concreteType1(v.node, analyzed, getAssignements)
			if isNil {
				if concreteTypeDebug {
					base.WarnfAt(src.NoXPos, "concreteType1(%v) name: %v assigned with nil", nn, name)
				}
				if t != nil {
					base.Fatalf("t = %v; want = <nil>", t)
				}
				continue
			}
		}
		if concreteTypeDebug {
			base.WarnfAt(src.NoXPos, "concreteType1(%v) name: %v assigned with %v", nn, name, t)
		}
		if t == nil || (typ != nil && !types.Identical(typ, t)) {
			return nil, false
		}
		typ = t
	}

	if typ == nil {
		// Variable either declared with zero value, or only assigned with nil.
		return nil, true
	}

	analyzed[name] = typ
	return typ, false
}

type valOrTyp struct {
	// TODO: improve comment
	// either typ or node is populated, neither both, or
	// both are nil (either interface was assigned
	// or a basic type without methods (i.e. int))
	typ  *types.Type
	node ir.Node
}

// ifaceAssignments returns a map containg every assignement to variables
// directly declared in the provieded func that are of interface types.
func ifaceAssignments(fun *ir.Func) map[*ir.Name][]valOrTyp {
	out := make(map[*ir.Name][]valOrTyp)

	assign := func(name ir.Node, value valOrTyp) {
		if name == nil || name.Op() != ir.ONAME {
			return
		}

		// TODO: is this needed?
		n, ok := ir.OuterValue(name).(*ir.Name)
		if !ok {
			return
		}

		n = n.Canonical()
		if n.Op() != ir.ONAME {
			base.Fatalf("reassigned %v", n)
		}

		// Do not track variables not declared directly in fun, and names that are not interfaces.
		// For devirtualization they are unnecessary, we will not even look them up.
		if n.Curfn != fun || !n.Type().IsInterface() {
			return
		}

		// n is assigned with nil, we can safely ignore them, see [StaticCall].
		if ir.IsNil(value.node) {
			return
		}

		// TODO: add test case that fails w/o this.
		if value.typ != nil && value.typ.IsInterface() {
			value.typ = nil
		}

		out[n] = append(out[n], value)
	}

	var do func(n ir.Node)
	do = func(n ir.Node) {
		switch n.Op() {
		case ir.OAS:
			n := n.(*ir.AssignStmt)
			if n.Y != nil {
				assign(n.X, valOrTyp{node: n.Y})
			}
		case ir.OAS2:
			n := n.(*ir.AssignListStmt)
			for i, p := range n.Lhs {
				if n.Rhs[i] != nil {
					assign(p, valOrTyp{node: n.Rhs[i]})
				}
			}
		case ir.OAS2DOTTYPE:
			n := n.(*ir.AssignListStmt)
			if n.Rhs[0] == nil {
				base.Fatalf("n.Rhs[0] == nil; n = %v", n)
			}
			assign(n.Lhs[0], valOrTyp{node: n.Rhs[0]})
			assign(n.Lhs[1], valOrTyp{}) // boolean does not have methods to devirtualize
		case ir.OAS2MAPR, ir.OAS2RECV, ir.OSELRECV2:
			n := n.(*ir.AssignListStmt)
			if n.Rhs[0] == nil {
				base.Fatalf("n.Rhs[0] == nil; n = %v", n)
			}
			assign(n.Lhs[0], valOrTyp{typ: n.Rhs[0].Type()})
			assign(n.Lhs[1], valOrTyp{}) // boolean does not have methods to devirtualize
		case ir.OAS2FUNC:
			n := n.(*ir.AssignListStmt)
			for i, p := range n.Lhs {
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
					assign(p, valOrTyp{typ: retTyp})
				} else if call, ok := rhs.(*ir.InlinedCallExpr); ok {
					assign(p, valOrTyp{node: call.Result(i)})
				} else {
					// TODO: can we reach here? Fatal?
					assign(p, valOrTyp{})
				}
			}
		case ir.ORANGE:
			n := n.(*ir.RangeStmt)
			xTyp := n.X.Type()

			// range over an array pointer
			if xTyp.IsPtr() && xTyp.Elem().IsArray() {
				xTyp = xTyp.Elem()
			}

			if xTyp.IsArray() || xTyp.IsSlice() {
				assign(n.Key, valOrTyp{}) // boolean
				assign(n.Value, valOrTyp{typ: xTyp.Elem()})
			} else if xTyp.IsChan() {
				assign(n.Key, valOrTyp{typ: xTyp.Elem()})
				base.Assertf(n.Value == nil, "n.Value != nil in range over chan")
			} else if xTyp.IsMap() {
				assign(n.Key, valOrTyp{typ: xTyp.Key()})
				assign(n.Value, valOrTyp{typ: xTyp.Elem()})
			} else if xTyp.IsInteger() || xTyp.IsString() {
				// range over int/string, results do not have methods, so nothing to devirtualize.
				assign(n.Key, valOrTyp{})
				assign(n.Value, valOrTyp{})
			} else {
				base.Fatalf("range over unexpected type %v", n.X.Type())
			}
		case ir.OSWITCH:
			n := n.(*ir.SwitchStmt)
			if guard, ok := n.Tag.(*ir.TypeSwitchGuard); ok {
				for _, v := range n.Cases {
					if v.Var == nil {
						base.Assert(guard.Tag == nil)
						continue
					}
					assign(v.Var, valOrTyp{node: guard.X})
				}
			}
		case ir.OCLOSURE:
			ir.Visit(n.(*ir.ClosureExpr).Func, do)
		}
	}
	ir.Visit(fun, do)
	return out
}
