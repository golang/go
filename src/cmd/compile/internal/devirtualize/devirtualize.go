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
)

const go126ImprovedConcreteTypeAnalysis = true

// StaticCall devirtualizes the given call if possible when the concrete callee
// is available statically.
func StaticCall(s *State, call *ir.CallExpr) {
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
	if go126ImprovedConcreteTypeAnalysis {
		typ = concreteType(s, sel.X)
		if typ == nil {
			return
		}

		// Don't create type-assertions that would be impossible at compile-time.
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

	if go126ImprovedConcreteTypeAnalysis {
		// Consider:
		//
		//	var v Iface
		//	v.A()
		//	v = &Impl{}
		//
		// Here in the devirtualizer, we determine the concrete type of v as being an *Impl,
		// but it can still be a nil interface, we have not detected that. The v.(*Impl)
		// type assertion that we make here would also have failed, but with a different
		// panic "pkg.Iface is nil, not *pkg.Impl", where previously we would get a nil panic.
		// We fix this, by introducing an additional nilcheck on the itab.
		// Calling a method on an nil interface (in most cases) is a bug in a program, so it is fine
		// to devirtualize and further (possibly) inline them, even though we would never reach
		// the called function.
		dt.UseNilPanic = true
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
func concreteType(s *State, n ir.Node) (typ *types.Type) {
	typ = concreteType1(s, n, make(map[*ir.Name]struct{}))
	if typ == &noType {
		return nil
	}
	if typ != nil && typ.IsInterface() {
		base.FatalfAt(n.Pos(), "typ.IsInterface() = true; want = false; typ = %v", typ)
	}
	return typ
}

// noType is a sentinel value returned by [concreteType1].
var noType types.Type

// concreteType1 analyzes the node n and returns its concrete type if it is statically known.
// Otherwise, it returns a nil Type, indicating that a concrete type was not determined.
// When n is known to be statically nil or a self-assignment is detected, in returns a sentinel [noType] type instead.
func concreteType1(s *State, n ir.Node, seen map[*ir.Name]struct{}) (outT *types.Type) {
	nn := n // for debug messages

	if concreteTypeDebug {
		defer func() {
			t := "&noType"
			if outT != &noType {
				t = outT.String()
			}
			base.Warn("concreteType1(%v) -> %v", nn, t)
		}()
	}

	for {
		if concreteTypeDebug {
			base.Warn("concreteType1(%v): analyzing %v", nn, n)
		}

		if !n.Type().IsInterface() {
			return n.Type()
		}

		switch n1 := n.(type) {
		case *ir.ConvExpr:
			if n1.Op() == ir.OCONVNOP {
				if !n1.Type().IsInterface() || !types.Identical(n1.Type().Underlying(), n1.X.Type().Underlying()) {
					// As we check (directly before this switch) whether n is an interface, thus we should only reach
					// here for iface conversions where both operands are the same.
					base.FatalfAt(n1.Pos(), "not identical/interface types found n1.Type = %v; n1.X.Type = %v", n1.Type(), n1.X.Type())
				}
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
			n = n1.X
			continue
		}
		break
	}

	if n.Op() != ir.ONAME {
		return nil
	}

	name := n.(*ir.Name).Canonical()
	if name.Class != ir.PAUTO {
		return nil
	}

	if name.Op() != ir.ONAME {
		base.FatalfAt(name.Pos(), "name.Op = %v; want = ONAME", n.Op())
	}

	// name.Curfn must be set, as we checked name.Class != ir.PAUTO before.
	if name.Curfn == nil {
		base.FatalfAt(name.Pos(), "name.Curfn = nil; want not nil")
	}

	if name.Addrtaken() {
		return nil // conservatively assume it's reassigned with a different type indirectly
	}

	if _, ok := seen[name]; ok {
		return &noType // Already analyzed assignments to name, no need to do that twice.
	}
	seen[name] = struct{}{}

	if concreteTypeDebug {
		base.Warn("concreteType1(%v): analyzing assignments to %v", nn, name)
	}

	var typ *types.Type
	for _, v := range s.assignments(name) {
		var t *types.Type
		switch v := v.(type) {
		case *types.Type:
			t = v
		case ir.Node:
			t = concreteType1(s, v, seen)
			if t == &noType {
				continue
			}
		}
		if t == nil || (typ != nil && !types.Identical(typ, t)) {
			return nil
		}
		typ = t
	}

	if typ == nil {
		// Variable either declared with zero value, or only assigned with nil.
		return &noType
	}

	return typ
}

// assignment can be one of:
// - nil - assignment from an interface type.
// - *types.Type - assignment from a concrete type (non-interface).
// - ir.Node - assignment from a ir.Node.
//
// In most cases assignment should be an [ir.Node], but in cases where we
// do not follow the data-flow, we return either a concrete type (*types.Type) or a nil.
// For example in range over a slice, if the slice elem is of an interface type, then we return
// a nil, otherwise the elem's concrete type (We do so because we do not analyze assignment to the
// slice being ranged-over).
type assignment any

// State holds precomputed state for use in [StaticCall].
type State struct {
	// ifaceAssignments maps interface variables to all their assignments
	// defined inside functions stored in the analyzedFuncs set.
	// Note: it does not include direct assignments to nil.
	ifaceAssignments map[*ir.Name][]assignment

	// ifaceCallExprAssigns stores every [*ir.CallExpr], which has an interface
	// result, that is assigned to a variable.
	ifaceCallExprAssigns map[*ir.CallExpr][]ifaceAssignRef

	// analyzedFuncs is a set of Funcs that were analyzed for iface assignments.
	analyzedFuncs map[*ir.Func]struct{}
}

type ifaceAssignRef struct {
	name            *ir.Name // ifaceAssignments[name]
	assignmentIndex int      // ifaceAssignments[name][assignmentIndex]
	returnIndex     int      // (*ir.CallExpr).Result(returnIndex)
}

// InlinedCall updates the [State] to take into account a newly inlined call.
func (s *State) InlinedCall(fun *ir.Func, origCall *ir.CallExpr, inlinedCall *ir.InlinedCallExpr) {
	if _, ok := s.analyzedFuncs[fun]; !ok {
		// Full analyze has not been yet executed for the provided function, so we can skip it for now.
		// When no devirtualization happens in a function, it is unnecessary to analyze it.
		return
	}

	// Analyze assignments in the newly inlined function.
	s.analyze(inlinedCall.Init())
	s.analyze(inlinedCall.Body)

	refs, ok := s.ifaceCallExprAssigns[origCall]
	if !ok {
		return
	}
	delete(s.ifaceCallExprAssigns, origCall)

	// Update assignments to reference the new ReturnVars of the inlined call.
	for _, ref := range refs {
		vt := &s.ifaceAssignments[ref.name][ref.assignmentIndex]
		if *vt != nil {
			base.Fatalf("unexpected non-nil assignment")
		}
		if concreteTypeDebug {
			base.Warn(
				"InlinedCall(%v, %v): replacing interface node in (%v,%v) to %v (typ %v)",
				origCall, inlinedCall, ref.name, ref.assignmentIndex,
				inlinedCall.ReturnVars[ref.returnIndex],
				inlinedCall.ReturnVars[ref.returnIndex].Type(),
			)
		}

		// Update ifaceAssignments with an ir.Node from the inlined function’s ReturnVars.
		// This may enable future devirtualization of calls that reference ref.name.
		// We will get calls to [StaticCall] from the interleaved package,
		// to try devirtualize such calls afterwards.
		*vt = inlinedCall.ReturnVars[ref.returnIndex]
	}
}

// assignments returns all assignments to n.
func (s *State) assignments(n *ir.Name) []assignment {
	fun := n.Curfn
	if fun == nil {
		base.FatalfAt(n.Pos(), "n.Curfn = <nil>")
	}
	if n.Class != ir.PAUTO {
		base.FatalfAt(n.Pos(), "n.Class = %v; want = PAUTO", n.Class)
	}

	if !n.Type().IsInterface() {
		base.FatalfAt(n.Pos(), "name passed to assignments is not of an interface type: %v", n.Type())
	}

	// Analyze assignments in func, if not analyzed before.
	if _, ok := s.analyzedFuncs[fun]; !ok {
		if concreteTypeDebug {
			base.Warn("assignments(): analyzing assignments in %v func", fun)
		}
		if s.analyzedFuncs == nil {
			s.ifaceAssignments = make(map[*ir.Name][]assignment)
			s.ifaceCallExprAssigns = make(map[*ir.CallExpr][]ifaceAssignRef)
			s.analyzedFuncs = make(map[*ir.Func]struct{})
		}
		s.analyzedFuncs[fun] = struct{}{}
		s.analyze(fun.Init())
		s.analyze(fun.Body)
	}

	return s.ifaceAssignments[n]
}

// analyze analyzes every assignment to interface variables in nodes, updating [State].
func (s *State) analyze(nodes ir.Nodes) {
	assign := func(name ir.Node, assignment assignment) (*ir.Name, int) {
		if name == nil || name.Op() != ir.ONAME || ir.IsBlank(name) {
			return nil, -1
		}

		n, ok := ir.OuterValue(name).(*ir.Name)
		if !ok || n.Curfn == nil {
			return nil, -1
		}

		// Do not track variables that are not of interface types.
		// For devirtualization they are unnecessary, we will not even look them up.
		if !n.Type().IsInterface() {
			return nil, -1
		}

		n = n.Canonical()
		if n.Op() != ir.ONAME {
			base.FatalfAt(n.Pos(), "n.Op = %v; want = ONAME", n.Op())
		}
		if n.Class != ir.PAUTO {
			return nil, -1
		}

		switch a := assignment.(type) {
		case nil:
		case *types.Type:
			if a != nil && a.IsInterface() {
				assignment = nil // non-concrete type
			}
		case ir.Node:
			// nil assignment, we can safely ignore them, see [StaticCall].
			if ir.IsNil(a) {
				return nil, -1
			}
		default:
			base.Fatalf("unexpected type: %v", assignment)
		}

		if concreteTypeDebug {
			base.Warn("analyze(): assignment found %v = %v", name, assignment)
		}

		s.ifaceAssignments[n] = append(s.ifaceAssignments[n], assignment)
		return n, len(s.ifaceAssignments[n]) - 1
	}

	var do func(n ir.Node)
	do = func(n ir.Node) {
		switch n.Op() {
		case ir.OAS:
			n := n.(*ir.AssignStmt)
			if rhs := n.Y; rhs != nil {
				for {
					if r, ok := rhs.(*ir.ParenExpr); ok {
						rhs = r.X
						continue
					}
					break
				}
				if call, ok := rhs.(*ir.CallExpr); ok && call.Fun != nil {
					retTyp := call.Fun.Type().Results()[0].Type
					n, idx := assign(n.X, retTyp)
					if n != nil && retTyp.IsInterface() {
						// We have a call expression, that returns an interface, store it for later evaluation.
						// In case this func gets inlined later, we will update the assignment (added before)
						// with a reference to ReturnVars, see [State.InlinedCall], which might allow for future devirtualizing of n.X.
						s.ifaceCallExprAssigns[call] = append(s.ifaceCallExprAssigns[call], ifaceAssignRef{n, idx, 0})
					}
				} else {
					assign(n.X, rhs)
				}
			}
		case ir.OAS2:
			n := n.(*ir.AssignListStmt)
			for i, p := range n.Lhs {
				if n.Rhs[i] != nil {
					assign(p, n.Rhs[i])
				}
			}
		case ir.OAS2DOTTYPE:
			n := n.(*ir.AssignListStmt)
			if n.Rhs[0] == nil {
				base.FatalfAt(n.Pos(), "n.Rhs[0] == nil; n = %v", n)
			}
			assign(n.Lhs[0], n.Rhs[0])
			assign(n.Lhs[1], nil) // boolean does not have methods to devirtualize
		case ir.OAS2MAPR, ir.OAS2RECV, ir.OSELRECV2:
			n := n.(*ir.AssignListStmt)
			if n.Rhs[0] == nil {
				base.FatalfAt(n.Pos(), "n.Rhs[0] == nil; n = %v", n)
			}
			assign(n.Lhs[0], n.Rhs[0].Type())
			assign(n.Lhs[1], nil) // boolean does not have methods to devirtualize
		case ir.OAS2FUNC:
			n := n.(*ir.AssignListStmt)
			rhs := n.Rhs[0]
			for {
				if r, ok := rhs.(*ir.ParenExpr); ok {
					rhs = r.X
					continue
				}
				break
			}
			if call, ok := rhs.(*ir.CallExpr); ok {
				for i, p := range n.Lhs {
					retTyp := call.Fun.Type().Results()[i].Type
					n, idx := assign(p, retTyp)
					if n != nil && retTyp.IsInterface() {
						// We have a call expression, that returns an interface, store it for later evaluation.
						// In case this func gets inlined later, we will update the assignment (added before)
						// with a reference to ReturnVars, see [State.InlinedCall], which might allow for future devirtualizing of n.X.
						s.ifaceCallExprAssigns[call] = append(s.ifaceCallExprAssigns[call], ifaceAssignRef{n, idx, i})
					}
				}
			} else if call, ok := rhs.(*ir.InlinedCallExpr); ok {
				for i, p := range n.Lhs {
					assign(p, call.ReturnVars[i])
				}
			} else {
				base.FatalfAt(n.Pos(), "unexpected type %T in OAS2FUNC Rhs[0]", call)
			}
		case ir.ORANGE:
			n := n.(*ir.RangeStmt)
			xTyp := n.X.Type()

			// Range over an array pointer.
			if xTyp.IsPtr() && xTyp.Elem().IsArray() {
				xTyp = xTyp.Elem()
			}

			if xTyp.IsArray() || xTyp.IsSlice() {
				assign(n.Key, nil) // integer does not have methods to devirtualize
				assign(n.Value, xTyp.Elem())
			} else if xTyp.IsChan() {
				assign(n.Key, xTyp.Elem())
				base.AssertfAt(n.Value == nil, n.Pos(), "n.Value != nil in range over chan")
			} else if xTyp.IsMap() {
				assign(n.Key, xTyp.Key())
				assign(n.Value, xTyp.Elem())
			} else if xTyp.IsInteger() || xTyp.IsString() {
				// Range over int/string, results do not have methods, so nothing to devirtualize.
				assign(n.Key, nil)
				assign(n.Value, nil)
			} else {
				// We will not reach here in case of an range-over-func, as it is
				// rewrtten to function calls in the noder package.
				base.FatalfAt(n.Pos(), "range over unexpected type %v", n.X.Type())
			}
		case ir.OSWITCH:
			n := n.(*ir.SwitchStmt)
			if guard, ok := n.Tag.(*ir.TypeSwitchGuard); ok {
				for _, v := range n.Cases {
					if v.Var == nil {
						base.Assert(guard.Tag == nil)
						continue
					}
					assign(v.Var, guard.X)
				}
			}
		case ir.OCLOSURE:
			n := n.(*ir.ClosureExpr)
			if _, ok := s.analyzedFuncs[n.Func]; !ok {
				s.analyzedFuncs[n.Func] = struct{}{}
				ir.Visit(n.Func, do)
			}
		}
	}
	ir.VisitList(nodes, do)
}
