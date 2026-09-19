// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package devirtualize

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// This file implements a type-set analysis shared by both interface
// devirtualizers.
//
// exprTypeSet determines the set of dynamic types an interface-typed
// expression may hold, following conversions, inlined calls, and the
// assignments [State.analyze] records for local variables.
//
// AnalyzeResultTypes determines, for every interface-typed result of
// every function, the set of dynamic types the function's return
// statements supply for that result. A function whose set is one
// concrete type is a devirtualization opportunity for its callers:
// exprTypeSet resolves static calls through the recorded sets, so a
// method call on the result can be rewritten into a direct call by
// [StaticCall] without inlining the function's body. A set like
// {*T, nil} could become a guarded opportunity, for error or nil
// results in particular (currently not implemented).
//
// The interleaved devirtualization and inlining pass drives the
// analysis bottom-up, one scc at a time. When a return statement returns the
// results of a static call, the callee has hopefully been analyzed already,
// so we can use its dynamic result sets to analyze that call itself.
// Currently, we do not try to break sccs in this analysis.
//
// Unlike receiver devirtualization, the return-value analysis must
// treat the nil interface as a dynamic type of its own. concreteType
// may ignore nil members because calling a method on a nil interface
// panics no matter what. A caller of a returned interface value may
// instead compare it against nil, so a function that returns both nil
// and *T values must not be recorded as always returning *T. This is
// why [State.analyze] records nil assignments and exprTypeSet tracks
// them as set members, with concreteType alone discarding them.

// typeSet is the working alias for [ir.TypeSet]; the type and its
// sentinels live in ir so that [ir.Func] can hold them.
type typeSet = ir.TypeSet

// Sentinels for typeSet.
var nilType, unknownType = ir.TypeSetNil, ir.TypeSetUnknown

// exprTypeSet adds the abstract dynamic types of the expression e
// into set.
//
// Recursive; seen guards against cycles.
func (s *State) exprTypeSet(set typeSet, e ir.Node, seen map[*ir.Name]struct{}) (out typeSet) {
	nn := e // for debug messages

	if concreteTypeDebug {
		defer func() {
			base.Warn("exprTypeSet(%v) -> {%v}", nn, out)
		}()
	}

	for {
		if concreteTypeDebug {
			base.Warn("exprTypeSet(%v): analyzing %v", nn, e)
		}

		if !e.Type().IsInterface() {
			return set.Add(e.Type())
		}

		switch n := e.(type) {
		case *ir.ConvExpr:
			if n.Op() == ir.OCONVNOP {
				if !n.Type().IsInterface() || !types.Identical(n.Type().Underlying(), n.X.Type().Underlying()) {
					// As we check (directly before this switch) whether n is an interface, thus we should only reach
					// here for iface conversions where both operands are the same.
					base.FatalfAt(n.Pos(), "not identical/interface types found n.Type = %v; n.X.Type = %v", n.Type(), n.X.Type())
				}
				e = n.X
				continue
			}
			if n.Op() == ir.OCONVIFACE {
				e = n.X
				continue
			}
		case *ir.InlinedCallExpr:
			if n.Op() == ir.OINLCALL {
				e = n.SingleResult()
				continue
			}
		case *ir.ParenExpr:
			e = n.X
			continue
		case *ir.TypeAssertExpr:
			e = n.X
			continue
		}
		break
	}

	if ir.IsNil(e) {
		return set.Add(nilType)
	}

	switch e := e.(type) {
	case *ir.CallExpr:
		return s.callResultTypeSet(set, e, 0)
	case *ir.Name:
		return s.nameTypeSet(set, e, seen)
	}

	return set.Add(unknownType)
}

// nameTypeSet adds the abstract dynamic types of every assignment to
// the local variable n into set.
func (s *State) nameTypeSet(set typeSet, n *ir.Name, seen map[*ir.Name]struct{}) typeSet {
	name := n.Canonical()
	if name.Class != ir.PAUTO {
		return set.Add(unknownType)
	}

	if name.Op() != ir.ONAME {
		base.FatalfAt(name.Pos(), "name.Op = %v; want = ONAME", n.Op())
	}

	// name.Curfn must be set, as we checked name.Class != ir.PAUTO before.
	if name.Curfn == nil {
		base.FatalfAt(name.Pos(), "name.Curfn = nil; want not nil")
	}

	if name.Addrtaken() {
		return set.Add(unknownType) // conservatively assume it's reassigned with a different type indirectly
	}

	if _, ok := seen[name]; ok {
		return set // already analyzed assignments to name; a cycle adds no members
	}
	seen[name] = struct{}{}

	if concreteTypeDebug {
		base.Warn("nameTypeSet: analyzing assignments to %v", name)
	}

	for _, v := range s.assignments(name) {
		switch v := v.(type) {
		case nil:
			set = set.Add(unknownType)
		case *types.Type:
			set = set.Add(v)
		case result:
			set = s.callResultTypeSet(set, v.call, v.index)
		case ir.Node:
			set = s.exprTypeSet(set, v, seen)
		}
	}
	return set
}

// callResultTypeSet adds the abstract dynamic types of the i'th result of a
// call into set, using the result sets recorded for the statically known callee
// by [State.AnalyzeResultTypes].
func (s *State) callResultTypeSet(set typeSet, call *ir.CallExpr, i int) typeSet {
	callee := staticCallee(call)
	if callee == nil {
		if concreteTypeDebug {
			base.Warn("callResultTypeSet(%v): callee not statically known", call)
		}
		return set.Add(unknownType)
	}

	rts := callee.ResultTypeSets
	if rts == nil || len(rts[i]) == 0 {
		if concreteTypeDebug {
			base.Warn("callResultTypeSet(%v): no recorded result types for %v", call, callee)
		}
		return set.Add(unknownType)
	}

	if concreteTypeDebug {
		base.Warn("callResultTypeSet(%v): result #%d of %v is {%v}", call, i, callee, rts[i])
	}
	for _, t := range rts[i] {
		set = set.Add(t)
	}
	return set
}

// staticCallee finds the function a call statically resolves to, or
// nil if the callee is not statically known.
func staticCallee(call *ir.CallExpr) *ir.Func {
	if call.Op() != ir.OCALLFUNC {
		return nil
	}

	switch fun := ir.StaticValue(call.Fun); fun.Op() {
	case ir.ONAME:
		if fun := fun.(*ir.Name); fun.Class == ir.PFUNC {
			return fun.Func
		}
	case ir.OMETHEXPR:
		if name := ir.MethodExprName(fun); name != nil {
			return name.Func
		}
	}
	return nil
}

// AnalyzeResultTypes records the sets of dynamic types fn's interface-typed
// results may hold.
//
// It must run after fn's body has reached its final form for the interleaved
// pass, and after the functions fn calls have been analyzed; the pass's
// bottom-up walk over strongly connected components provides both.
func (s *State) AnalyzeResultTypes(fn *ir.Func) {
	if len(fn.Body) == 0 {
		return // Impl in assembly or elsewhere, result type unknowable.
	}

	results := fn.Type().Results()
	hasIface := false
	for _, f := range results {
		if f.Type.IsInterface() {
			hasIface = true
			break
		}
	}
	if !hasIface {
		return // Nothing to do.
	}

	// Add the returned value's abstract types into rets at every
	// return statement, and watch for defers along the way. The walk
	// does not descend into closure bodies, so every ORETURN seen
	// here returns from fn itself.
	rets := make([]typeSet, len(results))
	hasDefer := false
	ir.VisitList(fn.Body, func(n ir.Node) {
		switch n.Op() {
		case ir.ODEFER:
			hasDefer = true
		case ir.ORETURN:
			ret := n.(*ir.ReturnStmt)
			if len(ret.Results) != len(rets) {
				// A bare return reads the named result variables, whose
				// assignments this analysis does not follow. Multi-valued
				// "return g()" does not reach here: typecheck has already
				// rewritten it into temporaries defined by an OAS2FUNC,
				// which exprTypeSet follows through their recorded
				// assignments.
				//
				// TODO(mcy): follow assignments to named results;
				// [State.analyze] only tracks PAUTO variables today.
				for i := range rets {
					if results[i].Type.IsInterface() {
						rets[i] = rets[i].Add(unknownType)
					}
				}
				return
			}

			for i, e := range ret.Results {
				if results[i].Type.IsInterface() {
					rets[i] = s.exprTypeSet(rets[i], e, make(map[*ir.Name]struct{}))
				}
			}
		}
	})

	if hasDefer {
		// A deferred call may recover a panic, making fn return the
		// zero value of every result, and the zero value of an
		// interface is nil. The recorded sets would be missing nil.
		//
		// This cannot use fn.HasDefer, which walk has not set yet.
		//
		// TODO(mcy): relax this. A defer only invalidates the
		// analysis when it can call recover or write to a named
		// result. When every result is unnamed, a recover only adds
		// nil to each set; and a deferred static call to a function
		// known not to recover changes nothing at all.
		if concreteTypeDebug {
			base.Warn("AnalyzeResultTypes(%v): has defer, results not recorded", fn)
		}
		return
	}

	fn.ResultTypeSets = rets

	if concreteTypeDebug {
		for i, set := range rets {
			if results[i].Type.IsInterface() {
				base.Warn("AnalyzeResultTypes(%v): result #%d = {%v}", fn, i, set)
			}
		}
	}

	if base.Flag.LowerM != 0 {
		for i, set := range rets {
			switch {
			case len(set) == 0, set[0] == unknownType:
			case len(set) == 1:
				base.WarnfAt(fn.Pos(), "result #%d of %v is always %s", i, fn, set)
			default:
				base.WarnfAt(fn.Pos(), "result #%d of %v is one of %v", i, fn, set)
			}
		}
	}
}
