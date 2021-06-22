// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// call evaluates a call expressions, including builtin calls. ks
// should contain the holes representing where the function callee's
// results flows.
func (e *escape) call(ks []hole, call ir.Node) {
	e.callCommon(ks, call, nil)
}

func (e *escape) callCommon(ks []hole, call ir.Node, where *ir.GoDeferStmt) {
	argument := func(k hole, argp *ir.Node) {
		if where != nil {
			if where.Esc() == ir.EscNever {
				// Top-level defers arguments don't escape to heap,
				// but they do need to last until end of function.
				k = e.later(k)
			} else {
				k = e.heapHole()
			}
		}

		e.expr(k.note(call, "call parameter"), *argp)
	}

	switch call.Op() {
	default:
		ir.Dump("esc", call)
		base.Fatalf("unexpected call op: %v", call.Op())

	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		call := call.(*ir.CallExpr)
		typecheck.FixVariadicCall(call)

		// Pick out the function callee, if statically known.
		var fn *ir.Name
		switch call.Op() {
		case ir.OCALLFUNC:
			switch v := ir.StaticValue(call.X); {
			case v.Op() == ir.ONAME && v.(*ir.Name).Class == ir.PFUNC:
				fn = v.(*ir.Name)
			case v.Op() == ir.OCLOSURE:
				fn = v.(*ir.ClosureExpr).Func.Nname
			}
		case ir.OCALLMETH:
			fn = ir.MethodExprName(call.X)
		}

		fntype := call.X.Type()
		if fn != nil {
			fntype = fn.Type()
		}

		if ks != nil && fn != nil && e.inMutualBatch(fn) {
			for i, result := range fn.Type().Results().FieldSlice() {
				e.expr(ks[i], ir.AsNode(result.Nname))
			}
		}

		if r := fntype.Recv(); r != nil {
			argument(e.tagHole(ks, fn, r), &call.X.(*ir.SelectorExpr).X)
		} else {
			// Evaluate callee function expression.
			argument(e.discardHole(), &call.X)
		}

		args := call.Args
		for i, param := range fntype.Params().FieldSlice() {
			argument(e.tagHole(ks, fn, param), &args[i])
		}

	case ir.OAPPEND:
		call := call.(*ir.CallExpr)
		args := call.Args

		// Appendee slice may flow directly to the result, if
		// it has enough capacity. Alternatively, a new heap
		// slice might be allocated, and all slice elements
		// might flow to heap.
		appendeeK := ks[0]
		if args[0].Type().Elem().HasPointers() {
			appendeeK = e.teeHole(appendeeK, e.heapHole().deref(call, "appendee slice"))
		}
		argument(appendeeK, &args[0])

		if call.IsDDD {
			appendedK := e.discardHole()
			if args[1].Type().IsSlice() && args[1].Type().Elem().HasPointers() {
				appendedK = e.heapHole().deref(call, "appended slice...")
			}
			argument(appendedK, &args[1])
		} else {
			for i := 1; i < len(args); i++ {
				argument(e.heapHole(), &args[i])
			}
		}

	case ir.OCOPY:
		call := call.(*ir.BinaryExpr)
		argument(e.discardHole(), &call.X)

		copiedK := e.discardHole()
		if call.Y.Type().IsSlice() && call.Y.Type().Elem().HasPointers() {
			copiedK = e.heapHole().deref(call, "copied slice")
		}
		argument(copiedK, &call.Y)

	case ir.OPANIC:
		call := call.(*ir.UnaryExpr)
		argument(e.heapHole(), &call.X)

	case ir.OCOMPLEX:
		call := call.(*ir.BinaryExpr)
		argument(e.discardHole(), &call.X)
		argument(e.discardHole(), &call.Y)

	case ir.ODELETE, ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
		call := call.(*ir.CallExpr)
		fixRecoverCall(call)
		for i := range call.Args {
			argument(e.discardHole(), &call.Args[i])
		}

	case ir.OLEN, ir.OCAP, ir.OREAL, ir.OIMAG, ir.OCLOSE:
		call := call.(*ir.UnaryExpr)
		argument(e.discardHole(), &call.X)

	case ir.OUNSAFEADD, ir.OUNSAFESLICE:
		call := call.(*ir.BinaryExpr)
		argument(ks[0], &call.X)
		argument(e.discardHole(), &call.Y)
	}
}

func (e *escape) goDeferStmt(n *ir.GoDeferStmt) {
	topLevelDefer := n.Op() == ir.ODEFER && e.loopDepth == 1
	if topLevelDefer {
		// force stack allocation of defer record, unless
		// open-coded defers are used (see ssa.go)
		n.SetEsc(ir.EscNever)
	}

	e.stmts(n.Call.Init())
	e.callCommon(nil, n.Call, n)
}

// tagHole returns a hole for evaluating an argument passed to param.
// ks should contain the holes representing where the function
// callee's results flows. fn is the statically-known callee function,
// if any.
func (e *escape) tagHole(ks []hole, fn *ir.Name, param *types.Field) hole {
	// If this is a dynamic call, we can't rely on param.Note.
	if fn == nil {
		return e.heapHole()
	}

	if e.inMutualBatch(fn) {
		return e.addr(ir.AsNode(param.Nname))
	}

	// Call to previously tagged function.

	if fn.Func != nil && fn.Func.Pragma&ir.UintptrEscapes != 0 && (param.Type.IsUintptr() || param.IsDDD() && param.Type.Elem().IsUintptr()) {
		k := e.heapHole()
		k.uintptrEscapesHack = true
		return k
	}

	var tagKs []hole

	esc := parseLeaks(param.Note)
	if x := esc.Heap(); x >= 0 {
		tagKs = append(tagKs, e.heapHole().shift(x))
	}

	if ks != nil {
		for i := 0; i < numEscResults; i++ {
			if x := esc.Result(i); x >= 0 {
				tagKs = append(tagKs, ks[i].shift(x))
			}
		}
	}

	return e.teeHole(tagKs...)
}
