// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// expr models evaluating an expression n and flowing the result into
// hole k.
func (e *escape) expr(k hole, n ir.Node) {
	if n == nil {
		return
	}
	e.stmts(n.Init())
	e.exprSkipInit(k, n)
}

func (e *escape) exprSkipInit(k hole, n ir.Node) {
	if n == nil {
		return
	}

	lno := ir.SetPos(n)
	defer func() {
		base.Pos = lno
	}()

	if k.derefs >= 0 && !n.Type().IsUntyped() && !n.Type().HasPointers() {
		k.dst = &e.blankLoc
	}

	switch n.Op() {
	default:
		base.Fatalf("unexpected expr: %s %v", n.Op().String(), n)

	case ir.OLITERAL, ir.ONIL, ir.OGETG, ir.OGETCALLERPC, ir.OGETCALLERSP, ir.OTYPE, ir.OMETHEXPR, ir.OLINKSYMOFFSET:
		// nop

	case ir.ONAME:
		n := n.(*ir.Name)
		if n.Class == ir.PFUNC || n.Class == ir.PEXTERN {
			return
		}
		e.flow(k, e.oldLoc(n))

	case ir.OPLUS, ir.ONEG, ir.OBITNOT, ir.ONOT:
		n := n.(*ir.UnaryExpr)
		e.discard(n.X)
	case ir.OADD, ir.OSUB, ir.OOR, ir.OXOR, ir.OMUL, ir.ODIV, ir.OMOD, ir.OLSH, ir.ORSH, ir.OAND, ir.OANDNOT, ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		n := n.(*ir.BinaryExpr)
		e.discard(n.X)
		e.discard(n.Y)
	case ir.OANDAND, ir.OOROR:
		n := n.(*ir.LogicalExpr)
		e.discard(n.X)
		e.discard(n.Y)
	case ir.OADDR:
		n := n.(*ir.AddrExpr)
		e.expr(k.addr(n, "address-of"), n.X) // "address-of"
	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		e.expr(k.deref(n, "indirection"), n.X) // "indirection"
	case ir.ODOT, ir.ODOTMETH, ir.ODOTINTER:
		n := n.(*ir.SelectorExpr)
		e.expr(k.note(n, "dot"), n.X)
	case ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		e.expr(k.deref(n, "dot of pointer"), n.X) // "dot of pointer"
	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n := n.(*ir.TypeAssertExpr)
		e.expr(k.dotType(n.Type(), n, "dot"), n.X)
	case ir.ODYNAMICDOTTYPE, ir.ODYNAMICDOTTYPE2:
		n := n.(*ir.DynamicTypeAssertExpr)
		e.expr(k.dotType(n.Type(), n, "dot"), n.X)
		// n.T doesn't need to be tracked; it always points to read-only storage.
	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		if n.X.Type().IsArray() {
			e.expr(k.note(n, "fixed-array-index-of"), n.X)
		} else {
			// TODO(mdempsky): Fix why reason text.
			e.expr(k.deref(n, "dot of pointer"), n.X)
		}
		e.discard(n.Index)
	case ir.OINDEXMAP:
		n := n.(*ir.IndexExpr)
		e.discard(n.X)
		e.discard(n.Index)
	case ir.OSLICE, ir.OSLICEARR, ir.OSLICE3, ir.OSLICE3ARR, ir.OSLICESTR:
		n := n.(*ir.SliceExpr)
		e.expr(k.note(n, "slice"), n.X)
		e.discard(n.Low)
		e.discard(n.High)
		e.discard(n.Max)

	case ir.OCONV, ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		if (ir.ShouldCheckPtr(e.curfn, 2) || ir.ShouldAsanCheckPtr(e.curfn)) && n.Type().IsUnsafePtr() && n.X.Type().IsPtr() {
			// When -d=checkptr=2 or -asan is enabled,
			// treat conversions to unsafe.Pointer as an
			// escaping operation. This allows better
			// runtime instrumentation, since we can more
			// easily detect object boundaries on the heap
			// than the stack.
			e.assignHeap(n.X, "conversion to unsafe.Pointer", n)
		} else if n.Type().IsUnsafePtr() && n.X.Type().IsUintptr() {
			e.unsafeValue(k, n.X)
		} else {
			e.expr(k, n.X)
		}
	case ir.OCONVIFACE, ir.OCONVIDATA:
		n := n.(*ir.ConvExpr)
		if !n.X.Type().IsInterface() && !types.IsDirectIface(n.X.Type()) {
			k = e.spill(k, n)
		}
		e.expr(k.note(n, "interface-converted"), n.X)
	case ir.OEFACE:
		n := n.(*ir.BinaryExpr)
		// Note: n.X is not needed because it can never point to memory that might escape.
		e.expr(k, n.Y)
	case ir.OIDATA, ir.OSPTR:
		n := n.(*ir.UnaryExpr)
		e.expr(k, n.X)
	case ir.OSLICE2ARRPTR:
		// the slice pointer flows directly to the result
		n := n.(*ir.ConvExpr)
		e.expr(k, n.X)
	case ir.ORECV:
		n := n.(*ir.UnaryExpr)
		e.discard(n.X)

	case ir.OCALLMETH, ir.OCALLFUNC, ir.OCALLINTER, ir.OINLCALL, ir.OLEN, ir.OCAP, ir.OCOMPLEX, ir.OREAL, ir.OIMAG, ir.OAPPEND, ir.OCOPY, ir.ORECOVER, ir.OUNSAFEADD, ir.OUNSAFESLICE:
		e.call([]hole{k}, n)

	case ir.ONEW:
		n := n.(*ir.UnaryExpr)
		e.spill(k, n)

	case ir.OMAKESLICE:
		n := n.(*ir.MakeExpr)
		e.spill(k, n)
		e.discard(n.Len)
		e.discard(n.Cap)
	case ir.OMAKECHAN:
		n := n.(*ir.MakeExpr)
		e.discard(n.Len)
	case ir.OMAKEMAP:
		n := n.(*ir.MakeExpr)
		e.spill(k, n)
		e.discard(n.Len)

	case ir.OMETHVALUE:
		// Flow the receiver argument to both the closure and
		// to the receiver parameter.

		n := n.(*ir.SelectorExpr)
		closureK := e.spill(k, n)

		m := n.Selection

		// We don't know how the method value will be called
		// later, so conservatively assume the result
		// parameters all flow to the heap.
		//
		// TODO(mdempsky): Change ks into a callback, so that
		// we don't have to create this slice?
		var ks []hole
		for i := m.Type.NumResults(); i > 0; i-- {
			ks = append(ks, e.heapHole())
		}
		name, _ := m.Nname.(*ir.Name)
		paramK := e.tagHole(ks, name, m.Type.Recv())

		e.expr(e.teeHole(paramK, closureK), n.X)

	case ir.OPTRLIT:
		n := n.(*ir.AddrExpr)
		e.expr(e.spill(k, n), n.X)

	case ir.OARRAYLIT:
		n := n.(*ir.CompLitExpr)
		for _, elt := range n.List {
			if elt.Op() == ir.OKEY {
				elt = elt.(*ir.KeyExpr).Value
			}
			e.expr(k.note(n, "array literal element"), elt)
		}

	case ir.OSLICELIT:
		n := n.(*ir.CompLitExpr)
		k = e.spill(k, n)

		for _, elt := range n.List {
			if elt.Op() == ir.OKEY {
				elt = elt.(*ir.KeyExpr).Value
			}
			e.expr(k.note(n, "slice-literal-element"), elt)
		}

	case ir.OSTRUCTLIT:
		n := n.(*ir.CompLitExpr)
		for _, elt := range n.List {
			e.expr(k.note(n, "struct literal element"), elt.(*ir.StructKeyExpr).Value)
		}

	case ir.OMAPLIT:
		n := n.(*ir.CompLitExpr)
		e.spill(k, n)

		// Map keys and values are always stored in the heap.
		for _, elt := range n.List {
			elt := elt.(*ir.KeyExpr)
			e.assignHeap(elt.Key, "map literal key", n)
			e.assignHeap(elt.Value, "map literal value", n)
		}

	case ir.OCLOSURE:
		n := n.(*ir.ClosureExpr)
		k = e.spill(k, n)
		e.closures = append(e.closures, closure{k, n})

		if fn := n.Func; fn.IsHiddenClosure() {
			for _, cv := range fn.ClosureVars {
				if loc := e.oldLoc(cv); !loc.captured {
					loc.captured = true

					// Ignore reassignments to the variable in straightline code
					// preceding the first capture by a closure.
					if loc.loopDepth == e.loopDepth {
						loc.reassigned = false
					}
				}
			}

			for _, n := range fn.Dcl {
				// Add locations for local variables of the
				// closure, if needed, in case we're not including
				// the closure func in the batch for escape
				// analysis (happens for escape analysis called
				// from reflectdata.methodWrapper)
				if n.Op() == ir.ONAME && n.Opt == nil {
					e.with(fn).newLoc(n, false)
				}
			}
			e.walkFunc(fn)
		}

	case ir.ORUNES2STR, ir.OBYTES2STR, ir.OSTR2RUNES, ir.OSTR2BYTES, ir.ORUNESTR:
		n := n.(*ir.ConvExpr)
		e.spill(k, n)
		e.discard(n.X)

	case ir.OADDSTR:
		n := n.(*ir.AddStringExpr)
		e.spill(k, n)

		// Arguments of OADDSTR never escape;
		// runtime.concatstrings makes sure of that.
		e.discards(n.List)

	case ir.ODYNAMICTYPE:
		// Nothing to do - argument is a *runtime._type (+ maybe a *runtime.itab) pointing to static data section
	}
}

// unsafeValue evaluates a uintptr-typed arithmetic expression looking
// for conversions from an unsafe.Pointer.
func (e *escape) unsafeValue(k hole, n ir.Node) {
	if n.Type().Kind() != types.TUINTPTR {
		base.Fatalf("unexpected type %v for %v", n.Type(), n)
	}
	if k.addrtaken {
		base.Fatalf("unexpected addrtaken")
	}

	e.stmts(n.Init())

	switch n.Op() {
	case ir.OCONV, ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		if n.X.Type().IsUnsafePtr() {
			e.expr(k, n.X)
		} else {
			e.discard(n.X)
		}
	case ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		if ir.IsReflectHeaderDataField(n) {
			e.expr(k.deref(n, "reflect.Header.Data"), n.X)
		} else {
			e.discard(n.X)
		}
	case ir.OPLUS, ir.ONEG, ir.OBITNOT:
		n := n.(*ir.UnaryExpr)
		e.unsafeValue(k, n.X)
	case ir.OADD, ir.OSUB, ir.OOR, ir.OXOR, ir.OMUL, ir.ODIV, ir.OMOD, ir.OAND, ir.OANDNOT:
		n := n.(*ir.BinaryExpr)
		e.unsafeValue(k, n.X)
		e.unsafeValue(k, n.Y)
	case ir.OLSH, ir.ORSH:
		n := n.(*ir.BinaryExpr)
		e.unsafeValue(k, n.X)
		// RHS need not be uintptr-typed (#32959) and can't meaningfully
		// flow pointers anyway.
		e.discard(n.Y)
	default:
		e.exprSkipInit(e.discardHole(), n)
	}
}

// discard evaluates an expression n for side-effects, but discards
// its value.
func (e *escape) discard(n ir.Node) {
	e.expr(e.discardHole(), n)
}

func (e *escape) discards(l ir.Nodes) {
	for _, n := range l {
		e.discard(n)
	}
}

// spill allocates a new location associated with expression n, flows
// its address to k, and returns a hole that flows values to it. It's
// intended for use with most expressions that allocate storage.
func (e *escape) spill(k hole, n ir.Node) hole {
	loc := e.newLoc(n, true)
	e.flow(k.addr(n, "spill"), loc)
	return loc.asHole()
}
