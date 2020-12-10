// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"encoding/binary"
	"fmt"
	"go/constant"
	"go/token"
	"strings"
)

// The constant is known to runtime.
const tmpstringbufsize = 32
const zeroValSize = 1024 // must match value of runtime/map.go:maxZero

func walk(fn *ir.Func) {
	Curfn = fn
	errorsBefore := base.Errors()

	if base.Flag.W != 0 {
		s := fmt.Sprintf("\nbefore walk %v", Curfn.Sym())
		ir.DumpList(s, Curfn.Body())
	}

	lno := base.Pos

	// Final typecheck for any unused variables.
	for i, ln := range fn.Dcl {
		if ln.Op() == ir.ONAME && (ln.Class() == ir.PAUTO || ln.Class() == ir.PAUTOHEAP) {
			ln = typecheck(ln, ctxExpr|ctxAssign).(*ir.Name)
			fn.Dcl[i] = ln
		}
	}

	// Propagate the used flag for typeswitch variables up to the NONAME in its definition.
	for _, ln := range fn.Dcl {
		if ln.Op() == ir.ONAME && (ln.Class() == ir.PAUTO || ln.Class() == ir.PAUTOHEAP) && ln.Defn != nil && ln.Defn.Op() == ir.OTYPESW && ln.Used() {
			ln.Defn.(*ir.TypeSwitchGuard).Used = true
		}
	}

	for _, ln := range fn.Dcl {
		if ln.Op() != ir.ONAME || (ln.Class() != ir.PAUTO && ln.Class() != ir.PAUTOHEAP) || ln.Sym().Name[0] == '&' || ln.Used() {
			continue
		}
		if defn, ok := ln.Defn.(*ir.TypeSwitchGuard); ok {
			if defn.Used {
				continue
			}
			base.ErrorfAt(defn.Tag.Pos(), "%v declared but not used", ln.Sym())
			defn.Used = true // suppress repeats
		} else {
			base.ErrorfAt(ln.Pos(), "%v declared but not used", ln.Sym())
		}
	}

	base.Pos = lno
	if base.Errors() > errorsBefore {
		return
	}
	walkstmtlist(Curfn.Body().Slice())
	if base.Flag.W != 0 {
		s := fmt.Sprintf("after walk %v", Curfn.Sym())
		ir.DumpList(s, Curfn.Body())
	}

	zeroResults()
	heapmoves()
	if base.Flag.W != 0 && Curfn.Enter.Len() > 0 {
		s := fmt.Sprintf("enter %v", Curfn.Sym())
		ir.DumpList(s, Curfn.Enter)
	}
}

func walkstmtlist(s []ir.Node) {
	for i := range s {
		s[i] = walkstmt(s[i])
	}
}

func paramoutheap(fn *ir.Func) bool {
	for _, ln := range fn.Dcl {
		switch ln.Class() {
		case ir.PPARAMOUT:
			if isParamStackCopy(ln) || ln.Addrtaken() {
				return true
			}

		case ir.PAUTO:
			// stop early - parameters are over
			return false
		}
	}

	return false
}

// The result of walkstmt MUST be assigned back to n, e.g.
// 	n.Left = walkstmt(n.Left)
func walkstmt(n ir.Node) ir.Node {
	if n == nil {
		return n
	}

	setlineno(n)

	walkstmtlist(n.Init().Slice())

	switch n.Op() {
	default:
		if n.Op() == ir.ONAME {
			base.Errorf("%v is not a top level statement", n.Sym())
		} else {
			base.Errorf("%v is not a top level statement", n.Op())
		}
		ir.Dump("nottop", n)
		return n

	case ir.OAS,
		ir.OASOP,
		ir.OAS2,
		ir.OAS2DOTTYPE,
		ir.OAS2RECV,
		ir.OAS2FUNC,
		ir.OAS2MAPR,
		ir.OCLOSE,
		ir.OCOPY,
		ir.OCALLMETH,
		ir.OCALLINTER,
		ir.OCALL,
		ir.OCALLFUNC,
		ir.ODELETE,
		ir.OSEND,
		ir.OPRINT,
		ir.OPRINTN,
		ir.OPANIC,
		ir.ORECOVER,
		ir.OGETG:
		if n.Typecheck() == 0 {
			base.Fatalf("missing typecheck: %+v", n)
		}
		init := n.Init()
		n.PtrInit().Set(nil)
		n = walkexpr(n, &init)
		if n.Op() == ir.ONAME {
			// copy rewrote to a statement list and a temp for the length.
			// Throw away the temp to avoid plain values as statements.
			n = ir.NewBlockStmt(n.Pos(), init.Slice())
			init.Set(nil)
		}
		if init.Len() > 0 {
			switch n.Op() {
			case ir.OAS, ir.OAS2, ir.OBLOCK:
				n.PtrInit().Prepend(init.Slice()...)

			default:
				init.Append(n)
				n = ir.NewBlockStmt(n.Pos(), init.Slice())
			}
		}
		return n

	// special case for a receive where we throw away
	// the value received.
	case ir.ORECV:
		if n.Typecheck() == 0 {
			base.Fatalf("missing typecheck: %+v", n)
		}
		init := n.Init()
		n.PtrInit().Set(nil)

		n.SetLeft(walkexpr(n.Left(), &init))
		n = mkcall1(chanfn("chanrecv1", 2, n.Left().Type()), nil, &init, n.Left(), nodnil())
		n = walkexpr(n, &init)
		return initExpr(init.Slice(), n)

	case ir.OBREAK,
		ir.OCONTINUE,
		ir.OFALL,
		ir.OGOTO,
		ir.OLABEL,
		ir.ODCLCONST,
		ir.ODCLTYPE,
		ir.OCHECKNIL,
		ir.OVARDEF,
		ir.OVARKILL,
		ir.OVARLIVE:
		return n

	case ir.ODCL:
		v := n.Left()
		if v.Class() == ir.PAUTOHEAP {
			if base.Flag.CompilingRuntime {
				base.Errorf("%v escapes to heap, not allowed in runtime", v)
			}
			if prealloc[v] == nil {
				prealloc[v] = callnew(v.Type())
			}
			nn := ir.Nod(ir.OAS, v.Name().Heapaddr, prealloc[v])
			nn.SetColas(true)
			return walkstmt(typecheck(nn, ctxStmt))
		}
		return n

	case ir.OBLOCK:
		walkstmtlist(n.List().Slice())
		return n

	case ir.OCASE:
		base.Errorf("case statement out of place")
		panic("unreachable")

	case ir.ODEFER:
		Curfn.SetHasDefer(true)
		Curfn.NumDefers++
		if Curfn.NumDefers > maxOpenDefers {
			// Don't allow open-coded defers if there are more than
			// 8 defers in the function, since we use a single
			// byte to record active defers.
			Curfn.SetOpenCodedDeferDisallowed(true)
		}
		if n.Esc() != EscNever {
			// If n.Esc is not EscNever, then this defer occurs in a loop,
			// so open-coded defers cannot be used in this function.
			Curfn.SetOpenCodedDeferDisallowed(true)
		}
		fallthrough
	case ir.OGO:
		var init ir.Nodes
		switch n.Left().Op() {
		case ir.OPRINT, ir.OPRINTN:
			n.SetLeft(wrapCall(n.Left(), &init))

		case ir.ODELETE:
			if mapfast(n.Left().List().First().Type()) == mapslow {
				n.SetLeft(wrapCall(n.Left(), &init))
			} else {
				n.SetLeft(walkexpr(n.Left(), &init))
			}

		case ir.OCOPY:
			n.SetLeft(copyany(n.Left(), &init, true))

		case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
			if n.Left().Body().Len() > 0 {
				n.SetLeft(wrapCall(n.Left(), &init))
			} else {
				n.SetLeft(walkexpr(n.Left(), &init))
			}

		default:
			n.SetLeft(walkexpr(n.Left(), &init))
		}
		if init.Len() > 0 {
			init.Append(n)
			n = ir.NewBlockStmt(n.Pos(), init.Slice())
		}
		return n

	case ir.OFOR, ir.OFORUNTIL:
		if n.Left() != nil {
			walkstmtlist(n.Left().Init().Slice())
			init := n.Left().Init()
			n.Left().PtrInit().Set(nil)
			n.SetLeft(walkexpr(n.Left(), &init))
			n.SetLeft(initExpr(init.Slice(), n.Left()))
		}

		n.SetRight(walkstmt(n.Right()))
		if n.Op() == ir.OFORUNTIL {
			walkstmtlist(n.List().Slice())
		}
		walkstmtlist(n.Body().Slice())
		return n

	case ir.OIF:
		n.SetLeft(walkexpr(n.Left(), n.PtrInit()))
		walkstmtlist(n.Body().Slice())
		walkstmtlist(n.Rlist().Slice())
		return n

	case ir.ORETURN:
		Curfn.NumReturns++
		if n.List().Len() == 0 {
			return n
		}
		if (hasNamedResults(Curfn) && n.List().Len() > 1) || paramoutheap(Curfn) {
			// assign to the function out parameters,
			// so that reorder3 can fix up conflicts
			var rl []ir.Node

			for _, ln := range Curfn.Dcl {
				cl := ln.Class()
				if cl == ir.PAUTO || cl == ir.PAUTOHEAP {
					break
				}
				if cl == ir.PPARAMOUT {
					var ln ir.Node = ln
					if isParamStackCopy(ln) {
						ln = walkexpr(typecheck(ir.Nod(ir.ODEREF, ln.Name().Heapaddr, nil), ctxExpr), nil)
					}
					rl = append(rl, ln)
				}
			}

			if got, want := n.List().Len(), len(rl); got != want {
				// order should have rewritten multi-value function calls
				// with explicit OAS2FUNC nodes.
				base.Fatalf("expected %v return arguments, have %v", want, got)
			}

			// move function calls out, to make reorder3's job easier.
			walkexprlistsafe(n.List().Slice(), n.PtrInit())

			ll := ascompatee(n.Op(), rl, n.List().Slice(), n.PtrInit())
			n.PtrList().Set(reorder3(ll))
			return n
		}
		walkexprlist(n.List().Slice(), n.PtrInit())

		// For each return parameter (lhs), assign the corresponding result (rhs).
		lhs := Curfn.Type().Results()
		rhs := n.List().Slice()
		res := make([]ir.Node, lhs.NumFields())
		for i, nl := range lhs.FieldSlice() {
			nname := ir.AsNode(nl.Nname)
			if isParamHeapCopy(nname) {
				nname = nname.Name().Stackcopy
			}
			a := ir.Nod(ir.OAS, nname, rhs[i])
			res[i] = convas(a, n.PtrInit())
		}
		n.PtrList().Set(res)
		return n

	case ir.ORETJMP:
		return n

	case ir.OINLMARK:
		return n

	case ir.OSELECT:
		n := n.(*ir.SelectStmt)
		walkselect(n)
		return n

	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		walkswitch(n)
		return n

	case ir.ORANGE:
		n := n.(*ir.RangeStmt)
		return walkrange(n)
	}

	// No return! Each case must return (or panic),
	// to avoid confusion about what gets returned
	// in the presence of type assertions.
}

// walk the whole tree of the body of an
// expression or simple statement.
// the types expressions are calculated.
// compile-time constants are evaluated.
// complex side effects like statements are appended to init
func walkexprlist(s []ir.Node, init *ir.Nodes) {
	for i := range s {
		s[i] = walkexpr(s[i], init)
	}
}

func walkexprlistsafe(s []ir.Node, init *ir.Nodes) {
	for i, n := range s {
		s[i] = safeexpr(n, init)
		s[i] = walkexpr(s[i], init)
	}
}

func walkexprlistcheap(s []ir.Node, init *ir.Nodes) {
	for i, n := range s {
		s[i] = cheapexpr(n, init)
		s[i] = walkexpr(s[i], init)
	}
}

// convFuncName builds the runtime function name for interface conversion.
// It also reports whether the function expects the data by address.
// Not all names are possible. For example, we never generate convE2E or convE2I.
func convFuncName(from, to *types.Type) (fnname string, needsaddr bool) {
	tkind := to.Tie()
	switch from.Tie() {
	case 'I':
		if tkind == 'I' {
			return "convI2I", false
		}
	case 'T':
		switch {
		case from.Size() == 2 && from.Align == 2:
			return "convT16", false
		case from.Size() == 4 && from.Align == 4 && !from.HasPointers():
			return "convT32", false
		case from.Size() == 8 && from.Align == types.Types[types.TUINT64].Align && !from.HasPointers():
			return "convT64", false
		}
		if sc := from.SoleComponent(); sc != nil {
			switch {
			case sc.IsString():
				return "convTstring", false
			case sc.IsSlice():
				return "convTslice", false
			}
		}

		switch tkind {
		case 'E':
			if !from.HasPointers() {
				return "convT2Enoptr", true
			}
			return "convT2E", true
		case 'I':
			if !from.HasPointers() {
				return "convT2Inoptr", true
			}
			return "convT2I", true
		}
	}
	base.Fatalf("unknown conv func %c2%c", from.Tie(), to.Tie())
	panic("unreachable")
}

// The result of walkexpr MUST be assigned back to n, e.g.
// 	n.Left = walkexpr(n.Left, init)
func walkexpr(n ir.Node, init *ir.Nodes) ir.Node {
	if n == nil {
		return n
	}

	// Eagerly checkwidth all expressions for the back end.
	if n.Type() != nil && !n.Type().WidthCalculated() {
		switch n.Type().Kind() {
		case types.TBLANK, types.TNIL, types.TIDEAL:
		default:
			checkwidth(n.Type())
		}
	}

	if init == n.PtrInit() {
		// not okay to use n->ninit when walking n,
		// because we might replace n with some other node
		// and would lose the init list.
		base.Fatalf("walkexpr init == &n->ninit")
	}

	if n.Init().Len() != 0 {
		walkstmtlist(n.Init().Slice())
		init.AppendNodes(n.PtrInit())
	}

	lno := setlineno(n)

	if base.Flag.LowerW > 1 {
		ir.Dump("before walk expr", n)
	}

	if n.Typecheck() != 1 {
		base.Fatalf("missed typecheck: %+v", n)
	}

	if n.Type().IsUntyped() {
		base.Fatalf("expression has untyped type: %+v", n)
	}

	if n.Op() == ir.ONAME && n.Class() == ir.PAUTOHEAP {
		nn := ir.Nod(ir.ODEREF, n.Name().Heapaddr, nil)
		nn.Left().MarkNonNil()
		return walkexpr(typecheck(nn, ctxExpr), init)
	}

	n = walkexpr1(n, init)

	// Expressions that are constant at run time but not
	// considered const by the language spec are not turned into
	// constants until walk. For example, if n is y%1 == 0, the
	// walk of y%1 may have replaced it by 0.
	// Check whether n with its updated args is itself now a constant.
	t := n.Type()
	n = evalConst(n)
	if n.Type() != t {
		base.Fatalf("evconst changed Type: %v had type %v, now %v", n, t, n.Type())
	}
	if n.Op() == ir.OLITERAL {
		n = typecheck(n, ctxExpr)
		// Emit string symbol now to avoid emitting
		// any concurrently during the backend.
		if v := n.Val(); v.Kind() == constant.String {
			_ = stringsym(n.Pos(), constant.StringVal(v))
		}
	}

	updateHasCall(n)

	if base.Flag.LowerW != 0 && n != nil {
		ir.Dump("after walk expr", n)
	}

	base.Pos = lno
	return n
}

func walkexpr1(n ir.Node, init *ir.Nodes) ir.Node {
	switch n.Op() {
	default:
		ir.Dump("walk", n)
		base.Fatalf("walkexpr: switch 1 unknown op %+v", n.Op())
		panic("unreachable")

	case ir.ONONAME, ir.OGETG, ir.ONEWOBJ, ir.OMETHEXPR:
		return n

	case ir.OTYPE, ir.ONAME, ir.OLITERAL, ir.ONIL:
		// TODO(mdempsky): Just return n; see discussion on CL 38655.
		// Perhaps refactor to use Node.mayBeShared for these instead.
		// If these return early, make sure to still call
		// stringsym for constant strings.
		return n

	case ir.ONOT, ir.ONEG, ir.OPLUS, ir.OBITNOT, ir.OREAL, ir.OIMAG, ir.ODOTMETH, ir.ODOTINTER,
		ir.ODEREF, ir.OSPTR, ir.OITAB, ir.OIDATA, ir.OADDR:
		n.SetLeft(walkexpr(n.Left(), init))
		return n

	case ir.OEFACE, ir.OAND, ir.OANDNOT, ir.OSUB, ir.OMUL, ir.OADD, ir.OOR, ir.OXOR, ir.OLSH, ir.ORSH:
		n.SetLeft(walkexpr(n.Left(), init))
		n.SetRight(walkexpr(n.Right(), init))
		return n

	case ir.ODOT, ir.ODOTPTR:
		usefield(n)
		n.SetLeft(walkexpr(n.Left(), init))
		return n

	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n.SetLeft(walkexpr(n.Left(), init))
		// Set up interface type addresses for back end.
		n.SetRight(typename(n.Type()))
		if n.Op() == ir.ODOTTYPE {
			n.Right().SetRight(typename(n.Left().Type()))
		}
		if !n.Type().IsInterface() && !n.Left().Type().IsEmptyInterface() {
			n.PtrList().Set1(itabname(n.Type(), n.Left().Type()))
		}
		return n

	case ir.OLEN, ir.OCAP:
		if isRuneCount(n) {
			// Replace len([]rune(string)) with runtime.countrunes(string).
			return mkcall("countrunes", n.Type(), init, conv(n.Left().Left(), types.Types[types.TSTRING]))
		}

		n.SetLeft(walkexpr(n.Left(), init))

		// replace len(*[10]int) with 10.
		// delayed until now to preserve side effects.
		t := n.Left().Type()

		if t.IsPtr() {
			t = t.Elem()
		}
		if t.IsArray() {
			safeexpr(n.Left(), init)
			n = origIntConst(n, t.NumElem())
			n.SetTypecheck(1)
		}
		return n

	case ir.OCOMPLEX:
		// Use results from call expression as arguments for complex.
		if n.Left() == nil && n.Right() == nil {
			n.SetLeft(n.List().First())
			n.SetRight(n.List().Second())
		}
		n.SetLeft(walkexpr(n.Left(), init))
		n.SetRight(walkexpr(n.Right(), init))
		return n

	case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		return walkcompare(n, init)

	case ir.OANDAND, ir.OOROR:
		n.SetLeft(walkexpr(n.Left(), init))

		// cannot put side effects from n.Right on init,
		// because they cannot run before n.Left is checked.
		// save elsewhere and store on the eventual n.Right.
		var ll ir.Nodes

		n.SetRight(walkexpr(n.Right(), &ll))
		n.SetRight(initExpr(ll.Slice(), n.Right()))
		return n

	case ir.OPRINT, ir.OPRINTN:
		return walkprint(n, init)

	case ir.OPANIC:
		return mkcall("gopanic", nil, init, n.Left())

	case ir.ORECOVER:
		return mkcall("gorecover", n.Type(), init, nodAddr(nodfp))

	case ir.OCLOSUREREAD, ir.OCFUNC:
		return n

	case ir.OCALLINTER, ir.OCALLFUNC, ir.OCALLMETH:
		if n.Op() == ir.OCALLINTER {
			usemethod(n)
			markUsedIfaceMethod(n)
		}

		if n.Op() == ir.OCALLFUNC && n.Left().Op() == ir.OCLOSURE {
			// Transform direct call of a closure to call of a normal function.
			// transformclosure already did all preparation work.

			// Prepend captured variables to argument list.
			n.PtrList().Prepend(n.Left().Func().ClosureEnter.Slice()...)
			n.Left().Func().ClosureEnter.Set(nil)

			// Replace OCLOSURE with ONAME/PFUNC.
			n.SetLeft(n.Left().Func().Nname)

			// Update type of OCALLFUNC node.
			// Output arguments had not changed, but their offsets could.
			if n.Left().Type().NumResults() == 1 {
				n.SetType(n.Left().Type().Results().Field(0).Type)
			} else {
				n.SetType(n.Left().Type().Results())
			}
		}

		walkCall(n, init)
		return n

	case ir.OAS, ir.OASOP:
		init.AppendNodes(n.PtrInit())

		// Recognize m[k] = append(m[k], ...) so we can reuse
		// the mapassign call.
		mapAppend := n.Left().Op() == ir.OINDEXMAP && n.Right().Op() == ir.OAPPEND
		if mapAppend && !samesafeexpr(n.Left(), n.Right().List().First()) {
			base.Fatalf("not same expressions: %v != %v", n.Left(), n.Right().List().First())
		}

		n.SetLeft(walkexpr(n.Left(), init))
		n.SetLeft(safeexpr(n.Left(), init))

		if mapAppend {
			n.Right().List().SetFirst(n.Left())
		}

		if n.Op() == ir.OASOP {
			// Rewrite x op= y into x = x op y.
			n = ir.Nod(ir.OAS, n.Left(),
				typecheck(ir.NewBinaryExpr(base.Pos, n.SubOp(), n.Left(), n.Right()), ctxExpr))
		}

		if oaslit(n, init) {
			return ir.NodAt(n.Pos(), ir.OBLOCK, nil, nil)
		}

		if n.Right() == nil {
			// TODO(austin): Check all "implicit zeroing"
			return n
		}

		if !instrumenting && isZero(n.Right()) {
			return n
		}

		switch n.Right().Op() {
		default:
			n.SetRight(walkexpr(n.Right(), init))

		case ir.ORECV:
			// x = <-c; n.Left is x, n.Right.Left is c.
			// order.stmt made sure x is addressable.
			n.Right().SetLeft(walkexpr(n.Right().Left(), init))

			n1 := nodAddr(n.Left())
			r := n.Right().Left() // the channel
			return mkcall1(chanfn("chanrecv1", 2, r.Type()), nil, init, r, n1)

		case ir.OAPPEND:
			// x = append(...)
			r := n.Right()
			if r.Type().Elem().NotInHeap() {
				base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", r.Type().Elem())
			}
			switch {
			case isAppendOfMake(r):
				// x = append(y, make([]T, y)...)
				r = extendslice(r, init)
			case r.IsDDD():
				r = appendslice(r, init) // also works for append(slice, string).
			default:
				r = walkappend(r, init, n)
			}
			n.SetRight(r)
			if r.Op() == ir.OAPPEND {
				// Left in place for back end.
				// Do not add a new write barrier.
				// Set up address of type for back end.
				r.SetLeft(typename(r.Type().Elem()))
				return n
			}
			// Otherwise, lowered for race detector.
			// Treat as ordinary assignment.
		}

		if n.Left() != nil && n.Right() != nil {
			n = convas(n, init)
		}
		return n

	case ir.OAS2:
		init.AppendNodes(n.PtrInit())
		walkexprlistsafe(n.List().Slice(), init)
		walkexprlistsafe(n.Rlist().Slice(), init)
		ll := ascompatee(ir.OAS, n.List().Slice(), n.Rlist().Slice(), init)
		ll = reorder3(ll)
		return liststmt(ll)

	// a,b,... = fn()
	case ir.OAS2FUNC:
		init.AppendNodes(n.PtrInit())

		r := n.Rlist().First()
		walkexprlistsafe(n.List().Slice(), init)
		r = walkexpr(r, init)

		if isIntrinsicCall(r.(*ir.CallExpr)) {
			n.PtrRlist().Set1(r)
			return n
		}
		init.Append(r)

		ll := ascompatet(n.List(), r.Type())
		return liststmt(ll)

	// x, y = <-c
	// order.stmt made sure x is addressable or blank.
	case ir.OAS2RECV:
		init.AppendNodes(n.PtrInit())

		r := n.Rlist().First()
		walkexprlistsafe(n.List().Slice(), init)
		r.SetLeft(walkexpr(r.Left(), init))
		var n1 ir.Node
		if ir.IsBlank(n.List().First()) {
			n1 = nodnil()
		} else {
			n1 = nodAddr(n.List().First())
		}
		fn := chanfn("chanrecv2", 2, r.Left().Type())
		ok := n.List().Second()
		call := mkcall1(fn, types.Types[types.TBOOL], init, r.Left(), n1)
		n = ir.Nod(ir.OAS, ok, call)
		return typecheck(n, ctxStmt)

	// a,b = m[i]
	case ir.OAS2MAPR:
		init.AppendNodes(n.PtrInit())

		r := n.Rlist().First()
		walkexprlistsafe(n.List().Slice(), init)
		r.SetLeft(walkexpr(r.Left(), init))
		r.SetRight(walkexpr(r.Right(), init))
		t := r.Left().Type()

		fast := mapfast(t)
		var key ir.Node
		if fast != mapslow {
			// fast versions take key by value
			key = r.Right()
		} else {
			// standard version takes key by reference
			// order.expr made sure key is addressable.
			key = nodAddr(r.Right())
		}

		// from:
		//   a,b = m[i]
		// to:
		//   var,b = mapaccess2*(t, m, i)
		//   a = *var
		a := n.List().First()

		if w := t.Elem().Width; w <= zeroValSize {
			fn := mapfn(mapaccess2[fast], t)
			r = mkcall1(fn, fn.Type().Results(), init, typename(t), r.Left(), key)
		} else {
			fn := mapfn("mapaccess2_fat", t)
			z := zeroaddr(w)
			r = mkcall1(fn, fn.Type().Results(), init, typename(t), r.Left(), key, z)
		}

		// mapaccess2* returns a typed bool, but due to spec changes,
		// the boolean result of i.(T) is now untyped so we make it the
		// same type as the variable on the lhs.
		if ok := n.List().Second(); !ir.IsBlank(ok) && ok.Type().IsBoolean() {
			r.Type().Field(1).Type = ok.Type()
		}
		n.PtrRlist().Set1(r)
		n.SetOp(ir.OAS2FUNC)

		// don't generate a = *var if a is _
		if !ir.IsBlank(a) {
			var_ := temp(types.NewPtr(t.Elem()))
			var_.SetTypecheck(1)
			var_.MarkNonNil() // mapaccess always returns a non-nil pointer
			n.List().SetFirst(var_)
			n = walkexpr(n, init)
			init.Append(n)
			n = ir.Nod(ir.OAS, a, ir.Nod(ir.ODEREF, var_, nil))
		}

		n = typecheck(n, ctxStmt)
		return walkexpr(n, init)

	case ir.ODELETE:
		init.AppendNodes(n.PtrInit())
		map_ := n.List().First()
		key := n.List().Second()
		map_ = walkexpr(map_, init)
		key = walkexpr(key, init)

		t := map_.Type()
		fast := mapfast(t)
		if fast == mapslow {
			// order.stmt made sure key is addressable.
			key = nodAddr(key)
		}
		return mkcall1(mapfndel(mapdelete[fast], t), nil, init, typename(t), map_, key)

	case ir.OAS2DOTTYPE:
		walkexprlistsafe(n.List().Slice(), init)
		n.PtrRlist().SetIndex(0, walkexpr(n.Rlist().First(), init))
		return n

	case ir.OCONVIFACE:
		n.SetLeft(walkexpr(n.Left(), init))

		fromType := n.Left().Type()
		toType := n.Type()

		if !fromType.IsInterface() && !ir.IsBlank(Curfn.Nname) { // skip unnamed functions (func _())
			markTypeUsedInInterface(fromType, Curfn.LSym)
		}

		// typeword generates the type word of the interface value.
		typeword := func() ir.Node {
			if toType.IsEmptyInterface() {
				return typename(fromType)
			}
			return itabname(fromType, toType)
		}

		// Optimize convT2E or convT2I as a two-word copy when T is pointer-shaped.
		if isdirectiface(fromType) {
			l := ir.Nod(ir.OEFACE, typeword(), n.Left())
			l.SetType(toType)
			l.SetTypecheck(n.Typecheck())
			return l
		}

		if staticuint64s == nil {
			staticuint64s = NewName(Runtimepkg.Lookup("staticuint64s"))
			staticuint64s.SetClass(ir.PEXTERN)
			// The actual type is [256]uint64, but we use [256*8]uint8 so we can address
			// individual bytes.
			staticuint64s.SetType(types.NewArray(types.Types[types.TUINT8], 256*8))
			zerobase = NewName(Runtimepkg.Lookup("zerobase"))
			zerobase.SetClass(ir.PEXTERN)
			zerobase.SetType(types.Types[types.TUINTPTR])
		}

		// Optimize convT2{E,I} for many cases in which T is not pointer-shaped,
		// by using an existing addressable value identical to n.Left
		// or creating one on the stack.
		var value ir.Node
		switch {
		case fromType.Size() == 0:
			// n.Left is zero-sized. Use zerobase.
			cheapexpr(n.Left(), init) // Evaluate n.Left for side-effects. See issue 19246.
			value = zerobase
		case fromType.IsBoolean() || (fromType.Size() == 1 && fromType.IsInteger()):
			// n.Left is a bool/byte. Use staticuint64s[n.Left * 8] on little-endian
			// and staticuint64s[n.Left * 8 + 7] on big-endian.
			n.SetLeft(cheapexpr(n.Left(), init))
			// byteindex widens n.Left so that the multiplication doesn't overflow.
			index := ir.Nod(ir.OLSH, byteindex(n.Left()), nodintconst(3))
			if thearch.LinkArch.ByteOrder == binary.BigEndian {
				index = ir.Nod(ir.OADD, index, nodintconst(7))
			}
			value = ir.Nod(ir.OINDEX, staticuint64s, index)
			value.SetBounded(true)
		case n.Left().Name() != nil && n.Left().Class() == ir.PEXTERN && n.Left().Name().Readonly():
			// n.Left is a readonly global; use it directly.
			value = n.Left()
		case !fromType.IsInterface() && n.Esc() == EscNone && fromType.Width <= 1024:
			// n.Left does not escape. Use a stack temporary initialized to n.Left.
			value = temp(fromType)
			init.Append(typecheck(ir.Nod(ir.OAS, value, n.Left()), ctxStmt))
		}

		if value != nil {
			// Value is identical to n.Left.
			// Construct the interface directly: {type/itab, &value}.
			l := ir.Nod(ir.OEFACE, typeword(), typecheck(nodAddr(value), ctxExpr))
			l.SetType(toType)
			l.SetTypecheck(n.Typecheck())
			return l
		}

		// Implement interface to empty interface conversion.
		// tmp = i.itab
		// if tmp != nil {
		//    tmp = tmp.type
		// }
		// e = iface{tmp, i.data}
		if toType.IsEmptyInterface() && fromType.IsInterface() && !fromType.IsEmptyInterface() {
			// Evaluate the input interface.
			c := temp(fromType)
			init.Append(ir.Nod(ir.OAS, c, n.Left()))

			// Get the itab out of the interface.
			tmp := temp(types.NewPtr(types.Types[types.TUINT8]))
			init.Append(ir.Nod(ir.OAS, tmp, typecheck(ir.Nod(ir.OITAB, c, nil), ctxExpr)))

			// Get the type out of the itab.
			nif := ir.Nod(ir.OIF, typecheck(ir.Nod(ir.ONE, tmp, nodnil()), ctxExpr), nil)
			nif.PtrBody().Set1(ir.Nod(ir.OAS, tmp, itabType(tmp)))
			init.Append(nif)

			// Build the result.
			e := ir.Nod(ir.OEFACE, tmp, ifaceData(n.Pos(), c, types.NewPtr(types.Types[types.TUINT8])))
			e.SetType(toType) // assign type manually, typecheck doesn't understand OEFACE.
			e.SetTypecheck(1)
			return e
		}

		fnname, needsaddr := convFuncName(fromType, toType)

		if !needsaddr && !fromType.IsInterface() {
			// Use a specialized conversion routine that only returns a data pointer.
			// ptr = convT2X(val)
			// e = iface{typ/tab, ptr}
			fn := syslook(fnname)
			dowidth(fromType)
			fn = substArgTypes(fn, fromType)
			dowidth(fn.Type())
			call := ir.Nod(ir.OCALL, fn, nil)
			call.PtrList().Set1(n.Left())
			e := ir.Nod(ir.OEFACE, typeword(), safeexpr(walkexpr(typecheck(call, ctxExpr), init), init))
			e.SetType(toType)
			e.SetTypecheck(1)
			return e
		}

		var tab ir.Node
		if fromType.IsInterface() {
			// convI2I
			tab = typename(toType)
		} else {
			// convT2x
			tab = typeword()
		}

		v := n.Left()
		if needsaddr {
			// Types of large or unknown size are passed by reference.
			// Orderexpr arranged for n.Left to be a temporary for all
			// the conversions it could see. Comparison of an interface
			// with a non-interface, especially in a switch on interface value
			// with non-interface cases, is not visible to order.stmt, so we
			// have to fall back on allocating a temp here.
			if !islvalue(v) {
				v = copyexpr(v, v.Type(), init)
			}
			v = nodAddr(v)
		}

		dowidth(fromType)
		fn := syslook(fnname)
		fn = substArgTypes(fn, fromType, toType)
		dowidth(fn.Type())
		n = ir.Nod(ir.OCALL, fn, nil)
		n.PtrList().Set2(tab, v)
		n = typecheck(n, ctxExpr)
		return walkexpr(n, init)

	case ir.OCONV, ir.OCONVNOP:
		n.SetLeft(walkexpr(n.Left(), init))
		if n.Op() == ir.OCONVNOP && n.Type() == n.Left().Type() {
			return n.Left()
		}
		if n.Op() == ir.OCONVNOP && checkPtr(Curfn, 1) {
			if n.Type().IsPtr() && n.Left().Type().IsUnsafePtr() { // unsafe.Pointer to *T
				return walkCheckPtrAlignment(n, init, nil)
			}
			if n.Type().IsUnsafePtr() && n.Left().Type().IsUintptr() { // uintptr to unsafe.Pointer
				return walkCheckPtrArithmetic(n, init)
			}
		}
		param, result := rtconvfn(n.Left().Type(), n.Type())
		if param == types.Txxx {
			return n
		}
		fn := types.BasicTypeNames[param] + "to" + types.BasicTypeNames[result]
		return conv(mkcall(fn, types.Types[result], init, conv(n.Left(), types.Types[param])), n.Type())

	case ir.ODIV, ir.OMOD:
		n.SetLeft(walkexpr(n.Left(), init))
		n.SetRight(walkexpr(n.Right(), init))

		// rewrite complex div into function call.
		et := n.Left().Type().Kind()

		if isComplex[et] && n.Op() == ir.ODIV {
			t := n.Type()
			n = mkcall("complex128div", types.Types[types.TCOMPLEX128], init, conv(n.Left(), types.Types[types.TCOMPLEX128]), conv(n.Right(), types.Types[types.TCOMPLEX128]))
			return conv(n, t)
		}

		// Nothing to do for float divisions.
		if isFloat[et] {
			return n
		}

		// rewrite 64-bit div and mod on 32-bit architectures.
		// TODO: Remove this code once we can introduce
		// runtime calls late in SSA processing.
		if Widthreg < 8 && (et == types.TINT64 || et == types.TUINT64) {
			if n.Right().Op() == ir.OLITERAL {
				// Leave div/mod by constant powers of 2 or small 16-bit constants.
				// The SSA backend will handle those.
				switch et {
				case types.TINT64:
					c := ir.Int64Val(n.Right())
					if c < 0 {
						c = -c
					}
					if c != 0 && c&(c-1) == 0 {
						return n
					}
				case types.TUINT64:
					c := ir.Uint64Val(n.Right())
					if c < 1<<16 {
						return n
					}
					if c != 0 && c&(c-1) == 0 {
						return n
					}
				}
			}
			var fn string
			if et == types.TINT64 {
				fn = "int64"
			} else {
				fn = "uint64"
			}
			if n.Op() == ir.ODIV {
				fn += "div"
			} else {
				fn += "mod"
			}
			return mkcall(fn, n.Type(), init, conv(n.Left(), types.Types[et]), conv(n.Right(), types.Types[et]))
		}
		return n

	case ir.OINDEX:
		n.SetLeft(walkexpr(n.Left(), init))

		// save the original node for bounds checking elision.
		// If it was a ODIV/OMOD walk might rewrite it.
		r := n.Right()

		n.SetRight(walkexpr(n.Right(), init))

		// if range of type cannot exceed static array bound,
		// disable bounds check.
		if n.Bounded() {
			return n
		}
		t := n.Left().Type()
		if t != nil && t.IsPtr() {
			t = t.Elem()
		}
		if t.IsArray() {
			n.SetBounded(bounded(r, t.NumElem()))
			if base.Flag.LowerM != 0 && n.Bounded() && !ir.IsConst(n.Right(), constant.Int) {
				base.Warn("index bounds check elided")
			}
			if smallintconst(n.Right()) && !n.Bounded() {
				base.Errorf("index out of bounds")
			}
		} else if ir.IsConst(n.Left(), constant.String) {
			n.SetBounded(bounded(r, int64(len(ir.StringVal(n.Left())))))
			if base.Flag.LowerM != 0 && n.Bounded() && !ir.IsConst(n.Right(), constant.Int) {
				base.Warn("index bounds check elided")
			}
			if smallintconst(n.Right()) && !n.Bounded() {
				base.Errorf("index out of bounds")
			}
		}

		if ir.IsConst(n.Right(), constant.Int) {
			if v := n.Right().Val(); constant.Sign(v) < 0 || doesoverflow(v, types.Types[types.TINT]) {
				base.Errorf("index out of bounds")
			}
		}
		return n

	case ir.OINDEXMAP:
		// Replace m[k] with *map{access1,assign}(maptype, m, &k)
		n.SetLeft(walkexpr(n.Left(), init))
		n.SetRight(walkexpr(n.Right(), init))
		map_ := n.Left()
		key := n.Right()
		t := map_.Type()
		if n.IndexMapLValue() {
			// This m[k] expression is on the left-hand side of an assignment.
			fast := mapfast(t)
			if fast == mapslow {
				// standard version takes key by reference.
				// order.expr made sure key is addressable.
				key = nodAddr(key)
			}
			n = mkcall1(mapfn(mapassign[fast], t), nil, init, typename(t), map_, key)
		} else {
			// m[k] is not the target of an assignment.
			fast := mapfast(t)
			if fast == mapslow {
				// standard version takes key by reference.
				// order.expr made sure key is addressable.
				key = nodAddr(key)
			}

			if w := t.Elem().Width; w <= zeroValSize {
				n = mkcall1(mapfn(mapaccess1[fast], t), types.NewPtr(t.Elem()), init, typename(t), map_, key)
			} else {
				z := zeroaddr(w)
				n = mkcall1(mapfn("mapaccess1_fat", t), types.NewPtr(t.Elem()), init, typename(t), map_, key, z)
			}
		}
		n.SetType(types.NewPtr(t.Elem()))
		n.MarkNonNil() // mapaccess1* and mapassign always return non-nil pointers.
		n = ir.Nod(ir.ODEREF, n, nil)
		n.SetType(t.Elem())
		n.SetTypecheck(1)
		return n

	case ir.ORECV:
		base.Fatalf("walkexpr ORECV") // should see inside OAS only
		panic("unreachable")

	case ir.OSLICEHEADER:
		n.SetLeft(walkexpr(n.Left(), init))
		n.List().SetFirst(walkexpr(n.List().First(), init))
		n.List().SetSecond(walkexpr(n.List().Second(), init))
		return n

	case ir.OSLICE, ir.OSLICEARR, ir.OSLICESTR, ir.OSLICE3, ir.OSLICE3ARR:
		checkSlice := checkPtr(Curfn, 1) && n.Op() == ir.OSLICE3ARR && n.Left().Op() == ir.OCONVNOP && n.Left().Left().Type().IsUnsafePtr()
		if checkSlice {
			n.Left().SetLeft(walkexpr(n.Left().Left(), init))
		} else {
			n.SetLeft(walkexpr(n.Left(), init))
		}
		low, high, max := n.SliceBounds()
		low = walkexpr(low, init)
		if low != nil && isZero(low) {
			// Reduce x[0:j] to x[:j] and x[0:j:k] to x[:j:k].
			low = nil
		}
		high = walkexpr(high, init)
		max = walkexpr(max, init)
		n.SetSliceBounds(low, high, max)
		if checkSlice {
			n.SetLeft(walkCheckPtrAlignment(n.Left(), init, max))
		}
		if n.Op().IsSlice3() {
			if max != nil && max.Op() == ir.OCAP && samesafeexpr(n.Left(), max.Left()) {
				// Reduce x[i:j:cap(x)] to x[i:j].
				if n.Op() == ir.OSLICE3 {
					n.SetOp(ir.OSLICE)
				} else {
					n.SetOp(ir.OSLICEARR)
				}
				return reduceSlice(n)
			}
			return n
		}
		return reduceSlice(n)

	case ir.ONEW:
		if n.Type().Elem().NotInHeap() {
			base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", n.Type().Elem())
		}
		if n.Esc() == EscNone {
			if n.Type().Elem().Width >= maxImplicitStackVarSize {
				base.Fatalf("large ONEW with EscNone: %v", n)
			}
			r := ir.Node(temp(n.Type().Elem()))
			r = ir.Nod(ir.OAS, r, nil) // zero temp
			r = typecheck(r, ctxStmt)
			init.Append(r)
			r = nodAddr(r.Left())
			return typecheck(r, ctxExpr)
		}
		return callnew(n.Type().Elem())

	case ir.OADDSTR:
		return addstr(n, init)

	case ir.OAPPEND:
		// order should make sure we only see OAS(node, OAPPEND), which we handle above.
		base.Fatalf("append outside assignment")
		panic("unreachable")

	case ir.OCOPY:
		return copyany(n, init, instrumenting && !base.Flag.CompilingRuntime)

	case ir.OCLOSE:
		// cannot use chanfn - closechan takes any, not chan any
		fn := syslook("closechan")
		fn = substArgTypes(fn, n.Left().Type())
		return mkcall1(fn, nil, init, n.Left())

	case ir.OMAKECHAN:
		// When size fits into int, use makechan instead of
		// makechan64, which is faster and shorter on 32 bit platforms.
		size := n.Left()
		fnname := "makechan64"
		argtype := types.Types[types.TINT64]

		// Type checking guarantees that TIDEAL size is positive and fits in an int.
		// The case of size overflow when converting TUINT or TUINTPTR to TINT
		// will be handled by the negative range checks in makechan during runtime.
		if size.Type().IsKind(types.TIDEAL) || size.Type().Size() <= types.Types[types.TUINT].Size() {
			fnname = "makechan"
			argtype = types.Types[types.TINT]
		}

		return mkcall1(chanfn(fnname, 1, n.Type()), n.Type(), init, typename(n.Type()), conv(size, argtype))

	case ir.OMAKEMAP:
		t := n.Type()
		hmapType := hmap(t)
		hint := n.Left()

		// var h *hmap
		var h ir.Node
		if n.Esc() == EscNone {
			// Allocate hmap on stack.

			// var hv hmap
			hv := temp(hmapType)
			init.Append(typecheck(ir.Nod(ir.OAS, hv, nil), ctxStmt))
			// h = &hv
			h = nodAddr(hv)

			// Allocate one bucket pointed to by hmap.buckets on stack if hint
			// is not larger than BUCKETSIZE. In case hint is larger than
			// BUCKETSIZE runtime.makemap will allocate the buckets on the heap.
			// Maximum key and elem size is 128 bytes, larger objects
			// are stored with an indirection. So max bucket size is 2048+eps.
			if !ir.IsConst(hint, constant.Int) ||
				constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(BUCKETSIZE)) {

				// In case hint is larger than BUCKETSIZE runtime.makemap
				// will allocate the buckets on the heap, see #20184
				//
				// if hint <= BUCKETSIZE {
				//     var bv bmap
				//     b = &bv
				//     h.buckets = b
				// }

				nif := ir.Nod(ir.OIF, ir.Nod(ir.OLE, hint, nodintconst(BUCKETSIZE)), nil)
				nif.SetLikely(true)

				// var bv bmap
				bv := temp(bmap(t))
				nif.PtrBody().Append(ir.Nod(ir.OAS, bv, nil))

				// b = &bv
				b := nodAddr(bv)

				// h.buckets = b
				bsym := hmapType.Field(5).Sym // hmap.buckets see reflect.go:hmap
				na := ir.Nod(ir.OAS, nodSym(ir.ODOT, h, bsym), b)
				nif.PtrBody().Append(na)
				appendWalkStmt(init, nif)
			}
		}

		if ir.IsConst(hint, constant.Int) && constant.Compare(hint.Val(), token.LEQ, constant.MakeInt64(BUCKETSIZE)) {
			// Handling make(map[any]any) and
			// make(map[any]any, hint) where hint <= BUCKETSIZE
			// special allows for faster map initialization and
			// improves binary size by using calls with fewer arguments.
			// For hint <= BUCKETSIZE overLoadFactor(hint, 0) is false
			// and no buckets will be allocated by makemap. Therefore,
			// no buckets need to be allocated in this code path.
			if n.Esc() == EscNone {
				// Only need to initialize h.hash0 since
				// hmap h has been allocated on the stack already.
				// h.hash0 = fastrand()
				rand := mkcall("fastrand", types.Types[types.TUINT32], init)
				hashsym := hmapType.Field(4).Sym // hmap.hash0 see reflect.go:hmap
				appendWalkStmt(init, ir.Nod(ir.OAS, nodSym(ir.ODOT, h, hashsym), rand))
				return convnop(h, t)
			}
			// Call runtime.makehmap to allocate an
			// hmap on the heap and initialize hmap's hash0 field.
			fn := syslook("makemap_small")
			fn = substArgTypes(fn, t.Key(), t.Elem())
			return mkcall1(fn, n.Type(), init)
		}

		if n.Esc() != EscNone {
			h = nodnil()
		}
		// Map initialization with a variable or large hint is
		// more complicated. We therefore generate a call to
		// runtime.makemap to initialize hmap and allocate the
		// map buckets.

		// When hint fits into int, use makemap instead of
		// makemap64, which is faster and shorter on 32 bit platforms.
		fnname := "makemap64"
		argtype := types.Types[types.TINT64]

		// Type checking guarantees that TIDEAL hint is positive and fits in an int.
		// See checkmake call in TMAP case of OMAKE case in OpSwitch in typecheck1 function.
		// The case of hint overflow when converting TUINT or TUINTPTR to TINT
		// will be handled by the negative range checks in makemap during runtime.
		if hint.Type().IsKind(types.TIDEAL) || hint.Type().Size() <= types.Types[types.TUINT].Size() {
			fnname = "makemap"
			argtype = types.Types[types.TINT]
		}

		fn := syslook(fnname)
		fn = substArgTypes(fn, hmapType, t.Key(), t.Elem())
		return mkcall1(fn, n.Type(), init, typename(n.Type()), conv(hint, argtype), h)

	case ir.OMAKESLICE:
		l := n.Left()
		r := n.Right()
		if r == nil {
			r = safeexpr(l, init)
			l = r
		}
		t := n.Type()
		if t.Elem().NotInHeap() {
			base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", t.Elem())
		}
		if n.Esc() == EscNone {
			if why := heapAllocReason(n); why != "" {
				base.Fatalf("%v has EscNone, but %v", n, why)
			}
			// var arr [r]T
			// n = arr[:l]
			i := indexconst(r)
			if i < 0 {
				base.Fatalf("walkexpr: invalid index %v", r)
			}

			// cap is constrained to [0,2^31) or [0,2^63) depending on whether
			// we're in 32-bit or 64-bit systems. So it's safe to do:
			//
			// if uint64(len) > cap {
			//     if len < 0 { panicmakeslicelen() }
			//     panicmakeslicecap()
			// }
			nif := ir.Nod(ir.OIF, ir.Nod(ir.OGT, conv(l, types.Types[types.TUINT64]), nodintconst(i)), nil)
			niflen := ir.Nod(ir.OIF, ir.Nod(ir.OLT, l, nodintconst(0)), nil)
			niflen.PtrBody().Set1(mkcall("panicmakeslicelen", nil, init))
			nif.PtrBody().Append(niflen, mkcall("panicmakeslicecap", nil, init))
			init.Append(typecheck(nif, ctxStmt))

			t = types.NewArray(t.Elem(), i) // [r]T
			var_ := temp(t)
			appendWalkStmt(init, ir.Nod(ir.OAS, var_, nil)) // zero temp
			r := ir.Nod(ir.OSLICE, var_, nil)               // arr[:l]
			r.SetSliceBounds(nil, l, nil)
			// The conv is necessary in case n.Type is named.
			return walkexpr(typecheck(conv(r, n.Type()), ctxExpr), init)
		}

		// n escapes; set up a call to makeslice.
		// When len and cap can fit into int, use makeslice instead of
		// makeslice64, which is faster and shorter on 32 bit platforms.

		len, cap := l, r

		fnname := "makeslice64"
		argtype := types.Types[types.TINT64]

		// Type checking guarantees that TIDEAL len/cap are positive and fit in an int.
		// The case of len or cap overflow when converting TUINT or TUINTPTR to TINT
		// will be handled by the negative range checks in makeslice during runtime.
		if (len.Type().IsKind(types.TIDEAL) || len.Type().Size() <= types.Types[types.TUINT].Size()) &&
			(cap.Type().IsKind(types.TIDEAL) || cap.Type().Size() <= types.Types[types.TUINT].Size()) {
			fnname = "makeslice"
			argtype = types.Types[types.TINT]
		}

		m := ir.Nod(ir.OSLICEHEADER, nil, nil)
		m.SetType(t)

		fn := syslook(fnname)
		m.SetLeft(mkcall1(fn, types.Types[types.TUNSAFEPTR], init, typename(t.Elem()), conv(len, argtype), conv(cap, argtype)))
		m.Left().MarkNonNil()
		m.PtrList().Set2(conv(len, types.Types[types.TINT]), conv(cap, types.Types[types.TINT]))
		return walkexpr(typecheck(m, ctxExpr), init)

	case ir.OMAKESLICECOPY:
		if n.Esc() == EscNone {
			base.Fatalf("OMAKESLICECOPY with EscNone: %v", n)
		}

		t := n.Type()
		if t.Elem().NotInHeap() {
			base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", t.Elem())
		}

		length := conv(n.Left(), types.Types[types.TINT])
		copylen := ir.Nod(ir.OLEN, n.Right(), nil)
		copyptr := ir.Nod(ir.OSPTR, n.Right(), nil)

		if !t.Elem().HasPointers() && n.Bounded() {
			// When len(to)==len(from) and elements have no pointers:
			// replace make+copy with runtime.mallocgc+runtime.memmove.

			// We do not check for overflow of len(to)*elem.Width here
			// since len(from) is an existing checked slice capacity
			// with same elem.Width for the from slice.
			size := ir.Nod(ir.OMUL, conv(length, types.Types[types.TUINTPTR]), conv(nodintconst(t.Elem().Width), types.Types[types.TUINTPTR]))

			// instantiate mallocgc(size uintptr, typ *byte, needszero bool) unsafe.Pointer
			fn := syslook("mallocgc")
			sh := ir.Nod(ir.OSLICEHEADER, nil, nil)
			sh.SetLeft(mkcall1(fn, types.Types[types.TUNSAFEPTR], init, size, nodnil(), nodbool(false)))
			sh.Left().MarkNonNil()
			sh.PtrList().Set2(length, length)
			sh.SetType(t)

			s := temp(t)
			r := typecheck(ir.Nod(ir.OAS, s, sh), ctxStmt)
			r = walkexpr(r, init)
			init.Append(r)

			// instantiate memmove(to *any, frm *any, size uintptr)
			fn = syslook("memmove")
			fn = substArgTypes(fn, t.Elem(), t.Elem())
			ncopy := mkcall1(fn, nil, init, ir.Nod(ir.OSPTR, s, nil), copyptr, size)
			ncopy = typecheck(ncopy, ctxStmt)
			ncopy = walkexpr(ncopy, init)
			init.Append(ncopy)

			return s
		}
		// Replace make+copy with runtime.makeslicecopy.
		// instantiate makeslicecopy(typ *byte, tolen int, fromlen int, from unsafe.Pointer) unsafe.Pointer
		fn := syslook("makeslicecopy")
		s := ir.Nod(ir.OSLICEHEADER, nil, nil)
		s.SetLeft(mkcall1(fn, types.Types[types.TUNSAFEPTR], init, typename(t.Elem()), length, copylen, conv(copyptr, types.Types[types.TUNSAFEPTR])))
		s.Left().MarkNonNil()
		s.PtrList().Set2(length, length)
		s.SetType(t)
		n = typecheck(s, ctxExpr)
		return walkexpr(n, init)

	case ir.ORUNESTR:
		a := nodnil()
		if n.Esc() == EscNone {
			t := types.NewArray(types.Types[types.TUINT8], 4)
			a = nodAddr(temp(t))
		}
		// intstring(*[4]byte, rune)
		return mkcall("intstring", n.Type(), init, a, conv(n.Left(), types.Types[types.TINT64]))

	case ir.OBYTES2STR, ir.ORUNES2STR:
		a := nodnil()
		if n.Esc() == EscNone {
			// Create temporary buffer for string on stack.
			t := types.NewArray(types.Types[types.TUINT8], tmpstringbufsize)
			a = nodAddr(temp(t))
		}
		if n.Op() == ir.ORUNES2STR {
			// slicerunetostring(*[32]byte, []rune) string
			return mkcall("slicerunetostring", n.Type(), init, a, n.Left())
		}
		// slicebytetostring(*[32]byte, ptr *byte, n int) string
		n.SetLeft(cheapexpr(n.Left(), init))
		ptr, len := backingArrayPtrLen(n.Left())
		return mkcall("slicebytetostring", n.Type(), init, a, ptr, len)

	case ir.OBYTES2STRTMP:
		n.SetLeft(walkexpr(n.Left(), init))
		if !instrumenting {
			// Let the backend handle OBYTES2STRTMP directly
			// to avoid a function call to slicebytetostringtmp.
			return n
		}
		// slicebytetostringtmp(ptr *byte, n int) string
		n.SetLeft(cheapexpr(n.Left(), init))
		ptr, len := backingArrayPtrLen(n.Left())
		return mkcall("slicebytetostringtmp", n.Type(), init, ptr, len)

	case ir.OSTR2BYTES:
		s := n.Left()
		if ir.IsConst(s, constant.String) {
			sc := ir.StringVal(s)

			// Allocate a [n]byte of the right size.
			t := types.NewArray(types.Types[types.TUINT8], int64(len(sc)))
			var a ir.Node
			if n.Esc() == EscNone && len(sc) <= int(maxImplicitStackVarSize) {
				a = nodAddr(temp(t))
			} else {
				a = callnew(t)
			}
			p := temp(t.PtrTo()) // *[n]byte
			init.Append(typecheck(ir.Nod(ir.OAS, p, a), ctxStmt))

			// Copy from the static string data to the [n]byte.
			if len(sc) > 0 {
				as := ir.Nod(ir.OAS,
					ir.Nod(ir.ODEREF, p, nil),
					ir.Nod(ir.ODEREF, convnop(ir.Nod(ir.OSPTR, s, nil), t.PtrTo()), nil))
				appendWalkStmt(init, as)
			}

			// Slice the [n]byte to a []byte.
			slice := ir.NodAt(n.Pos(), ir.OSLICEARR, p, nil)
			slice.SetType(n.Type())
			slice.SetTypecheck(1)
			return walkexpr(slice, init)
		}

		a := nodnil()
		if n.Esc() == EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[types.TUINT8], tmpstringbufsize)
			a = nodAddr(temp(t))
		}
		// stringtoslicebyte(*32[byte], string) []byte
		return mkcall("stringtoslicebyte", n.Type(), init, a, conv(s, types.Types[types.TSTRING]))

	case ir.OSTR2BYTESTMP:
		// []byte(string) conversion that creates a slice
		// referring to the actual string bytes.
		// This conversion is handled later by the backend and
		// is only for use by internal compiler optimizations
		// that know that the slice won't be mutated.
		// The only such case today is:
		// for i, c := range []byte(string)
		n.SetLeft(walkexpr(n.Left(), init))
		return n

	case ir.OSTR2RUNES:
		a := nodnil()
		if n.Esc() == EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[types.TINT32], tmpstringbufsize)
			a = nodAddr(temp(t))
		}
		// stringtoslicerune(*[32]rune, string) []rune
		return mkcall("stringtoslicerune", n.Type(), init, a, conv(n.Left(), types.Types[types.TSTRING]))

	case ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT, ir.OSTRUCTLIT, ir.OPTRLIT:
		if isStaticCompositeLiteral(n) && !canSSAType(n.Type()) {
			// n can be directly represented in the read-only data section.
			// Make direct reference to the static data. See issue 12841.
			vstat := readonlystaticname(n.Type())
			fixedlit(inInitFunction, initKindStatic, n, vstat, init)
			return typecheck(vstat, ctxExpr)
		}
		var_ := temp(n.Type())
		anylit(n, var_, init)
		return var_

	case ir.OSEND:
		n1 := n.Right()
		n1 = assignconv(n1, n.Left().Type().Elem(), "chan send")
		n1 = walkexpr(n1, init)
		n1 = nodAddr(n1)
		return mkcall1(chanfn("chansend1", 2, n.Left().Type()), nil, init, n.Left(), n1)

	case ir.OCLOSURE:
		return walkclosure(n, init)

	case ir.OCALLPART:
		return walkpartialcall(n.(*ir.CallPartExpr), init)
	}

	// No return! Each case must return (or panic),
	// to avoid confusion about what gets returned
	// in the presence of type assertions.
}

// markTypeUsedInInterface marks that type t is converted to an interface.
// This information is used in the linker in dead method elimination.
func markTypeUsedInInterface(t *types.Type, from *obj.LSym) {
	tsym := typenamesym(t).Linksym()
	// Emit a marker relocation. The linker will know the type is converted
	// to an interface if "from" is reachable.
	r := obj.Addrel(from)
	r.Sym = tsym
	r.Type = objabi.R_USEIFACE
}

// markUsedIfaceMethod marks that an interface method is used in the current
// function. n is OCALLINTER node.
func markUsedIfaceMethod(n ir.Node) {
	ityp := n.Left().Left().Type()
	tsym := typenamesym(ityp).Linksym()
	r := obj.Addrel(Curfn.LSym)
	r.Sym = tsym
	// n.Left.Xoffset is the method index * Widthptr (the offset of code pointer
	// in itab).
	midx := n.Left().Offset() / int64(Widthptr)
	r.Add = ifaceMethodOffset(ityp, midx)
	r.Type = objabi.R_USEIFACEMETHOD
}

// rtconvfn returns the parameter and result types that will be used by a
// runtime function to convert from type src to type dst. The runtime function
// name can be derived from the names of the returned types.
//
// If no such function is necessary, it returns (Txxx, Txxx).
func rtconvfn(src, dst *types.Type) (param, result types.Kind) {
	if thearch.SoftFloat {
		return types.Txxx, types.Txxx
	}

	switch thearch.LinkArch.Family {
	case sys.ARM, sys.MIPS:
		if src.IsFloat() {
			switch dst.Kind() {
			case types.TINT64, types.TUINT64:
				return types.TFLOAT64, dst.Kind()
			}
		}
		if dst.IsFloat() {
			switch src.Kind() {
			case types.TINT64, types.TUINT64:
				return src.Kind(), types.TFLOAT64
			}
		}

	case sys.I386:
		if src.IsFloat() {
			switch dst.Kind() {
			case types.TINT64, types.TUINT64:
				return types.TFLOAT64, dst.Kind()
			case types.TUINT32, types.TUINT, types.TUINTPTR:
				return types.TFLOAT64, types.TUINT32
			}
		}
		if dst.IsFloat() {
			switch src.Kind() {
			case types.TINT64, types.TUINT64:
				return src.Kind(), types.TFLOAT64
			case types.TUINT32, types.TUINT, types.TUINTPTR:
				return types.TUINT32, types.TFLOAT64
			}
		}
	}
	return types.Txxx, types.Txxx
}

// TODO(josharian): combine this with its caller and simplify
func reduceSlice(n ir.Node) ir.Node {
	low, high, max := n.SliceBounds()
	if high != nil && high.Op() == ir.OLEN && samesafeexpr(n.Left(), high.Left()) {
		// Reduce x[i:len(x)] to x[i:].
		high = nil
	}
	n.SetSliceBounds(low, high, max)
	if (n.Op() == ir.OSLICE || n.Op() == ir.OSLICESTR) && low == nil && high == nil {
		// Reduce x[:] to x.
		if base.Debug.Slice > 0 {
			base.Warn("slice: omit slice operation")
		}
		return n.Left()
	}
	return n
}

func ascompatee1(l ir.Node, r ir.Node, init *ir.Nodes) ir.Node {
	// convas will turn map assigns into function calls,
	// making it impossible for reorder3 to work.
	n := ir.Nod(ir.OAS, l, r)

	if l.Op() == ir.OINDEXMAP {
		return n
	}

	return convas(n, init)
}

func ascompatee(op ir.Op, nl, nr []ir.Node, init *ir.Nodes) []ir.Node {
	// check assign expression list to
	// an expression list. called in
	//	expr-list = expr-list

	// ensure order of evaluation for function calls
	for i := range nl {
		nl[i] = safeexpr(nl[i], init)
	}
	for i1 := range nr {
		nr[i1] = safeexpr(nr[i1], init)
	}

	var nn []ir.Node
	i := 0
	for ; i < len(nl); i++ {
		if i >= len(nr) {
			break
		}
		// Do not generate 'x = x' during return. See issue 4014.
		if op == ir.ORETURN && samesafeexpr(nl[i], nr[i]) {
			continue
		}
		nn = append(nn, ascompatee1(nl[i], nr[i], init))
	}

	// cannot happen: caller checked that lists had same length
	if i < len(nl) || i < len(nr) {
		var nln, nrn ir.Nodes
		nln.Set(nl)
		nrn.Set(nr)
		base.Fatalf("error in shape across %+v %v %+v / %d %d [%s]", nln, op, nrn, len(nl), len(nr), ir.FuncName(Curfn))
	}
	return nn
}

// fncall reports whether assigning an rvalue of type rt to an lvalue l might involve a function call.
func fncall(l ir.Node, rt *types.Type) bool {
	if l.HasCall() || l.Op() == ir.OINDEXMAP {
		return true
	}
	if types.Identical(l.Type(), rt) {
		return false
	}
	// There might be a conversion required, which might involve a runtime call.
	return true
}

// check assign type list to
// an expression list. called in
//	expr-list = func()
func ascompatet(nl ir.Nodes, nr *types.Type) []ir.Node {
	if nl.Len() != nr.NumFields() {
		base.Fatalf("ascompatet: assignment count mismatch: %d = %d", nl.Len(), nr.NumFields())
	}

	var nn, mm ir.Nodes
	for i, l := range nl.Slice() {
		if ir.IsBlank(l) {
			continue
		}
		r := nr.Field(i)

		// Any assignment to an lvalue that might cause a function call must be
		// deferred until all the returned values have been read.
		if fncall(l, r.Type) {
			tmp := ir.Node(temp(r.Type))
			tmp = typecheck(tmp, ctxExpr)
			a := convas(ir.Nod(ir.OAS, l, tmp), &mm)
			mm.Append(a)
			l = tmp
		}

		res := ir.Nod(ir.ORESULT, nil, nil)
		res.SetOffset(base.Ctxt.FixedFrameSize() + r.Offset)
		res.SetType(r.Type)
		res.SetTypecheck(1)

		a := convas(ir.Nod(ir.OAS, l, res), &nn)
		updateHasCall(a)
		if a.HasCall() {
			ir.Dump("ascompatet ucount", a)
			base.Fatalf("ascompatet: too many function calls evaluating parameters")
		}

		nn.Append(a)
	}
	return append(nn.Slice(), mm.Slice()...)
}

// package all the arguments that match a ... T parameter into a []T.
func mkdotargslice(typ *types.Type, args []ir.Node) ir.Node {
	var n ir.Node
	if len(args) == 0 {
		n = nodnil()
		n.SetType(typ)
	} else {
		n = ir.Nod(ir.OCOMPLIT, nil, ir.TypeNode(typ))
		n.PtrList().Append(args...)
		n.SetImplicit(true)
	}

	n = typecheck(n, ctxExpr)
	if n.Type() == nil {
		base.Fatalf("mkdotargslice: typecheck failed")
	}
	return n
}

// fixVariadicCall rewrites calls to variadic functions to use an
// explicit ... argument if one is not already present.
func fixVariadicCall(call ir.Node) {
	fntype := call.Left().Type()
	if !fntype.IsVariadic() || call.IsDDD() {
		return
	}

	vi := fntype.NumParams() - 1
	vt := fntype.Params().Field(vi).Type

	args := call.List().Slice()
	extra := args[vi:]
	slice := mkdotargslice(vt, extra)
	for i := range extra {
		extra[i] = nil // allow GC
	}

	call.PtrList().Set(append(args[:vi], slice))
	call.SetIsDDD(true)
}

func walkCall(n ir.Node, init *ir.Nodes) {
	if n.Rlist().Len() != 0 {
		return // already walked
	}

	params := n.Left().Type().Params()
	args := n.List().Slice()

	n.SetLeft(walkexpr(n.Left(), init))
	walkexprlist(args, init)

	// If this is a method call, add the receiver at the beginning of the args.
	if n.Op() == ir.OCALLMETH {
		withRecv := make([]ir.Node, len(args)+1)
		withRecv[0] = n.Left().Left()
		n.Left().SetLeft(nil)
		copy(withRecv[1:], args)
		args = withRecv
	}

	// For any argument whose evaluation might require a function call,
	// store that argument into a temporary variable,
	// to prevent that calls from clobbering arguments already on the stack.
	// When instrumenting, all arguments might require function calls.
	var tempAssigns []ir.Node
	for i, arg := range args {
		updateHasCall(arg)
		// Determine param type.
		var t *types.Type
		if n.Op() == ir.OCALLMETH {
			if i == 0 {
				t = n.Left().Type().Recv().Type
			} else {
				t = params.Field(i - 1).Type
			}
		} else {
			t = params.Field(i).Type
		}
		if instrumenting || fncall(arg, t) {
			// make assignment of fncall to tempAt
			tmp := temp(t)
			a := convas(ir.Nod(ir.OAS, tmp, arg), init)
			tempAssigns = append(tempAssigns, a)
			// replace arg with temp
			args[i] = tmp
		}
	}

	n.PtrList().Set(tempAssigns)
	n.PtrRlist().Set(args)
}

// generate code for print
func walkprint(nn ir.Node, init *ir.Nodes) ir.Node {
	// Hoist all the argument evaluation up before the lock.
	walkexprlistcheap(nn.List().Slice(), init)

	// For println, add " " between elements and "\n" at the end.
	if nn.Op() == ir.OPRINTN {
		s := nn.List().Slice()
		t := make([]ir.Node, 0, len(s)*2)
		for i, n := range s {
			if i != 0 {
				t = append(t, nodstr(" "))
			}
			t = append(t, n)
		}
		t = append(t, nodstr("\n"))
		nn.PtrList().Set(t)
	}

	// Collapse runs of constant strings.
	s := nn.List().Slice()
	t := make([]ir.Node, 0, len(s))
	for i := 0; i < len(s); {
		var strs []string
		for i < len(s) && ir.IsConst(s[i], constant.String) {
			strs = append(strs, ir.StringVal(s[i]))
			i++
		}
		if len(strs) > 0 {
			t = append(t, nodstr(strings.Join(strs, "")))
		}
		if i < len(s) {
			t = append(t, s[i])
			i++
		}
	}
	nn.PtrList().Set(t)

	calls := []ir.Node{mkcall("printlock", nil, init)}
	for i, n := range nn.List().Slice() {
		if n.Op() == ir.OLITERAL {
			if n.Type() == types.UntypedRune {
				n = defaultlit(n, types.RuneType)
			}

			switch n.Val().Kind() {
			case constant.Int:
				n = defaultlit(n, types.Types[types.TINT64])

			case constant.Float:
				n = defaultlit(n, types.Types[types.TFLOAT64])
			}
		}

		if n.Op() != ir.OLITERAL && n.Type() != nil && n.Type().Kind() == types.TIDEAL {
			n = defaultlit(n, types.Types[types.TINT64])
		}
		n = defaultlit(n, nil)
		nn.List().SetIndex(i, n)
		if n.Type() == nil || n.Type().Kind() == types.TFORW {
			continue
		}

		var on ir.Node
		switch n.Type().Kind() {
		case types.TINTER:
			if n.Type().IsEmptyInterface() {
				on = syslook("printeface")
			} else {
				on = syslook("printiface")
			}
			on = substArgTypes(on, n.Type()) // any-1
		case types.TPTR:
			if n.Type().Elem().NotInHeap() {
				on = syslook("printuintptr")
				n = ir.Nod(ir.OCONV, n, nil)
				n.SetType(types.Types[types.TUNSAFEPTR])
				n = ir.Nod(ir.OCONV, n, nil)
				n.SetType(types.Types[types.TUINTPTR])
				break
			}
			fallthrough
		case types.TCHAN, types.TMAP, types.TFUNC, types.TUNSAFEPTR:
			on = syslook("printpointer")
			on = substArgTypes(on, n.Type()) // any-1
		case types.TSLICE:
			on = syslook("printslice")
			on = substArgTypes(on, n.Type()) // any-1
		case types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64, types.TUINTPTR:
			if isRuntimePkg(n.Type().Sym().Pkg) && n.Type().Sym().Name == "hex" {
				on = syslook("printhex")
			} else {
				on = syslook("printuint")
			}
		case types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64:
			on = syslook("printint")
		case types.TFLOAT32, types.TFLOAT64:
			on = syslook("printfloat")
		case types.TCOMPLEX64, types.TCOMPLEX128:
			on = syslook("printcomplex")
		case types.TBOOL:
			on = syslook("printbool")
		case types.TSTRING:
			cs := ""
			if ir.IsConst(n, constant.String) {
				cs = ir.StringVal(n)
			}
			switch cs {
			case " ":
				on = syslook("printsp")
			case "\n":
				on = syslook("printnl")
			default:
				on = syslook("printstring")
			}
		default:
			badtype(ir.OPRINT, n.Type(), nil)
			continue
		}

		r := ir.Nod(ir.OCALL, on, nil)
		if params := on.Type().Params().FieldSlice(); len(params) > 0 {
			t := params[0].Type
			if !types.Identical(t, n.Type()) {
				n = ir.Nod(ir.OCONV, n, nil)
				n.SetType(t)
			}
			r.PtrList().Append(n)
		}
		calls = append(calls, r)
	}

	calls = append(calls, mkcall("printunlock", nil, init))

	typecheckslice(calls, ctxStmt)
	walkexprlist(calls, init)

	r := ir.Nod(ir.OBLOCK, nil, nil)
	r.PtrList().Set(calls)
	return walkstmt(typecheck(r, ctxStmt))
}

func callnew(t *types.Type) ir.Node {
	dowidth(t)
	n := ir.Nod(ir.ONEWOBJ, typename(t), nil)
	n.SetType(types.NewPtr(t))
	n.SetTypecheck(1)
	n.MarkNonNil()
	return n
}

// isReflectHeaderDataField reports whether l is an expression p.Data
// where p has type reflect.SliceHeader or reflect.StringHeader.
func isReflectHeaderDataField(l ir.Node) bool {
	if l.Type() != types.Types[types.TUINTPTR] {
		return false
	}

	var tsym *types.Sym
	switch l.Op() {
	case ir.ODOT:
		tsym = l.Left().Type().Sym()
	case ir.ODOTPTR:
		tsym = l.Left().Type().Elem().Sym()
	default:
		return false
	}

	if tsym == nil || l.Sym().Name != "Data" || tsym.Pkg.Path != "reflect" {
		return false
	}
	return tsym.Name == "SliceHeader" || tsym.Name == "StringHeader"
}

func convas(n ir.Node, init *ir.Nodes) ir.Node {
	if n.Op() != ir.OAS {
		base.Fatalf("convas: not OAS %v", n.Op())
	}
	defer updateHasCall(n)

	n.SetTypecheck(1)

	if n.Left() == nil || n.Right() == nil {
		return n
	}

	lt := n.Left().Type()
	rt := n.Right().Type()
	if lt == nil || rt == nil {
		return n
	}

	if ir.IsBlank(n.Left()) {
		n.SetRight(defaultlit(n.Right(), nil))
		return n
	}

	if !types.Identical(lt, rt) {
		n.SetRight(assignconv(n.Right(), lt, "assignment"))
		n.SetRight(walkexpr(n.Right(), init))
	}
	dowidth(n.Right().Type())

	return n
}

// from ascompat[ee]
//	a,b = c,d
// simultaneous assignment. there cannot
// be later use of an earlier lvalue.
//
// function calls have been removed.
func reorder3(all []ir.Node) []ir.Node {
	// If a needed expression may be affected by an
	// earlier assignment, make an early copy of that
	// expression and use the copy instead.
	var early []ir.Node

	var mapinit ir.Nodes
	for i, n := range all {
		l := n.Left()

		// Save subexpressions needed on left side.
		// Drill through non-dereferences.
		for {
			if l.Op() == ir.ODOT || l.Op() == ir.OPAREN {
				l = l.Left()
				continue
			}

			if l.Op() == ir.OINDEX && l.Left().Type().IsArray() {
				l.SetRight(reorder3save(l.Right(), all, i, &early))
				l = l.Left()
				continue
			}

			break
		}

		switch l.Op() {
		default:
			base.Fatalf("reorder3 unexpected lvalue %v", l.Op())

		case ir.ONAME:
			break

		case ir.OINDEX, ir.OINDEXMAP:
			l.SetLeft(reorder3save(l.Left(), all, i, &early))
			l.SetRight(reorder3save(l.Right(), all, i, &early))
			if l.Op() == ir.OINDEXMAP {
				all[i] = convas(all[i], &mapinit)
			}

		case ir.ODEREF, ir.ODOTPTR:
			l.SetLeft(reorder3save(l.Left(), all, i, &early))
		}

		// Save expression on right side.
		all[i].SetRight(reorder3save(all[i].Right(), all, i, &early))
	}

	early = append(mapinit.Slice(), early...)
	return append(early, all...)
}

// if the evaluation of *np would be affected by the
// assignments in all up to but not including the ith assignment,
// copy into a temporary during *early and
// replace *np with that temp.
// The result of reorder3save MUST be assigned back to n, e.g.
// 	n.Left = reorder3save(n.Left, all, i, early)
func reorder3save(n ir.Node, all []ir.Node, i int, early *[]ir.Node) ir.Node {
	if !aliased(n, all[:i]) {
		return n
	}

	q := ir.Node(temp(n.Type()))
	q = ir.Nod(ir.OAS, q, n)
	q = typecheck(q, ctxStmt)
	*early = append(*early, q)
	return q.Left()
}

// what's the outer value that a write to n affects?
// outer value means containing struct or array.
func outervalue(n ir.Node) ir.Node {
	for {
		switch n.Op() {
		case ir.OXDOT:
			base.Fatalf("OXDOT in walk")
		case ir.ODOT, ir.OPAREN, ir.OCONVNOP:
			n = n.Left()
			continue
		case ir.OINDEX:
			if n.Left().Type() != nil && n.Left().Type().IsArray() {
				n = n.Left()
				continue
			}
		}

		return n
	}
}

// Is it possible that the computation of r might be
// affected by assignments in all?
func aliased(r ir.Node, all []ir.Node) bool {
	if r == nil {
		return false
	}

	// Treat all fields of a struct as referring to the whole struct.
	// We could do better but we would have to keep track of the fields.
	for r.Op() == ir.ODOT {
		r = r.Left()
	}

	// Look for obvious aliasing: a variable being assigned
	// during the all list and appearing in n.
	// Also record whether there are any writes to addressable
	// memory (either main memory or variables whose addresses
	// have been taken).
	memwrite := false
	for _, as := range all {
		// We can ignore assignments to blank.
		if ir.IsBlank(as.Left()) {
			continue
		}

		l := outervalue(as.Left())
		if l.Op() != ir.ONAME {
			memwrite = true
			continue
		}

		switch l.Class() {
		default:
			base.Fatalf("unexpected class: %v, %v", l, l.Class())

		case ir.PAUTOHEAP, ir.PEXTERN:
			memwrite = true
			continue

		case ir.PAUTO, ir.PPARAM, ir.PPARAMOUT:
			if l.Name().Addrtaken() {
				memwrite = true
				continue
			}

			if vmatch2(l, r) {
				// Direct hit: l appears in r.
				return true
			}
		}
	}

	// The variables being written do not appear in r.
	// However, r might refer to computed addresses
	// that are being written.

	// If no computed addresses are affected by the writes, no aliasing.
	if !memwrite {
		return false
	}

	// If r does not refer to computed addresses
	// (that is, if r only refers to variables whose addresses
	// have not been taken), no aliasing.
	if varexpr(r) {
		return false
	}

	// Otherwise, both the writes and r refer to computed memory addresses.
	// Assume that they might conflict.
	return true
}

// does the evaluation of n only refer to variables
// whose addresses have not been taken?
// (and no other memory)
func varexpr(n ir.Node) bool {
	if n == nil {
		return true
	}

	switch n.Op() {
	case ir.OLITERAL, ir.ONIL:
		return true

	case ir.ONAME:
		switch n.Class() {
		case ir.PAUTO, ir.PPARAM, ir.PPARAMOUT:
			if !n.Name().Addrtaken() {
				return true
			}
		}

		return false

	case ir.OADD,
		ir.OSUB,
		ir.OOR,
		ir.OXOR,
		ir.OMUL,
		ir.ODIV,
		ir.OMOD,
		ir.OLSH,
		ir.ORSH,
		ir.OAND,
		ir.OANDNOT,
		ir.OPLUS,
		ir.ONEG,
		ir.OBITNOT,
		ir.OPAREN,
		ir.OANDAND,
		ir.OOROR,
		ir.OCONV,
		ir.OCONVNOP,
		ir.OCONVIFACE,
		ir.ODOTTYPE:
		return varexpr(n.Left()) && varexpr(n.Right())

	case ir.ODOT: // but not ODOTPTR
		// Should have been handled in aliased.
		base.Fatalf("varexpr unexpected ODOT")
	}

	// Be conservative.
	return false
}

// is the name l mentioned in r?
func vmatch2(l ir.Node, r ir.Node) bool {
	if r == nil {
		return false
	}
	switch r.Op() {
	// match each right given left
	case ir.ONAME:
		return l == r

	case ir.OLITERAL, ir.ONIL:
		return false
	}

	if vmatch2(l, r.Left()) {
		return true
	}
	if vmatch2(l, r.Right()) {
		return true
	}
	for _, n := range r.List().Slice() {
		if vmatch2(l, n) {
			return true
		}
	}
	return false
}

// is any name mentioned in l also mentioned in r?
// called by sinit.go
func vmatch1(l ir.Node, r ir.Node) bool {
	// isolate all left sides
	if l == nil || r == nil {
		return false
	}
	switch l.Op() {
	case ir.ONAME:
		switch l.Class() {
		case ir.PPARAM, ir.PAUTO:
			break

		default:
			// assignment to non-stack variable must be
			// delayed if right has function calls.
			if r.HasCall() {
				return true
			}
		}

		return vmatch2(l, r)

	case ir.OLITERAL, ir.ONIL:
		return false
	}

	if vmatch1(l.Left(), r) {
		return true
	}
	if vmatch1(l.Right(), r) {
		return true
	}
	for _, n := range l.List().Slice() {
		if vmatch1(n, r) {
			return true
		}
	}
	return false
}

// paramstoheap returns code to allocate memory for heap-escaped parameters
// and to copy non-result parameters' values from the stack.
func paramstoheap(params *types.Type) []ir.Node {
	var nn []ir.Node
	for _, t := range params.Fields().Slice() {
		v := ir.AsNode(t.Nname)
		if v != nil && v.Sym() != nil && strings.HasPrefix(v.Sym().Name, "~r") { // unnamed result
			v = nil
		}
		if v == nil {
			continue
		}

		if stackcopy := v.Name().Stackcopy; stackcopy != nil {
			nn = append(nn, walkstmt(ir.Nod(ir.ODCL, v, nil)))
			if stackcopy.Class() == ir.PPARAM {
				nn = append(nn, walkstmt(typecheck(ir.Nod(ir.OAS, v, stackcopy), ctxStmt)))
			}
		}
	}

	return nn
}

// zeroResults zeros the return values at the start of the function.
// We need to do this very early in the function.  Defer might stop a
// panic and show the return values as they exist at the time of
// panic.  For precise stacks, the garbage collector assumes results
// are always live, so we need to zero them before any allocations,
// even allocations to move params/results to the heap.
// The generated code is added to Curfn's Enter list.
func zeroResults() {
	for _, f := range Curfn.Type().Results().Fields().Slice() {
		v := ir.AsNode(f.Nname)
		if v != nil && v.Name().Heapaddr != nil {
			// The local which points to the return value is the
			// thing that needs zeroing. This is already handled
			// by a Needzero annotation in plive.go:livenessepilogue.
			continue
		}
		if isParamHeapCopy(v) {
			// TODO(josharian/khr): Investigate whether we can switch to "continue" here,
			// and document more in either case.
			// In the review of CL 114797, Keith wrote (roughly):
			// I don't think the zeroing below matters.
			// The stack return value will never be marked as live anywhere in the function.
			// It is not written to until deferreturn returns.
			v = v.Name().Stackcopy
		}
		// Zero the stack location containing f.
		Curfn.Enter.Append(ir.NodAt(Curfn.Pos(), ir.OAS, v, nil))
	}
}

// returnsfromheap returns code to copy values for heap-escaped parameters
// back to the stack.
func returnsfromheap(params *types.Type) []ir.Node {
	var nn []ir.Node
	for _, t := range params.Fields().Slice() {
		v := ir.AsNode(t.Nname)
		if v == nil {
			continue
		}
		if stackcopy := v.Name().Stackcopy; stackcopy != nil && stackcopy.Class() == ir.PPARAMOUT {
			nn = append(nn, walkstmt(typecheck(ir.Nod(ir.OAS, stackcopy, v), ctxStmt)))
		}
	}

	return nn
}

// heapmoves generates code to handle migrating heap-escaped parameters
// between the stack and the heap. The generated code is added to Curfn's
// Enter and Exit lists.
func heapmoves() {
	lno := base.Pos
	base.Pos = Curfn.Pos()
	nn := paramstoheap(Curfn.Type().Recvs())
	nn = append(nn, paramstoheap(Curfn.Type().Params())...)
	nn = append(nn, paramstoheap(Curfn.Type().Results())...)
	Curfn.Enter.Append(nn...)
	base.Pos = Curfn.Endlineno
	Curfn.Exit.Append(returnsfromheap(Curfn.Type().Results())...)
	base.Pos = lno
}

func vmkcall(fn ir.Node, t *types.Type, init *ir.Nodes, va []ir.Node) ir.Node {
	if fn.Type() == nil || fn.Type().Kind() != types.TFUNC {
		base.Fatalf("mkcall %v %v", fn, fn.Type())
	}

	n := fn.Type().NumParams()
	if n != len(va) {
		base.Fatalf("vmkcall %v needs %v args got %v", fn, n, len(va))
	}

	call := ir.Nod(ir.OCALL, fn, nil)
	call.PtrList().Set(va)
	ctx := ctxStmt
	if fn.Type().NumResults() > 0 {
		ctx = ctxExpr | ctxMultiOK
	}
	r1 := typecheck(call, ctx)
	r1.SetType(t)
	return walkexpr(r1, init)
}

func mkcall(name string, t *types.Type, init *ir.Nodes, args ...ir.Node) ir.Node {
	return vmkcall(syslook(name), t, init, args)
}

func mkcall1(fn ir.Node, t *types.Type, init *ir.Nodes, args ...ir.Node) ir.Node {
	return vmkcall(fn, t, init, args)
}

func conv(n ir.Node, t *types.Type) ir.Node {
	if types.Identical(n.Type(), t) {
		return n
	}
	n = ir.Nod(ir.OCONV, n, nil)
	n.SetType(t)
	n = typecheck(n, ctxExpr)
	return n
}

// convnop converts node n to type t using the OCONVNOP op
// and typechecks the result with ctxExpr.
func convnop(n ir.Node, t *types.Type) ir.Node {
	if types.Identical(n.Type(), t) {
		return n
	}
	n = ir.Nod(ir.OCONVNOP, n, nil)
	n.SetType(t)
	n = typecheck(n, ctxExpr)
	return n
}

// byteindex converts n, which is byte-sized, to an int used to index into an array.
// We cannot use conv, because we allow converting bool to int here,
// which is forbidden in user code.
func byteindex(n ir.Node) ir.Node {
	// We cannot convert from bool to int directly.
	// While converting from int8 to int is possible, it would yield
	// the wrong result for negative values.
	// Reinterpreting the value as an unsigned byte solves both cases.
	if !types.Identical(n.Type(), types.Types[types.TUINT8]) {
		n = ir.Nod(ir.OCONV, n, nil)
		n.SetType(types.Types[types.TUINT8])
		n.SetTypecheck(1)
	}
	n = ir.Nod(ir.OCONV, n, nil)
	n.SetType(types.Types[types.TINT])
	n.SetTypecheck(1)
	return n
}

func chanfn(name string, n int, t *types.Type) ir.Node {
	if !t.IsChan() {
		base.Fatalf("chanfn %v", t)
	}
	fn := syslook(name)
	switch n {
	default:
		base.Fatalf("chanfn %d", n)
	case 1:
		fn = substArgTypes(fn, t.Elem())
	case 2:
		fn = substArgTypes(fn, t.Elem(), t.Elem())
	}
	return fn
}

func mapfn(name string, t *types.Type) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	fn := syslook(name)
	fn = substArgTypes(fn, t.Key(), t.Elem(), t.Key(), t.Elem())
	return fn
}

func mapfndel(name string, t *types.Type) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	fn := syslook(name)
	fn = substArgTypes(fn, t.Key(), t.Elem(), t.Key())
	return fn
}

const (
	mapslow = iota
	mapfast32
	mapfast32ptr
	mapfast64
	mapfast64ptr
	mapfaststr
	nmapfast
)

type mapnames [nmapfast]string

func mkmapnames(base string, ptr string) mapnames {
	return mapnames{base, base + "_fast32", base + "_fast32" + ptr, base + "_fast64", base + "_fast64" + ptr, base + "_faststr"}
}

var mapaccess1 = mkmapnames("mapaccess1", "")
var mapaccess2 = mkmapnames("mapaccess2", "")
var mapassign = mkmapnames("mapassign", "ptr")
var mapdelete = mkmapnames("mapdelete", "")

func mapfast(t *types.Type) int {
	// Check runtime/map.go:maxElemSize before changing.
	if t.Elem().Width > 128 {
		return mapslow
	}
	switch algtype(t.Key()) {
	case AMEM32:
		if !t.Key().HasPointers() {
			return mapfast32
		}
		if Widthptr == 4 {
			return mapfast32ptr
		}
		base.Fatalf("small pointer %v", t.Key())
	case AMEM64:
		if !t.Key().HasPointers() {
			return mapfast64
		}
		if Widthptr == 8 {
			return mapfast64ptr
		}
		// Two-word object, at least one of which is a pointer.
		// Use the slow path.
	case ASTRING:
		return mapfaststr
	}
	return mapslow
}

func writebarrierfn(name string, l *types.Type, r *types.Type) ir.Node {
	fn := syslook(name)
	fn = substArgTypes(fn, l, r)
	return fn
}

func addstr(n ir.Node, init *ir.Nodes) ir.Node {
	// order.expr rewrote OADDSTR to have a list of strings.
	c := n.List().Len()

	if c < 2 {
		base.Fatalf("addstr count %d too small", c)
	}

	buf := nodnil()
	if n.Esc() == EscNone {
		sz := int64(0)
		for _, n1 := range n.List().Slice() {
			if n1.Op() == ir.OLITERAL {
				sz += int64(len(ir.StringVal(n1)))
			}
		}

		// Don't allocate the buffer if the result won't fit.
		if sz < tmpstringbufsize {
			// Create temporary buffer for result string on stack.
			t := types.NewArray(types.Types[types.TUINT8], tmpstringbufsize)
			buf = nodAddr(temp(t))
		}
	}

	// build list of string arguments
	args := []ir.Node{buf}
	for _, n2 := range n.List().Slice() {
		args = append(args, conv(n2, types.Types[types.TSTRING]))
	}

	var fn string
	if c <= 5 {
		// small numbers of strings use direct runtime helpers.
		// note: order.expr knows this cutoff too.
		fn = fmt.Sprintf("concatstring%d", c)
	} else {
		// large numbers of strings are passed to the runtime as a slice.
		fn = "concatstrings"

		t := types.NewSlice(types.Types[types.TSTRING])
		slice := ir.Nod(ir.OCOMPLIT, nil, ir.TypeNode(t))
		if prealloc[n] != nil {
			prealloc[slice] = prealloc[n]
		}
		slice.PtrList().Set(args[1:]) // skip buf arg
		args = []ir.Node{buf, slice}
		slice.SetEsc(EscNone)
	}

	cat := syslook(fn)
	r := ir.Nod(ir.OCALL, cat, nil)
	r.PtrList().Set(args)
	r1 := typecheck(r, ctxExpr)
	r1 = walkexpr(r1, init)
	r1.SetType(n.Type())

	return r1
}

func walkAppendArgs(n ir.Node, init *ir.Nodes) {
	walkexprlistsafe(n.List().Slice(), init)

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	ls := n.List().Slice()
	for i1, n1 := range ls {
		ls[i1] = cheapexpr(n1, init)
	}
}

// expand append(l1, l2...) to
//   init {
//     s := l1
//     n := len(s) + len(l2)
//     // Compare as uint so growslice can panic on overflow.
//     if uint(n) > uint(cap(s)) {
//       s = growslice(s, n)
//     }
//     s = s[:n]
//     memmove(&s[len(l1)], &l2[0], len(l2)*sizeof(T))
//   }
//   s
//
// l2 is allowed to be a string.
func appendslice(n ir.Node, init *ir.Nodes) ir.Node {
	walkAppendArgs(n, init)

	l1 := n.List().First()
	l2 := n.List().Second()
	l2 = cheapexpr(l2, init)
	n.List().SetSecond(l2)

	var nodes ir.Nodes

	// var s []T
	s := temp(l1.Type())
	nodes.Append(ir.Nod(ir.OAS, s, l1)) // s = l1

	elemtype := s.Type().Elem()

	// n := len(s) + len(l2)
	nn := temp(types.Types[types.TINT])
	nodes.Append(ir.Nod(ir.OAS, nn, ir.Nod(ir.OADD, ir.Nod(ir.OLEN, s, nil), ir.Nod(ir.OLEN, l2, nil))))

	// if uint(n) > uint(cap(s))
	nif := ir.Nod(ir.OIF, nil, nil)
	nuint := conv(nn, types.Types[types.TUINT])
	scapuint := conv(ir.Nod(ir.OCAP, s, nil), types.Types[types.TUINT])
	nif.SetLeft(ir.Nod(ir.OGT, nuint, scapuint))

	// instantiate growslice(typ *type, []any, int) []any
	fn := syslook("growslice")
	fn = substArgTypes(fn, elemtype, elemtype)

	// s = growslice(T, s, n)
	nif.PtrBody().Set1(ir.Nod(ir.OAS, s, mkcall1(fn, s.Type(), nif.PtrInit(), typename(elemtype), s, nn)))
	nodes.Append(nif)

	// s = s[:n]
	nt := ir.Nod(ir.OSLICE, s, nil)
	nt.SetSliceBounds(nil, nn, nil)
	nt.SetBounded(true)
	nodes.Append(ir.Nod(ir.OAS, s, nt))

	var ncopy ir.Node
	if elemtype.HasPointers() {
		// copy(s[len(l1):], l2)
		slice := ir.Nod(ir.OSLICE, s, nil)
		slice.SetType(s.Type())
		slice.SetSliceBounds(ir.Nod(ir.OLEN, l1, nil), nil, nil)

		Curfn.SetWBPos(n.Pos())

		// instantiate typedslicecopy(typ *type, dstPtr *any, dstLen int, srcPtr *any, srcLen int) int
		fn := syslook("typedslicecopy")
		fn = substArgTypes(fn, l1.Type().Elem(), l2.Type().Elem())
		ptr1, len1 := backingArrayPtrLen(cheapexpr(slice, &nodes))
		ptr2, len2 := backingArrayPtrLen(l2)
		ncopy = mkcall1(fn, types.Types[types.TINT], &nodes, typename(elemtype), ptr1, len1, ptr2, len2)
	} else if instrumenting && !base.Flag.CompilingRuntime {
		// rely on runtime to instrument:
		//  copy(s[len(l1):], l2)
		// l2 can be a slice or string.
		slice := ir.Nod(ir.OSLICE, s, nil)
		slice.SetType(s.Type())
		slice.SetSliceBounds(ir.Nod(ir.OLEN, l1, nil), nil, nil)

		ptr1, len1 := backingArrayPtrLen(cheapexpr(slice, &nodes))
		ptr2, len2 := backingArrayPtrLen(l2)

		fn := syslook("slicecopy")
		fn = substArgTypes(fn, ptr1.Type().Elem(), ptr2.Type().Elem())
		ncopy = mkcall1(fn, types.Types[types.TINT], &nodes, ptr1, len1, ptr2, len2, nodintconst(elemtype.Width))
	} else {
		// memmove(&s[len(l1)], &l2[0], len(l2)*sizeof(T))
		ix := ir.Nod(ir.OINDEX, s, ir.Nod(ir.OLEN, l1, nil))
		ix.SetBounded(true)
		addr := nodAddr(ix)

		sptr := ir.Nod(ir.OSPTR, l2, nil)

		nwid := cheapexpr(conv(ir.Nod(ir.OLEN, l2, nil), types.Types[types.TUINTPTR]), &nodes)
		nwid = ir.Nod(ir.OMUL, nwid, nodintconst(elemtype.Width))

		// instantiate func memmove(to *any, frm *any, length uintptr)
		fn := syslook("memmove")
		fn = substArgTypes(fn, elemtype, elemtype)
		ncopy = mkcall1(fn, nil, &nodes, addr, sptr, nwid)
	}
	ln := append(nodes.Slice(), ncopy)

	typecheckslice(ln, ctxStmt)
	walkstmtlist(ln)
	init.Append(ln...)
	return s
}

// isAppendOfMake reports whether n is of the form append(x , make([]T, y)...).
// isAppendOfMake assumes n has already been typechecked.
func isAppendOfMake(n ir.Node) bool {
	if base.Flag.N != 0 || instrumenting {
		return false
	}

	if n.Typecheck() == 0 {
		base.Fatalf("missing typecheck: %+v", n)
	}

	if n.Op() != ir.OAPPEND || !n.IsDDD() || n.List().Len() != 2 {
		return false
	}

	second := n.List().Second()
	if second.Op() != ir.OMAKESLICE || second.Right() != nil {
		return false
	}

	// y must be either an integer constant or the largest possible positive value
	// of variable y needs to fit into an uint.

	// typecheck made sure that constant arguments to make are not negative and fit into an int.

	// The care of overflow of the len argument to make will be handled by an explicit check of int(len) < 0 during runtime.
	y := second.Left()
	if !ir.IsConst(y, constant.Int) && y.Type().Size() > types.Types[types.TUINT].Size() {
		return false
	}

	return true
}

// extendslice rewrites append(l1, make([]T, l2)...) to
//   init {
//     if l2 >= 0 { // Empty if block here for more meaningful node.SetLikely(true)
//     } else {
//       panicmakeslicelen()
//     }
//     s := l1
//     n := len(s) + l2
//     // Compare n and s as uint so growslice can panic on overflow of len(s) + l2.
//     // cap is a positive int and n can become negative when len(s) + l2
//     // overflows int. Interpreting n when negative as uint makes it larger
//     // than cap(s). growslice will check the int n arg and panic if n is
//     // negative. This prevents the overflow from being undetected.
//     if uint(n) > uint(cap(s)) {
//       s = growslice(T, s, n)
//     }
//     s = s[:n]
//     lptr := &l1[0]
//     sptr := &s[0]
//     if lptr == sptr || !T.HasPointers() {
//       // growslice did not clear the whole underlying array (or did not get called)
//       hp := &s[len(l1)]
//       hn := l2 * sizeof(T)
//       memclr(hp, hn)
//     }
//   }
//   s
func extendslice(n ir.Node, init *ir.Nodes) ir.Node {
	// isAppendOfMake made sure all possible positive values of l2 fit into an uint.
	// The case of l2 overflow when converting from e.g. uint to int is handled by an explicit
	// check of l2 < 0 at runtime which is generated below.
	l2 := conv(n.List().Second().Left(), types.Types[types.TINT])
	l2 = typecheck(l2, ctxExpr)
	n.List().SetSecond(l2) // walkAppendArgs expects l2 in n.List.Second().

	walkAppendArgs(n, init)

	l1 := n.List().First()
	l2 = n.List().Second() // re-read l2, as it may have been updated by walkAppendArgs

	var nodes []ir.Node

	// if l2 >= 0 (likely happens), do nothing
	nifneg := ir.Nod(ir.OIF, ir.Nod(ir.OGE, l2, nodintconst(0)), nil)
	nifneg.SetLikely(true)

	// else panicmakeslicelen()
	nifneg.PtrRlist().Set1(mkcall("panicmakeslicelen", nil, init))
	nodes = append(nodes, nifneg)

	// s := l1
	s := temp(l1.Type())
	nodes = append(nodes, ir.Nod(ir.OAS, s, l1))

	elemtype := s.Type().Elem()

	// n := len(s) + l2
	nn := temp(types.Types[types.TINT])
	nodes = append(nodes, ir.Nod(ir.OAS, nn, ir.Nod(ir.OADD, ir.Nod(ir.OLEN, s, nil), l2)))

	// if uint(n) > uint(cap(s))
	nuint := conv(nn, types.Types[types.TUINT])
	capuint := conv(ir.Nod(ir.OCAP, s, nil), types.Types[types.TUINT])
	nif := ir.Nod(ir.OIF, ir.Nod(ir.OGT, nuint, capuint), nil)

	// instantiate growslice(typ *type, old []any, newcap int) []any
	fn := syslook("growslice")
	fn = substArgTypes(fn, elemtype, elemtype)

	// s = growslice(T, s, n)
	nif.PtrBody().Set1(ir.Nod(ir.OAS, s, mkcall1(fn, s.Type(), nif.PtrInit(), typename(elemtype), s, nn)))
	nodes = append(nodes, nif)

	// s = s[:n]
	nt := ir.Nod(ir.OSLICE, s, nil)
	nt.SetSliceBounds(nil, nn, nil)
	nt.SetBounded(true)
	nodes = append(nodes, ir.Nod(ir.OAS, s, nt))

	// lptr := &l1[0]
	l1ptr := temp(l1.Type().Elem().PtrTo())
	tmp := ir.Nod(ir.OSPTR, l1, nil)
	nodes = append(nodes, ir.Nod(ir.OAS, l1ptr, tmp))

	// sptr := &s[0]
	sptr := temp(elemtype.PtrTo())
	tmp = ir.Nod(ir.OSPTR, s, nil)
	nodes = append(nodes, ir.Nod(ir.OAS, sptr, tmp))

	// hp := &s[len(l1)]
	ix := ir.Nod(ir.OINDEX, s, ir.Nod(ir.OLEN, l1, nil))
	ix.SetBounded(true)
	hp := convnop(nodAddr(ix), types.Types[types.TUNSAFEPTR])

	// hn := l2 * sizeof(elem(s))
	hn := conv(ir.Nod(ir.OMUL, l2, nodintconst(elemtype.Width)), types.Types[types.TUINTPTR])

	clrname := "memclrNoHeapPointers"
	hasPointers := elemtype.HasPointers()
	if hasPointers {
		clrname = "memclrHasPointers"
		Curfn.SetWBPos(n.Pos())
	}

	var clr ir.Nodes
	clrfn := mkcall(clrname, nil, &clr, hp, hn)
	clr.Append(clrfn)

	if hasPointers {
		// if l1ptr == sptr
		nifclr := ir.Nod(ir.OIF, ir.Nod(ir.OEQ, l1ptr, sptr), nil)
		nifclr.SetBody(clr)
		nodes = append(nodes, nifclr)
	} else {
		nodes = append(nodes, clr.Slice()...)
	}

	typecheckslice(nodes, ctxStmt)
	walkstmtlist(nodes)
	init.Append(nodes...)
	return s
}

// Rewrite append(src, x, y, z) so that any side effects in
// x, y, z (including runtime panics) are evaluated in
// initialization statements before the append.
// For normal code generation, stop there and leave the
// rest to cgen_append.
//
// For race detector, expand append(src, a [, b]* ) to
//
//   init {
//     s := src
//     const argc = len(args) - 1
//     if cap(s) - len(s) < argc {
//	    s = growslice(s, len(s)+argc)
//     }
//     n := len(s)
//     s = s[:n+argc]
//     s[n] = a
//     s[n+1] = b
//     ...
//   }
//   s
func walkappend(n ir.Node, init *ir.Nodes, dst ir.Node) ir.Node {
	if !samesafeexpr(dst, n.List().First()) {
		n.List().SetFirst(safeexpr(n.List().First(), init))
		n.List().SetFirst(walkexpr(n.List().First(), init))
	}
	walkexprlistsafe(n.List().Slice()[1:], init)

	nsrc := n.List().First()

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	// Using cheapexpr also makes sure that the evaluation
	// of all arguments (and especially any panics) happen
	// before we begin to modify the slice in a visible way.
	ls := n.List().Slice()[1:]
	for i, n := range ls {
		n = cheapexpr(n, init)
		if !types.Identical(n.Type(), nsrc.Type().Elem()) {
			n = assignconv(n, nsrc.Type().Elem(), "append")
			n = walkexpr(n, init)
		}
		ls[i] = n
	}

	argc := n.List().Len() - 1
	if argc < 1 {
		return nsrc
	}

	// General case, with no function calls left as arguments.
	// Leave for gen, except that instrumentation requires old form.
	if !instrumenting || base.Flag.CompilingRuntime {
		return n
	}

	var l []ir.Node

	ns := temp(nsrc.Type())
	l = append(l, ir.Nod(ir.OAS, ns, nsrc)) // s = src

	na := nodintconst(int64(argc))  // const argc
	nif := ir.Nod(ir.OIF, nil, nil) // if cap(s) - len(s) < argc
	nif.SetLeft(ir.Nod(ir.OLT, ir.Nod(ir.OSUB, ir.Nod(ir.OCAP, ns, nil), ir.Nod(ir.OLEN, ns, nil)), na))

	fn := syslook("growslice") //   growslice(<type>, old []T, mincap int) (ret []T)
	fn = substArgTypes(fn, ns.Type().Elem(), ns.Type().Elem())

	nif.PtrBody().Set1(ir.Nod(ir.OAS, ns,
		mkcall1(fn, ns.Type(), nif.PtrInit(), typename(ns.Type().Elem()), ns,
			ir.Nod(ir.OADD, ir.Nod(ir.OLEN, ns, nil), na))))

	l = append(l, nif)

	nn := temp(types.Types[types.TINT])
	l = append(l, ir.Nod(ir.OAS, nn, ir.Nod(ir.OLEN, ns, nil))) // n = len(s)

	slice := ir.Nod(ir.OSLICE, ns, nil) // ...s[:n+argc]
	slice.SetSliceBounds(nil, ir.Nod(ir.OADD, nn, na), nil)
	slice.SetBounded(true)
	l = append(l, ir.Nod(ir.OAS, ns, slice)) // s = s[:n+argc]

	ls = n.List().Slice()[1:]
	for i, n := range ls {
		ix := ir.Nod(ir.OINDEX, ns, nn) // s[n] ...
		ix.SetBounded(true)
		l = append(l, ir.Nod(ir.OAS, ix, n)) // s[n] = arg
		if i+1 < len(ls) {
			l = append(l, ir.Nod(ir.OAS, nn, ir.Nod(ir.OADD, nn, nodintconst(1)))) // n = n + 1
		}
	}

	typecheckslice(l, ctxStmt)
	walkstmtlist(l)
	init.Append(l...)
	return ns
}

// Lower copy(a, b) to a memmove call or a runtime call.
//
// init {
//   n := len(a)
//   if n > len(b) { n = len(b) }
//   if a.ptr != b.ptr { memmove(a.ptr, b.ptr, n*sizeof(elem(a))) }
// }
// n;
//
// Also works if b is a string.
//
func copyany(n ir.Node, init *ir.Nodes, runtimecall bool) ir.Node {
	if n.Left().Type().Elem().HasPointers() {
		Curfn.SetWBPos(n.Pos())
		fn := writebarrierfn("typedslicecopy", n.Left().Type().Elem(), n.Right().Type().Elem())
		n.SetLeft(cheapexpr(n.Left(), init))
		ptrL, lenL := backingArrayPtrLen(n.Left())
		n.SetRight(cheapexpr(n.Right(), init))
		ptrR, lenR := backingArrayPtrLen(n.Right())
		return mkcall1(fn, n.Type(), init, typename(n.Left().Type().Elem()), ptrL, lenL, ptrR, lenR)
	}

	if runtimecall {
		// rely on runtime to instrument:
		//  copy(n.Left, n.Right)
		// n.Right can be a slice or string.

		n.SetLeft(cheapexpr(n.Left(), init))
		ptrL, lenL := backingArrayPtrLen(n.Left())
		n.SetRight(cheapexpr(n.Right(), init))
		ptrR, lenR := backingArrayPtrLen(n.Right())

		fn := syslook("slicecopy")
		fn = substArgTypes(fn, ptrL.Type().Elem(), ptrR.Type().Elem())

		return mkcall1(fn, n.Type(), init, ptrL, lenL, ptrR, lenR, nodintconst(n.Left().Type().Elem().Width))
	}

	n.SetLeft(walkexpr(n.Left(), init))
	n.SetRight(walkexpr(n.Right(), init))
	nl := temp(n.Left().Type())
	nr := temp(n.Right().Type())
	var l []ir.Node
	l = append(l, ir.Nod(ir.OAS, nl, n.Left()))
	l = append(l, ir.Nod(ir.OAS, nr, n.Right()))

	nfrm := ir.Nod(ir.OSPTR, nr, nil)
	nto := ir.Nod(ir.OSPTR, nl, nil)

	nlen := temp(types.Types[types.TINT])

	// n = len(to)
	l = append(l, ir.Nod(ir.OAS, nlen, ir.Nod(ir.OLEN, nl, nil)))

	// if n > len(frm) { n = len(frm) }
	nif := ir.Nod(ir.OIF, nil, nil)

	nif.SetLeft(ir.Nod(ir.OGT, nlen, ir.Nod(ir.OLEN, nr, nil)))
	nif.PtrBody().Append(ir.Nod(ir.OAS, nlen, ir.Nod(ir.OLEN, nr, nil)))
	l = append(l, nif)

	// if to.ptr != frm.ptr { memmove( ... ) }
	ne := ir.Nod(ir.OIF, ir.Nod(ir.ONE, nto, nfrm), nil)
	ne.SetLikely(true)
	l = append(l, ne)

	fn := syslook("memmove")
	fn = substArgTypes(fn, nl.Type().Elem(), nl.Type().Elem())
	nwid := ir.Node(temp(types.Types[types.TUINTPTR]))
	setwid := ir.Nod(ir.OAS, nwid, conv(nlen, types.Types[types.TUINTPTR]))
	ne.PtrBody().Append(setwid)
	nwid = ir.Nod(ir.OMUL, nwid, nodintconst(nl.Type().Elem().Width))
	call := mkcall1(fn, nil, init, nto, nfrm, nwid)
	ne.PtrBody().Append(call)

	typecheckslice(l, ctxStmt)
	walkstmtlist(l)
	init.Append(l...)
	return nlen
}

func eqfor(t *types.Type) (n ir.Node, needsize bool) {
	// Should only arrive here with large memory or
	// a struct/array containing a non-memory field/element.
	// Small memory is handled inline, and single non-memory
	// is handled by walkcompare.
	switch a, _ := algtype1(t); a {
	case AMEM:
		n := syslook("memequal")
		n = substArgTypes(n, t, t)
		return n, true
	case ASPECIAL:
		sym := typesymprefix(".eq", t)
		n := NewName(sym)
		setNodeNameFunc(n)
		n.SetType(functype(nil, []*ir.Field{
			anonfield(types.NewPtr(t)),
			anonfield(types.NewPtr(t)),
		}, []*ir.Field{
			anonfield(types.Types[types.TBOOL]),
		}))
		return n, false
	}
	base.Fatalf("eqfor %v", t)
	return nil, false
}

// The result of walkcompare MUST be assigned back to n, e.g.
// 	n.Left = walkcompare(n.Left, init)
func walkcompare(n ir.Node, init *ir.Nodes) ir.Node {
	if n.Left().Type().IsInterface() && n.Right().Type().IsInterface() && n.Left().Op() != ir.ONIL && n.Right().Op() != ir.ONIL {
		return walkcompareInterface(n, init)
	}

	if n.Left().Type().IsString() && n.Right().Type().IsString() {
		return walkcompareString(n, init)
	}

	n.SetLeft(walkexpr(n.Left(), init))
	n.SetRight(walkexpr(n.Right(), init))

	// Given mixed interface/concrete comparison,
	// rewrite into types-equal && data-equal.
	// This is efficient, avoids allocations, and avoids runtime calls.
	if n.Left().Type().IsInterface() != n.Right().Type().IsInterface() {
		// Preserve side-effects in case of short-circuiting; see #32187.
		l := cheapexpr(n.Left(), init)
		r := cheapexpr(n.Right(), init)
		// Swap so that l is the interface value and r is the concrete value.
		if n.Right().Type().IsInterface() {
			l, r = r, l
		}

		// Handle both == and !=.
		eq := n.Op()
		andor := ir.OOROR
		if eq == ir.OEQ {
			andor = ir.OANDAND
		}
		// Check for types equal.
		// For empty interface, this is:
		//   l.tab == type(r)
		// For non-empty interface, this is:
		//   l.tab != nil && l.tab._type == type(r)
		var eqtype ir.Node
		tab := ir.Nod(ir.OITAB, l, nil)
		rtyp := typename(r.Type())
		if l.Type().IsEmptyInterface() {
			tab.SetType(types.NewPtr(types.Types[types.TUINT8]))
			tab.SetTypecheck(1)
			eqtype = ir.NewBinaryExpr(base.Pos, eq, tab, rtyp)
		} else {
			nonnil := ir.NewBinaryExpr(base.Pos, brcom(eq), nodnil(), tab)
			match := ir.NewBinaryExpr(base.Pos, eq, itabType(tab), rtyp)
			eqtype = ir.NewLogicalExpr(base.Pos, andor, nonnil, match)
		}
		// Check for data equal.
		eqdata := ir.NewBinaryExpr(base.Pos, eq, ifaceData(n.Pos(), l, r.Type()), r)
		// Put it all together.
		expr := ir.NewLogicalExpr(base.Pos, andor, eqtype, eqdata)
		n = finishcompare(n, expr, init)
		return n
	}

	// Must be comparison of array or struct.
	// Otherwise back end handles it.
	// While we're here, decide whether to
	// inline or call an eq alg.
	t := n.Left().Type()
	var inline bool

	maxcmpsize := int64(4)
	unalignedLoad := canMergeLoads()
	if unalignedLoad {
		// Keep this low enough to generate less code than a function call.
		maxcmpsize = 2 * int64(thearch.LinkArch.RegSize)
	}

	switch t.Kind() {
	default:
		if base.Debug.Libfuzzer != 0 && t.IsInteger() {
			n.SetLeft(cheapexpr(n.Left(), init))
			n.SetRight(cheapexpr(n.Right(), init))

			// If exactly one comparison operand is
			// constant, invoke the constcmp functions
			// instead, and arrange for the constant
			// operand to be the first argument.
			l, r := n.Left(), n.Right()
			if r.Op() == ir.OLITERAL {
				l, r = r, l
			}
			constcmp := l.Op() == ir.OLITERAL && r.Op() != ir.OLITERAL

			var fn string
			var paramType *types.Type
			switch t.Size() {
			case 1:
				fn = "libfuzzerTraceCmp1"
				if constcmp {
					fn = "libfuzzerTraceConstCmp1"
				}
				paramType = types.Types[types.TUINT8]
			case 2:
				fn = "libfuzzerTraceCmp2"
				if constcmp {
					fn = "libfuzzerTraceConstCmp2"
				}
				paramType = types.Types[types.TUINT16]
			case 4:
				fn = "libfuzzerTraceCmp4"
				if constcmp {
					fn = "libfuzzerTraceConstCmp4"
				}
				paramType = types.Types[types.TUINT32]
			case 8:
				fn = "libfuzzerTraceCmp8"
				if constcmp {
					fn = "libfuzzerTraceConstCmp8"
				}
				paramType = types.Types[types.TUINT64]
			default:
				base.Fatalf("unexpected integer size %d for %v", t.Size(), t)
			}
			init.Append(mkcall(fn, nil, init, tracecmpArg(l, paramType, init), tracecmpArg(r, paramType, init)))
		}
		return n
	case types.TARRAY:
		// We can compare several elements at once with 2/4/8 byte integer compares
		inline = t.NumElem() <= 1 || (issimple[t.Elem().Kind()] && (t.NumElem() <= 4 || t.Elem().Width*t.NumElem() <= maxcmpsize))
	case types.TSTRUCT:
		inline = t.NumComponents(types.IgnoreBlankFields) <= 4
	}

	cmpl := n.Left()
	for cmpl != nil && cmpl.Op() == ir.OCONVNOP {
		cmpl = cmpl.Left()
	}
	cmpr := n.Right()
	for cmpr != nil && cmpr.Op() == ir.OCONVNOP {
		cmpr = cmpr.Left()
	}

	// Chose not to inline. Call equality function directly.
	if !inline {
		// eq algs take pointers; cmpl and cmpr must be addressable
		if !islvalue(cmpl) || !islvalue(cmpr) {
			base.Fatalf("arguments of comparison must be lvalues - %v %v", cmpl, cmpr)
		}

		fn, needsize := eqfor(t)
		call := ir.Nod(ir.OCALL, fn, nil)
		call.PtrList().Append(nodAddr(cmpl))
		call.PtrList().Append(nodAddr(cmpr))
		if needsize {
			call.PtrList().Append(nodintconst(t.Width))
		}
		res := ir.Node(call)
		if n.Op() != ir.OEQ {
			res = ir.Nod(ir.ONOT, res, nil)
		}
		n = finishcompare(n, res, init)
		return n
	}

	// inline: build boolean expression comparing element by element
	andor := ir.OANDAND
	if n.Op() == ir.ONE {
		andor = ir.OOROR
	}
	var expr ir.Node
	compare := func(el, er ir.Node) {
		a := ir.NewBinaryExpr(base.Pos, n.Op(), el, er)
		if expr == nil {
			expr = a
		} else {
			expr = ir.NewLogicalExpr(base.Pos, andor, expr, a)
		}
	}
	cmpl = safeexpr(cmpl, init)
	cmpr = safeexpr(cmpr, init)
	if t.IsStruct() {
		for _, f := range t.Fields().Slice() {
			sym := f.Sym
			if sym.IsBlank() {
				continue
			}
			compare(
				nodSym(ir.OXDOT, cmpl, sym),
				nodSym(ir.OXDOT, cmpr, sym),
			)
		}
	} else {
		step := int64(1)
		remains := t.NumElem() * t.Elem().Width
		combine64bit := unalignedLoad && Widthreg == 8 && t.Elem().Width <= 4 && t.Elem().IsInteger()
		combine32bit := unalignedLoad && t.Elem().Width <= 2 && t.Elem().IsInteger()
		combine16bit := unalignedLoad && t.Elem().Width == 1 && t.Elem().IsInteger()
		for i := int64(0); remains > 0; {
			var convType *types.Type
			switch {
			case remains >= 8 && combine64bit:
				convType = types.Types[types.TINT64]
				step = 8 / t.Elem().Width
			case remains >= 4 && combine32bit:
				convType = types.Types[types.TUINT32]
				step = 4 / t.Elem().Width
			case remains >= 2 && combine16bit:
				convType = types.Types[types.TUINT16]
				step = 2 / t.Elem().Width
			default:
				step = 1
			}
			if step == 1 {
				compare(
					ir.Nod(ir.OINDEX, cmpl, nodintconst(i)),
					ir.Nod(ir.OINDEX, cmpr, nodintconst(i)),
				)
				i++
				remains -= t.Elem().Width
			} else {
				elemType := t.Elem().ToUnsigned()
				cmplw := ir.Node(ir.Nod(ir.OINDEX, cmpl, nodintconst(i)))
				cmplw = conv(cmplw, elemType) // convert to unsigned
				cmplw = conv(cmplw, convType) // widen
				cmprw := ir.Node(ir.Nod(ir.OINDEX, cmpr, nodintconst(i)))
				cmprw = conv(cmprw, elemType)
				cmprw = conv(cmprw, convType)
				// For code like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will generate a single large load.
				for offset := int64(1); offset < step; offset++ {
					lb := ir.Node(ir.Nod(ir.OINDEX, cmpl, nodintconst(i+offset)))
					lb = conv(lb, elemType)
					lb = conv(lb, convType)
					lb = ir.Nod(ir.OLSH, lb, nodintconst(8*t.Elem().Width*offset))
					cmplw = ir.Nod(ir.OOR, cmplw, lb)
					rb := ir.Node(ir.Nod(ir.OINDEX, cmpr, nodintconst(i+offset)))
					rb = conv(rb, elemType)
					rb = conv(rb, convType)
					rb = ir.Nod(ir.OLSH, rb, nodintconst(8*t.Elem().Width*offset))
					cmprw = ir.Nod(ir.OOR, cmprw, rb)
				}
				compare(cmplw, cmprw)
				i += step
				remains -= step * t.Elem().Width
			}
		}
	}
	if expr == nil {
		expr = nodbool(n.Op() == ir.OEQ)
		// We still need to use cmpl and cmpr, in case they contain
		// an expression which might panic. See issue 23837.
		t := temp(cmpl.Type())
		a1 := typecheck(ir.Nod(ir.OAS, t, cmpl), ctxStmt)
		a2 := typecheck(ir.Nod(ir.OAS, t, cmpr), ctxStmt)
		init.Append(a1, a2)
	}
	n = finishcompare(n, expr, init)
	return n
}

func tracecmpArg(n ir.Node, t *types.Type, init *ir.Nodes) ir.Node {
	// Ugly hack to avoid "constant -1 overflows uintptr" errors, etc.
	if n.Op() == ir.OLITERAL && n.Type().IsSigned() && ir.Int64Val(n) < 0 {
		n = copyexpr(n, n.Type(), init)
	}

	return conv(n, t)
}

func walkcompareInterface(n ir.Node, init *ir.Nodes) ir.Node {
	n.SetRight(cheapexpr(n.Right(), init))
	n.SetLeft(cheapexpr(n.Left(), init))
	eqtab, eqdata := eqinterface(n.Left(), n.Right())
	var cmp ir.Node
	if n.Op() == ir.OEQ {
		cmp = ir.Nod(ir.OANDAND, eqtab, eqdata)
	} else {
		eqtab.SetOp(ir.ONE)
		cmp = ir.Nod(ir.OOROR, eqtab, ir.Nod(ir.ONOT, eqdata, nil))
	}
	return finishcompare(n, cmp, init)
}

func walkcompareString(n ir.Node, init *ir.Nodes) ir.Node {
	// Rewrite comparisons to short constant strings as length+byte-wise comparisons.
	var cs, ncs ir.Node // const string, non-const string
	switch {
	case ir.IsConst(n.Left(), constant.String) && ir.IsConst(n.Right(), constant.String):
		// ignore; will be constant evaluated
	case ir.IsConst(n.Left(), constant.String):
		cs = n.Left()
		ncs = n.Right()
	case ir.IsConst(n.Right(), constant.String):
		cs = n.Right()
		ncs = n.Left()
	}
	if cs != nil {
		cmp := n.Op()
		// Our comparison below assumes that the non-constant string
		// is on the left hand side, so rewrite "" cmp x to x cmp "".
		// See issue 24817.
		if ir.IsConst(n.Left(), constant.String) {
			cmp = brrev(cmp)
		}

		// maxRewriteLen was chosen empirically.
		// It is the value that minimizes cmd/go file size
		// across most architectures.
		// See the commit description for CL 26758 for details.
		maxRewriteLen := 6
		// Some architectures can load unaligned byte sequence as 1 word.
		// So we can cover longer strings with the same amount of code.
		canCombineLoads := canMergeLoads()
		combine64bit := false
		if canCombineLoads {
			// Keep this low enough to generate less code than a function call.
			maxRewriteLen = 2 * thearch.LinkArch.RegSize
			combine64bit = thearch.LinkArch.RegSize >= 8
		}

		var and ir.Op
		switch cmp {
		case ir.OEQ:
			and = ir.OANDAND
		case ir.ONE:
			and = ir.OOROR
		default:
			// Don't do byte-wise comparisons for <, <=, etc.
			// They're fairly complicated.
			// Length-only checks are ok, though.
			maxRewriteLen = 0
		}
		if s := ir.StringVal(cs); len(s) <= maxRewriteLen {
			if len(s) > 0 {
				ncs = safeexpr(ncs, init)
			}
			r := ir.Node(ir.NewBinaryExpr(base.Pos, cmp, ir.Nod(ir.OLEN, ncs, nil), nodintconst(int64(len(s)))))
			remains := len(s)
			for i := 0; remains > 0; {
				if remains == 1 || !canCombineLoads {
					cb := nodintconst(int64(s[i]))
					ncb := ir.Nod(ir.OINDEX, ncs, nodintconst(int64(i)))
					r = ir.NewLogicalExpr(base.Pos, and, r, ir.NewBinaryExpr(base.Pos, cmp, ncb, cb))
					remains--
					i++
					continue
				}
				var step int
				var convType *types.Type
				switch {
				case remains >= 8 && combine64bit:
					convType = types.Types[types.TINT64]
					step = 8
				case remains >= 4:
					convType = types.Types[types.TUINT32]
					step = 4
				case remains >= 2:
					convType = types.Types[types.TUINT16]
					step = 2
				}
				ncsubstr := conv(ir.Nod(ir.OINDEX, ncs, nodintconst(int64(i))), convType)
				csubstr := int64(s[i])
				// Calculate large constant from bytes as sequence of shifts and ors.
				// Like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will combine this into a single large load.
				for offset := 1; offset < step; offset++ {
					b := conv(ir.Nod(ir.OINDEX, ncs, nodintconst(int64(i+offset))), convType)
					b = ir.Nod(ir.OLSH, b, nodintconst(int64(8*offset)))
					ncsubstr = ir.Nod(ir.OOR, ncsubstr, b)
					csubstr |= int64(s[i+offset]) << uint8(8*offset)
				}
				csubstrPart := nodintconst(csubstr)
				// Compare "step" bytes as once
				r = ir.NewLogicalExpr(base.Pos, and, r, ir.NewBinaryExpr(base.Pos, cmp, csubstrPart, ncsubstr))
				remains -= step
				i += step
			}
			return finishcompare(n, r, init)
		}
	}

	var r ir.Node
	if n.Op() == ir.OEQ || n.Op() == ir.ONE {
		// prepare for rewrite below
		n.SetLeft(cheapexpr(n.Left(), init))
		n.SetRight(cheapexpr(n.Right(), init))
		eqlen, eqmem := eqstring(n.Left(), n.Right())
		// quick check of len before full compare for == or !=.
		// memequal then tests equality up to length len.
		if n.Op() == ir.OEQ {
			// len(left) == len(right) && memequal(left, right, len)
			r = ir.Nod(ir.OANDAND, eqlen, eqmem)
		} else {
			// len(left) != len(right) || !memequal(left, right, len)
			eqlen.SetOp(ir.ONE)
			r = ir.Nod(ir.OOROR, eqlen, ir.Nod(ir.ONOT, eqmem, nil))
		}
	} else {
		// sys_cmpstring(s1, s2) :: 0
		r = mkcall("cmpstring", types.Types[types.TINT], init, conv(n.Left(), types.Types[types.TSTRING]), conv(n.Right(), types.Types[types.TSTRING]))
		r = ir.NewBinaryExpr(base.Pos, n.Op(), r, nodintconst(0))
	}

	return finishcompare(n, r, init)
}

// The result of finishcompare MUST be assigned back to n, e.g.
// 	n.Left = finishcompare(n.Left, x, r, init)
func finishcompare(n, r ir.Node, init *ir.Nodes) ir.Node {
	r = typecheck(r, ctxExpr)
	r = conv(r, n.Type())
	r = walkexpr(r, init)
	return r
}

// return 1 if integer n must be in range [0, max), 0 otherwise
func bounded(n ir.Node, max int64) bool {
	if n.Type() == nil || !n.Type().IsInteger() {
		return false
	}

	sign := n.Type().IsSigned()
	bits := int32(8 * n.Type().Width)

	if smallintconst(n) {
		v := ir.Int64Val(n)
		return 0 <= v && v < max
	}

	switch n.Op() {
	case ir.OAND, ir.OANDNOT:
		v := int64(-1)
		switch {
		case smallintconst(n.Left()):
			v = ir.Int64Val(n.Left())
		case smallintconst(n.Right()):
			v = ir.Int64Val(n.Right())
			if n.Op() == ir.OANDNOT {
				v = ^v
				if !sign {
					v &= 1<<uint(bits) - 1
				}
			}
		}
		if 0 <= v && v < max {
			return true
		}

	case ir.OMOD:
		if !sign && smallintconst(n.Right()) {
			v := ir.Int64Val(n.Right())
			if 0 <= v && v <= max {
				return true
			}
		}

	case ir.ODIV:
		if !sign && smallintconst(n.Right()) {
			v := ir.Int64Val(n.Right())
			for bits > 0 && v >= 2 {
				bits--
				v >>= 1
			}
		}

	case ir.ORSH:
		if !sign && smallintconst(n.Right()) {
			v := ir.Int64Val(n.Right())
			if v > int64(bits) {
				return true
			}
			bits -= int32(v)
		}
	}

	if !sign && bits <= 62 && 1<<uint(bits) <= max {
		return true
	}

	return false
}

// usemethod checks interface method calls for uses of reflect.Type.Method.
func usemethod(n ir.Node) {
	t := n.Left().Type()

	// Looking for either of:
	//	Method(int) reflect.Method
	//	MethodByName(string) (reflect.Method, bool)
	//
	// TODO(crawshaw): improve precision of match by working out
	//                 how to check the method name.
	if n := t.NumParams(); n != 1 {
		return
	}
	if n := t.NumResults(); n != 1 && n != 2 {
		return
	}
	p0 := t.Params().Field(0)
	res0 := t.Results().Field(0)
	var res1 *types.Field
	if t.NumResults() == 2 {
		res1 = t.Results().Field(1)
	}

	if res1 == nil {
		if p0.Type.Kind() != types.TINT {
			return
		}
	} else {
		if !p0.Type.IsString() {
			return
		}
		if !res1.Type.IsBoolean() {
			return
		}
	}

	// Note: Don't rely on res0.Type.String() since its formatting depends on multiple factors
	//       (including global variables such as numImports - was issue #19028).
	// Also need to check for reflect package itself (see Issue #38515).
	if s := res0.Type.Sym(); s != nil && s.Name == "Method" && isReflectPkg(s.Pkg) {
		Curfn.SetReflectMethod(true)
		// The LSym is initialized at this point. We need to set the attribute on the LSym.
		Curfn.LSym.Set(obj.AttrReflectMethod, true)
	}
}

func usefield(n ir.Node) {
	if objabi.Fieldtrack_enabled == 0 {
		return
	}

	switch n.Op() {
	default:
		base.Fatalf("usefield %v", n.Op())

	case ir.ODOT, ir.ODOTPTR:
		break
	}
	if n.Sym() == nil {
		// No field name.  This DOTPTR was built by the compiler for access
		// to runtime data structures.  Ignore.
		return
	}

	t := n.Left().Type()
	if t.IsPtr() {
		t = t.Elem()
	}
	field := n.(*ir.SelectorExpr).Selection
	if field == nil {
		base.Fatalf("usefield %v %v without paramfld", n.Left().Type(), n.Sym())
	}
	if field.Sym != n.Sym() || field.Offset != n.Offset() {
		base.Fatalf("field inconsistency: %v,%v != %v,%v", field.Sym, field.Offset, n.Sym(), n.Offset())
	}
	if !strings.Contains(field.Note, "go:\"track\"") {
		return
	}

	outer := n.Left().Type()
	if outer.IsPtr() {
		outer = outer.Elem()
	}
	if outer.Sym() == nil {
		base.Errorf("tracked field must be in named struct type")
	}
	if !types.IsExported(field.Sym.Name) {
		base.Errorf("tracked field must be exported (upper case)")
	}

	sym := tracksym(outer, field)
	if Curfn.FieldTrack == nil {
		Curfn.FieldTrack = make(map[*types.Sym]struct{})
	}
	Curfn.FieldTrack[sym] = struct{}{}
}

// hasSideEffects reports whether n contains any operations that could have observable side effects.
func hasSideEffects(n ir.Node) bool {
	return ir.Find(n, func(n ir.Node) bool {
		switch n.Op() {
		// Assume side effects unless we know otherwise.
		default:
			return true

		// No side effects here (arguments are checked separately).
		case ir.ONAME,
			ir.ONONAME,
			ir.OTYPE,
			ir.OPACK,
			ir.OLITERAL,
			ir.ONIL,
			ir.OADD,
			ir.OSUB,
			ir.OOR,
			ir.OXOR,
			ir.OADDSTR,
			ir.OADDR,
			ir.OANDAND,
			ir.OBYTES2STR,
			ir.ORUNES2STR,
			ir.OSTR2BYTES,
			ir.OSTR2RUNES,
			ir.OCAP,
			ir.OCOMPLIT,
			ir.OMAPLIT,
			ir.OSTRUCTLIT,
			ir.OARRAYLIT,
			ir.OSLICELIT,
			ir.OPTRLIT,
			ir.OCONV,
			ir.OCONVIFACE,
			ir.OCONVNOP,
			ir.ODOT,
			ir.OEQ,
			ir.ONE,
			ir.OLT,
			ir.OLE,
			ir.OGT,
			ir.OGE,
			ir.OKEY,
			ir.OSTRUCTKEY,
			ir.OLEN,
			ir.OMUL,
			ir.OLSH,
			ir.ORSH,
			ir.OAND,
			ir.OANDNOT,
			ir.ONEW,
			ir.ONOT,
			ir.OBITNOT,
			ir.OPLUS,
			ir.ONEG,
			ir.OOROR,
			ir.OPAREN,
			ir.ORUNESTR,
			ir.OREAL,
			ir.OIMAG,
			ir.OCOMPLEX:
			return false

		// Only possible side effect is division by zero.
		case ir.ODIV, ir.OMOD:
			if n.Right().Op() != ir.OLITERAL || constant.Sign(n.Right().Val()) == 0 {
				return true
			}

		// Only possible side effect is panic on invalid size,
		// but many makechan and makemap use size zero, which is definitely OK.
		case ir.OMAKECHAN, ir.OMAKEMAP:
			if !ir.IsConst(n.Left(), constant.Int) || constant.Sign(n.Left().Val()) != 0 {
				return true
			}

		// Only possible side effect is panic on invalid size.
		// TODO(rsc): Merge with previous case (probably breaks toolstash -cmp).
		case ir.OMAKESLICE, ir.OMAKESLICECOPY:
			return true
		}
		return false
	})
}

// Rewrite
//	go builtin(x, y, z)
// into
//	go func(a1, a2, a3) {
//		builtin(a1, a2, a3)
//	}(x, y, z)
// for print, println, and delete.
//
// Rewrite
//	go f(x, y, uintptr(unsafe.Pointer(z)))
// into
//	go func(a1, a2, a3) {
//		builtin(a1, a2, uintptr(a3))
//	}(x, y, unsafe.Pointer(z))
// for function contains unsafe-uintptr arguments.

var wrapCall_prgen int

// The result of wrapCall MUST be assigned back to n, e.g.
// 	n.Left = wrapCall(n.Left, init)
func wrapCall(n ir.Node, init *ir.Nodes) ir.Node {
	if n.Init().Len() != 0 {
		walkstmtlist(n.Init().Slice())
		init.AppendNodes(n.PtrInit())
	}

	isBuiltinCall := n.Op() != ir.OCALLFUNC && n.Op() != ir.OCALLMETH && n.Op() != ir.OCALLINTER

	// Turn f(a, b, []T{c, d, e}...) back into f(a, b, c, d, e).
	if !isBuiltinCall && n.IsDDD() {
		last := n.List().Len() - 1
		if va := n.List().Index(last); va.Op() == ir.OSLICELIT {
			n.PtrList().Set(append(n.List().Slice()[:last], va.List().Slice()...))
			n.SetIsDDD(false)
		}
	}

	// origArgs keeps track of what argument is uintptr-unsafe/unsafe-uintptr conversion.
	origArgs := make([]ir.Node, n.List().Len())
	var funcArgs []*ir.Field
	for i, arg := range n.List().Slice() {
		s := lookupN("a", i)
		if !isBuiltinCall && arg.Op() == ir.OCONVNOP && arg.Type().IsUintptr() && arg.Left().Type().IsUnsafePtr() {
			origArgs[i] = arg
			arg = arg.Left()
			n.List().SetIndex(i, arg)
		}
		funcArgs = append(funcArgs, symfield(s, arg.Type()))
	}
	t := ir.NewFuncType(base.Pos, nil, funcArgs, nil)

	wrapCall_prgen++
	sym := lookupN("wrap·", wrapCall_prgen)
	fn := dclfunc(sym, t)

	args := paramNnames(t.Type())
	for i, origArg := range origArgs {
		if origArg == nil {
			continue
		}
		args[i] = ir.NewConvExpr(base.Pos, origArg.Op(), origArg.Type(), args[i])
	}
	call := ir.NewCallExpr(base.Pos, n.Op(), n.Left(), args)
	if !isBuiltinCall {
		call.SetOp(ir.OCALL)
		call.SetIsDDD(n.IsDDD())
	}
	fn.PtrBody().Set1(call)

	funcbody()

	typecheckFunc(fn)
	typecheckslice(fn.Body().Slice(), ctxStmt)
	xtop = append(xtop, fn)

	call = ir.NewCallExpr(base.Pos, ir.OCALL, fn.Nname, n.List().Slice())
	return walkexpr(typecheck(call, ctxStmt), init)
}

// substArgTypes substitutes the given list of types for
// successive occurrences of the "any" placeholder in the
// type syntax expression n.Type.
// The result of substArgTypes MUST be assigned back to old, e.g.
// 	n.Left = substArgTypes(n.Left, t1, t2)
func substArgTypes(old ir.Node, types_ ...*types.Type) ir.Node {
	n := ir.Copy(old)

	for _, t := range types_ {
		dowidth(t)
	}
	n.SetType(types.SubstAny(n.Type(), &types_))
	if len(types_) > 0 {
		base.Fatalf("substArgTypes: too many argument types")
	}
	return n
}

// canMergeLoads reports whether the backend optimization passes for
// the current architecture can combine adjacent loads into a single
// larger, possibly unaligned, load. Note that currently the
// optimizations must be able to handle little endian byte order.
func canMergeLoads() bool {
	switch thearch.LinkArch.Family {
	case sys.ARM64, sys.AMD64, sys.I386, sys.S390X:
		return true
	case sys.PPC64:
		// Load combining only supported on ppc64le.
		return thearch.LinkArch.ByteOrder == binary.LittleEndian
	}
	return false
}

// isRuneCount reports whether n is of the form len([]rune(string)).
// These are optimized into a call to runtime.countrunes.
func isRuneCount(n ir.Node) bool {
	return base.Flag.N == 0 && !instrumenting && n.Op() == ir.OLEN && n.Left().Op() == ir.OSTR2RUNES
}

func walkCheckPtrAlignment(n ir.Node, init *ir.Nodes, count ir.Node) ir.Node {
	if !n.Type().IsPtr() {
		base.Fatalf("expected pointer type: %v", n.Type())
	}
	elem := n.Type().Elem()
	if count != nil {
		if !elem.IsArray() {
			base.Fatalf("expected array type: %v", elem)
		}
		elem = elem.Elem()
	}

	size := elem.Size()
	if elem.Alignment() == 1 && (size == 0 || size == 1 && count == nil) {
		return n
	}

	if count == nil {
		count = nodintconst(1)
	}

	n.SetLeft(cheapexpr(n.Left(), init))
	init.Append(mkcall("checkptrAlignment", nil, init, convnop(n.Left(), types.Types[types.TUNSAFEPTR]), typename(elem), conv(count, types.Types[types.TUINTPTR])))
	return n
}

var walkCheckPtrArithmeticMarker byte

func walkCheckPtrArithmetic(n ir.Node, init *ir.Nodes) ir.Node {
	// Calling cheapexpr(n, init) below leads to a recursive call
	// to walkexpr, which leads us back here again. Use n.Opt to
	// prevent infinite loops.
	if opt := n.Opt(); opt == &walkCheckPtrArithmeticMarker {
		return n
	} else if opt != nil {
		// We use n.Opt() here because today it's not used for OCONVNOP. If that changes,
		// there's no guarantee that temporarily replacing it is safe, so just hard fail here.
		base.Fatalf("unexpected Opt: %v", opt)
	}
	n.SetOpt(&walkCheckPtrArithmeticMarker)
	defer n.SetOpt(nil)

	// TODO(mdempsky): Make stricter. We only need to exempt
	// reflect.Value.Pointer and reflect.Value.UnsafeAddr.
	switch n.Left().Op() {
	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		return n
	}

	if n.Left().Op() == ir.ODOTPTR && isReflectHeaderDataField(n.Left()) {
		return n
	}

	// Find original unsafe.Pointer operands involved in this
	// arithmetic expression.
	//
	// "It is valid both to add and to subtract offsets from a
	// pointer in this way. It is also valid to use &^ to round
	// pointers, usually for alignment."
	var originals []ir.Node
	var walk func(n ir.Node)
	walk = func(n ir.Node) {
		switch n.Op() {
		case ir.OADD:
			walk(n.Left())
			walk(n.Right())
		case ir.OSUB, ir.OANDNOT:
			walk(n.Left())
		case ir.OCONVNOP:
			if n.Left().Type().IsUnsafePtr() {
				n.SetLeft(cheapexpr(n.Left(), init))
				originals = append(originals, convnop(n.Left(), types.Types[types.TUNSAFEPTR]))
			}
		}
	}
	walk(n.Left())

	n = cheapexpr(n, init)

	slice := mkdotargslice(types.NewSlice(types.Types[types.TUNSAFEPTR]), originals)
	slice.SetEsc(EscNone)

	init.Append(mkcall("checkptrArithmetic", nil, init, convnop(n, types.Types[types.TUNSAFEPTR]), slice))
	// TODO(khr): Mark backing store of slice as dead. This will allow us to reuse
	// the backing store for multiple calls to checkptrArithmetic.

	return n
}

// checkPtr reports whether pointer checking should be enabled for
// function fn at a given level. See debugHelpFooter for defined
// levels.
func checkPtr(fn *ir.Func, level int) bool {
	return base.Debug.Checkptr >= level && fn.Pragma&ir.NoCheckPtr == 0
}

// appendWalkStmt typechecks and walks stmt and then appends it to init.
func appendWalkStmt(init *ir.Nodes, stmt ir.Node) {
	op := stmt.Op()
	n := typecheck(stmt, ctxStmt)
	if op == ir.OAS || op == ir.OAS2 {
		// If the assignment has side effects, walkexpr will append them
		// directly to init for us, while walkstmt will wrap it in an OBLOCK.
		// We need to append them directly.
		// TODO(rsc): Clean this up.
		n = walkexpr(n, init)
	} else {
		n = walkstmt(n)
	}
	init.Append(n)
}
