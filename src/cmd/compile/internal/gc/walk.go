// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"encoding/binary"
	"errors"
	"fmt"
	"go/constant"
	"go/token"
	"strings"
)

// The constant is known to runtime.
const tmpstringbufsize = 32
const zeroValSize = 1024 // must match value of runtime/map.go:maxZero

func walk(fn *ir.Func) {
	ir.CurFunc = fn
	errorsBefore := base.Errors()
	order(fn)
	if base.Errors() > errorsBefore {
		return
	}

	if base.Flag.W != 0 {
		s := fmt.Sprintf("\nbefore walk %v", ir.CurFunc.Sym())
		ir.DumpList(s, ir.CurFunc.Body)
	}

	lno := base.Pos

	// Final typecheck for any unused variables.
	for i, ln := range fn.Dcl {
		if ln.Op() == ir.ONAME && (ln.Class_ == ir.PAUTO || ln.Class_ == ir.PAUTOHEAP) {
			ln = typecheck.AssignExpr(ln).(*ir.Name)
			fn.Dcl[i] = ln
		}
	}

	// Propagate the used flag for typeswitch variables up to the NONAME in its definition.
	for _, ln := range fn.Dcl {
		if ln.Op() == ir.ONAME && (ln.Class_ == ir.PAUTO || ln.Class_ == ir.PAUTOHEAP) && ln.Defn != nil && ln.Defn.Op() == ir.OTYPESW && ln.Used() {
			ln.Defn.(*ir.TypeSwitchGuard).Used = true
		}
	}

	for _, ln := range fn.Dcl {
		if ln.Op() != ir.ONAME || (ln.Class_ != ir.PAUTO && ln.Class_ != ir.PAUTOHEAP) || ln.Sym().Name[0] == '&' || ln.Used() {
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
	walkstmtlist(ir.CurFunc.Body)
	if base.Flag.W != 0 {
		s := fmt.Sprintf("after walk %v", ir.CurFunc.Sym())
		ir.DumpList(s, ir.CurFunc.Body)
	}

	zeroResults()
	heapmoves()
	if base.Flag.W != 0 && len(ir.CurFunc.Enter) > 0 {
		s := fmt.Sprintf("enter %v", ir.CurFunc.Sym())
		ir.DumpList(s, ir.CurFunc.Enter)
	}

	if base.Flag.Cfg.Instrumenting {
		instrument(fn)
	}
}

func walkstmtlist(s []ir.Node) {
	for i := range s {
		s[i] = walkstmt(s[i])
	}
}

func paramoutheap(fn *ir.Func) bool {
	for _, ln := range fn.Dcl {
		switch ln.Class_ {
		case ir.PPARAMOUT:
			if ir.IsParamStackCopy(ln) || ln.Addrtaken() {
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

	ir.SetPos(n)

	walkstmtlist(n.Init())

	switch n.Op() {
	default:
		if n.Op() == ir.ONAME {
			n := n.(*ir.Name)
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
			n = ir.NewBlockStmt(n.Pos(), init)
			init.Set(nil)
		}
		if len(init) > 0 {
			switch n.Op() {
			case ir.OAS, ir.OAS2, ir.OBLOCK:
				n.PtrInit().Prepend(init...)

			default:
				init.Append(n)
				n = ir.NewBlockStmt(n.Pos(), init)
			}
		}
		return n

	// special case for a receive where we throw away
	// the value received.
	case ir.ORECV:
		n := n.(*ir.UnaryExpr)
		if n.Typecheck() == 0 {
			base.Fatalf("missing typecheck: %+v", n)
		}
		init := n.Init()
		n.PtrInit().Set(nil)

		n.X = walkexpr(n.X, &init)
		call := walkexpr(mkcall1(chanfn("chanrecv1", 2, n.X.Type()), nil, &init, n.X, typecheck.NodNil()), &init)
		return ir.InitExpr(init, call)

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
		n := n.(*ir.Decl)
		v := n.X.(*ir.Name)
		if v.Class_ == ir.PAUTOHEAP {
			if base.Flag.CompilingRuntime {
				base.Errorf("%v escapes to heap, not allowed in runtime", v)
			}
			nn := ir.NewAssignStmt(base.Pos, v.Name().Heapaddr, callnew(v.Type()))
			nn.Def = true
			return walkstmt(typecheck.Stmt(nn))
		}
		return n

	case ir.OBLOCK:
		n := n.(*ir.BlockStmt)
		walkstmtlist(n.List)
		return n

	case ir.OCASE:
		base.Errorf("case statement out of place")
		panic("unreachable")

	case ir.ODEFER:
		n := n.(*ir.GoDeferStmt)
		ir.CurFunc.SetHasDefer(true)
		ir.CurFunc.NumDefers++
		if ir.CurFunc.NumDefers > maxOpenDefers {
			// Don't allow open-coded defers if there are more than
			// 8 defers in the function, since we use a single
			// byte to record active defers.
			ir.CurFunc.SetOpenCodedDeferDisallowed(true)
		}
		if n.Esc() != ir.EscNever {
			// If n.Esc is not EscNever, then this defer occurs in a loop,
			// so open-coded defers cannot be used in this function.
			ir.CurFunc.SetOpenCodedDeferDisallowed(true)
		}
		fallthrough
	case ir.OGO:
		n := n.(*ir.GoDeferStmt)
		var init ir.Nodes
		switch call := n.Call; call.Op() {
		case ir.OPRINT, ir.OPRINTN:
			call := call.(*ir.CallExpr)
			n.Call = wrapCall(call, &init)

		case ir.ODELETE:
			call := call.(*ir.CallExpr)
			if mapfast(call.Args[0].Type()) == mapslow {
				n.Call = wrapCall(call, &init)
			} else {
				n.Call = walkexpr(call, &init)
			}

		case ir.OCOPY:
			call := call.(*ir.BinaryExpr)
			n.Call = copyany(call, &init, true)

		case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
			call := call.(*ir.CallExpr)
			if len(call.Body) > 0 {
				n.Call = wrapCall(call, &init)
			} else {
				n.Call = walkexpr(call, &init)
			}

		default:
			n.Call = walkexpr(call, &init)
		}
		if len(init) > 0 {
			init.Append(n)
			return ir.NewBlockStmt(n.Pos(), init)
		}
		return n

	case ir.OFOR, ir.OFORUNTIL:
		n := n.(*ir.ForStmt)
		if n.Cond != nil {
			walkstmtlist(n.Cond.Init())
			init := n.Cond.Init()
			n.Cond.PtrInit().Set(nil)
			n.Cond = walkexpr(n.Cond, &init)
			n.Cond = ir.InitExpr(init, n.Cond)
		}

		n.Post = walkstmt(n.Post)
		if n.Op() == ir.OFORUNTIL {
			walkstmtlist(n.Late)
		}
		walkstmtlist(n.Body)
		return n

	case ir.OIF:
		n := n.(*ir.IfStmt)
		n.Cond = walkexpr(n.Cond, n.PtrInit())
		walkstmtlist(n.Body)
		walkstmtlist(n.Else)
		return n

	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		ir.CurFunc.NumReturns++
		if len(n.Results) == 0 {
			return n
		}
		if (ir.HasNamedResults(ir.CurFunc) && len(n.Results) > 1) || paramoutheap(ir.CurFunc) {
			// assign to the function out parameters,
			// so that ascompatee can fix up conflicts
			var rl []ir.Node

			for _, ln := range ir.CurFunc.Dcl {
				cl := ln.Class_
				if cl == ir.PAUTO || cl == ir.PAUTOHEAP {
					break
				}
				if cl == ir.PPARAMOUT {
					var ln ir.Node = ln
					if ir.IsParamStackCopy(ln) {
						ln = walkexpr(typecheck.Expr(ir.NewStarExpr(base.Pos, ln.Name().Heapaddr)), nil)
					}
					rl = append(rl, ln)
				}
			}

			if got, want := len(n.Results), len(rl); got != want {
				// order should have rewritten multi-value function calls
				// with explicit OAS2FUNC nodes.
				base.Fatalf("expected %v return arguments, have %v", want, got)
			}

			// move function calls out, to make ascompatee's job easier.
			walkexprlistsafe(n.Results, n.PtrInit())

			n.Results.Set(ascompatee(n.Op(), rl, n.Results, n.PtrInit()))
			return n
		}
		walkexprlist(n.Results, n.PtrInit())

		// For each return parameter (lhs), assign the corresponding result (rhs).
		lhs := ir.CurFunc.Type().Results()
		rhs := n.Results
		res := make([]ir.Node, lhs.NumFields())
		for i, nl := range lhs.FieldSlice() {
			nname := ir.AsNode(nl.Nname)
			if ir.IsParamHeapCopy(nname) {
				nname = nname.Name().Stackcopy
			}
			a := ir.NewAssignStmt(base.Pos, nname, rhs[i])
			res[i] = convas(a, n.PtrInit())
		}
		n.Results.Set(res)
		return n

	case ir.ORETJMP:
		n := n.(*ir.BranchStmt)
		return n

	case ir.OINLMARK:
		n := n.(*ir.InlineMarkStmt)
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
			types.CheckSize(n.Type())
		}
	}

	if init == n.PtrInit() {
		// not okay to use n->ninit when walking n,
		// because we might replace n with some other node
		// and would lose the init list.
		base.Fatalf("walkexpr init == &n->ninit")
	}

	if len(n.Init()) != 0 {
		walkstmtlist(n.Init())
		init.Append(n.PtrInit().Take()...)
	}

	lno := ir.SetPos(n)

	if base.Flag.LowerW > 1 {
		ir.Dump("before walk expr", n)
	}

	if n.Typecheck() != 1 {
		base.Fatalf("missed typecheck: %+v", n)
	}

	if n.Type().IsUntyped() {
		base.Fatalf("expression has untyped type: %+v", n)
	}

	if n.Op() == ir.ONAME && n.(*ir.Name).Class_ == ir.PAUTOHEAP {
		n := n.(*ir.Name)
		nn := ir.NewStarExpr(base.Pos, n.Name().Heapaddr)
		nn.X.MarkNonNil()
		return walkexpr(typecheck.Expr(nn), init)
	}

	n = walkexpr1(n, init)

	// Expressions that are constant at run time but not
	// considered const by the language spec are not turned into
	// constants until walk. For example, if n is y%1 == 0, the
	// walk of y%1 may have replaced it by 0.
	// Check whether n with its updated args is itself now a constant.
	t := n.Type()
	n = typecheck.EvalConst(n)
	if n.Type() != t {
		base.Fatalf("evconst changed Type: %v had type %v, now %v", n, t, n.Type())
	}
	if n.Op() == ir.OLITERAL {
		n = typecheck.Expr(n)
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

	case ir.OTYPE, ir.ONAME, ir.OLITERAL, ir.ONIL, ir.ONAMEOFFSET:
		// TODO(mdempsky): Just return n; see discussion on CL 38655.
		// Perhaps refactor to use Node.mayBeShared for these instead.
		// If these return early, make sure to still call
		// stringsym for constant strings.
		return n

	case ir.ONOT, ir.ONEG, ir.OPLUS, ir.OBITNOT, ir.OREAL, ir.OIMAG, ir.OSPTR, ir.OITAB, ir.OIDATA:
		n := n.(*ir.UnaryExpr)
		n.X = walkexpr(n.X, init)
		return n

	case ir.ODOTMETH, ir.ODOTINTER:
		n := n.(*ir.SelectorExpr)
		n.X = walkexpr(n.X, init)
		return n

	case ir.OADDR:
		n := n.(*ir.AddrExpr)
		n.X = walkexpr(n.X, init)
		return n

	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		n.X = walkexpr(n.X, init)
		return n

	case ir.OEFACE, ir.OAND, ir.OANDNOT, ir.OSUB, ir.OMUL, ir.OADD, ir.OOR, ir.OXOR, ir.OLSH, ir.ORSH:
		n := n.(*ir.BinaryExpr)
		n.X = walkexpr(n.X, init)
		n.Y = walkexpr(n.Y, init)
		return n

	case ir.ODOT, ir.ODOTPTR:
		n := n.(*ir.SelectorExpr)
		usefield(n)
		n.X = walkexpr(n.X, init)
		return n

	case ir.ODOTTYPE, ir.ODOTTYPE2:
		n := n.(*ir.TypeAssertExpr)
		n.X = walkexpr(n.X, init)
		// Set up interface type addresses for back end.
		n.Ntype = typename(n.Type())
		if n.Op() == ir.ODOTTYPE {
			n.Ntype.(*ir.AddrExpr).Alloc = typename(n.X.Type())
		}
		if !n.Type().IsInterface() && !n.X.Type().IsEmptyInterface() {
			n.Itab = []ir.Node{itabname(n.Type(), n.X.Type())}
		}
		return n

	case ir.OLEN, ir.OCAP:
		n := n.(*ir.UnaryExpr)
		if isRuneCount(n) {
			// Replace len([]rune(string)) with runtime.countrunes(string).
			return mkcall("countrunes", n.Type(), init, typecheck.Conv(n.X.(*ir.ConvExpr).X, types.Types[types.TSTRING]))
		}

		n.X = walkexpr(n.X, init)

		// replace len(*[10]int) with 10.
		// delayed until now to preserve side effects.
		t := n.X.Type()

		if t.IsPtr() {
			t = t.Elem()
		}
		if t.IsArray() {
			safeexpr(n.X, init)
			con := typecheck.OrigInt(n, t.NumElem())
			con.SetTypecheck(1)
			return con
		}
		return n

	case ir.OCOMPLEX:
		n := n.(*ir.BinaryExpr)
		n.X = walkexpr(n.X, init)
		n.Y = walkexpr(n.Y, init)
		return n

	case ir.OEQ, ir.ONE, ir.OLT, ir.OLE, ir.OGT, ir.OGE:
		n := n.(*ir.BinaryExpr)
		return walkcompare(n, init)

	case ir.OANDAND, ir.OOROR:
		n := n.(*ir.LogicalExpr)
		n.X = walkexpr(n.X, init)

		// cannot put side effects from n.Right on init,
		// because they cannot run before n.Left is checked.
		// save elsewhere and store on the eventual n.Right.
		var ll ir.Nodes

		n.Y = walkexpr(n.Y, &ll)
		n.Y = ir.InitExpr(ll, n.Y)
		return n

	case ir.OPRINT, ir.OPRINTN:
		return walkprint(n.(*ir.CallExpr), init)

	case ir.OPANIC:
		n := n.(*ir.UnaryExpr)
		return mkcall("gopanic", nil, init, n.X)

	case ir.ORECOVER:
		n := n.(*ir.CallExpr)
		return mkcall("gorecover", n.Type(), init, typecheck.NodAddr(ir.RegFP))

	case ir.OCLOSUREREAD, ir.OCFUNC:
		return n

	case ir.OCALLINTER, ir.OCALLFUNC, ir.OCALLMETH:
		n := n.(*ir.CallExpr)
		if n.Op() == ir.OCALLINTER {
			usemethod(n)
			markUsedIfaceMethod(n)
		}

		if n.Op() == ir.OCALLFUNC && n.X.Op() == ir.OCLOSURE {
			// Transform direct call of a closure to call of a normal function.
			// transformclosure already did all preparation work.

			// Prepend captured variables to argument list.
			clo := n.X.(*ir.ClosureExpr)
			n.Args.Prepend(clo.Func.ClosureEnter...)
			clo.Func.ClosureEnter.Set(nil)

			// Replace OCLOSURE with ONAME/PFUNC.
			n.X = clo.Func.Nname

			// Update type of OCALLFUNC node.
			// Output arguments had not changed, but their offsets could.
			if n.X.Type().NumResults() == 1 {
				n.SetType(n.X.Type().Results().Field(0).Type)
			} else {
				n.SetType(n.X.Type().Results())
			}
		}

		walkCall(n, init)
		return n

	case ir.OAS, ir.OASOP:
		init.Append(n.PtrInit().Take()...)

		var left, right ir.Node
		switch n.Op() {
		case ir.OAS:
			n := n.(*ir.AssignStmt)
			left, right = n.X, n.Y
		case ir.OASOP:
			n := n.(*ir.AssignOpStmt)
			left, right = n.X, n.Y
		}

		// Recognize m[k] = append(m[k], ...) so we can reuse
		// the mapassign call.
		var mapAppend *ir.CallExpr
		if left.Op() == ir.OINDEXMAP && right.Op() == ir.OAPPEND {
			left := left.(*ir.IndexExpr)
			mapAppend = right.(*ir.CallExpr)
			if !ir.SameSafeExpr(left, mapAppend.Args[0]) {
				base.Fatalf("not same expressions: %v != %v", left, mapAppend.Args[0])
			}
		}

		left = walkexpr(left, init)
		left = safeexpr(left, init)
		if mapAppend != nil {
			mapAppend.Args[0] = left
		}

		if n.Op() == ir.OASOP {
			// Rewrite x op= y into x = x op y.
			n = ir.NewAssignStmt(base.Pos, left, typecheck.Expr(ir.NewBinaryExpr(base.Pos, n.(*ir.AssignOpStmt).AsOp, left, right)))
		} else {
			n.(*ir.AssignStmt).X = left
		}
		as := n.(*ir.AssignStmt)

		if oaslit(as, init) {
			return ir.NewBlockStmt(as.Pos(), nil)
		}

		if as.Y == nil {
			// TODO(austin): Check all "implicit zeroing"
			return as
		}

		if !base.Flag.Cfg.Instrumenting && ir.IsZero(as.Y) {
			return as
		}

		switch as.Y.Op() {
		default:
			as.Y = walkexpr(as.Y, init)

		case ir.ORECV:
			// x = <-c; as.Left is x, as.Right.Left is c.
			// order.stmt made sure x is addressable.
			recv := as.Y.(*ir.UnaryExpr)
			recv.X = walkexpr(recv.X, init)

			n1 := typecheck.NodAddr(as.X)
			r := recv.X // the channel
			return mkcall1(chanfn("chanrecv1", 2, r.Type()), nil, init, r, n1)

		case ir.OAPPEND:
			// x = append(...)
			call := as.Y.(*ir.CallExpr)
			if call.Type().Elem().NotInHeap() {
				base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", call.Type().Elem())
			}
			var r ir.Node
			switch {
			case isAppendOfMake(call):
				// x = append(y, make([]T, y)...)
				r = extendslice(call, init)
			case call.IsDDD:
				r = appendslice(call, init) // also works for append(slice, string).
			default:
				r = walkappend(call, init, as)
			}
			as.Y = r
			if r.Op() == ir.OAPPEND {
				// Left in place for back end.
				// Do not add a new write barrier.
				// Set up address of type for back end.
				r.(*ir.CallExpr).X = typename(r.Type().Elem())
				return as
			}
			// Otherwise, lowered for race detector.
			// Treat as ordinary assignment.
		}

		if as.X != nil && as.Y != nil {
			return convas(as, init)
		}
		return as

	case ir.OAS2:
		n := n.(*ir.AssignListStmt)
		init.Append(n.PtrInit().Take()...)
		walkexprlistsafe(n.Lhs, init)
		walkexprlistsafe(n.Rhs, init)
		return ir.NewBlockStmt(src.NoXPos, ascompatee(ir.OAS, n.Lhs, n.Rhs, init))

	// a,b,... = fn()
	case ir.OAS2FUNC:
		n := n.(*ir.AssignListStmt)
		init.Append(n.PtrInit().Take()...)

		r := n.Rhs[0]
		walkexprlistsafe(n.Lhs, init)
		r = walkexpr(r, init)

		if ir.IsIntrinsicCall(r.(*ir.CallExpr)) {
			n.Rhs = []ir.Node{r}
			return n
		}
		init.Append(r)

		ll := ascompatet(n.Lhs, r.Type())
		return ir.NewBlockStmt(src.NoXPos, ll)

	// x, y = <-c
	// order.stmt made sure x is addressable or blank.
	case ir.OAS2RECV:
		n := n.(*ir.AssignListStmt)
		init.Append(n.PtrInit().Take()...)

		r := n.Rhs[0].(*ir.UnaryExpr) // recv
		walkexprlistsafe(n.Lhs, init)
		r.X = walkexpr(r.X, init)
		var n1 ir.Node
		if ir.IsBlank(n.Lhs[0]) {
			n1 = typecheck.NodNil()
		} else {
			n1 = typecheck.NodAddr(n.Lhs[0])
		}
		fn := chanfn("chanrecv2", 2, r.X.Type())
		ok := n.Lhs[1]
		call := mkcall1(fn, types.Types[types.TBOOL], init, r.X, n1)
		return typecheck.Stmt(ir.NewAssignStmt(base.Pos, ok, call))

	// a,b = m[i]
	case ir.OAS2MAPR:
		n := n.(*ir.AssignListStmt)
		init.Append(n.PtrInit().Take()...)

		r := n.Rhs[0].(*ir.IndexExpr)
		walkexprlistsafe(n.Lhs, init)
		r.X = walkexpr(r.X, init)
		r.Index = walkexpr(r.Index, init)
		t := r.X.Type()

		fast := mapfast(t)
		var key ir.Node
		if fast != mapslow {
			// fast versions take key by value
			key = r.Index
		} else {
			// standard version takes key by reference
			// order.expr made sure key is addressable.
			key = typecheck.NodAddr(r.Index)
		}

		// from:
		//   a,b = m[i]
		// to:
		//   var,b = mapaccess2*(t, m, i)
		//   a = *var
		a := n.Lhs[0]

		var call *ir.CallExpr
		if w := t.Elem().Width; w <= zeroValSize {
			fn := mapfn(mapaccess2[fast], t)
			call = mkcall1(fn, fn.Type().Results(), init, typename(t), r.X, key)
		} else {
			fn := mapfn("mapaccess2_fat", t)
			z := zeroaddr(w)
			call = mkcall1(fn, fn.Type().Results(), init, typename(t), r.X, key, z)
		}

		// mapaccess2* returns a typed bool, but due to spec changes,
		// the boolean result of i.(T) is now untyped so we make it the
		// same type as the variable on the lhs.
		if ok := n.Lhs[1]; !ir.IsBlank(ok) && ok.Type().IsBoolean() {
			call.Type().Field(1).Type = ok.Type()
		}
		n.Rhs = []ir.Node{call}
		n.SetOp(ir.OAS2FUNC)

		// don't generate a = *var if a is _
		if ir.IsBlank(a) {
			return walkexpr(typecheck.Stmt(n), init)
		}

		var_ := typecheck.Temp(types.NewPtr(t.Elem()))
		var_.SetTypecheck(1)
		var_.MarkNonNil() // mapaccess always returns a non-nil pointer

		n.Lhs[0] = var_
		init.Append(walkexpr(n, init))

		as := ir.NewAssignStmt(base.Pos, a, ir.NewStarExpr(base.Pos, var_))
		return walkexpr(typecheck.Stmt(as), init)

	case ir.ODELETE:
		n := n.(*ir.CallExpr)
		init.Append(n.PtrInit().Take()...)
		map_ := n.Args[0]
		key := n.Args[1]
		map_ = walkexpr(map_, init)
		key = walkexpr(key, init)

		t := map_.Type()
		fast := mapfast(t)
		if fast == mapslow {
			// order.stmt made sure key is addressable.
			key = typecheck.NodAddr(key)
		}
		return mkcall1(mapfndel(mapdelete[fast], t), nil, init, typename(t), map_, key)

	case ir.OAS2DOTTYPE:
		n := n.(*ir.AssignListStmt)
		walkexprlistsafe(n.Lhs, init)
		n.Rhs[0] = walkexpr(n.Rhs[0], init)
		return n

	case ir.OCONVIFACE:
		n := n.(*ir.ConvExpr)
		n.X = walkexpr(n.X, init)

		fromType := n.X.Type()
		toType := n.Type()

		if !fromType.IsInterface() && !ir.IsBlank(ir.CurFunc.Nname) { // skip unnamed functions (func _())
			markTypeUsedInInterface(fromType, ir.CurFunc.LSym)
		}

		// typeword generates the type word of the interface value.
		typeword := func() ir.Node {
			if toType.IsEmptyInterface() {
				return typename(fromType)
			}
			return itabname(fromType, toType)
		}

		// Optimize convT2E or convT2I as a two-word copy when T is pointer-shaped.
		if types.IsDirectIface(fromType) {
			l := ir.NewBinaryExpr(base.Pos, ir.OEFACE, typeword(), n.X)
			l.SetType(toType)
			l.SetTypecheck(n.Typecheck())
			return l
		}

		if ir.Names.Staticuint64s == nil {
			ir.Names.Staticuint64s = typecheck.NewName(ir.Pkgs.Runtime.Lookup("staticuint64s"))
			ir.Names.Staticuint64s.Class_ = ir.PEXTERN
			// The actual type is [256]uint64, but we use [256*8]uint8 so we can address
			// individual bytes.
			ir.Names.Staticuint64s.SetType(types.NewArray(types.Types[types.TUINT8], 256*8))
			ir.Names.Zerobase = typecheck.NewName(ir.Pkgs.Runtime.Lookup("zerobase"))
			ir.Names.Zerobase.Class_ = ir.PEXTERN
			ir.Names.Zerobase.SetType(types.Types[types.TUINTPTR])
		}

		// Optimize convT2{E,I} for many cases in which T is not pointer-shaped,
		// by using an existing addressable value identical to n.Left
		// or creating one on the stack.
		var value ir.Node
		switch {
		case fromType.Size() == 0:
			// n.Left is zero-sized. Use zerobase.
			cheapexpr(n.X, init) // Evaluate n.Left for side-effects. See issue 19246.
			value = ir.Names.Zerobase
		case fromType.IsBoolean() || (fromType.Size() == 1 && fromType.IsInteger()):
			// n.Left is a bool/byte. Use staticuint64s[n.Left * 8] on little-endian
			// and staticuint64s[n.Left * 8 + 7] on big-endian.
			n.X = cheapexpr(n.X, init)
			// byteindex widens n.Left so that the multiplication doesn't overflow.
			index := ir.NewBinaryExpr(base.Pos, ir.OLSH, byteindex(n.X), ir.NewInt(3))
			if thearch.LinkArch.ByteOrder == binary.BigEndian {
				index = ir.NewBinaryExpr(base.Pos, ir.OADD, index, ir.NewInt(7))
			}
			xe := ir.NewIndexExpr(base.Pos, ir.Names.Staticuint64s, index)
			xe.SetBounded(true)
			value = xe
		case n.X.Op() == ir.ONAME && n.X.(*ir.Name).Class_ == ir.PEXTERN && n.X.(*ir.Name).Readonly():
			// n.Left is a readonly global; use it directly.
			value = n.X
		case !fromType.IsInterface() && n.Esc() == ir.EscNone && fromType.Width <= 1024:
			// n.Left does not escape. Use a stack temporary initialized to n.Left.
			value = typecheck.Temp(fromType)
			init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, value, n.X)))
		}

		if value != nil {
			// Value is identical to n.Left.
			// Construct the interface directly: {type/itab, &value}.
			l := ir.NewBinaryExpr(base.Pos, ir.OEFACE, typeword(), typecheck.Expr(typecheck.NodAddr(value)))
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
			c := typecheck.Temp(fromType)
			init.Append(ir.NewAssignStmt(base.Pos, c, n.X))

			// Get the itab out of the interface.
			tmp := typecheck.Temp(types.NewPtr(types.Types[types.TUINT8]))
			init.Append(ir.NewAssignStmt(base.Pos, tmp, typecheck.Expr(ir.NewUnaryExpr(base.Pos, ir.OITAB, c))))

			// Get the type out of the itab.
			nif := ir.NewIfStmt(base.Pos, typecheck.Expr(ir.NewBinaryExpr(base.Pos, ir.ONE, tmp, typecheck.NodNil())), nil, nil)
			nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, tmp, itabType(tmp))}
			init.Append(nif)

			// Build the result.
			e := ir.NewBinaryExpr(base.Pos, ir.OEFACE, tmp, ifaceData(n.Pos(), c, types.NewPtr(types.Types[types.TUINT8])))
			e.SetType(toType) // assign type manually, typecheck doesn't understand OEFACE.
			e.SetTypecheck(1)
			return e
		}

		fnname, needsaddr := convFuncName(fromType, toType)

		if !needsaddr && !fromType.IsInterface() {
			// Use a specialized conversion routine that only returns a data pointer.
			// ptr = convT2X(val)
			// e = iface{typ/tab, ptr}
			fn := typecheck.LookupRuntime(fnname)
			types.CalcSize(fromType)
			fn = typecheck.SubstArgTypes(fn, fromType)
			types.CalcSize(fn.Type())
			call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, nil)
			call.Args = []ir.Node{n.X}
			e := ir.NewBinaryExpr(base.Pos, ir.OEFACE, typeword(), safeexpr(walkexpr(typecheck.Expr(call), init), init))
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

		v := n.X
		if needsaddr {
			// Types of large or unknown size are passed by reference.
			// Orderexpr arranged for n.Left to be a temporary for all
			// the conversions it could see. Comparison of an interface
			// with a non-interface, especially in a switch on interface value
			// with non-interface cases, is not visible to order.stmt, so we
			// have to fall back on allocating a temp here.
			if !ir.IsAssignable(v) {
				v = copyexpr(v, v.Type(), init)
			}
			v = typecheck.NodAddr(v)
		}

		types.CalcSize(fromType)
		fn := typecheck.LookupRuntime(fnname)
		fn = typecheck.SubstArgTypes(fn, fromType, toType)
		types.CalcSize(fn.Type())
		call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, nil)
		call.Args = []ir.Node{tab, v}
		return walkexpr(typecheck.Expr(call), init)

	case ir.OCONV, ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		n.X = walkexpr(n.X, init)
		if n.Op() == ir.OCONVNOP && n.Type() == n.X.Type() {
			return n.X
		}
		if n.Op() == ir.OCONVNOP && ir.ShouldCheckPtr(ir.CurFunc, 1) {
			if n.Type().IsPtr() && n.X.Type().IsUnsafePtr() { // unsafe.Pointer to *T
				return walkCheckPtrAlignment(n, init, nil)
			}
			if n.Type().IsUnsafePtr() && n.X.Type().IsUintptr() { // uintptr to unsafe.Pointer
				return walkCheckPtrArithmetic(n, init)
			}
		}
		param, result := rtconvfn(n.X.Type(), n.Type())
		if param == types.Txxx {
			return n
		}
		fn := types.BasicTypeNames[param] + "to" + types.BasicTypeNames[result]
		return typecheck.Conv(mkcall(fn, types.Types[result], init, typecheck.Conv(n.X, types.Types[param])), n.Type())

	case ir.ODIV, ir.OMOD:
		n := n.(*ir.BinaryExpr)
		n.X = walkexpr(n.X, init)
		n.Y = walkexpr(n.Y, init)

		// rewrite complex div into function call.
		et := n.X.Type().Kind()

		if types.IsComplex[et] && n.Op() == ir.ODIV {
			t := n.Type()
			call := mkcall("complex128div", types.Types[types.TCOMPLEX128], init, typecheck.Conv(n.X, types.Types[types.TCOMPLEX128]), typecheck.Conv(n.Y, types.Types[types.TCOMPLEX128]))
			return typecheck.Conv(call, t)
		}

		// Nothing to do for float divisions.
		if types.IsFloat[et] {
			return n
		}

		// rewrite 64-bit div and mod on 32-bit architectures.
		// TODO: Remove this code once we can introduce
		// runtime calls late in SSA processing.
		if types.RegSize < 8 && (et == types.TINT64 || et == types.TUINT64) {
			if n.Y.Op() == ir.OLITERAL {
				// Leave div/mod by constant powers of 2 or small 16-bit constants.
				// The SSA backend will handle those.
				switch et {
				case types.TINT64:
					c := ir.Int64Val(n.Y)
					if c < 0 {
						c = -c
					}
					if c != 0 && c&(c-1) == 0 {
						return n
					}
				case types.TUINT64:
					c := ir.Uint64Val(n.Y)
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
			return mkcall(fn, n.Type(), init, typecheck.Conv(n.X, types.Types[et]), typecheck.Conv(n.Y, types.Types[et]))
		}
		return n

	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		n.X = walkexpr(n.X, init)

		// save the original node for bounds checking elision.
		// If it was a ODIV/OMOD walk might rewrite it.
		r := n.Index

		n.Index = walkexpr(n.Index, init)

		// if range of type cannot exceed static array bound,
		// disable bounds check.
		if n.Bounded() {
			return n
		}
		t := n.X.Type()
		if t != nil && t.IsPtr() {
			t = t.Elem()
		}
		if t.IsArray() {
			n.SetBounded(bounded(r, t.NumElem()))
			if base.Flag.LowerM != 0 && n.Bounded() && !ir.IsConst(n.Index, constant.Int) {
				base.Warn("index bounds check elided")
			}
			if ir.IsSmallIntConst(n.Index) && !n.Bounded() {
				base.Errorf("index out of bounds")
			}
		} else if ir.IsConst(n.X, constant.String) {
			n.SetBounded(bounded(r, int64(len(ir.StringVal(n.X)))))
			if base.Flag.LowerM != 0 && n.Bounded() && !ir.IsConst(n.Index, constant.Int) {
				base.Warn("index bounds check elided")
			}
			if ir.IsSmallIntConst(n.Index) && !n.Bounded() {
				base.Errorf("index out of bounds")
			}
		}

		if ir.IsConst(n.Index, constant.Int) {
			if v := n.Index.Val(); constant.Sign(v) < 0 || ir.ConstOverflow(v, types.Types[types.TINT]) {
				base.Errorf("index out of bounds")
			}
		}
		return n

	case ir.OINDEXMAP:
		// Replace m[k] with *map{access1,assign}(maptype, m, &k)
		n := n.(*ir.IndexExpr)
		n.X = walkexpr(n.X, init)
		n.Index = walkexpr(n.Index, init)
		map_ := n.X
		key := n.Index
		t := map_.Type()
		var call *ir.CallExpr
		if n.Assigned {
			// This m[k] expression is on the left-hand side of an assignment.
			fast := mapfast(t)
			if fast == mapslow {
				// standard version takes key by reference.
				// order.expr made sure key is addressable.
				key = typecheck.NodAddr(key)
			}
			call = mkcall1(mapfn(mapassign[fast], t), nil, init, typename(t), map_, key)
		} else {
			// m[k] is not the target of an assignment.
			fast := mapfast(t)
			if fast == mapslow {
				// standard version takes key by reference.
				// order.expr made sure key is addressable.
				key = typecheck.NodAddr(key)
			}

			if w := t.Elem().Width; w <= zeroValSize {
				call = mkcall1(mapfn(mapaccess1[fast], t), types.NewPtr(t.Elem()), init, typename(t), map_, key)
			} else {
				z := zeroaddr(w)
				call = mkcall1(mapfn("mapaccess1_fat", t), types.NewPtr(t.Elem()), init, typename(t), map_, key, z)
			}
		}
		call.SetType(types.NewPtr(t.Elem()))
		call.MarkNonNil() // mapaccess1* and mapassign always return non-nil pointers.
		star := ir.NewStarExpr(base.Pos, call)
		star.SetType(t.Elem())
		star.SetTypecheck(1)
		return star

	case ir.ORECV:
		base.Fatalf("walkexpr ORECV") // should see inside OAS only
		panic("unreachable")

	case ir.OSLICEHEADER:
		n := n.(*ir.SliceHeaderExpr)
		n.Ptr = walkexpr(n.Ptr, init)
		n.LenCap[0] = walkexpr(n.LenCap[0], init)
		n.LenCap[1] = walkexpr(n.LenCap[1], init)
		return n

	case ir.OSLICE, ir.OSLICEARR, ir.OSLICESTR, ir.OSLICE3, ir.OSLICE3ARR:
		n := n.(*ir.SliceExpr)

		checkSlice := ir.ShouldCheckPtr(ir.CurFunc, 1) && n.Op() == ir.OSLICE3ARR && n.X.Op() == ir.OCONVNOP && n.X.(*ir.ConvExpr).X.Type().IsUnsafePtr()
		if checkSlice {
			conv := n.X.(*ir.ConvExpr)
			conv.X = walkexpr(conv.X, init)
		} else {
			n.X = walkexpr(n.X, init)
		}

		low, high, max := n.SliceBounds()
		low = walkexpr(low, init)
		if low != nil && ir.IsZero(low) {
			// Reduce x[0:j] to x[:j] and x[0:j:k] to x[:j:k].
			low = nil
		}
		high = walkexpr(high, init)
		max = walkexpr(max, init)
		n.SetSliceBounds(low, high, max)
		if checkSlice {
			n.X = walkCheckPtrAlignment(n.X.(*ir.ConvExpr), init, max)
		}

		if n.Op().IsSlice3() {
			if max != nil && max.Op() == ir.OCAP && ir.SameSafeExpr(n.X, max.(*ir.UnaryExpr).X) {
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
		n := n.(*ir.UnaryExpr)
		if n.Type().Elem().NotInHeap() {
			base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", n.Type().Elem())
		}
		if n.Esc() == ir.EscNone {
			if n.Type().Elem().Width >= ir.MaxImplicitStackVarSize {
				base.Fatalf("large ONEW with EscNone: %v", n)
			}
			r := typecheck.Temp(n.Type().Elem())
			init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, r, nil))) // zero temp
			return typecheck.Expr(typecheck.NodAddr(r))
		}
		return callnew(n.Type().Elem())

	case ir.OADDSTR:
		return addstr(n.(*ir.AddStringExpr), init)

	case ir.OAPPEND:
		// order should make sure we only see OAS(node, OAPPEND), which we handle above.
		base.Fatalf("append outside assignment")
		panic("unreachable")

	case ir.OCOPY:
		return copyany(n.(*ir.BinaryExpr), init, base.Flag.Cfg.Instrumenting && !base.Flag.CompilingRuntime)

	case ir.OCLOSE:
		// cannot use chanfn - closechan takes any, not chan any
		n := n.(*ir.UnaryExpr)
		fn := typecheck.LookupRuntime("closechan")
		fn = typecheck.SubstArgTypes(fn, n.X.Type())
		return mkcall1(fn, nil, init, n.X)

	case ir.OMAKECHAN:
		// When size fits into int, use makechan instead of
		// makechan64, which is faster and shorter on 32 bit platforms.
		n := n.(*ir.MakeExpr)
		size := n.Len
		fnname := "makechan64"
		argtype := types.Types[types.TINT64]

		// Type checking guarantees that TIDEAL size is positive and fits in an int.
		// The case of size overflow when converting TUINT or TUINTPTR to TINT
		// will be handled by the negative range checks in makechan during runtime.
		if size.Type().IsKind(types.TIDEAL) || size.Type().Size() <= types.Types[types.TUINT].Size() {
			fnname = "makechan"
			argtype = types.Types[types.TINT]
		}

		return mkcall1(chanfn(fnname, 1, n.Type()), n.Type(), init, typename(n.Type()), typecheck.Conv(size, argtype))

	case ir.OMAKEMAP:
		n := n.(*ir.MakeExpr)
		t := n.Type()
		hmapType := hmap(t)
		hint := n.Len

		// var h *hmap
		var h ir.Node
		if n.Esc() == ir.EscNone {
			// Allocate hmap on stack.

			// var hv hmap
			hv := typecheck.Temp(hmapType)
			init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, hv, nil)))
			// h = &hv
			h = typecheck.NodAddr(hv)

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

				nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLE, hint, ir.NewInt(BUCKETSIZE)), nil, nil)
				nif.Likely = true

				// var bv bmap
				bv := typecheck.Temp(bmap(t))
				nif.Body.Append(ir.NewAssignStmt(base.Pos, bv, nil))

				// b = &bv
				b := typecheck.NodAddr(bv)

				// h.buckets = b
				bsym := hmapType.Field(5).Sym // hmap.buckets see reflect.go:hmap
				na := ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, h, bsym), b)
				nif.Body.Append(na)
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
			if n.Esc() == ir.EscNone {
				// Only need to initialize h.hash0 since
				// hmap h has been allocated on the stack already.
				// h.hash0 = fastrand()
				rand := mkcall("fastrand", types.Types[types.TUINT32], init)
				hashsym := hmapType.Field(4).Sym // hmap.hash0 see reflect.go:hmap
				appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ir.NewSelectorExpr(base.Pos, ir.ODOT, h, hashsym), rand))
				return typecheck.ConvNop(h, t)
			}
			// Call runtime.makehmap to allocate an
			// hmap on the heap and initialize hmap's hash0 field.
			fn := typecheck.LookupRuntime("makemap_small")
			fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem())
			return mkcall1(fn, n.Type(), init)
		}

		if n.Esc() != ir.EscNone {
			h = typecheck.NodNil()
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

		fn := typecheck.LookupRuntime(fnname)
		fn = typecheck.SubstArgTypes(fn, hmapType, t.Key(), t.Elem())
		return mkcall1(fn, n.Type(), init, typename(n.Type()), typecheck.Conv(hint, argtype), h)

	case ir.OMAKESLICE:
		n := n.(*ir.MakeExpr)
		l := n.Len
		r := n.Cap
		if r == nil {
			r = safeexpr(l, init)
			l = r
		}
		t := n.Type()
		if t.Elem().NotInHeap() {
			base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", t.Elem())
		}
		if n.Esc() == ir.EscNone {
			if why := escape.HeapAllocReason(n); why != "" {
				base.Fatalf("%v has EscNone, but %v", n, why)
			}
			// var arr [r]T
			// n = arr[:l]
			i := typecheck.IndexConst(r)
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
			nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGT, typecheck.Conv(l, types.Types[types.TUINT64]), ir.NewInt(i)), nil, nil)
			niflen := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OLT, l, ir.NewInt(0)), nil, nil)
			niflen.Body = []ir.Node{mkcall("panicmakeslicelen", nil, init)}
			nif.Body.Append(niflen, mkcall("panicmakeslicecap", nil, init))
			init.Append(typecheck.Stmt(nif))

			t = types.NewArray(t.Elem(), i) // [r]T
			var_ := typecheck.Temp(t)
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, var_, nil)) // zero temp
			r := ir.NewSliceExpr(base.Pos, ir.OSLICE, var_)             // arr[:l]
			r.SetSliceBounds(nil, l, nil)
			// The conv is necessary in case n.Type is named.
			return walkexpr(typecheck.Expr(typecheck.Conv(r, n.Type())), init)
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

		m := ir.NewSliceHeaderExpr(base.Pos, nil, nil, nil, nil)
		m.SetType(t)

		fn := typecheck.LookupRuntime(fnname)
		m.Ptr = mkcall1(fn, types.Types[types.TUNSAFEPTR], init, typename(t.Elem()), typecheck.Conv(len, argtype), typecheck.Conv(cap, argtype))
		m.Ptr.MarkNonNil()
		m.LenCap = []ir.Node{typecheck.Conv(len, types.Types[types.TINT]), typecheck.Conv(cap, types.Types[types.TINT])}
		return walkexpr(typecheck.Expr(m), init)

	case ir.OMAKESLICECOPY:
		n := n.(*ir.MakeExpr)
		if n.Esc() == ir.EscNone {
			base.Fatalf("OMAKESLICECOPY with EscNone: %v", n)
		}

		t := n.Type()
		if t.Elem().NotInHeap() {
			base.Errorf("%v can't be allocated in Go; it is incomplete (or unallocatable)", t.Elem())
		}

		length := typecheck.Conv(n.Len, types.Types[types.TINT])
		copylen := ir.NewUnaryExpr(base.Pos, ir.OLEN, n.Cap)
		copyptr := ir.NewUnaryExpr(base.Pos, ir.OSPTR, n.Cap)

		if !t.Elem().HasPointers() && n.Bounded() {
			// When len(to)==len(from) and elements have no pointers:
			// replace make+copy with runtime.mallocgc+runtime.memmove.

			// We do not check for overflow of len(to)*elem.Width here
			// since len(from) is an existing checked slice capacity
			// with same elem.Width for the from slice.
			size := ir.NewBinaryExpr(base.Pos, ir.OMUL, typecheck.Conv(length, types.Types[types.TUINTPTR]), typecheck.Conv(ir.NewInt(t.Elem().Width), types.Types[types.TUINTPTR]))

			// instantiate mallocgc(size uintptr, typ *byte, needszero bool) unsafe.Pointer
			fn := typecheck.LookupRuntime("mallocgc")
			sh := ir.NewSliceHeaderExpr(base.Pos, nil, nil, nil, nil)
			sh.Ptr = mkcall1(fn, types.Types[types.TUNSAFEPTR], init, size, typecheck.NodNil(), ir.NewBool(false))
			sh.Ptr.MarkNonNil()
			sh.LenCap = []ir.Node{length, length}
			sh.SetType(t)

			s := typecheck.Temp(t)
			r := typecheck.Stmt(ir.NewAssignStmt(base.Pos, s, sh))
			r = walkexpr(r, init)
			init.Append(r)

			// instantiate memmove(to *any, frm *any, size uintptr)
			fn = typecheck.LookupRuntime("memmove")
			fn = typecheck.SubstArgTypes(fn, t.Elem(), t.Elem())
			ncopy := mkcall1(fn, nil, init, ir.NewUnaryExpr(base.Pos, ir.OSPTR, s), copyptr, size)
			init.Append(walkexpr(typecheck.Stmt(ncopy), init))

			return s
		}
		// Replace make+copy with runtime.makeslicecopy.
		// instantiate makeslicecopy(typ *byte, tolen int, fromlen int, from unsafe.Pointer) unsafe.Pointer
		fn := typecheck.LookupRuntime("makeslicecopy")
		s := ir.NewSliceHeaderExpr(base.Pos, nil, nil, nil, nil)
		s.Ptr = mkcall1(fn, types.Types[types.TUNSAFEPTR], init, typename(t.Elem()), length, copylen, typecheck.Conv(copyptr, types.Types[types.TUNSAFEPTR]))
		s.Ptr.MarkNonNil()
		s.LenCap = []ir.Node{length, length}
		s.SetType(t)
		return walkexpr(typecheck.Expr(s), init)

	case ir.ORUNESTR:
		n := n.(*ir.ConvExpr)
		a := typecheck.NodNil()
		if n.Esc() == ir.EscNone {
			t := types.NewArray(types.Types[types.TUINT8], 4)
			a = typecheck.NodAddr(typecheck.Temp(t))
		}
		// intstring(*[4]byte, rune)
		return mkcall("intstring", n.Type(), init, a, typecheck.Conv(n.X, types.Types[types.TINT64]))

	case ir.OBYTES2STR, ir.ORUNES2STR:
		n := n.(*ir.ConvExpr)
		a := typecheck.NodNil()
		if n.Esc() == ir.EscNone {
			// Create temporary buffer for string on stack.
			t := types.NewArray(types.Types[types.TUINT8], tmpstringbufsize)
			a = typecheck.NodAddr(typecheck.Temp(t))
		}
		if n.Op() == ir.ORUNES2STR {
			// slicerunetostring(*[32]byte, []rune) string
			return mkcall("slicerunetostring", n.Type(), init, a, n.X)
		}
		// slicebytetostring(*[32]byte, ptr *byte, n int) string
		n.X = cheapexpr(n.X, init)
		ptr, len := backingArrayPtrLen(n.X)
		return mkcall("slicebytetostring", n.Type(), init, a, ptr, len)

	case ir.OBYTES2STRTMP:
		n := n.(*ir.ConvExpr)
		n.X = walkexpr(n.X, init)
		if !base.Flag.Cfg.Instrumenting {
			// Let the backend handle OBYTES2STRTMP directly
			// to avoid a function call to slicebytetostringtmp.
			return n
		}
		// slicebytetostringtmp(ptr *byte, n int) string
		n.X = cheapexpr(n.X, init)
		ptr, len := backingArrayPtrLen(n.X)
		return mkcall("slicebytetostringtmp", n.Type(), init, ptr, len)

	case ir.OSTR2BYTES:
		n := n.(*ir.ConvExpr)
		s := n.X
		if ir.IsConst(s, constant.String) {
			sc := ir.StringVal(s)

			// Allocate a [n]byte of the right size.
			t := types.NewArray(types.Types[types.TUINT8], int64(len(sc)))
			var a ir.Node
			if n.Esc() == ir.EscNone && len(sc) <= int(ir.MaxImplicitStackVarSize) {
				a = typecheck.NodAddr(typecheck.Temp(t))
			} else {
				a = callnew(t)
			}
			p := typecheck.Temp(t.PtrTo()) // *[n]byte
			init.Append(typecheck.Stmt(ir.NewAssignStmt(base.Pos, p, a)))

			// Copy from the static string data to the [n]byte.
			if len(sc) > 0 {
				as := ir.NewAssignStmt(base.Pos, ir.NewStarExpr(base.Pos, p), ir.NewStarExpr(base.Pos, typecheck.ConvNop(ir.NewUnaryExpr(base.Pos, ir.OSPTR, s), t.PtrTo())))
				appendWalkStmt(init, as)
			}

			// Slice the [n]byte to a []byte.
			slice := ir.NewSliceExpr(n.Pos(), ir.OSLICEARR, p)
			slice.SetType(n.Type())
			slice.SetTypecheck(1)
			return walkexpr(slice, init)
		}

		a := typecheck.NodNil()
		if n.Esc() == ir.EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[types.TUINT8], tmpstringbufsize)
			a = typecheck.NodAddr(typecheck.Temp(t))
		}
		// stringtoslicebyte(*32[byte], string) []byte
		return mkcall("stringtoslicebyte", n.Type(), init, a, typecheck.Conv(s, types.Types[types.TSTRING]))

	case ir.OSTR2BYTESTMP:
		// []byte(string) conversion that creates a slice
		// referring to the actual string bytes.
		// This conversion is handled later by the backend and
		// is only for use by internal compiler optimizations
		// that know that the slice won't be mutated.
		// The only such case today is:
		// for i, c := range []byte(string)
		n := n.(*ir.ConvExpr)
		n.X = walkexpr(n.X, init)
		return n

	case ir.OSTR2RUNES:
		n := n.(*ir.ConvExpr)
		a := typecheck.NodNil()
		if n.Esc() == ir.EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[types.TINT32], tmpstringbufsize)
			a = typecheck.NodAddr(typecheck.Temp(t))
		}
		// stringtoslicerune(*[32]rune, string) []rune
		return mkcall("stringtoslicerune", n.Type(), init, a, typecheck.Conv(n.X, types.Types[types.TSTRING]))

	case ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT, ir.OSTRUCTLIT, ir.OPTRLIT:
		if isStaticCompositeLiteral(n) && !canSSAType(n.Type()) {
			n := n.(*ir.CompLitExpr) // not OPTRLIT
			// n can be directly represented in the read-only data section.
			// Make direct reference to the static data. See issue 12841.
			vstat := readonlystaticname(n.Type())
			fixedlit(inInitFunction, initKindStatic, n, vstat, init)
			return typecheck.Expr(vstat)
		}
		var_ := typecheck.Temp(n.Type())
		anylit(n, var_, init)
		return var_

	case ir.OSEND:
		n := n.(*ir.SendStmt)
		n1 := n.Value
		n1 = typecheck.AssignConv(n1, n.Chan.Type().Elem(), "chan send")
		n1 = walkexpr(n1, init)
		n1 = typecheck.NodAddr(n1)
		return mkcall1(chanfn("chansend1", 2, n.Chan.Type()), nil, init, n.Chan, n1)

	case ir.OCLOSURE:
		return walkclosure(n.(*ir.ClosureExpr), init)

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
func markUsedIfaceMethod(n *ir.CallExpr) {
	dot := n.X.(*ir.SelectorExpr)
	ityp := dot.X.Type()
	tsym := typenamesym(ityp).Linksym()
	r := obj.Addrel(ir.CurFunc.LSym)
	r.Sym = tsym
	// dot.Xoffset is the method index * Widthptr (the offset of code pointer
	// in itab).
	midx := dot.Offset / int64(types.PtrSize)
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
func reduceSlice(n *ir.SliceExpr) ir.Node {
	low, high, max := n.SliceBounds()
	if high != nil && high.Op() == ir.OLEN && ir.SameSafeExpr(n.X, high.(*ir.UnaryExpr).X) {
		// Reduce x[i:len(x)] to x[i:].
		high = nil
	}
	n.SetSliceBounds(low, high, max)
	if (n.Op() == ir.OSLICE || n.Op() == ir.OSLICESTR) && low == nil && high == nil {
		// Reduce x[:] to x.
		if base.Debug.Slice > 0 {
			base.Warn("slice: omit slice operation")
		}
		return n.X
	}
	return n
}

func ascompatee1(l ir.Node, r ir.Node, init *ir.Nodes) *ir.AssignStmt {
	// convas will turn map assigns into function calls,
	// making it impossible for reorder3 to work.
	n := ir.NewAssignStmt(base.Pos, l, r)

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

	var nn []*ir.AssignStmt
	i := 0
	for ; i < len(nl); i++ {
		if i >= len(nr) {
			break
		}
		// Do not generate 'x = x' during return. See issue 4014.
		if op == ir.ORETURN && ir.SameSafeExpr(nl[i], nr[i]) {
			continue
		}
		nn = append(nn, ascompatee1(nl[i], nr[i], init))
	}

	// cannot happen: caller checked that lists had same length
	if i < len(nl) || i < len(nr) {
		var nln, nrn ir.Nodes
		nln.Set(nl)
		nrn.Set(nr)
		base.Fatalf("error in shape across %+v %v %+v / %d %d [%s]", nln, op, nrn, len(nl), len(nr), ir.FuncName(ir.CurFunc))
	}
	return reorder3(nn)
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
	if len(nl) != nr.NumFields() {
		base.Fatalf("ascompatet: assignment count mismatch: %d = %d", len(nl), nr.NumFields())
	}

	var nn, mm ir.Nodes
	for i, l := range nl {
		if ir.IsBlank(l) {
			continue
		}
		r := nr.Field(i)

		// Any assignment to an lvalue that might cause a function call must be
		// deferred until all the returned values have been read.
		if fncall(l, r.Type) {
			tmp := ir.Node(typecheck.Temp(r.Type))
			tmp = typecheck.Expr(tmp)
			a := convas(ir.NewAssignStmt(base.Pos, l, tmp), &mm)
			mm.Append(a)
			l = tmp
		}

		res := ir.NewResultExpr(base.Pos, nil, types.BADWIDTH)
		res.Offset = base.Ctxt.FixedFrameSize() + r.Offset
		res.SetType(r.Type)
		res.SetTypecheck(1)

		a := convas(ir.NewAssignStmt(base.Pos, l, res), &nn)
		updateHasCall(a)
		if a.HasCall() {
			ir.Dump("ascompatet ucount", a)
			base.Fatalf("ascompatet: too many function calls evaluating parameters")
		}

		nn.Append(a)
	}
	return append(nn, mm...)
}

func walkCall(n *ir.CallExpr, init *ir.Nodes) {
	if len(n.Rargs) != 0 {
		return // already walked
	}

	params := n.X.Type().Params()
	args := n.Args

	n.X = walkexpr(n.X, init)
	walkexprlist(args, init)

	// If this is a method call, add the receiver at the beginning of the args.
	if n.Op() == ir.OCALLMETH {
		withRecv := make([]ir.Node, len(args)+1)
		dot := n.X.(*ir.SelectorExpr)
		withRecv[0] = dot.X
		dot.X = nil
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
				t = n.X.Type().Recv().Type
			} else {
				t = params.Field(i - 1).Type
			}
		} else {
			t = params.Field(i).Type
		}
		if base.Flag.Cfg.Instrumenting || fncall(arg, t) {
			// make assignment of fncall to tempAt
			tmp := typecheck.Temp(t)
			a := convas(ir.NewAssignStmt(base.Pos, tmp, arg), init)
			tempAssigns = append(tempAssigns, a)
			// replace arg with temp
			args[i] = tmp
		}
	}

	n.Args.Set(tempAssigns)
	n.Rargs.Set(args)
}

// generate code for print
func walkprint(nn *ir.CallExpr, init *ir.Nodes) ir.Node {
	// Hoist all the argument evaluation up before the lock.
	walkexprlistcheap(nn.Args, init)

	// For println, add " " between elements and "\n" at the end.
	if nn.Op() == ir.OPRINTN {
		s := nn.Args
		t := make([]ir.Node, 0, len(s)*2)
		for i, n := range s {
			if i != 0 {
				t = append(t, ir.NewString(" "))
			}
			t = append(t, n)
		}
		t = append(t, ir.NewString("\n"))
		nn.Args.Set(t)
	}

	// Collapse runs of constant strings.
	s := nn.Args
	t := make([]ir.Node, 0, len(s))
	for i := 0; i < len(s); {
		var strs []string
		for i < len(s) && ir.IsConst(s[i], constant.String) {
			strs = append(strs, ir.StringVal(s[i]))
			i++
		}
		if len(strs) > 0 {
			t = append(t, ir.NewString(strings.Join(strs, "")))
		}
		if i < len(s) {
			t = append(t, s[i])
			i++
		}
	}
	nn.Args.Set(t)

	calls := []ir.Node{mkcall("printlock", nil, init)}
	for i, n := range nn.Args {
		if n.Op() == ir.OLITERAL {
			if n.Type() == types.UntypedRune {
				n = typecheck.DefaultLit(n, types.RuneType)
			}

			switch n.Val().Kind() {
			case constant.Int:
				n = typecheck.DefaultLit(n, types.Types[types.TINT64])

			case constant.Float:
				n = typecheck.DefaultLit(n, types.Types[types.TFLOAT64])
			}
		}

		if n.Op() != ir.OLITERAL && n.Type() != nil && n.Type().Kind() == types.TIDEAL {
			n = typecheck.DefaultLit(n, types.Types[types.TINT64])
		}
		n = typecheck.DefaultLit(n, nil)
		nn.Args[i] = n
		if n.Type() == nil || n.Type().Kind() == types.TFORW {
			continue
		}

		var on *ir.Name
		switch n.Type().Kind() {
		case types.TINTER:
			if n.Type().IsEmptyInterface() {
				on = typecheck.LookupRuntime("printeface")
			} else {
				on = typecheck.LookupRuntime("printiface")
			}
			on = typecheck.SubstArgTypes(on, n.Type()) // any-1
		case types.TPTR:
			if n.Type().Elem().NotInHeap() {
				on = typecheck.LookupRuntime("printuintptr")
				n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
				n.SetType(types.Types[types.TUNSAFEPTR])
				n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
				n.SetType(types.Types[types.TUINTPTR])
				break
			}
			fallthrough
		case types.TCHAN, types.TMAP, types.TFUNC, types.TUNSAFEPTR:
			on = typecheck.LookupRuntime("printpointer")
			on = typecheck.SubstArgTypes(on, n.Type()) // any-1
		case types.TSLICE:
			on = typecheck.LookupRuntime("printslice")
			on = typecheck.SubstArgTypes(on, n.Type()) // any-1
		case types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64, types.TUINTPTR:
			if types.IsRuntimePkg(n.Type().Sym().Pkg) && n.Type().Sym().Name == "hex" {
				on = typecheck.LookupRuntime("printhex")
			} else {
				on = typecheck.LookupRuntime("printuint")
			}
		case types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64:
			on = typecheck.LookupRuntime("printint")
		case types.TFLOAT32, types.TFLOAT64:
			on = typecheck.LookupRuntime("printfloat")
		case types.TCOMPLEX64, types.TCOMPLEX128:
			on = typecheck.LookupRuntime("printcomplex")
		case types.TBOOL:
			on = typecheck.LookupRuntime("printbool")
		case types.TSTRING:
			cs := ""
			if ir.IsConst(n, constant.String) {
				cs = ir.StringVal(n)
			}
			switch cs {
			case " ":
				on = typecheck.LookupRuntime("printsp")
			case "\n":
				on = typecheck.LookupRuntime("printnl")
			default:
				on = typecheck.LookupRuntime("printstring")
			}
		default:
			badtype(ir.OPRINT, n.Type(), nil)
			continue
		}

		r := ir.NewCallExpr(base.Pos, ir.OCALL, on, nil)
		if params := on.Type().Params().FieldSlice(); len(params) > 0 {
			t := params[0].Type
			if !types.Identical(t, n.Type()) {
				n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
				n.SetType(t)
			}
			r.Args.Append(n)
		}
		calls = append(calls, r)
	}

	calls = append(calls, mkcall("printunlock", nil, init))

	typecheck.Stmts(calls)
	walkexprlist(calls, init)

	r := ir.NewBlockStmt(base.Pos, nil)
	r.List.Set(calls)
	return walkstmt(typecheck.Stmt(r))
}

func callnew(t *types.Type) ir.Node {
	types.CalcSize(t)
	n := ir.NewUnaryExpr(base.Pos, ir.ONEWOBJ, typename(t))
	n.SetType(types.NewPtr(t))
	n.SetTypecheck(1)
	n.MarkNonNil()
	return n
}

func convas(n *ir.AssignStmt, init *ir.Nodes) *ir.AssignStmt {
	if n.Op() != ir.OAS {
		base.Fatalf("convas: not OAS %v", n.Op())
	}
	defer updateHasCall(n)

	n.SetTypecheck(1)

	if n.X == nil || n.Y == nil {
		return n
	}

	lt := n.X.Type()
	rt := n.Y.Type()
	if lt == nil || rt == nil {
		return n
	}

	if ir.IsBlank(n.X) {
		n.Y = typecheck.DefaultLit(n.Y, nil)
		return n
	}

	if !types.Identical(lt, rt) {
		n.Y = typecheck.AssignConv(n.Y, lt, "assignment")
		n.Y = walkexpr(n.Y, init)
	}
	types.CalcSize(n.Y.Type())

	return n
}

// reorder3
// from ascompatee
//	a,b = c,d
// simultaneous assignment. there cannot
// be later use of an earlier lvalue.
//
// function calls have been removed.
func reorder3(all []*ir.AssignStmt) []ir.Node {
	// If a needed expression may be affected by an
	// earlier assignment, make an early copy of that
	// expression and use the copy instead.
	var early []ir.Node

	var mapinit ir.Nodes
	for i, n := range all {
		l := n.X

		// Save subexpressions needed on left side.
		// Drill through non-dereferences.
		for {
			switch ll := l; ll.Op() {
			case ir.ODOT:
				ll := ll.(*ir.SelectorExpr)
				l = ll.X
				continue
			case ir.OPAREN:
				ll := ll.(*ir.ParenExpr)
				l = ll.X
				continue
			case ir.OINDEX:
				ll := ll.(*ir.IndexExpr)
				if ll.X.Type().IsArray() {
					ll.Index = reorder3save(ll.Index, all, i, &early)
					l = ll.X
					continue
				}
			}
			break
		}

		switch l.Op() {
		default:
			base.Fatalf("reorder3 unexpected lvalue %v", l.Op())

		case ir.ONAME:
			break

		case ir.OINDEX, ir.OINDEXMAP:
			l := l.(*ir.IndexExpr)
			l.X = reorder3save(l.X, all, i, &early)
			l.Index = reorder3save(l.Index, all, i, &early)
			if l.Op() == ir.OINDEXMAP {
				all[i] = convas(all[i], &mapinit)
			}

		case ir.ODEREF:
			l := l.(*ir.StarExpr)
			l.X = reorder3save(l.X, all, i, &early)
		case ir.ODOTPTR:
			l := l.(*ir.SelectorExpr)
			l.X = reorder3save(l.X, all, i, &early)
		}

		// Save expression on right side.
		all[i].Y = reorder3save(all[i].Y, all, i, &early)
	}

	early = append(mapinit, early...)
	for _, as := range all {
		early = append(early, as)
	}
	return early
}

// if the evaluation of *np would be affected by the
// assignments in all up to but not including the ith assignment,
// copy into a temporary during *early and
// replace *np with that temp.
// The result of reorder3save MUST be assigned back to n, e.g.
// 	n.Left = reorder3save(n.Left, all, i, early)
func reorder3save(n ir.Node, all []*ir.AssignStmt, i int, early *[]ir.Node) ir.Node {
	if !aliased(n, all[:i]) {
		return n
	}

	q := ir.Node(typecheck.Temp(n.Type()))
	as := typecheck.Stmt(ir.NewAssignStmt(base.Pos, q, n))
	*early = append(*early, as)
	return q
}

// Is it possible that the computation of r might be
// affected by assignments in all?
func aliased(r ir.Node, all []*ir.AssignStmt) bool {
	if r == nil {
		return false
	}

	// Treat all fields of a struct as referring to the whole struct.
	// We could do better but we would have to keep track of the fields.
	for r.Op() == ir.ODOT {
		r = r.(*ir.SelectorExpr).X
	}

	// Look for obvious aliasing: a variable being assigned
	// during the all list and appearing in n.
	// Also record whether there are any writes to addressable
	// memory (either main memory or variables whose addresses
	// have been taken).
	memwrite := false
	for _, as := range all {
		// We can ignore assignments to blank.
		if ir.IsBlank(as.X) {
			continue
		}

		lv := ir.OuterValue(as.X)
		if lv.Op() != ir.ONAME {
			memwrite = true
			continue
		}
		l := lv.(*ir.Name)

		switch l.Class_ {
		default:
			base.Fatalf("unexpected class: %v, %v", l, l.Class_)

		case ir.PAUTOHEAP, ir.PEXTERN:
			memwrite = true
			continue

		case ir.PAUTO, ir.PPARAM, ir.PPARAMOUT:
			if l.Name().Addrtaken() {
				memwrite = true
				continue
			}

			if refersToName(l, r) {
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

	// If r does not refer to any variables whose addresses have been taken,
	// then the only possible writes to r would be directly to the variables,
	// and we checked those above, so no aliasing problems.
	if !anyAddrTaken(r) {
		return false
	}

	// Otherwise, both the writes and r refer to computed memory addresses.
	// Assume that they might conflict.
	return true
}

// anyAddrTaken reports whether the evaluation n,
// which appears on the left side of an assignment,
// may refer to variables whose addresses have been taken.
func anyAddrTaken(n ir.Node) bool {
	return ir.Any(n, func(n ir.Node) bool {
		switch n.Op() {
		case ir.ONAME:
			n := n.(*ir.Name)
			return n.Class_ == ir.PEXTERN || n.Class_ == ir.PAUTOHEAP || n.Name().Addrtaken()

		case ir.ODOT: // but not ODOTPTR - should have been handled in aliased.
			base.Fatalf("anyAddrTaken unexpected ODOT")

		case ir.OADD,
			ir.OAND,
			ir.OANDAND,
			ir.OANDNOT,
			ir.OBITNOT,
			ir.OCONV,
			ir.OCONVIFACE,
			ir.OCONVNOP,
			ir.ODIV,
			ir.ODOTTYPE,
			ir.OLITERAL,
			ir.OLSH,
			ir.OMOD,
			ir.OMUL,
			ir.ONEG,
			ir.ONIL,
			ir.OOR,
			ir.OOROR,
			ir.OPAREN,
			ir.OPLUS,
			ir.ORSH,
			ir.OSUB,
			ir.OXOR:
			return false
		}
		// Be conservative.
		return true
	})
}

// refersToName reports whether r refers to name.
func refersToName(name *ir.Name, r ir.Node) bool {
	return ir.Any(r, func(r ir.Node) bool {
		return r.Op() == ir.ONAME && r == name
	})
}

var stop = errors.New("stop")

// refersToCommonName reports whether any name
// appears in common between l and r.
// This is called from sinit.go.
func refersToCommonName(l ir.Node, r ir.Node) bool {
	if l == nil || r == nil {
		return false
	}

	// This could be written elegantly as a Find nested inside a Find:
	//
	//	found := ir.Find(l, func(l ir.Node) interface{} {
	//		if l.Op() == ir.ONAME {
	//			return ir.Find(r, func(r ir.Node) interface{} {
	//				if r.Op() == ir.ONAME && l.Name() == r.Name() {
	//					return r
	//				}
	//				return nil
	//			})
	//		}
	//		return nil
	//	})
	//	return found != nil
	//
	// But that would allocate a new closure for the inner Find
	// for each name found on the left side.
	// It may not matter at all, but the below way of writing it
	// only allocates two closures, not O(|L|) closures.

	var doL, doR func(ir.Node) error
	var targetL *ir.Name
	doR = func(r ir.Node) error {
		if r.Op() == ir.ONAME && r.Name() == targetL {
			return stop
		}
		return ir.DoChildren(r, doR)
	}
	doL = func(l ir.Node) error {
		if l.Op() == ir.ONAME {
			l := l.(*ir.Name)
			targetL = l.Name()
			if doR(r) == stop {
				return stop
			}
		}
		return ir.DoChildren(l, doL)
	}
	return doL(l) == stop
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
			nn = append(nn, walkstmt(ir.NewDecl(base.Pos, ir.ODCL, v)))
			if stackcopy.Class_ == ir.PPARAM {
				nn = append(nn, walkstmt(typecheck.Stmt(ir.NewAssignStmt(base.Pos, v, stackcopy))))
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
	for _, f := range ir.CurFunc.Type().Results().Fields().Slice() {
		v := ir.AsNode(f.Nname)
		if v != nil && v.Name().Heapaddr != nil {
			// The local which points to the return value is the
			// thing that needs zeroing. This is already handled
			// by a Needzero annotation in plive.go:livenessepilogue.
			continue
		}
		if ir.IsParamHeapCopy(v) {
			// TODO(josharian/khr): Investigate whether we can switch to "continue" here,
			// and document more in either case.
			// In the review of CL 114797, Keith wrote (roughly):
			// I don't think the zeroing below matters.
			// The stack return value will never be marked as live anywhere in the function.
			// It is not written to until deferreturn returns.
			v = v.Name().Stackcopy
		}
		// Zero the stack location containing f.
		ir.CurFunc.Enter.Append(ir.NewAssignStmt(ir.CurFunc.Pos(), v, nil))
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
		if stackcopy := v.Name().Stackcopy; stackcopy != nil && stackcopy.Class_ == ir.PPARAMOUT {
			nn = append(nn, walkstmt(typecheck.Stmt(ir.NewAssignStmt(base.Pos, stackcopy, v))))
		}
	}

	return nn
}

// heapmoves generates code to handle migrating heap-escaped parameters
// between the stack and the heap. The generated code is added to Curfn's
// Enter and Exit lists.
func heapmoves() {
	lno := base.Pos
	base.Pos = ir.CurFunc.Pos()
	nn := paramstoheap(ir.CurFunc.Type().Recvs())
	nn = append(nn, paramstoheap(ir.CurFunc.Type().Params())...)
	nn = append(nn, paramstoheap(ir.CurFunc.Type().Results())...)
	ir.CurFunc.Enter.Append(nn...)
	base.Pos = ir.CurFunc.Endlineno
	ir.CurFunc.Exit.Append(returnsfromheap(ir.CurFunc.Type().Results())...)
	base.Pos = lno
}

func vmkcall(fn ir.Node, t *types.Type, init *ir.Nodes, va []ir.Node) *ir.CallExpr {
	if fn.Type() == nil || fn.Type().Kind() != types.TFUNC {
		base.Fatalf("mkcall %v %v", fn, fn.Type())
	}

	n := fn.Type().NumParams()
	if n != len(va) {
		base.Fatalf("vmkcall %v needs %v args got %v", fn, n, len(va))
	}

	call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, va)
	typecheck.Call(call)
	call.SetType(t)
	return walkexpr(call, init).(*ir.CallExpr)
}

func mkcall(name string, t *types.Type, init *ir.Nodes, args ...ir.Node) *ir.CallExpr {
	return vmkcall(typecheck.LookupRuntime(name), t, init, args)
}

func mkcall1(fn ir.Node, t *types.Type, init *ir.Nodes, args ...ir.Node) *ir.CallExpr {
	return vmkcall(fn, t, init, args)
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
		n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
		n.SetType(types.Types[types.TUINT8])
		n.SetTypecheck(1)
	}
	n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
	n.SetType(types.Types[types.TINT])
	n.SetTypecheck(1)
	return n
}

func chanfn(name string, n int, t *types.Type) ir.Node {
	if !t.IsChan() {
		base.Fatalf("chanfn %v", t)
	}
	fn := typecheck.LookupRuntime(name)
	switch n {
	default:
		base.Fatalf("chanfn %d", n)
	case 1:
		fn = typecheck.SubstArgTypes(fn, t.Elem())
	case 2:
		fn = typecheck.SubstArgTypes(fn, t.Elem(), t.Elem())
	}
	return fn
}

func mapfn(name string, t *types.Type) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	fn := typecheck.LookupRuntime(name)
	fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem(), t.Key(), t.Elem())
	return fn
}

func mapfndel(name string, t *types.Type) ir.Node {
	if !t.IsMap() {
		base.Fatalf("mapfn %v", t)
	}
	fn := typecheck.LookupRuntime(name)
	fn = typecheck.SubstArgTypes(fn, t.Key(), t.Elem(), t.Key())
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
	case types.AMEM32:
		if !t.Key().HasPointers() {
			return mapfast32
		}
		if types.PtrSize == 4 {
			return mapfast32ptr
		}
		base.Fatalf("small pointer %v", t.Key())
	case types.AMEM64:
		if !t.Key().HasPointers() {
			return mapfast64
		}
		if types.PtrSize == 8 {
			return mapfast64ptr
		}
		// Two-word object, at least one of which is a pointer.
		// Use the slow path.
	case types.ASTRING:
		return mapfaststr
	}
	return mapslow
}

func writebarrierfn(name string, l *types.Type, r *types.Type) ir.Node {
	fn := typecheck.LookupRuntime(name)
	fn = typecheck.SubstArgTypes(fn, l, r)
	return fn
}

func addstr(n *ir.AddStringExpr, init *ir.Nodes) ir.Node {
	c := len(n.List)

	if c < 2 {
		base.Fatalf("addstr count %d too small", c)
	}

	buf := typecheck.NodNil()
	if n.Esc() == ir.EscNone {
		sz := int64(0)
		for _, n1 := range n.List {
			if n1.Op() == ir.OLITERAL {
				sz += int64(len(ir.StringVal(n1)))
			}
		}

		// Don't allocate the buffer if the result won't fit.
		if sz < tmpstringbufsize {
			// Create temporary buffer for result string on stack.
			t := types.NewArray(types.Types[types.TUINT8], tmpstringbufsize)
			buf = typecheck.NodAddr(typecheck.Temp(t))
		}
	}

	// build list of string arguments
	args := []ir.Node{buf}
	for _, n2 := range n.List {
		args = append(args, typecheck.Conv(n2, types.Types[types.TSTRING]))
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
		// args[1:] to skip buf arg
		slice := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(t), args[1:])
		slice.Prealloc = n.Prealloc
		args = []ir.Node{buf, slice}
		slice.SetEsc(ir.EscNone)
	}

	cat := typecheck.LookupRuntime(fn)
	r := ir.NewCallExpr(base.Pos, ir.OCALL, cat, nil)
	r.Args.Set(args)
	r1 := typecheck.Expr(r)
	r1 = walkexpr(r1, init)
	r1.SetType(n.Type())

	return r1
}

func walkAppendArgs(n *ir.CallExpr, init *ir.Nodes) {
	walkexprlistsafe(n.Args, init)

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	ls := n.Args
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
func appendslice(n *ir.CallExpr, init *ir.Nodes) ir.Node {
	walkAppendArgs(n, init)

	l1 := n.Args[0]
	l2 := n.Args[1]
	l2 = cheapexpr(l2, init)
	n.Args[1] = l2

	var nodes ir.Nodes

	// var s []T
	s := typecheck.Temp(l1.Type())
	nodes.Append(ir.NewAssignStmt(base.Pos, s, l1)) // s = l1

	elemtype := s.Type().Elem()

	// n := len(s) + len(l2)
	nn := typecheck.Temp(types.Types[types.TINT])
	nodes.Append(ir.NewAssignStmt(base.Pos, nn, ir.NewBinaryExpr(base.Pos, ir.OADD, ir.NewUnaryExpr(base.Pos, ir.OLEN, s), ir.NewUnaryExpr(base.Pos, ir.OLEN, l2))))

	// if uint(n) > uint(cap(s))
	nif := ir.NewIfStmt(base.Pos, nil, nil, nil)
	nuint := typecheck.Conv(nn, types.Types[types.TUINT])
	scapuint := typecheck.Conv(ir.NewUnaryExpr(base.Pos, ir.OCAP, s), types.Types[types.TUINT])
	nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OGT, nuint, scapuint)

	// instantiate growslice(typ *type, []any, int) []any
	fn := typecheck.LookupRuntime("growslice")
	fn = typecheck.SubstArgTypes(fn, elemtype, elemtype)

	// s = growslice(T, s, n)
	nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, s, mkcall1(fn, s.Type(), nif.PtrInit(), typename(elemtype), s, nn))}
	nodes.Append(nif)

	// s = s[:n]
	nt := ir.NewSliceExpr(base.Pos, ir.OSLICE, s)
	nt.SetSliceBounds(nil, nn, nil)
	nt.SetBounded(true)
	nodes.Append(ir.NewAssignStmt(base.Pos, s, nt))

	var ncopy ir.Node
	if elemtype.HasPointers() {
		// copy(s[len(l1):], l2)
		slice := ir.NewSliceExpr(base.Pos, ir.OSLICE, s)
		slice.SetType(s.Type())
		slice.SetSliceBounds(ir.NewUnaryExpr(base.Pos, ir.OLEN, l1), nil, nil)

		ir.CurFunc.SetWBPos(n.Pos())

		// instantiate typedslicecopy(typ *type, dstPtr *any, dstLen int, srcPtr *any, srcLen int) int
		fn := typecheck.LookupRuntime("typedslicecopy")
		fn = typecheck.SubstArgTypes(fn, l1.Type().Elem(), l2.Type().Elem())
		ptr1, len1 := backingArrayPtrLen(cheapexpr(slice, &nodes))
		ptr2, len2 := backingArrayPtrLen(l2)
		ncopy = mkcall1(fn, types.Types[types.TINT], &nodes, typename(elemtype), ptr1, len1, ptr2, len2)
	} else if base.Flag.Cfg.Instrumenting && !base.Flag.CompilingRuntime {
		// rely on runtime to instrument:
		//  copy(s[len(l1):], l2)
		// l2 can be a slice or string.
		slice := ir.NewSliceExpr(base.Pos, ir.OSLICE, s)
		slice.SetType(s.Type())
		slice.SetSliceBounds(ir.NewUnaryExpr(base.Pos, ir.OLEN, l1), nil, nil)

		ptr1, len1 := backingArrayPtrLen(cheapexpr(slice, &nodes))
		ptr2, len2 := backingArrayPtrLen(l2)

		fn := typecheck.LookupRuntime("slicecopy")
		fn = typecheck.SubstArgTypes(fn, ptr1.Type().Elem(), ptr2.Type().Elem())
		ncopy = mkcall1(fn, types.Types[types.TINT], &nodes, ptr1, len1, ptr2, len2, ir.NewInt(elemtype.Width))
	} else {
		// memmove(&s[len(l1)], &l2[0], len(l2)*sizeof(T))
		ix := ir.NewIndexExpr(base.Pos, s, ir.NewUnaryExpr(base.Pos, ir.OLEN, l1))
		ix.SetBounded(true)
		addr := typecheck.NodAddr(ix)

		sptr := ir.NewUnaryExpr(base.Pos, ir.OSPTR, l2)

		nwid := cheapexpr(typecheck.Conv(ir.NewUnaryExpr(base.Pos, ir.OLEN, l2), types.Types[types.TUINTPTR]), &nodes)
		nwid = ir.NewBinaryExpr(base.Pos, ir.OMUL, nwid, ir.NewInt(elemtype.Width))

		// instantiate func memmove(to *any, frm *any, length uintptr)
		fn := typecheck.LookupRuntime("memmove")
		fn = typecheck.SubstArgTypes(fn, elemtype, elemtype)
		ncopy = mkcall1(fn, nil, &nodes, addr, sptr, nwid)
	}
	ln := append(nodes, ncopy)

	typecheck.Stmts(ln)
	walkstmtlist(ln)
	init.Append(ln...)
	return s
}

// isAppendOfMake reports whether n is of the form append(x , make([]T, y)...).
// isAppendOfMake assumes n has already been typechecked.
func isAppendOfMake(n ir.Node) bool {
	if base.Flag.N != 0 || base.Flag.Cfg.Instrumenting {
		return false
	}

	if n.Typecheck() == 0 {
		base.Fatalf("missing typecheck: %+v", n)
	}

	if n.Op() != ir.OAPPEND {
		return false
	}
	call := n.(*ir.CallExpr)
	if !call.IsDDD || len(call.Args) != 2 || call.Args[1].Op() != ir.OMAKESLICE {
		return false
	}

	mk := call.Args[1].(*ir.MakeExpr)
	if mk.Cap != nil {
		return false
	}

	// y must be either an integer constant or the largest possible positive value
	// of variable y needs to fit into an uint.

	// typecheck made sure that constant arguments to make are not negative and fit into an int.

	// The care of overflow of the len argument to make will be handled by an explicit check of int(len) < 0 during runtime.
	y := mk.Len
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
func extendslice(n *ir.CallExpr, init *ir.Nodes) ir.Node {
	// isAppendOfMake made sure all possible positive values of l2 fit into an uint.
	// The case of l2 overflow when converting from e.g. uint to int is handled by an explicit
	// check of l2 < 0 at runtime which is generated below.
	l2 := typecheck.Conv(n.Args[1].(*ir.MakeExpr).Len, types.Types[types.TINT])
	l2 = typecheck.Expr(l2)
	n.Args[1] = l2 // walkAppendArgs expects l2 in n.List.Second().

	walkAppendArgs(n, init)

	l1 := n.Args[0]
	l2 = n.Args[1] // re-read l2, as it may have been updated by walkAppendArgs

	var nodes []ir.Node

	// if l2 >= 0 (likely happens), do nothing
	nifneg := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGE, l2, ir.NewInt(0)), nil, nil)
	nifneg.Likely = true

	// else panicmakeslicelen()
	nifneg.Else = []ir.Node{mkcall("panicmakeslicelen", nil, init)}
	nodes = append(nodes, nifneg)

	// s := l1
	s := typecheck.Temp(l1.Type())
	nodes = append(nodes, ir.NewAssignStmt(base.Pos, s, l1))

	elemtype := s.Type().Elem()

	// n := len(s) + l2
	nn := typecheck.Temp(types.Types[types.TINT])
	nodes = append(nodes, ir.NewAssignStmt(base.Pos, nn, ir.NewBinaryExpr(base.Pos, ir.OADD, ir.NewUnaryExpr(base.Pos, ir.OLEN, s), l2)))

	// if uint(n) > uint(cap(s))
	nuint := typecheck.Conv(nn, types.Types[types.TUINT])
	capuint := typecheck.Conv(ir.NewUnaryExpr(base.Pos, ir.OCAP, s), types.Types[types.TUINT])
	nif := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OGT, nuint, capuint), nil, nil)

	// instantiate growslice(typ *type, old []any, newcap int) []any
	fn := typecheck.LookupRuntime("growslice")
	fn = typecheck.SubstArgTypes(fn, elemtype, elemtype)

	// s = growslice(T, s, n)
	nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, s, mkcall1(fn, s.Type(), nif.PtrInit(), typename(elemtype), s, nn))}
	nodes = append(nodes, nif)

	// s = s[:n]
	nt := ir.NewSliceExpr(base.Pos, ir.OSLICE, s)
	nt.SetSliceBounds(nil, nn, nil)
	nt.SetBounded(true)
	nodes = append(nodes, ir.NewAssignStmt(base.Pos, s, nt))

	// lptr := &l1[0]
	l1ptr := typecheck.Temp(l1.Type().Elem().PtrTo())
	tmp := ir.NewUnaryExpr(base.Pos, ir.OSPTR, l1)
	nodes = append(nodes, ir.NewAssignStmt(base.Pos, l1ptr, tmp))

	// sptr := &s[0]
	sptr := typecheck.Temp(elemtype.PtrTo())
	tmp = ir.NewUnaryExpr(base.Pos, ir.OSPTR, s)
	nodes = append(nodes, ir.NewAssignStmt(base.Pos, sptr, tmp))

	// hp := &s[len(l1)]
	ix := ir.NewIndexExpr(base.Pos, s, ir.NewUnaryExpr(base.Pos, ir.OLEN, l1))
	ix.SetBounded(true)
	hp := typecheck.ConvNop(typecheck.NodAddr(ix), types.Types[types.TUNSAFEPTR])

	// hn := l2 * sizeof(elem(s))
	hn := typecheck.Conv(ir.NewBinaryExpr(base.Pos, ir.OMUL, l2, ir.NewInt(elemtype.Width)), types.Types[types.TUINTPTR])

	clrname := "memclrNoHeapPointers"
	hasPointers := elemtype.HasPointers()
	if hasPointers {
		clrname = "memclrHasPointers"
		ir.CurFunc.SetWBPos(n.Pos())
	}

	var clr ir.Nodes
	clrfn := mkcall(clrname, nil, &clr, hp, hn)
	clr.Append(clrfn)

	if hasPointers {
		// if l1ptr == sptr
		nifclr := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.OEQ, l1ptr, sptr), nil, nil)
		nifclr.Body = clr
		nodes = append(nodes, nifclr)
	} else {
		nodes = append(nodes, clr...)
	}

	typecheck.Stmts(nodes)
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
func walkappend(n *ir.CallExpr, init *ir.Nodes, dst ir.Node) ir.Node {
	if !ir.SameSafeExpr(dst, n.Args[0]) {
		n.Args[0] = safeexpr(n.Args[0], init)
		n.Args[0] = walkexpr(n.Args[0], init)
	}
	walkexprlistsafe(n.Args[1:], init)

	nsrc := n.Args[0]

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	// Using cheapexpr also makes sure that the evaluation
	// of all arguments (and especially any panics) happen
	// before we begin to modify the slice in a visible way.
	ls := n.Args[1:]
	for i, n := range ls {
		n = cheapexpr(n, init)
		if !types.Identical(n.Type(), nsrc.Type().Elem()) {
			n = typecheck.AssignConv(n, nsrc.Type().Elem(), "append")
			n = walkexpr(n, init)
		}
		ls[i] = n
	}

	argc := len(n.Args) - 1
	if argc < 1 {
		return nsrc
	}

	// General case, with no function calls left as arguments.
	// Leave for gen, except that instrumentation requires old form.
	if !base.Flag.Cfg.Instrumenting || base.Flag.CompilingRuntime {
		return n
	}

	var l []ir.Node

	ns := typecheck.Temp(nsrc.Type())
	l = append(l, ir.NewAssignStmt(base.Pos, ns, nsrc)) // s = src

	na := ir.NewInt(int64(argc))                 // const argc
	nif := ir.NewIfStmt(base.Pos, nil, nil, nil) // if cap(s) - len(s) < argc
	nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OLT, ir.NewBinaryExpr(base.Pos, ir.OSUB, ir.NewUnaryExpr(base.Pos, ir.OCAP, ns), ir.NewUnaryExpr(base.Pos, ir.OLEN, ns)), na)

	fn := typecheck.LookupRuntime("growslice") //   growslice(<type>, old []T, mincap int) (ret []T)
	fn = typecheck.SubstArgTypes(fn, ns.Type().Elem(), ns.Type().Elem())

	nif.Body = []ir.Node{ir.NewAssignStmt(base.Pos, ns, mkcall1(fn, ns.Type(), nif.PtrInit(), typename(ns.Type().Elem()), ns,
		ir.NewBinaryExpr(base.Pos, ir.OADD, ir.NewUnaryExpr(base.Pos, ir.OLEN, ns), na)))}

	l = append(l, nif)

	nn := typecheck.Temp(types.Types[types.TINT])
	l = append(l, ir.NewAssignStmt(base.Pos, nn, ir.NewUnaryExpr(base.Pos, ir.OLEN, ns))) // n = len(s)

	slice := ir.NewSliceExpr(base.Pos, ir.OSLICE, ns) // ...s[:n+argc]
	slice.SetSliceBounds(nil, ir.NewBinaryExpr(base.Pos, ir.OADD, nn, na), nil)
	slice.SetBounded(true)
	l = append(l, ir.NewAssignStmt(base.Pos, ns, slice)) // s = s[:n+argc]

	ls = n.Args[1:]
	for i, n := range ls {
		ix := ir.NewIndexExpr(base.Pos, ns, nn) // s[n] ...
		ix.SetBounded(true)
		l = append(l, ir.NewAssignStmt(base.Pos, ix, n)) // s[n] = arg
		if i+1 < len(ls) {
			l = append(l, ir.NewAssignStmt(base.Pos, nn, ir.NewBinaryExpr(base.Pos, ir.OADD, nn, ir.NewInt(1)))) // n = n + 1
		}
	}

	typecheck.Stmts(l)
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
func copyany(n *ir.BinaryExpr, init *ir.Nodes, runtimecall bool) ir.Node {
	if n.X.Type().Elem().HasPointers() {
		ir.CurFunc.SetWBPos(n.Pos())
		fn := writebarrierfn("typedslicecopy", n.X.Type().Elem(), n.Y.Type().Elem())
		n.X = cheapexpr(n.X, init)
		ptrL, lenL := backingArrayPtrLen(n.X)
		n.Y = cheapexpr(n.Y, init)
		ptrR, lenR := backingArrayPtrLen(n.Y)
		return mkcall1(fn, n.Type(), init, typename(n.X.Type().Elem()), ptrL, lenL, ptrR, lenR)
	}

	if runtimecall {
		// rely on runtime to instrument:
		//  copy(n.Left, n.Right)
		// n.Right can be a slice or string.

		n.X = cheapexpr(n.X, init)
		ptrL, lenL := backingArrayPtrLen(n.X)
		n.Y = cheapexpr(n.Y, init)
		ptrR, lenR := backingArrayPtrLen(n.Y)

		fn := typecheck.LookupRuntime("slicecopy")
		fn = typecheck.SubstArgTypes(fn, ptrL.Type().Elem(), ptrR.Type().Elem())

		return mkcall1(fn, n.Type(), init, ptrL, lenL, ptrR, lenR, ir.NewInt(n.X.Type().Elem().Width))
	}

	n.X = walkexpr(n.X, init)
	n.Y = walkexpr(n.Y, init)
	nl := typecheck.Temp(n.X.Type())
	nr := typecheck.Temp(n.Y.Type())
	var l []ir.Node
	l = append(l, ir.NewAssignStmt(base.Pos, nl, n.X))
	l = append(l, ir.NewAssignStmt(base.Pos, nr, n.Y))

	nfrm := ir.NewUnaryExpr(base.Pos, ir.OSPTR, nr)
	nto := ir.NewUnaryExpr(base.Pos, ir.OSPTR, nl)

	nlen := typecheck.Temp(types.Types[types.TINT])

	// n = len(to)
	l = append(l, ir.NewAssignStmt(base.Pos, nlen, ir.NewUnaryExpr(base.Pos, ir.OLEN, nl)))

	// if n > len(frm) { n = len(frm) }
	nif := ir.NewIfStmt(base.Pos, nil, nil, nil)

	nif.Cond = ir.NewBinaryExpr(base.Pos, ir.OGT, nlen, ir.NewUnaryExpr(base.Pos, ir.OLEN, nr))
	nif.Body.Append(ir.NewAssignStmt(base.Pos, nlen, ir.NewUnaryExpr(base.Pos, ir.OLEN, nr)))
	l = append(l, nif)

	// if to.ptr != frm.ptr { memmove( ... ) }
	ne := ir.NewIfStmt(base.Pos, ir.NewBinaryExpr(base.Pos, ir.ONE, nto, nfrm), nil, nil)
	ne.Likely = true
	l = append(l, ne)

	fn := typecheck.LookupRuntime("memmove")
	fn = typecheck.SubstArgTypes(fn, nl.Type().Elem(), nl.Type().Elem())
	nwid := ir.Node(typecheck.Temp(types.Types[types.TUINTPTR]))
	setwid := ir.NewAssignStmt(base.Pos, nwid, typecheck.Conv(nlen, types.Types[types.TUINTPTR]))
	ne.Body.Append(setwid)
	nwid = ir.NewBinaryExpr(base.Pos, ir.OMUL, nwid, ir.NewInt(nl.Type().Elem().Width))
	call := mkcall1(fn, nil, init, nto, nfrm, nwid)
	ne.Body.Append(call)

	typecheck.Stmts(l)
	walkstmtlist(l)
	init.Append(l...)
	return nlen
}

func eqfor(t *types.Type) (n ir.Node, needsize bool) {
	// Should only arrive here with large memory or
	// a struct/array containing a non-memory field/element.
	// Small memory is handled inline, and single non-memory
	// is handled by walkcompare.
	switch a, _ := types.AlgType(t); a {
	case types.AMEM:
		n := typecheck.LookupRuntime("memequal")
		n = typecheck.SubstArgTypes(n, t, t)
		return n, true
	case types.ASPECIAL:
		sym := typesymprefix(".eq", t)
		n := typecheck.NewName(sym)
		ir.MarkFunc(n)
		n.SetType(typecheck.NewFuncType(nil, []*ir.Field{
			ir.NewField(base.Pos, nil, nil, types.NewPtr(t)),
			ir.NewField(base.Pos, nil, nil, types.NewPtr(t)),
		}, []*ir.Field{
			ir.NewField(base.Pos, nil, nil, types.Types[types.TBOOL]),
		}))
		return n, false
	}
	base.Fatalf("eqfor %v", t)
	return nil, false
}

// The result of walkcompare MUST be assigned back to n, e.g.
// 	n.Left = walkcompare(n.Left, init)
func walkcompare(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	if n.X.Type().IsInterface() && n.Y.Type().IsInterface() && n.X.Op() != ir.ONIL && n.Y.Op() != ir.ONIL {
		return walkcompareInterface(n, init)
	}

	if n.X.Type().IsString() && n.Y.Type().IsString() {
		return walkcompareString(n, init)
	}

	n.X = walkexpr(n.X, init)
	n.Y = walkexpr(n.Y, init)

	// Given mixed interface/concrete comparison,
	// rewrite into types-equal && data-equal.
	// This is efficient, avoids allocations, and avoids runtime calls.
	if n.X.Type().IsInterface() != n.Y.Type().IsInterface() {
		// Preserve side-effects in case of short-circuiting; see #32187.
		l := cheapexpr(n.X, init)
		r := cheapexpr(n.Y, init)
		// Swap so that l is the interface value and r is the concrete value.
		if n.Y.Type().IsInterface() {
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
		tab := ir.NewUnaryExpr(base.Pos, ir.OITAB, l)
		rtyp := typename(r.Type())
		if l.Type().IsEmptyInterface() {
			tab.SetType(types.NewPtr(types.Types[types.TUINT8]))
			tab.SetTypecheck(1)
			eqtype = ir.NewBinaryExpr(base.Pos, eq, tab, rtyp)
		} else {
			nonnil := ir.NewBinaryExpr(base.Pos, brcom(eq), typecheck.NodNil(), tab)
			match := ir.NewBinaryExpr(base.Pos, eq, itabType(tab), rtyp)
			eqtype = ir.NewLogicalExpr(base.Pos, andor, nonnil, match)
		}
		// Check for data equal.
		eqdata := ir.NewBinaryExpr(base.Pos, eq, ifaceData(n.Pos(), l, r.Type()), r)
		// Put it all together.
		expr := ir.NewLogicalExpr(base.Pos, andor, eqtype, eqdata)
		return finishcompare(n, expr, init)
	}

	// Must be comparison of array or struct.
	// Otherwise back end handles it.
	// While we're here, decide whether to
	// inline or call an eq alg.
	t := n.X.Type()
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
			n.X = cheapexpr(n.X, init)
			n.Y = cheapexpr(n.Y, init)

			// If exactly one comparison operand is
			// constant, invoke the constcmp functions
			// instead, and arrange for the constant
			// operand to be the first argument.
			l, r := n.X, n.Y
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
		inline = t.NumElem() <= 1 || (types.IsSimple[t.Elem().Kind()] && (t.NumElem() <= 4 || t.Elem().Width*t.NumElem() <= maxcmpsize))
	case types.TSTRUCT:
		inline = t.NumComponents(types.IgnoreBlankFields) <= 4
	}

	cmpl := n.X
	for cmpl != nil && cmpl.Op() == ir.OCONVNOP {
		cmpl = cmpl.(*ir.ConvExpr).X
	}
	cmpr := n.Y
	for cmpr != nil && cmpr.Op() == ir.OCONVNOP {
		cmpr = cmpr.(*ir.ConvExpr).X
	}

	// Chose not to inline. Call equality function directly.
	if !inline {
		// eq algs take pointers; cmpl and cmpr must be addressable
		if !ir.IsAssignable(cmpl) || !ir.IsAssignable(cmpr) {
			base.Fatalf("arguments of comparison must be lvalues - %v %v", cmpl, cmpr)
		}

		fn, needsize := eqfor(t)
		call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, nil)
		call.Args.Append(typecheck.NodAddr(cmpl))
		call.Args.Append(typecheck.NodAddr(cmpr))
		if needsize {
			call.Args.Append(ir.NewInt(t.Width))
		}
		res := ir.Node(call)
		if n.Op() != ir.OEQ {
			res = ir.NewUnaryExpr(base.Pos, ir.ONOT, res)
		}
		return finishcompare(n, res, init)
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
				ir.NewSelectorExpr(base.Pos, ir.OXDOT, cmpl, sym),
				ir.NewSelectorExpr(base.Pos, ir.OXDOT, cmpr, sym),
			)
		}
	} else {
		step := int64(1)
		remains := t.NumElem() * t.Elem().Width
		combine64bit := unalignedLoad && types.RegSize == 8 && t.Elem().Width <= 4 && t.Elem().IsInteger()
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
					ir.NewIndexExpr(base.Pos, cmpl, ir.NewInt(i)),
					ir.NewIndexExpr(base.Pos, cmpr, ir.NewInt(i)),
				)
				i++
				remains -= t.Elem().Width
			} else {
				elemType := t.Elem().ToUnsigned()
				cmplw := ir.Node(ir.NewIndexExpr(base.Pos, cmpl, ir.NewInt(i)))
				cmplw = typecheck.Conv(cmplw, elemType) // convert to unsigned
				cmplw = typecheck.Conv(cmplw, convType) // widen
				cmprw := ir.Node(ir.NewIndexExpr(base.Pos, cmpr, ir.NewInt(i)))
				cmprw = typecheck.Conv(cmprw, elemType)
				cmprw = typecheck.Conv(cmprw, convType)
				// For code like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will generate a single large load.
				for offset := int64(1); offset < step; offset++ {
					lb := ir.Node(ir.NewIndexExpr(base.Pos, cmpl, ir.NewInt(i+offset)))
					lb = typecheck.Conv(lb, elemType)
					lb = typecheck.Conv(lb, convType)
					lb = ir.NewBinaryExpr(base.Pos, ir.OLSH, lb, ir.NewInt(8*t.Elem().Width*offset))
					cmplw = ir.NewBinaryExpr(base.Pos, ir.OOR, cmplw, lb)
					rb := ir.Node(ir.NewIndexExpr(base.Pos, cmpr, ir.NewInt(i+offset)))
					rb = typecheck.Conv(rb, elemType)
					rb = typecheck.Conv(rb, convType)
					rb = ir.NewBinaryExpr(base.Pos, ir.OLSH, rb, ir.NewInt(8*t.Elem().Width*offset))
					cmprw = ir.NewBinaryExpr(base.Pos, ir.OOR, cmprw, rb)
				}
				compare(cmplw, cmprw)
				i += step
				remains -= step * t.Elem().Width
			}
		}
	}
	if expr == nil {
		expr = ir.NewBool(n.Op() == ir.OEQ)
		// We still need to use cmpl and cmpr, in case they contain
		// an expression which might panic. See issue 23837.
		t := typecheck.Temp(cmpl.Type())
		a1 := typecheck.Stmt(ir.NewAssignStmt(base.Pos, t, cmpl))
		a2 := typecheck.Stmt(ir.NewAssignStmt(base.Pos, t, cmpr))
		init.Append(a1, a2)
	}
	return finishcompare(n, expr, init)
}

func tracecmpArg(n ir.Node, t *types.Type, init *ir.Nodes) ir.Node {
	// Ugly hack to avoid "constant -1 overflows uintptr" errors, etc.
	if n.Op() == ir.OLITERAL && n.Type().IsSigned() && ir.Int64Val(n) < 0 {
		n = copyexpr(n, n.Type(), init)
	}

	return typecheck.Conv(n, t)
}

func walkcompareInterface(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	n.Y = cheapexpr(n.Y, init)
	n.X = cheapexpr(n.X, init)
	eqtab, eqdata := eqinterface(n.X, n.Y)
	var cmp ir.Node
	if n.Op() == ir.OEQ {
		cmp = ir.NewLogicalExpr(base.Pos, ir.OANDAND, eqtab, eqdata)
	} else {
		eqtab.SetOp(ir.ONE)
		cmp = ir.NewLogicalExpr(base.Pos, ir.OOROR, eqtab, ir.NewUnaryExpr(base.Pos, ir.ONOT, eqdata))
	}
	return finishcompare(n, cmp, init)
}

func walkcompareString(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	// Rewrite comparisons to short constant strings as length+byte-wise comparisons.
	var cs, ncs ir.Node // const string, non-const string
	switch {
	case ir.IsConst(n.X, constant.String) && ir.IsConst(n.Y, constant.String):
		// ignore; will be constant evaluated
	case ir.IsConst(n.X, constant.String):
		cs = n.X
		ncs = n.Y
	case ir.IsConst(n.Y, constant.String):
		cs = n.Y
		ncs = n.X
	}
	if cs != nil {
		cmp := n.Op()
		// Our comparison below assumes that the non-constant string
		// is on the left hand side, so rewrite "" cmp x to x cmp "".
		// See issue 24817.
		if ir.IsConst(n.X, constant.String) {
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
			r := ir.Node(ir.NewBinaryExpr(base.Pos, cmp, ir.NewUnaryExpr(base.Pos, ir.OLEN, ncs), ir.NewInt(int64(len(s)))))
			remains := len(s)
			for i := 0; remains > 0; {
				if remains == 1 || !canCombineLoads {
					cb := ir.NewInt(int64(s[i]))
					ncb := ir.NewIndexExpr(base.Pos, ncs, ir.NewInt(int64(i)))
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
				ncsubstr := typecheck.Conv(ir.NewIndexExpr(base.Pos, ncs, ir.NewInt(int64(i))), convType)
				csubstr := int64(s[i])
				// Calculate large constant from bytes as sequence of shifts and ors.
				// Like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will combine this into a single large load.
				for offset := 1; offset < step; offset++ {
					b := typecheck.Conv(ir.NewIndexExpr(base.Pos, ncs, ir.NewInt(int64(i+offset))), convType)
					b = ir.NewBinaryExpr(base.Pos, ir.OLSH, b, ir.NewInt(int64(8*offset)))
					ncsubstr = ir.NewBinaryExpr(base.Pos, ir.OOR, ncsubstr, b)
					csubstr |= int64(s[i+offset]) << uint8(8*offset)
				}
				csubstrPart := ir.NewInt(csubstr)
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
		n.X = cheapexpr(n.X, init)
		n.Y = cheapexpr(n.Y, init)
		eqlen, eqmem := eqstring(n.X, n.Y)
		// quick check of len before full compare for == or !=.
		// memequal then tests equality up to length len.
		if n.Op() == ir.OEQ {
			// len(left) == len(right) && memequal(left, right, len)
			r = ir.NewLogicalExpr(base.Pos, ir.OANDAND, eqlen, eqmem)
		} else {
			// len(left) != len(right) || !memequal(left, right, len)
			eqlen.SetOp(ir.ONE)
			r = ir.NewLogicalExpr(base.Pos, ir.OOROR, eqlen, ir.NewUnaryExpr(base.Pos, ir.ONOT, eqmem))
		}
	} else {
		// sys_cmpstring(s1, s2) :: 0
		r = mkcall("cmpstring", types.Types[types.TINT], init, typecheck.Conv(n.X, types.Types[types.TSTRING]), typecheck.Conv(n.Y, types.Types[types.TSTRING]))
		r = ir.NewBinaryExpr(base.Pos, n.Op(), r, ir.NewInt(0))
	}

	return finishcompare(n, r, init)
}

// The result of finishcompare MUST be assigned back to n, e.g.
// 	n.Left = finishcompare(n.Left, x, r, init)
func finishcompare(n *ir.BinaryExpr, r ir.Node, init *ir.Nodes) ir.Node {
	r = typecheck.Expr(r)
	r = typecheck.Conv(r, n.Type())
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

	if ir.IsSmallIntConst(n) {
		v := ir.Int64Val(n)
		return 0 <= v && v < max
	}

	switch n.Op() {
	case ir.OAND, ir.OANDNOT:
		n := n.(*ir.BinaryExpr)
		v := int64(-1)
		switch {
		case ir.IsSmallIntConst(n.X):
			v = ir.Int64Val(n.X)
		case ir.IsSmallIntConst(n.Y):
			v = ir.Int64Val(n.Y)
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
		n := n.(*ir.BinaryExpr)
		if !sign && ir.IsSmallIntConst(n.Y) {
			v := ir.Int64Val(n.Y)
			if 0 <= v && v <= max {
				return true
			}
		}

	case ir.ODIV:
		n := n.(*ir.BinaryExpr)
		if !sign && ir.IsSmallIntConst(n.Y) {
			v := ir.Int64Val(n.Y)
			for bits > 0 && v >= 2 {
				bits--
				v >>= 1
			}
		}

	case ir.ORSH:
		n := n.(*ir.BinaryExpr)
		if !sign && ir.IsSmallIntConst(n.Y) {
			v := ir.Int64Val(n.Y)
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
func usemethod(n *ir.CallExpr) {
	t := n.X.Type()

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
	if s := res0.Type.Sym(); s != nil && s.Name == "Method" && types.IsReflectPkg(s.Pkg) {
		ir.CurFunc.SetReflectMethod(true)
		// The LSym is initialized at this point. We need to set the attribute on the LSym.
		ir.CurFunc.LSym.Set(obj.AttrReflectMethod, true)
	}
}

func usefield(n *ir.SelectorExpr) {
	if objabi.Fieldtrack_enabled == 0 {
		return
	}

	switch n.Op() {
	default:
		base.Fatalf("usefield %v", n.Op())

	case ir.ODOT, ir.ODOTPTR:
		break
	}
	if n.Sel == nil {
		// No field name.  This DOTPTR was built by the compiler for access
		// to runtime data structures.  Ignore.
		return
	}

	t := n.X.Type()
	if t.IsPtr() {
		t = t.Elem()
	}
	field := n.Selection
	if field == nil {
		base.Fatalf("usefield %v %v without paramfld", n.X.Type(), n.Sel)
	}
	if field.Sym != n.Sel || field.Offset != n.Offset {
		base.Fatalf("field inconsistency: %v,%v != %v,%v", field.Sym, field.Offset, n.Sel, n.Offset)
	}
	if !strings.Contains(field.Note, "go:\"track\"") {
		return
	}

	outer := n.X.Type()
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
	if ir.CurFunc.FieldTrack == nil {
		ir.CurFunc.FieldTrack = make(map[*types.Sym]struct{})
	}
	ir.CurFunc.FieldTrack[sym] = struct{}{}
}

// anySideEffects reports whether n contains any operations that could have observable side effects.
func anySideEffects(n ir.Node) bool {
	return ir.Any(n, func(n ir.Node) bool {
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
			n := n.(*ir.BinaryExpr)
			if n.Y.Op() != ir.OLITERAL || constant.Sign(n.Y.Val()) == 0 {
				return true
			}

		// Only possible side effect is panic on invalid size,
		// but many makechan and makemap use size zero, which is definitely OK.
		case ir.OMAKECHAN, ir.OMAKEMAP:
			n := n.(*ir.MakeExpr)
			if !ir.IsConst(n.Len, constant.Int) || constant.Sign(n.Len.Val()) != 0 {
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
func wrapCall(n *ir.CallExpr, init *ir.Nodes) ir.Node {
	if len(n.Init()) != 0 {
		walkstmtlist(n.Init())
		init.Append(n.PtrInit().Take()...)
	}

	isBuiltinCall := n.Op() != ir.OCALLFUNC && n.Op() != ir.OCALLMETH && n.Op() != ir.OCALLINTER

	// Turn f(a, b, []T{c, d, e}...) back into f(a, b, c, d, e).
	if !isBuiltinCall && n.IsDDD {
		last := len(n.Args) - 1
		if va := n.Args[last]; va.Op() == ir.OSLICELIT {
			va := va.(*ir.CompLitExpr)
			n.Args.Set(append(n.Args[:last], va.List...))
			n.IsDDD = false
		}
	}

	// origArgs keeps track of what argument is uintptr-unsafe/unsafe-uintptr conversion.
	origArgs := make([]ir.Node, len(n.Args))
	var funcArgs []*ir.Field
	for i, arg := range n.Args {
		s := typecheck.LookupNum("a", i)
		if !isBuiltinCall && arg.Op() == ir.OCONVNOP && arg.Type().IsUintptr() && arg.(*ir.ConvExpr).X.Type().IsUnsafePtr() {
			origArgs[i] = arg
			arg = arg.(*ir.ConvExpr).X
			n.Args[i] = arg
		}
		funcArgs = append(funcArgs, ir.NewField(base.Pos, s, nil, arg.Type()))
	}
	t := ir.NewFuncType(base.Pos, nil, funcArgs, nil)

	wrapCall_prgen++
	sym := typecheck.LookupNum("wrap·", wrapCall_prgen)
	fn := typecheck.DeclFunc(sym, t)

	args := ir.ParamNames(t.Type())
	for i, origArg := range origArgs {
		if origArg == nil {
			continue
		}
		args[i] = ir.NewConvExpr(base.Pos, origArg.Op(), origArg.Type(), args[i])
	}
	call := ir.NewCallExpr(base.Pos, n.Op(), n.X, args)
	if !isBuiltinCall {
		call.SetOp(ir.OCALL)
		call.IsDDD = n.IsDDD
	}
	fn.Body = []ir.Node{call}

	typecheck.FinishFuncBody()

	typecheck.Func(fn)
	typecheck.Stmts(fn.Body)
	typecheck.Target.Decls = append(typecheck.Target.Decls, fn)

	call = ir.NewCallExpr(base.Pos, ir.OCALL, fn.Nname, n.Args)
	return walkexpr(typecheck.Stmt(call), init)
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
	return base.Flag.N == 0 && !base.Flag.Cfg.Instrumenting && n.Op() == ir.OLEN && n.(*ir.UnaryExpr).X.Op() == ir.OSTR2RUNES
}

func walkCheckPtrAlignment(n *ir.ConvExpr, init *ir.Nodes, count ir.Node) ir.Node {
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
		count = ir.NewInt(1)
	}

	n.X = cheapexpr(n.X, init)
	init.Append(mkcall("checkptrAlignment", nil, init, typecheck.ConvNop(n.X, types.Types[types.TUNSAFEPTR]), typename(elem), typecheck.Conv(count, types.Types[types.TUINTPTR])))
	return n
}

var walkCheckPtrArithmeticMarker byte

func walkCheckPtrArithmetic(n *ir.ConvExpr, init *ir.Nodes) ir.Node {
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
	switch n.X.Op() {
	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		return n
	}

	if n.X.Op() == ir.ODOTPTR && ir.IsReflectHeaderDataField(n.X) {
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
			n := n.(*ir.BinaryExpr)
			walk(n.X)
			walk(n.Y)
		case ir.OSUB, ir.OANDNOT:
			n := n.(*ir.BinaryExpr)
			walk(n.X)
		case ir.OCONVNOP:
			n := n.(*ir.ConvExpr)
			if n.X.Type().IsUnsafePtr() {
				n.X = cheapexpr(n.X, init)
				originals = append(originals, typecheck.ConvNop(n.X, types.Types[types.TUNSAFEPTR]))
			}
		}
	}
	walk(n.X)

	cheap := cheapexpr(n, init)

	slice := typecheck.MakeDotArgs(types.NewSlice(types.Types[types.TUNSAFEPTR]), originals)
	slice.SetEsc(ir.EscNone)

	init.Append(mkcall("checkptrArithmetic", nil, init, typecheck.ConvNop(cheap, types.Types[types.TUNSAFEPTR]), slice))
	// TODO(khr): Mark backing store of slice as dead. This will allow us to reuse
	// the backing store for multiple calls to checkptrArithmetic.

	return cheap
}

// appendWalkStmt typechecks and walks stmt and then appends it to init.
func appendWalkStmt(init *ir.Nodes, stmt ir.Node) {
	op := stmt.Op()
	n := typecheck.Stmt(stmt)
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
