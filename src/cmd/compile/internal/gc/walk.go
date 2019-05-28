// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"encoding/binary"
	"fmt"
	"strings"
)

// The constant is known to runtime.
const tmpstringbufsize = 32

func walk(fn *Node) {
	Curfn = fn

	if Debug['W'] != 0 {
		s := fmt.Sprintf("\nbefore walk %v", Curfn.Func.Nname.Sym)
		dumplist(s, Curfn.Nbody)
	}

	lno := lineno

	// Final typecheck for any unused variables.
	for i, ln := range fn.Func.Dcl {
		if ln.Op == ONAME && (ln.Class() == PAUTO || ln.Class() == PAUTOHEAP) {
			ln = typecheck(ln, ctxExpr|ctxAssign)
			fn.Func.Dcl[i] = ln
		}
	}

	// Propagate the used flag for typeswitch variables up to the NONAME in its definition.
	for _, ln := range fn.Func.Dcl {
		if ln.Op == ONAME && (ln.Class() == PAUTO || ln.Class() == PAUTOHEAP) && ln.Name.Defn != nil && ln.Name.Defn.Op == OTYPESW && ln.Name.Used() {
			ln.Name.Defn.Left.Name.SetUsed(true)
		}
	}

	for _, ln := range fn.Func.Dcl {
		if ln.Op != ONAME || (ln.Class() != PAUTO && ln.Class() != PAUTOHEAP) || ln.Sym.Name[0] == '&' || ln.Name.Used() {
			continue
		}
		if defn := ln.Name.Defn; defn != nil && defn.Op == OTYPESW {
			if defn.Left.Name.Used() {
				continue
			}
			yyerrorl(defn.Left.Pos, "%v declared and not used", ln.Sym)
			defn.Left.Name.SetUsed(true) // suppress repeats
		} else {
			yyerrorl(ln.Pos, "%v declared and not used", ln.Sym)
		}
	}

	lineno = lno
	if nerrors != 0 {
		return
	}
	walkstmtlist(Curfn.Nbody.Slice())
	if Debug['W'] != 0 {
		s := fmt.Sprintf("after walk %v", Curfn.Func.Nname.Sym)
		dumplist(s, Curfn.Nbody)
	}

	zeroResults()
	heapmoves()
	if Debug['W'] != 0 && Curfn.Func.Enter.Len() > 0 {
		s := fmt.Sprintf("enter %v", Curfn.Func.Nname.Sym)
		dumplist(s, Curfn.Func.Enter)
	}
}

func walkstmtlist(s []*Node) {
	for i := range s {
		s[i] = walkstmt(s[i])
	}
}

func samelist(a, b []*Node) bool {
	if len(a) != len(b) {
		return false
	}
	for i, n := range a {
		if n != b[i] {
			return false
		}
	}
	return true
}

func paramoutheap(fn *Node) bool {
	for _, ln := range fn.Func.Dcl {
		switch ln.Class() {
		case PPARAMOUT:
			if ln.isParamStackCopy() || ln.Addrtaken() {
				return true
			}

		case PAUTO:
			// stop early - parameters are over
			return false
		}
	}

	return false
}

// The result of walkstmt MUST be assigned back to n, e.g.
// 	n.Left = walkstmt(n.Left)
func walkstmt(n *Node) *Node {
	if n == nil {
		return n
	}

	setlineno(n)

	walkstmtlist(n.Ninit.Slice())

	switch n.Op {
	default:
		if n.Op == ONAME {
			yyerror("%v is not a top level statement", n.Sym)
		} else {
			yyerror("%v is not a top level statement", n.Op)
		}
		Dump("nottop", n)

	case OAS,
		OASOP,
		OAS2,
		OAS2DOTTYPE,
		OAS2RECV,
		OAS2FUNC,
		OAS2MAPR,
		OCLOSE,
		OCOPY,
		OCALLMETH,
		OCALLINTER,
		OCALL,
		OCALLFUNC,
		ODELETE,
		OSEND,
		OPRINT,
		OPRINTN,
		OPANIC,
		OEMPTY,
		ORECOVER,
		OGETG:
		if n.Typecheck() == 0 {
			Fatalf("missing typecheck: %+v", n)
		}
		wascopy := n.Op == OCOPY
		init := n.Ninit
		n.Ninit.Set(nil)
		n = walkexpr(n, &init)
		n = addinit(n, init.Slice())
		if wascopy && n.Op == OCONVNOP {
			n.Op = OEMPTY // don't leave plain values as statements.
		}

	// special case for a receive where we throw away
	// the value received.
	case ORECV:
		if n.Typecheck() == 0 {
			Fatalf("missing typecheck: %+v", n)
		}
		init := n.Ninit
		n.Ninit.Set(nil)

		n.Left = walkexpr(n.Left, &init)
		n = mkcall1(chanfn("chanrecv1", 2, n.Left.Type), nil, &init, n.Left, nodnil())
		n = walkexpr(n, &init)

		n = addinit(n, init.Slice())

	case OBREAK,
		OCONTINUE,
		OFALL,
		OGOTO,
		OLABEL,
		ODCLCONST,
		ODCLTYPE,
		OCHECKNIL,
		OVARDEF,
		OVARKILL,
		OVARLIVE:
		break

	case ODCL:
		v := n.Left
		if v.Class() == PAUTOHEAP {
			if compiling_runtime {
				yyerror("%v escapes to heap, not allowed in runtime.", v)
			}
			if prealloc[v] == nil {
				prealloc[v] = callnew(v.Type)
			}
			nn := nod(OAS, v.Name.Param.Heapaddr, prealloc[v])
			nn.SetColas(true)
			nn = typecheck(nn, ctxStmt)
			return walkstmt(nn)
		}

	case OBLOCK:
		walkstmtlist(n.List.Slice())

	case OXCASE:
		yyerror("case statement out of place")
		n.Op = OCASE
		fallthrough

	case OCASE:
		n.Right = walkstmt(n.Right)

	case ODEFER:
		Curfn.Func.SetHasDefer(true)
		fallthrough
	case OGO:
		switch n.Left.Op {
		case OPRINT, OPRINTN:
			n.Left = wrapCall(n.Left, &n.Ninit)

		case ODELETE:
			if mapfast(n.Left.List.First().Type) == mapslow {
				n.Left = wrapCall(n.Left, &n.Ninit)
			} else {
				n.Left = walkexpr(n.Left, &n.Ninit)
			}

		case OCOPY:
			n.Left = copyany(n.Left, &n.Ninit, true)

		default:
			n.Left = walkexpr(n.Left, &n.Ninit)
		}

	case OFOR, OFORUNTIL:
		if n.Left != nil {
			walkstmtlist(n.Left.Ninit.Slice())
			init := n.Left.Ninit
			n.Left.Ninit.Set(nil)
			n.Left = walkexpr(n.Left, &init)
			n.Left = addinit(n.Left, init.Slice())
		}

		n.Right = walkstmt(n.Right)
		if n.Op == OFORUNTIL {
			walkstmtlist(n.List.Slice())
		}
		walkstmtlist(n.Nbody.Slice())

	case OIF:
		n.Left = walkexpr(n.Left, &n.Ninit)
		walkstmtlist(n.Nbody.Slice())
		walkstmtlist(n.Rlist.Slice())

	case ORETURN:
		if n.List.Len() == 0 {
			break
		}
		if (Curfn.Type.FuncType().Outnamed && n.List.Len() > 1) || paramoutheap(Curfn) {
			// assign to the function out parameters,
			// so that reorder3 can fix up conflicts
			var rl []*Node

			for _, ln := range Curfn.Func.Dcl {
				cl := ln.Class()
				if cl == PAUTO || cl == PAUTOHEAP {
					break
				}
				if cl == PPARAMOUT {
					if ln.isParamStackCopy() {
						ln = walkexpr(typecheck(nod(ODEREF, ln.Name.Param.Heapaddr, nil), ctxExpr), nil)
					}
					rl = append(rl, ln)
				}
			}

			if got, want := n.List.Len(), len(rl); got != want {
				// order should have rewritten multi-value function calls
				// with explicit OAS2FUNC nodes.
				Fatalf("expected %v return arguments, have %v", want, got)
			}

			if samelist(rl, n.List.Slice()) {
				// special return in disguise
				// TODO(josharian, 1.12): is "special return" still relevant?
				// Tests still pass w/o this. See comments on https://go-review.googlesource.com/c/go/+/118318
				walkexprlist(n.List.Slice(), &n.Ninit)
				n.List.Set(nil)

				break
			}

			// move function calls out, to make reorder3's job easier.
			walkexprlistsafe(n.List.Slice(), &n.Ninit)

			ll := ascompatee(n.Op, rl, n.List.Slice(), &n.Ninit)
			n.List.Set(reorder3(ll))
			break
		}
		walkexprlist(n.List.Slice(), &n.Ninit)

		// For each return parameter (lhs), assign the corresponding result (rhs).
		lhs := Curfn.Type.Results()
		rhs := n.List.Slice()
		res := make([]*Node, lhs.NumFields())
		for i, nl := range lhs.FieldSlice() {
			nname := asNode(nl.Nname)
			if nname.isParamHeapCopy() {
				nname = nname.Name.Param.Stackcopy
			}
			a := nod(OAS, nname, rhs[i])
			res[i] = convas(a, &n.Ninit)
		}
		n.List.Set(res)

	case ORETJMP:
		break

	case OINLMARK:
		break

	case OSELECT:
		walkselect(n)

	case OSWITCH:
		walkswitch(n)

	case ORANGE:
		n = walkrange(n)
	}

	if n.Op == ONAME {
		Fatalf("walkstmt ended up with name: %+v", n)
	}
	return n
}

func isSmallMakeSlice(n *Node) bool {
	if n.Op != OMAKESLICE {
		return false
	}
	l := n.Left
	r := n.Right
	if r == nil {
		r = l
	}
	t := n.Type

	return smallintconst(l) && smallintconst(r) && (t.Elem().Width == 0 || r.Int64() < maxImplicitStackVarSize/t.Elem().Width)
}

// walk the whole tree of the body of an
// expression or simple statement.
// the types expressions are calculated.
// compile-time constants are evaluated.
// complex side effects like statements are appended to init
func walkexprlist(s []*Node, init *Nodes) {
	for i := range s {
		s[i] = walkexpr(s[i], init)
	}
}

func walkexprlistsafe(s []*Node, init *Nodes) {
	for i, n := range s {
		s[i] = safeexpr(n, init)
		s[i] = walkexpr(s[i], init)
	}
}

func walkexprlistcheap(s []*Node, init *Nodes) {
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
		case from.Size() == 4 && from.Align == 4 && !types.Haspointers(from):
			return "convT32", false
		case from.Size() == 8 && from.Align == types.Types[TUINT64].Align && !types.Haspointers(from):
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
			if !types.Haspointers(from) {
				return "convT2Enoptr", true
			}
			return "convT2E", true
		case 'I':
			if !types.Haspointers(from) {
				return "convT2Inoptr", true
			}
			return "convT2I", true
		}
	}
	Fatalf("unknown conv func %c2%c", from.Tie(), to.Tie())
	panic("unreachable")
}

// The result of walkexpr MUST be assigned back to n, e.g.
// 	n.Left = walkexpr(n.Left, init)
func walkexpr(n *Node, init *Nodes) *Node {
	if n == nil {
		return n
	}

	// Eagerly checkwidth all expressions for the back end.
	if n.Type != nil && !n.Type.WidthCalculated() {
		switch n.Type.Etype {
		case TBLANK, TNIL, TIDEAL:
		default:
			checkwidth(n.Type)
		}
	}

	if init == &n.Ninit {
		// not okay to use n->ninit when walking n,
		// because we might replace n with some other node
		// and would lose the init list.
		Fatalf("walkexpr init == &n->ninit")
	}

	if n.Ninit.Len() != 0 {
		walkstmtlist(n.Ninit.Slice())
		init.AppendNodes(&n.Ninit)
	}

	lno := setlineno(n)

	if Debug['w'] > 1 {
		Dump("before walk expr", n)
	}

	if n.Typecheck() != 1 {
		Fatalf("missed typecheck: %+v", n)
	}

	if n.Type.IsUntyped() {
		Fatalf("expression has untyped type: %+v", n)
	}

	if n.Op == ONAME && n.Class() == PAUTOHEAP {
		nn := nod(ODEREF, n.Name.Param.Heapaddr, nil)
		nn = typecheck(nn, ctxExpr)
		nn = walkexpr(nn, init)
		nn.Left.SetNonNil(true)
		return nn
	}

opswitch:
	switch n.Op {
	default:
		Dump("walk", n)
		Fatalf("walkexpr: switch 1 unknown op %+S", n)

	case ONONAME, OEMPTY, OGETG, ONEWOBJ:

	case OTYPE, ONAME, OLITERAL:
		// TODO(mdempsky): Just return n; see discussion on CL 38655.
		// Perhaps refactor to use Node.mayBeShared for these instead.
		// If these return early, make sure to still call
		// stringsym for constant strings.

	case ONOT, ONEG, OPLUS, OBITNOT, OREAL, OIMAG, ODOTMETH, ODOTINTER,
		ODEREF, OSPTR, OITAB, OIDATA, OADDR:
		n.Left = walkexpr(n.Left, init)

	case OEFACE, OAND, OSUB, OMUL, OADD, OOR, OXOR, OLSH, ORSH:
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)

	case ODOT, ODOTPTR:
		usefield(n)
		n.Left = walkexpr(n.Left, init)

	case ODOTTYPE, ODOTTYPE2:
		n.Left = walkexpr(n.Left, init)
		// Set up interface type addresses for back end.
		n.Right = typename(n.Type)
		if n.Op == ODOTTYPE {
			n.Right.Right = typename(n.Left.Type)
		}
		if !n.Type.IsInterface() && !n.Left.Type.IsEmptyInterface() {
			n.List.Set1(itabname(n.Type, n.Left.Type))
		}

	case OLEN, OCAP:
		if isRuneCount(n) {
			// Replace len([]rune(string)) with runtime.countrunes(string).
			n = mkcall("countrunes", n.Type, init, conv(n.Left.Left, types.Types[TSTRING]))
			break
		}

		n.Left = walkexpr(n.Left, init)

		// replace len(*[10]int) with 10.
		// delayed until now to preserve side effects.
		t := n.Left.Type

		if t.IsPtr() {
			t = t.Elem()
		}
		if t.IsArray() {
			safeexpr(n.Left, init)
			setintconst(n, t.NumElem())
			n.SetTypecheck(1)
		}

	case OCOMPLEX:
		// Use results from call expression as arguments for complex.
		if n.Left == nil && n.Right == nil {
			n.Left = n.List.First()
			n.Right = n.List.Second()
		}
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)

	case OEQ, ONE, OLT, OLE, OGT, OGE:
		n = walkcompare(n, init)

	case OANDAND, OOROR:
		n.Left = walkexpr(n.Left, init)

		// cannot put side effects from n.Right on init,
		// because they cannot run before n.Left is checked.
		// save elsewhere and store on the eventual n.Right.
		var ll Nodes

		n.Right = walkexpr(n.Right, &ll)
		n.Right = addinit(n.Right, ll.Slice())
		n = walkinrange(n, init)

	case OPRINT, OPRINTN:
		n = walkprint(n, init)

	case OPANIC:
		n = mkcall("gopanic", nil, init, n.Left)

	case ORECOVER:
		n = mkcall("gorecover", n.Type, init, nod(OADDR, nodfp, nil))

	case OCLOSUREVAR, OCFUNC:
		n.SetAddable(true)

	case OCALLINTER, OCALLFUNC, OCALLMETH:
		if n.Op == OCALLINTER {
			usemethod(n)
		}

		if n.Op == OCALLFUNC && n.Left.Op == OCLOSURE {
			// Transform direct call of a closure to call of a normal function.
			// transformclosure already did all preparation work.

			// Prepend captured variables to argument list.
			n.List.Prepend(n.Left.Func.Enter.Slice()...)

			n.Left.Func.Enter.Set(nil)

			// Replace OCLOSURE with ONAME/PFUNC.
			n.Left = n.Left.Func.Closure.Func.Nname

			// Update type of OCALLFUNC node.
			// Output arguments had not changed, but their offsets could.
			if n.Left.Type.NumResults() == 1 {
				n.Type = n.Left.Type.Results().Field(0).Type
			} else {
				n.Type = n.Left.Type.Results()
			}
		}

		walkCall(n, init)

	case OAS, OASOP:
		init.AppendNodes(&n.Ninit)

		// Recognize m[k] = append(m[k], ...) so we can reuse
		// the mapassign call.
		mapAppend := n.Left.Op == OINDEXMAP && n.Right.Op == OAPPEND
		if mapAppend && !samesafeexpr(n.Left, n.Right.List.First()) {
			Fatalf("not same expressions: %v != %v", n.Left, n.Right.List.First())
		}

		n.Left = walkexpr(n.Left, init)
		n.Left = safeexpr(n.Left, init)

		if mapAppend {
			n.Right.List.SetFirst(n.Left)
		}

		if n.Op == OASOP {
			// Rewrite x op= y into x = x op y.
			n.Right = nod(n.SubOp(), n.Left, n.Right)
			n.Right = typecheck(n.Right, ctxExpr)

			n.Op = OAS
			n.ResetAux()
		}

		if oaslit(n, init) {
			break
		}

		if n.Right == nil {
			// TODO(austin): Check all "implicit zeroing"
			break
		}

		if !instrumenting && isZero(n.Right) {
			break
		}

		switch n.Right.Op {
		default:
			n.Right = walkexpr(n.Right, init)

		case ORECV:
			// x = <-c; n.Left is x, n.Right.Left is c.
			// orderstmt made sure x is addressable.
			n.Right.Left = walkexpr(n.Right.Left, init)

			n1 := nod(OADDR, n.Left, nil)
			r := n.Right.Left // the channel
			n = mkcall1(chanfn("chanrecv1", 2, r.Type), nil, init, r, n1)
			n = walkexpr(n, init)
			break opswitch

		case OAPPEND:
			// x = append(...)
			r := n.Right
			if r.Type.Elem().NotInHeap() {
				yyerror("%v is go:notinheap; heap allocation disallowed", r.Type.Elem())
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
			n.Right = r
			if r.Op == OAPPEND {
				// Left in place for back end.
				// Do not add a new write barrier.
				// Set up address of type for back end.
				r.Left = typename(r.Type.Elem())
				break opswitch
			}
			// Otherwise, lowered for race detector.
			// Treat as ordinary assignment.
		}

		if n.Left != nil && n.Right != nil {
			n = convas(n, init)
		}

	case OAS2:
		init.AppendNodes(&n.Ninit)
		walkexprlistsafe(n.List.Slice(), init)
		walkexprlistsafe(n.Rlist.Slice(), init)
		ll := ascompatee(OAS, n.List.Slice(), n.Rlist.Slice(), init)
		ll = reorder3(ll)
		n = liststmt(ll)

	// a,b,... = fn()
	case OAS2FUNC:
		init.AppendNodes(&n.Ninit)

		r := n.Rlist.First()
		walkexprlistsafe(n.List.Slice(), init)
		r = walkexpr(r, init)

		if isIntrinsicCall(r) {
			n.Rlist.Set1(r)
			break
		}
		init.Append(r)

		ll := ascompatet(n.List, r.Type)
		n = liststmt(ll)

	// x, y = <-c
	// orderstmt made sure x is addressable.
	case OAS2RECV:
		init.AppendNodes(&n.Ninit)

		r := n.Rlist.First()
		walkexprlistsafe(n.List.Slice(), init)
		r.Left = walkexpr(r.Left, init)
		var n1 *Node
		if n.List.First().isBlank() {
			n1 = nodnil()
		} else {
			n1 = nod(OADDR, n.List.First(), nil)
		}
		fn := chanfn("chanrecv2", 2, r.Left.Type)
		ok := n.List.Second()
		call := mkcall1(fn, ok.Type, init, r.Left, n1)
		n = nod(OAS, ok, call)
		n = typecheck(n, ctxStmt)

	// a,b = m[i]
	case OAS2MAPR:
		init.AppendNodes(&n.Ninit)

		r := n.Rlist.First()
		walkexprlistsafe(n.List.Slice(), init)
		r.Left = walkexpr(r.Left, init)
		r.Right = walkexpr(r.Right, init)
		t := r.Left.Type

		fast := mapfast(t)
		var key *Node
		if fast != mapslow {
			// fast versions take key by value
			key = r.Right
		} else {
			// standard version takes key by reference
			// orderexpr made sure key is addressable.
			key = nod(OADDR, r.Right, nil)
		}

		// from:
		//   a,b = m[i]
		// to:
		//   var,b = mapaccess2*(t, m, i)
		//   a = *var
		a := n.List.First()

		if w := t.Elem().Width; w <= 1024 { // 1024 must match runtime/map.go:maxZero
			fn := mapfn(mapaccess2[fast], t)
			r = mkcall1(fn, fn.Type.Results(), init, typename(t), r.Left, key)
		} else {
			fn := mapfn("mapaccess2_fat", t)
			z := zeroaddr(w)
			r = mkcall1(fn, fn.Type.Results(), init, typename(t), r.Left, key, z)
		}

		// mapaccess2* returns a typed bool, but due to spec changes,
		// the boolean result of i.(T) is now untyped so we make it the
		// same type as the variable on the lhs.
		if ok := n.List.Second(); !ok.isBlank() && ok.Type.IsBoolean() {
			r.Type.Field(1).Type = ok.Type
		}
		n.Rlist.Set1(r)
		n.Op = OAS2FUNC

		// don't generate a = *var if a is _
		if !a.isBlank() {
			var_ := temp(types.NewPtr(t.Elem()))
			var_.SetTypecheck(1)
			var_.SetNonNil(true) // mapaccess always returns a non-nil pointer
			n.List.SetFirst(var_)
			n = walkexpr(n, init)
			init.Append(n)
			n = nod(OAS, a, nod(ODEREF, var_, nil))
		}

		n = typecheck(n, ctxStmt)
		n = walkexpr(n, init)

	case ODELETE:
		init.AppendNodes(&n.Ninit)
		map_ := n.List.First()
		key := n.List.Second()
		map_ = walkexpr(map_, init)
		key = walkexpr(key, init)

		t := map_.Type
		fast := mapfast(t)
		if fast == mapslow {
			// orderstmt made sure key is addressable.
			key = nod(OADDR, key, nil)
		}
		n = mkcall1(mapfndel(mapdelete[fast], t), nil, init, typename(t), map_, key)

	case OAS2DOTTYPE:
		walkexprlistsafe(n.List.Slice(), init)
		n.Rlist.SetFirst(walkexpr(n.Rlist.First(), init))

	case OCONVIFACE:
		n.Left = walkexpr(n.Left, init)

		fromType := n.Left.Type
		toType := n.Type

		// typeword generates the type word of the interface value.
		typeword := func() *Node {
			if toType.IsEmptyInterface() {
				return typename(fromType)
			}
			return itabname(fromType, toType)
		}

		// Optimize convT2E or convT2I as a two-word copy when T is pointer-shaped.
		if isdirectiface(fromType) {
			l := nod(OEFACE, typeword(), n.Left)
			l.Type = toType
			l.SetTypecheck(n.Typecheck())
			n = l
			break
		}

		if staticbytes == nil {
			staticbytes = newname(Runtimepkg.Lookup("staticbytes"))
			staticbytes.SetClass(PEXTERN)
			staticbytes.Type = types.NewArray(types.Types[TUINT8], 256)
			zerobase = newname(Runtimepkg.Lookup("zerobase"))
			zerobase.SetClass(PEXTERN)
			zerobase.Type = types.Types[TUINTPTR]
		}

		// Optimize convT2{E,I} for many cases in which T is not pointer-shaped,
		// by using an existing addressable value identical to n.Left
		// or creating one on the stack.
		var value *Node
		switch {
		case fromType.Size() == 0:
			// n.Left is zero-sized. Use zerobase.
			cheapexpr(n.Left, init) // Evaluate n.Left for side-effects. See issue 19246.
			value = zerobase
		case fromType.IsBoolean() || (fromType.Size() == 1 && fromType.IsInteger()):
			// n.Left is a bool/byte. Use staticbytes[n.Left].
			n.Left = cheapexpr(n.Left, init)
			value = nod(OINDEX, staticbytes, byteindex(n.Left))
			value.SetBounded(true)
		case n.Left.Class() == PEXTERN && n.Left.Name != nil && n.Left.Name.Readonly():
			// n.Left is a readonly global; use it directly.
			value = n.Left
		case !fromType.IsInterface() && n.Esc == EscNone && fromType.Width <= 1024:
			// n.Left does not escape. Use a stack temporary initialized to n.Left.
			value = temp(fromType)
			init.Append(typecheck(nod(OAS, value, n.Left), ctxStmt))
		}

		if value != nil {
			// Value is identical to n.Left.
			// Construct the interface directly: {type/itab, &value}.
			l := nod(OEFACE, typeword(), typecheck(nod(OADDR, value, nil), ctxExpr))
			l.Type = toType
			l.SetTypecheck(n.Typecheck())
			n = l
			break
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
			init.Append(nod(OAS, c, n.Left))

			// Get the itab out of the interface.
			tmp := temp(types.NewPtr(types.Types[TUINT8]))
			init.Append(nod(OAS, tmp, typecheck(nod(OITAB, c, nil), ctxExpr)))

			// Get the type out of the itab.
			nif := nod(OIF, typecheck(nod(ONE, tmp, nodnil()), ctxExpr), nil)
			nif.Nbody.Set1(nod(OAS, tmp, itabType(tmp)))
			init.Append(nif)

			// Build the result.
			e := nod(OEFACE, tmp, ifaceData(c, types.NewPtr(types.Types[TUINT8])))
			e.Type = toType // assign type manually, typecheck doesn't understand OEFACE.
			e.SetTypecheck(1)
			n = e
			break
		}

		fnname, needsaddr := convFuncName(fromType, toType)

		if !needsaddr && !fromType.IsInterface() {
			// Use a specialized conversion routine that only returns a data pointer.
			// ptr = convT2X(val)
			// e = iface{typ/tab, ptr}
			fn := syslook(fnname)
			dowidth(fromType)
			fn = substArgTypes(fn, fromType)
			dowidth(fn.Type)
			call := nod(OCALL, fn, nil)
			call.List.Set1(n.Left)
			call = typecheck(call, ctxExpr)
			call = walkexpr(call, init)
			call = safeexpr(call, init)
			e := nod(OEFACE, typeword(), call)
			e.Type = toType
			e.SetTypecheck(1)
			n = e
			break
		}

		var tab *Node
		if fromType.IsInterface() {
			// convI2I
			tab = typename(toType)
		} else {
			// convT2x
			tab = typeword()
		}

		v := n.Left
		if needsaddr {
			// Types of large or unknown size are passed by reference.
			// Orderexpr arranged for n.Left to be a temporary for all
			// the conversions it could see. Comparison of an interface
			// with a non-interface, especially in a switch on interface value
			// with non-interface cases, is not visible to orderstmt, so we
			// have to fall back on allocating a temp here.
			if !islvalue(v) {
				v = copyexpr(v, v.Type, init)
			}
			v = nod(OADDR, v, nil)
		}

		dowidth(fromType)
		fn := syslook(fnname)
		fn = substArgTypes(fn, fromType, toType)
		dowidth(fn.Type)
		n = nod(OCALL, fn, nil)
		n.List.Set2(tab, v)
		n = typecheck(n, ctxExpr)
		n = walkexpr(n, init)

	case OCONV, OCONVNOP:
		n.Left = walkexpr(n.Left, init)
		param, result := rtconvfn(n.Left.Type, n.Type)
		if param == Txxx {
			break
		}
		fn := basicnames[param] + "to" + basicnames[result]
		n = conv(mkcall(fn, types.Types[result], init, conv(n.Left, types.Types[param])), n.Type)

	case OANDNOT:
		n.Left = walkexpr(n.Left, init)
		n.Op = OAND
		n.Right = nod(OBITNOT, n.Right, nil)
		n.Right = typecheck(n.Right, ctxExpr)
		n.Right = walkexpr(n.Right, init)

	case ODIV, OMOD:
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)

		// rewrite complex div into function call.
		et := n.Left.Type.Etype

		if isComplex[et] && n.Op == ODIV {
			t := n.Type
			n = mkcall("complex128div", types.Types[TCOMPLEX128], init, conv(n.Left, types.Types[TCOMPLEX128]), conv(n.Right, types.Types[TCOMPLEX128]))
			n = conv(n, t)
			break
		}

		// Nothing to do for float divisions.
		if isFloat[et] {
			break
		}

		// rewrite 64-bit div and mod on 32-bit architectures.
		// TODO: Remove this code once we can introduce
		// runtime calls late in SSA processing.
		if Widthreg < 8 && (et == TINT64 || et == TUINT64) {
			if n.Right.Op == OLITERAL {
				// Leave div/mod by constant powers of 2.
				// The SSA backend will handle those.
				switch et {
				case TINT64:
					c := n.Right.Int64()
					if c < 0 {
						c = -c
					}
					if c != 0 && c&(c-1) == 0 {
						break opswitch
					}
				case TUINT64:
					c := uint64(n.Right.Int64())
					if c != 0 && c&(c-1) == 0 {
						break opswitch
					}
				}
			}
			var fn string
			if et == TINT64 {
				fn = "int64"
			} else {
				fn = "uint64"
			}
			if n.Op == ODIV {
				fn += "div"
			} else {
				fn += "mod"
			}
			n = mkcall(fn, n.Type, init, conv(n.Left, types.Types[et]), conv(n.Right, types.Types[et]))
		}

	case OINDEX:
		n.Left = walkexpr(n.Left, init)

		// save the original node for bounds checking elision.
		// If it was a ODIV/OMOD walk might rewrite it.
		r := n.Right

		n.Right = walkexpr(n.Right, init)

		// if range of type cannot exceed static array bound,
		// disable bounds check.
		if n.Bounded() {
			break
		}
		t := n.Left.Type
		if t != nil && t.IsPtr() {
			t = t.Elem()
		}
		if t.IsArray() {
			n.SetBounded(bounded(r, t.NumElem()))
			if Debug['m'] != 0 && n.Bounded() && !Isconst(n.Right, CTINT) {
				Warn("index bounds check elided")
			}
			if smallintconst(n.Right) && !n.Bounded() {
				yyerror("index out of bounds")
			}
		} else if Isconst(n.Left, CTSTR) {
			n.SetBounded(bounded(r, int64(len(n.Left.Val().U.(string)))))
			if Debug['m'] != 0 && n.Bounded() && !Isconst(n.Right, CTINT) {
				Warn("index bounds check elided")
			}
			if smallintconst(n.Right) && !n.Bounded() {
				yyerror("index out of bounds")
			}
		}

		if Isconst(n.Right, CTINT) {
			if n.Right.Val().U.(*Mpint).CmpInt64(0) < 0 || n.Right.Val().U.(*Mpint).Cmp(maxintval[TINT]) > 0 {
				yyerror("index out of bounds")
			}
		}

	case OINDEXMAP:
		// Replace m[k] with *map{access1,assign}(maptype, m, &k)
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)
		map_ := n.Left
		key := n.Right
		t := map_.Type
		if n.IndexMapLValue() {
			// This m[k] expression is on the left-hand side of an assignment.
			fast := mapfast(t)
			if fast == mapslow {
				// standard version takes key by reference.
				// orderexpr made sure key is addressable.
				key = nod(OADDR, key, nil)
			}
			n = mkcall1(mapfn(mapassign[fast], t), nil, init, typename(t), map_, key)
		} else {
			// m[k] is not the target of an assignment.
			fast := mapfast(t)
			if fast == mapslow {
				// standard version takes key by reference.
				// orderexpr made sure key is addressable.
				key = nod(OADDR, key, nil)
			}

			if w := t.Elem().Width; w <= 1024 { // 1024 must match runtime/map.go:maxZero
				n = mkcall1(mapfn(mapaccess1[fast], t), types.NewPtr(t.Elem()), init, typename(t), map_, key)
			} else {
				z := zeroaddr(w)
				n = mkcall1(mapfn("mapaccess1_fat", t), types.NewPtr(t.Elem()), init, typename(t), map_, key, z)
			}
		}
		n.Type = types.NewPtr(t.Elem())
		n.SetNonNil(true) // mapaccess1* and mapassign always return non-nil pointers.
		n = nod(ODEREF, n, nil)
		n.Type = t.Elem()
		n.SetTypecheck(1)

	case ORECV:
		Fatalf("walkexpr ORECV") // should see inside OAS only

	case OSLICEHEADER:
		n.Left = walkexpr(n.Left, init)
		n.List.SetFirst(walkexpr(n.List.First(), init))
		n.List.SetSecond(walkexpr(n.List.Second(), init))

	case OSLICE, OSLICEARR, OSLICESTR, OSLICE3, OSLICE3ARR:
		n.Left = walkexpr(n.Left, init)
		low, high, max := n.SliceBounds()
		low = walkexpr(low, init)
		if low != nil && isZero(low) {
			// Reduce x[0:j] to x[:j] and x[0:j:k] to x[:j:k].
			low = nil
		}
		high = walkexpr(high, init)
		max = walkexpr(max, init)
		n.SetSliceBounds(low, high, max)
		if n.Op.IsSlice3() {
			if max != nil && max.Op == OCAP && samesafeexpr(n.Left, max.Left) {
				// Reduce x[i:j:cap(x)] to x[i:j].
				if n.Op == OSLICE3 {
					n.Op = OSLICE
				} else {
					n.Op = OSLICEARR
				}
				n = reduceSlice(n)
			}
		} else {
			n = reduceSlice(n)
		}

	case ONEW:
		if n.Esc == EscNone {
			if n.Type.Elem().Width >= maxImplicitStackVarSize {
				Fatalf("large ONEW with EscNone: %v", n)
			}
			r := temp(n.Type.Elem())
			r = nod(OAS, r, nil) // zero temp
			r = typecheck(r, ctxStmt)
			init.Append(r)
			r = nod(OADDR, r.Left, nil)
			r = typecheck(r, ctxExpr)
			n = r
		} else {
			n = callnew(n.Type.Elem())
		}

	case OADDSTR:
		n = addstr(n, init)

	case OAPPEND:
		// order should make sure we only see OAS(node, OAPPEND), which we handle above.
		Fatalf("append outside assignment")

	case OCOPY:
		n = copyany(n, init, instrumenting && !compiling_runtime)

		// cannot use chanfn - closechan takes any, not chan any
	case OCLOSE:
		fn := syslook("closechan")

		fn = substArgTypes(fn, n.Left.Type)
		n = mkcall1(fn, nil, init, n.Left)

	case OMAKECHAN:
		// When size fits into int, use makechan instead of
		// makechan64, which is faster and shorter on 32 bit platforms.
		size := n.Left
		fnname := "makechan64"
		argtype := types.Types[TINT64]

		// Type checking guarantees that TIDEAL size is positive and fits in an int.
		// The case of size overflow when converting TUINT or TUINTPTR to TINT
		// will be handled by the negative range checks in makechan during runtime.
		if size.Type.IsKind(TIDEAL) || maxintval[size.Type.Etype].Cmp(maxintval[TUINT]) <= 0 {
			fnname = "makechan"
			argtype = types.Types[TINT]
		}

		n = mkcall1(chanfn(fnname, 1, n.Type), n.Type, init, typename(n.Type), conv(size, argtype))

	case OMAKEMAP:
		t := n.Type
		hmapType := hmap(t)
		hint := n.Left

		// var h *hmap
		var h *Node
		if n.Esc == EscNone {
			// Allocate hmap on stack.

			// var hv hmap
			hv := temp(hmapType)
			zero := nod(OAS, hv, nil)
			zero = typecheck(zero, ctxStmt)
			init.Append(zero)
			// h = &hv
			h = nod(OADDR, hv, nil)

			// Allocate one bucket pointed to by hmap.buckets on stack if hint
			// is not larger than BUCKETSIZE. In case hint is larger than
			// BUCKETSIZE runtime.makemap will allocate the buckets on the heap.
			// Maximum key and elem size is 128 bytes, larger objects
			// are stored with an indirection. So max bucket size is 2048+eps.
			if !Isconst(hint, CTINT) ||
				hint.Val().U.(*Mpint).CmpInt64(BUCKETSIZE) <= 0 {
				// var bv bmap
				bv := temp(bmap(t))

				zero = nod(OAS, bv, nil)
				zero = typecheck(zero, ctxStmt)
				init.Append(zero)

				// b = &bv
				b := nod(OADDR, bv, nil)

				// h.buckets = b
				bsym := hmapType.Field(5).Sym // hmap.buckets see reflect.go:hmap
				na := nod(OAS, nodSym(ODOT, h, bsym), b)
				na = typecheck(na, ctxStmt)
				init.Append(na)
			}
		}

		if Isconst(hint, CTINT) && hint.Val().U.(*Mpint).CmpInt64(BUCKETSIZE) <= 0 {
			// Handling make(map[any]any) and
			// make(map[any]any, hint) where hint <= BUCKETSIZE
			// special allows for faster map initialization and
			// improves binary size by using calls with fewer arguments.
			// For hint <= BUCKETSIZE overLoadFactor(hint, 0) is false
			// and no buckets will be allocated by makemap. Therefore,
			// no buckets need to be allocated in this code path.
			if n.Esc == EscNone {
				// Only need to initialize h.hash0 since
				// hmap h has been allocated on the stack already.
				// h.hash0 = fastrand()
				rand := mkcall("fastrand", types.Types[TUINT32], init)
				hashsym := hmapType.Field(4).Sym // hmap.hash0 see reflect.go:hmap
				a := nod(OAS, nodSym(ODOT, h, hashsym), rand)
				a = typecheck(a, ctxStmt)
				a = walkexpr(a, init)
				init.Append(a)
				n = convnop(h, t)
			} else {
				// Call runtime.makehmap to allocate an
				// hmap on the heap and initialize hmap's hash0 field.
				fn := syslook("makemap_small")
				fn = substArgTypes(fn, t.Key(), t.Elem())
				n = mkcall1(fn, n.Type, init)
			}
		} else {
			if n.Esc != EscNone {
				h = nodnil()
			}
			// Map initialization with a variable or large hint is
			// more complicated. We therefore generate a call to
			// runtime.makemap to intialize hmap and allocate the
			// map buckets.

			// When hint fits into int, use makemap instead of
			// makemap64, which is faster and shorter on 32 bit platforms.
			fnname := "makemap64"
			argtype := types.Types[TINT64]

			// Type checking guarantees that TIDEAL hint is positive and fits in an int.
			// See checkmake call in TMAP case of OMAKE case in OpSwitch in typecheck1 function.
			// The case of hint overflow when converting TUINT or TUINTPTR to TINT
			// will be handled by the negative range checks in makemap during runtime.
			if hint.Type.IsKind(TIDEAL) || maxintval[hint.Type.Etype].Cmp(maxintval[TUINT]) <= 0 {
				fnname = "makemap"
				argtype = types.Types[TINT]
			}

			fn := syslook(fnname)
			fn = substArgTypes(fn, hmapType, t.Key(), t.Elem())
			n = mkcall1(fn, n.Type, init, typename(n.Type), conv(hint, argtype), h)
		}

	case OMAKESLICE:
		l := n.Left
		r := n.Right
		if r == nil {
			r = safeexpr(l, init)
			l = r
		}
		t := n.Type
		if n.Esc == EscNone {
			if !isSmallMakeSlice(n) {
				Fatalf("non-small OMAKESLICE with EscNone: %v", n)
			}
			// var arr [r]T
			// n = arr[:l]
			i := indexconst(r)
			if i < 0 {
				Fatalf("walkexpr: invalid index %v", r)
			}
			t = types.NewArray(t.Elem(), i) // [r]T
			var_ := temp(t)
			a := nod(OAS, var_, nil) // zero temp
			a = typecheck(a, ctxStmt)
			init.Append(a)
			r := nod(OSLICE, var_, nil) // arr[:l]
			r.SetSliceBounds(nil, l, nil)
			r = conv(r, n.Type) // in case n.Type is named.
			r = typecheck(r, ctxExpr)
			r = walkexpr(r, init)
			n = r
		} else {
			// n escapes; set up a call to makeslice.
			// When len and cap can fit into int, use makeslice instead of
			// makeslice64, which is faster and shorter on 32 bit platforms.

			if t.Elem().NotInHeap() {
				yyerror("%v is go:notinheap; heap allocation disallowed", t.Elem())
			}

			len, cap := l, r

			fnname := "makeslice64"
			argtype := types.Types[TINT64]

			// Type checking guarantees that TIDEAL len/cap are positive and fit in an int.
			// The case of len or cap overflow when converting TUINT or TUINTPTR to TINT
			// will be handled by the negative range checks in makeslice during runtime.
			if (len.Type.IsKind(TIDEAL) || maxintval[len.Type.Etype].Cmp(maxintval[TUINT]) <= 0) &&
				(cap.Type.IsKind(TIDEAL) || maxintval[cap.Type.Etype].Cmp(maxintval[TUINT]) <= 0) {
				fnname = "makeslice"
				argtype = types.Types[TINT]
			}

			m := nod(OSLICEHEADER, nil, nil)
			m.Type = t

			fn := syslook(fnname)
			m.Left = mkcall1(fn, types.Types[TUNSAFEPTR], init, typename(t.Elem()), conv(len, argtype), conv(cap, argtype))
			m.Left.SetNonNil(true)
			m.List.Set2(conv(len, types.Types[TINT]), conv(cap, types.Types[TINT]))

			m = typecheck(m, ctxExpr)
			m = walkexpr(m, init)
			n = m
		}

	case ORUNESTR:
		a := nodnil()
		if n.Esc == EscNone {
			t := types.NewArray(types.Types[TUINT8], 4)
			a = nod(OADDR, temp(t), nil)
		}
		// intstring(*[4]byte, rune)
		n = mkcall("intstring", n.Type, init, a, conv(n.Left, types.Types[TINT64]))

	case OBYTES2STR, ORUNES2STR:
		a := nodnil()
		if n.Esc == EscNone {
			// Create temporary buffer for string on stack.
			t := types.NewArray(types.Types[TUINT8], tmpstringbufsize)
			a = nod(OADDR, temp(t), nil)
		}
		fn := "slicebytetostring"
		if n.Op == ORUNES2STR {
			fn = "slicerunetostring"
		}
		// slicebytetostring(*[32]byte, []byte) string
		// slicerunetostring(*[32]byte, []rune) string
		n = mkcall(fn, n.Type, init, a, n.Left)

	case OBYTES2STRTMP:
		n.Left = walkexpr(n.Left, init)
		if !instrumenting {
			// Let the backend handle OBYTES2STRTMP directly
			// to avoid a function call to slicebytetostringtmp.
			break
		}
		// slicebytetostringtmp([]byte) string
		n = mkcall("slicebytetostringtmp", n.Type, init, n.Left)

	case OSTR2BYTES:
		s := n.Left
		if Isconst(s, CTSTR) {
			sc := s.Val().U.(string)

			// Allocate a [n]byte of the right size.
			t := types.NewArray(types.Types[TUINT8], int64(len(sc)))
			var a *Node
			if n.Esc == EscNone && len(sc) <= maxImplicitStackVarSize {
				a = nod(OADDR, temp(t), nil)
			} else {
				a = callnew(t)
			}
			p := temp(t.PtrTo()) // *[n]byte
			init.Append(typecheck(nod(OAS, p, a), ctxStmt))

			// Copy from the static string data to the [n]byte.
			if len(sc) > 0 {
				as := nod(OAS,
					nod(ODEREF, p, nil),
					nod(ODEREF, convnop(nod(OSPTR, s, nil), t.PtrTo()), nil))
				as = typecheck(as, ctxStmt)
				as = walkstmt(as)
				init.Append(as)
			}

			// Slice the [n]byte to a []byte.
			n.Op = OSLICEARR
			n.Left = p
			n = walkexpr(n, init)
			break
		}

		a := nodnil()
		if n.Esc == EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[TUINT8], tmpstringbufsize)
			a = nod(OADDR, temp(t), nil)
		}
		// stringtoslicebyte(*32[byte], string) []byte
		n = mkcall("stringtoslicebyte", n.Type, init, a, conv(s, types.Types[TSTRING]))

	case OSTR2BYTESTMP:
		// []byte(string) conversion that creates a slice
		// referring to the actual string bytes.
		// This conversion is handled later by the backend and
		// is only for use by internal compiler optimizations
		// that know that the slice won't be mutated.
		// The only such case today is:
		// for i, c := range []byte(string)
		n.Left = walkexpr(n.Left, init)

	case OSTR2RUNES:
		a := nodnil()
		if n.Esc == EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[TINT32], tmpstringbufsize)
			a = nod(OADDR, temp(t), nil)
		}
		// stringtoslicerune(*[32]rune, string) []rune
		n = mkcall("stringtoslicerune", n.Type, init, a, conv(n.Left, types.Types[TSTRING]))

	case OARRAYLIT, OSLICELIT, OMAPLIT, OSTRUCTLIT, OPTRLIT:
		if isStaticCompositeLiteral(n) && !canSSAType(n.Type) {
			// n can be directly represented in the read-only data section.
			// Make direct reference to the static data. See issue 12841.
			vstat := staticname(n.Type)
			vstat.Name.SetReadonly(true)
			fixedlit(inInitFunction, initKindStatic, n, vstat, init)
			n = vstat
			n = typecheck(n, ctxExpr)
			break
		}
		var_ := temp(n.Type)
		anylit(n, var_, init)
		n = var_

	case OSEND:
		n1 := n.Right
		n1 = assignconv(n1, n.Left.Type.Elem(), "chan send")
		n1 = walkexpr(n1, init)
		n1 = nod(OADDR, n1, nil)
		n = mkcall1(chanfn("chansend1", 2, n.Left.Type), nil, init, n.Left, n1)

	case OCLOSURE:
		n = walkclosure(n, init)

	case OCALLPART:
		n = walkpartialcall(n, init)
	}

	// Expressions that are constant at run time but not
	// considered const by the language spec are not turned into
	// constants until walk. For example, if n is y%1 == 0, the
	// walk of y%1 may have replaced it by 0.
	// Check whether n with its updated args is itself now a constant.
	t := n.Type
	evconst(n)
	if n.Type != t {
		Fatalf("evconst changed Type: %v had type %v, now %v", n, t, n.Type)
	}
	if n.Op == OLITERAL {
		n = typecheck(n, ctxExpr)
		// Emit string symbol now to avoid emitting
		// any concurrently during the backend.
		if s, ok := n.Val().U.(string); ok {
			_ = stringsym(n.Pos, s)
		}
	}

	updateHasCall(n)

	if Debug['w'] != 0 && n != nil {
		Dump("after walk expr", n)
	}

	lineno = lno
	return n
}

// rtconvfn returns the parameter and result types that will be used by a
// runtime function to convert from type src to type dst. The runtime function
// name can be derived from the names of the returned types.
//
// If no such function is necessary, it returns (Txxx, Txxx).
func rtconvfn(src, dst *types.Type) (param, result types.EType) {
	if thearch.SoftFloat {
		return Txxx, Txxx
	}

	switch thearch.LinkArch.Family {
	case sys.ARM, sys.MIPS:
		if src.IsFloat() {
			switch dst.Etype {
			case TINT64, TUINT64:
				return TFLOAT64, dst.Etype
			}
		}
		if dst.IsFloat() {
			switch src.Etype {
			case TINT64, TUINT64:
				return src.Etype, TFLOAT64
			}
		}

	case sys.I386:
		if src.IsFloat() {
			switch dst.Etype {
			case TINT64, TUINT64:
				return TFLOAT64, dst.Etype
			case TUINT32, TUINT, TUINTPTR:
				return TFLOAT64, TUINT32
			}
		}
		if dst.IsFloat() {
			switch src.Etype {
			case TINT64, TUINT64:
				return src.Etype, TFLOAT64
			case TUINT32, TUINT, TUINTPTR:
				return TUINT32, TFLOAT64
			}
		}
	}
	return Txxx, Txxx
}

// TODO(josharian): combine this with its caller and simplify
func reduceSlice(n *Node) *Node {
	low, high, max := n.SliceBounds()
	if high != nil && high.Op == OLEN && samesafeexpr(n.Left, high.Left) {
		// Reduce x[i:len(x)] to x[i:].
		high = nil
	}
	n.SetSliceBounds(low, high, max)
	if (n.Op == OSLICE || n.Op == OSLICESTR) && low == nil && high == nil {
		// Reduce x[:] to x.
		if Debug_slice > 0 {
			Warn("slice: omit slice operation")
		}
		return n.Left
	}
	return n
}

func ascompatee1(l *Node, r *Node, init *Nodes) *Node {
	// convas will turn map assigns into function calls,
	// making it impossible for reorder3 to work.
	n := nod(OAS, l, r)

	if l.Op == OINDEXMAP {
		return n
	}

	return convas(n, init)
}

func ascompatee(op Op, nl, nr []*Node, init *Nodes) []*Node {
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

	var nn []*Node
	i := 0
	for ; i < len(nl); i++ {
		if i >= len(nr) {
			break
		}
		// Do not generate 'x = x' during return. See issue 4014.
		if op == ORETURN && samesafeexpr(nl[i], nr[i]) {
			continue
		}
		nn = append(nn, ascompatee1(nl[i], nr[i], init))
	}

	// cannot happen: caller checked that lists had same length
	if i < len(nl) || i < len(nr) {
		var nln, nrn Nodes
		nln.Set(nl)
		nrn.Set(nr)
		Fatalf("error in shape across %+v %v %+v / %d %d [%s]", nln, op, nrn, len(nl), len(nr), Curfn.funcname())
	}
	return nn
}

// fncall reports whether assigning an rvalue of type rt to an lvalue l might involve a function call.
func fncall(l *Node, rt *types.Type) bool {
	if l.HasCall() || l.Op == OINDEXMAP {
		return true
	}
	if types.Identical(l.Type, rt) {
		return false
	}
	// There might be a conversion required, which might involve a runtime call.
	return true
}

// check assign type list to
// an expression list. called in
//	expr-list = func()
func ascompatet(nl Nodes, nr *types.Type) []*Node {
	if nl.Len() != nr.NumFields() {
		Fatalf("ascompatet: assignment count mismatch: %d = %d", nl.Len(), nr.NumFields())
	}

	var nn, mm Nodes
	for i, l := range nl.Slice() {
		if l.isBlank() {
			continue
		}
		r := nr.Field(i)

		// Any assignment to an lvalue that might cause a function call must be
		// deferred until all the returned values have been read.
		if fncall(l, r.Type) {
			tmp := temp(r.Type)
			tmp = typecheck(tmp, ctxExpr)
			a := nod(OAS, l, tmp)
			a = convas(a, &mm)
			mm.Append(a)
			l = tmp
		}

		res := nod(ORESULT, nil, nil)
		res.Xoffset = Ctxt.FixedFrameSize() + r.Offset
		res.Type = r.Type
		res.SetTypecheck(1)

		a := nod(OAS, l, res)
		a = convas(a, &nn)
		updateHasCall(a)
		if a.HasCall() {
			Dump("ascompatet ucount", a)
			Fatalf("ascompatet: too many function calls evaluating parameters")
		}

		nn.Append(a)
	}
	return append(nn.Slice(), mm.Slice()...)
}

// package all the arguments that match a ... T parameter into a []T.
func mkdotargslice(typ *types.Type, args []*Node, init *Nodes, ddd *Node) *Node {
	esc := uint16(EscUnknown)
	if ddd != nil {
		esc = ddd.Esc
	}

	if len(args) == 0 {
		n := nodnil()
		n.Type = typ
		return n
	}

	n := nod(OCOMPLIT, nil, typenod(typ))
	if ddd != nil && prealloc[ddd] != nil {
		prealloc[n] = prealloc[ddd] // temporary to use
	}
	n.List.Set(args)
	n.Esc = esc
	n = typecheck(n, ctxExpr)
	if n.Type == nil {
		Fatalf("mkdotargslice: typecheck failed")
	}
	n = walkexpr(n, init)
	return n
}

func walkCall(n *Node, init *Nodes) {
	if n.Rlist.Len() != 0 {
		return // already walked
	}
	n.Left = walkexpr(n.Left, init)
	walkexprlist(n.List.Slice(), init)

	params := n.Left.Type.Params()
	args := n.List.Slice()
	// If there's a ... parameter (which is only valid as the final
	// parameter) and this is not a ... call expression,
	// then assign the remaining arguments as a slice.
	if nf := params.NumFields(); nf > 0 {
		if last := params.Field(nf - 1); last.IsDDD() && !n.IsDDD() {
			tail := args[nf-1:]
			slice := mkdotargslice(last.Type, tail, init, n.Right)
			// Allow immediate GC.
			for i := range tail {
				tail[i] = nil
			}
			args = append(args[:nf-1], slice)
		}
	}

	// If this is a method call, add the receiver at the beginning of the args.
	if n.Op == OCALLMETH {
		withRecv := make([]*Node, len(args)+1)
		withRecv[0] = n.Left.Left
		n.Left.Left = nil
		copy(withRecv[1:], args)
		args = withRecv
	}

	// For any argument whose evaluation might require a function call,
	// store that argument into a temporary variable,
	// to prevent that calls from clobbering arguments already on the stack.
	// When instrumenting, all arguments might require function calls.
	var tempAssigns []*Node
	for i, arg := range args {
		updateHasCall(arg)
		// Determine param type.
		var t *types.Type
		if n.Op == OCALLMETH {
			if i == 0 {
				t = n.Left.Type.Recv().Type
			} else {
				t = params.Field(i - 1).Type
			}
		} else {
			t = params.Field(i).Type
		}
		if instrumenting || fncall(arg, t) {
			// make assignment of fncall to tempname
			tmp := temp(t)
			a := nod(OAS, tmp, arg)
			a = convas(a, init)
			tempAssigns = append(tempAssigns, a)
			// replace arg with temp
			args[i] = tmp
		}
	}

	n.List.Set(tempAssigns)
	n.Rlist.Set(args)
}

// generate code for print
func walkprint(nn *Node, init *Nodes) *Node {
	// Hoist all the argument evaluation up before the lock.
	walkexprlistcheap(nn.List.Slice(), init)

	// For println, add " " between elements and "\n" at the end.
	if nn.Op == OPRINTN {
		s := nn.List.Slice()
		t := make([]*Node, 0, len(s)*2)
		for i, n := range s {
			if i != 0 {
				t = append(t, nodstr(" "))
			}
			t = append(t, n)
		}
		t = append(t, nodstr("\n"))
		nn.List.Set(t)
	}

	// Collapse runs of constant strings.
	s := nn.List.Slice()
	t := make([]*Node, 0, len(s))
	for i := 0; i < len(s); {
		var strs []string
		for i < len(s) && Isconst(s[i], CTSTR) {
			strs = append(strs, s[i].Val().U.(string))
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
	nn.List.Set(t)

	calls := []*Node{mkcall("printlock", nil, init)}
	for i, n := range nn.List.Slice() {
		if n.Op == OLITERAL {
			switch n.Val().Ctype() {
			case CTRUNE:
				n = defaultlit(n, types.Runetype)

			case CTINT:
				n = defaultlit(n, types.Types[TINT64])

			case CTFLT:
				n = defaultlit(n, types.Types[TFLOAT64])
			}
		}

		if n.Op != OLITERAL && n.Type != nil && n.Type.Etype == TIDEAL {
			n = defaultlit(n, types.Types[TINT64])
		}
		n = defaultlit(n, nil)
		nn.List.SetIndex(i, n)
		if n.Type == nil || n.Type.Etype == TFORW {
			continue
		}

		var on *Node
		switch n.Type.Etype {
		case TINTER:
			if n.Type.IsEmptyInterface() {
				on = syslook("printeface")
			} else {
				on = syslook("printiface")
			}
			on = substArgTypes(on, n.Type) // any-1
		case TPTR, TCHAN, TMAP, TFUNC, TUNSAFEPTR:
			on = syslook("printpointer")
			on = substArgTypes(on, n.Type) // any-1
		case TSLICE:
			on = syslook("printslice")
			on = substArgTypes(on, n.Type) // any-1
		case TUINT, TUINT8, TUINT16, TUINT32, TUINT64, TUINTPTR:
			if isRuntimePkg(n.Type.Sym.Pkg) && n.Type.Sym.Name == "hex" {
				on = syslook("printhex")
			} else {
				on = syslook("printuint")
			}
		case TINT, TINT8, TINT16, TINT32, TINT64:
			on = syslook("printint")
		case TFLOAT32, TFLOAT64:
			on = syslook("printfloat")
		case TCOMPLEX64, TCOMPLEX128:
			on = syslook("printcomplex")
		case TBOOL:
			on = syslook("printbool")
		case TSTRING:
			cs := ""
			if Isconst(n, CTSTR) {
				cs = n.Val().U.(string)
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
			badtype(OPRINT, n.Type, nil)
			continue
		}

		r := nod(OCALL, on, nil)
		if params := on.Type.Params().FieldSlice(); len(params) > 0 {
			t := params[0].Type
			if !types.Identical(t, n.Type) {
				n = nod(OCONV, n, nil)
				n.Type = t
			}
			r.List.Append(n)
		}
		calls = append(calls, r)
	}

	calls = append(calls, mkcall("printunlock", nil, init))

	typecheckslice(calls, ctxStmt)
	walkexprlist(calls, init)

	r := nod(OEMPTY, nil, nil)
	r = typecheck(r, ctxStmt)
	r = walkexpr(r, init)
	r.Ninit.Set(calls)
	return r
}

func callnew(t *types.Type) *Node {
	if t.NotInHeap() {
		yyerror("%v is go:notinheap; heap allocation disallowed", t)
	}
	dowidth(t)
	n := nod(ONEWOBJ, typename(t), nil)
	n.Type = types.NewPtr(t)
	n.SetTypecheck(1)
	n.SetNonNil(true)
	return n
}

// isReflectHeaderDataField reports whether l is an expression p.Data
// where p has type reflect.SliceHeader or reflect.StringHeader.
func isReflectHeaderDataField(l *Node) bool {
	if l.Type != types.Types[TUINTPTR] {
		return false
	}

	var tsym *types.Sym
	switch l.Op {
	case ODOT:
		tsym = l.Left.Type.Sym
	case ODOTPTR:
		tsym = l.Left.Type.Elem().Sym
	default:
		return false
	}

	if tsym == nil || l.Sym.Name != "Data" || tsym.Pkg.Path != "reflect" {
		return false
	}
	return tsym.Name == "SliceHeader" || tsym.Name == "StringHeader"
}

func convas(n *Node, init *Nodes) *Node {
	if n.Op != OAS {
		Fatalf("convas: not OAS %v", n.Op)
	}
	defer updateHasCall(n)

	n.SetTypecheck(1)

	if n.Left == nil || n.Right == nil {
		return n
	}

	lt := n.Left.Type
	rt := n.Right.Type
	if lt == nil || rt == nil {
		return n
	}

	if n.Left.isBlank() {
		n.Right = defaultlit(n.Right, nil)
		return n
	}

	if !types.Identical(lt, rt) {
		n.Right = assignconv(n.Right, lt, "assignment")
		n.Right = walkexpr(n.Right, init)
	}
	dowidth(n.Right.Type)

	return n
}

// from ascompat[ee]
//	a,b = c,d
// simultaneous assignment. there cannot
// be later use of an earlier lvalue.
//
// function calls have been removed.
func reorder3(all []*Node) []*Node {
	// If a needed expression may be affected by an
	// earlier assignment, make an early copy of that
	// expression and use the copy instead.
	var early []*Node

	var mapinit Nodes
	for i, n := range all {
		l := n.Left

		// Save subexpressions needed on left side.
		// Drill through non-dereferences.
		for {
			if l.Op == ODOT || l.Op == OPAREN {
				l = l.Left
				continue
			}

			if l.Op == OINDEX && l.Left.Type.IsArray() {
				l.Right = reorder3save(l.Right, all, i, &early)
				l = l.Left
				continue
			}

			break
		}

		switch l.Op {
		default:
			Fatalf("reorder3 unexpected lvalue %#v", l.Op)

		case ONAME:
			break

		case OINDEX, OINDEXMAP:
			l.Left = reorder3save(l.Left, all, i, &early)
			l.Right = reorder3save(l.Right, all, i, &early)
			if l.Op == OINDEXMAP {
				all[i] = convas(all[i], &mapinit)
			}

		case ODEREF, ODOTPTR:
			l.Left = reorder3save(l.Left, all, i, &early)
		}

		// Save expression on right side.
		all[i].Right = reorder3save(all[i].Right, all, i, &early)
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
func reorder3save(n *Node, all []*Node, i int, early *[]*Node) *Node {
	if !aliased(n, all, i) {
		return n
	}

	q := temp(n.Type)
	q = nod(OAS, q, n)
	q = typecheck(q, ctxStmt)
	*early = append(*early, q)
	return q.Left
}

// what's the outer value that a write to n affects?
// outer value means containing struct or array.
func outervalue(n *Node) *Node {
	for {
		switch n.Op {
		case OXDOT:
			Fatalf("OXDOT in walk")
		case ODOT, OPAREN, OCONVNOP:
			n = n.Left
			continue
		case OINDEX:
			if n.Left.Type != nil && n.Left.Type.IsArray() {
				n = n.Left
				continue
			}
		}

		return n
	}
}

// Is it possible that the computation of n might be
// affected by writes in as up to but not including the ith element?
func aliased(n *Node, all []*Node, i int) bool {
	if n == nil {
		return false
	}

	// Treat all fields of a struct as referring to the whole struct.
	// We could do better but we would have to keep track of the fields.
	for n.Op == ODOT {
		n = n.Left
	}

	// Look for obvious aliasing: a variable being assigned
	// during the all list and appearing in n.
	// Also record whether there are any writes to main memory.
	// Also record whether there are any writes to variables
	// whose addresses have been taken.
	memwrite := false
	varwrite := false
	for _, an := range all[:i] {
		a := outervalue(an.Left)

		for a.Op == ODOT {
			a = a.Left
		}

		if a.Op != ONAME {
			memwrite = true
			continue
		}

		switch n.Class() {
		default:
			varwrite = true
			continue

		case PAUTO, PPARAM, PPARAMOUT:
			if n.Addrtaken() {
				varwrite = true
				continue
			}

			if vmatch2(a, n) {
				// Direct hit.
				return true
			}
		}
	}

	// The variables being written do not appear in n.
	// However, n might refer to computed addresses
	// that are being written.

	// If no computed addresses are affected by the writes, no aliasing.
	if !memwrite && !varwrite {
		return false
	}

	// If n does not refer to computed addresses
	// (that is, if n only refers to variables whose addresses
	// have not been taken), no aliasing.
	if varexpr(n) {
		return false
	}

	// Otherwise, both the writes and n refer to computed memory addresses.
	// Assume that they might conflict.
	return true
}

// does the evaluation of n only refer to variables
// whose addresses have not been taken?
// (and no other memory)
func varexpr(n *Node) bool {
	if n == nil {
		return true
	}

	switch n.Op {
	case OLITERAL:
		return true

	case ONAME:
		switch n.Class() {
		case PAUTO, PPARAM, PPARAMOUT:
			if !n.Addrtaken() {
				return true
			}
		}

		return false

	case OADD,
		OSUB,
		OOR,
		OXOR,
		OMUL,
		ODIV,
		OMOD,
		OLSH,
		ORSH,
		OAND,
		OANDNOT,
		OPLUS,
		ONEG,
		OBITNOT,
		OPAREN,
		OANDAND,
		OOROR,
		OCONV,
		OCONVNOP,
		OCONVIFACE,
		ODOTTYPE:
		return varexpr(n.Left) && varexpr(n.Right)

	case ODOT: // but not ODOTPTR
		// Should have been handled in aliased.
		Fatalf("varexpr unexpected ODOT")
	}

	// Be conservative.
	return false
}

// is the name l mentioned in r?
func vmatch2(l *Node, r *Node) bool {
	if r == nil {
		return false
	}
	switch r.Op {
	// match each right given left
	case ONAME:
		return l == r

	case OLITERAL:
		return false
	}

	if vmatch2(l, r.Left) {
		return true
	}
	if vmatch2(l, r.Right) {
		return true
	}
	for _, n := range r.List.Slice() {
		if vmatch2(l, n) {
			return true
		}
	}
	return false
}

// is any name mentioned in l also mentioned in r?
// called by sinit.go
func vmatch1(l *Node, r *Node) bool {
	// isolate all left sides
	if l == nil || r == nil {
		return false
	}
	switch l.Op {
	case ONAME:
		switch l.Class() {
		case PPARAM, PAUTO:
			break

		default:
			// assignment to non-stack variable must be
			// delayed if right has function calls.
			if r.HasCall() {
				return true
			}
		}

		return vmatch2(l, r)

	case OLITERAL:
		return false
	}

	if vmatch1(l.Left, r) {
		return true
	}
	if vmatch1(l.Right, r) {
		return true
	}
	for _, n := range l.List.Slice() {
		if vmatch1(n, r) {
			return true
		}
	}
	return false
}

// paramstoheap returns code to allocate memory for heap-escaped parameters
// and to copy non-result parameters' values from the stack.
func paramstoheap(params *types.Type) []*Node {
	var nn []*Node
	for _, t := range params.Fields().Slice() {
		v := asNode(t.Nname)
		if v != nil && v.Sym != nil && strings.HasPrefix(v.Sym.Name, "~r") { // unnamed result
			v = nil
		}
		if v == nil {
			continue
		}

		if stackcopy := v.Name.Param.Stackcopy; stackcopy != nil {
			nn = append(nn, walkstmt(nod(ODCL, v, nil)))
			if stackcopy.Class() == PPARAM {
				nn = append(nn, walkstmt(typecheck(nod(OAS, v, stackcopy), ctxStmt)))
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
	for _, f := range Curfn.Type.Results().Fields().Slice() {
		v := asNode(f.Nname)
		if v != nil && v.Name.Param.Heapaddr != nil {
			// The local which points to the return value is the
			// thing that needs zeroing. This is already handled
			// by a Needzero annotation in plive.go:livenessepilogue.
			continue
		}
		if v.isParamHeapCopy() {
			// TODO(josharian/khr): Investigate whether we can switch to "continue" here,
			// and document more in either case.
			// In the review of CL 114797, Keith wrote (roughly):
			// I don't think the zeroing below matters.
			// The stack return value will never be marked as live anywhere in the function.
			// It is not written to until deferreturn returns.
			v = v.Name.Param.Stackcopy
		}
		// Zero the stack location containing f.
		Curfn.Func.Enter.Append(nodl(Curfn.Pos, OAS, v, nil))
	}
}

// returnsfromheap returns code to copy values for heap-escaped parameters
// back to the stack.
func returnsfromheap(params *types.Type) []*Node {
	var nn []*Node
	for _, t := range params.Fields().Slice() {
		v := asNode(t.Nname)
		if v == nil {
			continue
		}
		if stackcopy := v.Name.Param.Stackcopy; stackcopy != nil && stackcopy.Class() == PPARAMOUT {
			nn = append(nn, walkstmt(typecheck(nod(OAS, stackcopy, v), ctxStmt)))
		}
	}

	return nn
}

// heapmoves generates code to handle migrating heap-escaped parameters
// between the stack and the heap. The generated code is added to Curfn's
// Enter and Exit lists.
func heapmoves() {
	lno := lineno
	lineno = Curfn.Pos
	nn := paramstoheap(Curfn.Type.Recvs())
	nn = append(nn, paramstoheap(Curfn.Type.Params())...)
	nn = append(nn, paramstoheap(Curfn.Type.Results())...)
	Curfn.Func.Enter.Append(nn...)
	lineno = Curfn.Func.Endlineno
	Curfn.Func.Exit.Append(returnsfromheap(Curfn.Type.Results())...)
	lineno = lno
}

func vmkcall(fn *Node, t *types.Type, init *Nodes, va []*Node) *Node {
	if fn.Type == nil || fn.Type.Etype != TFUNC {
		Fatalf("mkcall %v %v", fn, fn.Type)
	}

	n := fn.Type.NumParams()
	if n != len(va) {
		Fatalf("vmkcall %v needs %v args got %v", fn, n, len(va))
	}

	r := nod(OCALL, fn, nil)
	r.List.Set(va)
	if fn.Type.NumResults() > 0 {
		r = typecheck(r, ctxExpr|ctxMultiOK)
	} else {
		r = typecheck(r, ctxStmt)
	}
	r = walkexpr(r, init)
	r.Type = t
	return r
}

func mkcall(name string, t *types.Type, init *Nodes, args ...*Node) *Node {
	return vmkcall(syslook(name), t, init, args)
}

func mkcall1(fn *Node, t *types.Type, init *Nodes, args ...*Node) *Node {
	return vmkcall(fn, t, init, args)
}

func conv(n *Node, t *types.Type) *Node {
	if types.Identical(n.Type, t) {
		return n
	}
	n = nod(OCONV, n, nil)
	n.Type = t
	n = typecheck(n, ctxExpr)
	return n
}

// convnop converts node n to type t using the OCONVNOP op
// and typechecks the result with ctxExpr.
func convnop(n *Node, t *types.Type) *Node {
	n = nod(OCONVNOP, n, nil)
	n.Type = t
	n = typecheck(n, ctxExpr)
	return n
}

// byteindex converts n, which is byte-sized, to a uint8.
// We cannot use conv, because we allow converting bool to uint8 here,
// which is forbidden in user code.
func byteindex(n *Node) *Node {
	if types.Identical(n.Type, types.Types[TUINT8]) {
		return n
	}
	n = nod(OCONV, n, nil)
	n.Type = types.Types[TUINT8]
	n.SetTypecheck(1)
	return n
}

func chanfn(name string, n int, t *types.Type) *Node {
	if !t.IsChan() {
		Fatalf("chanfn %v", t)
	}
	fn := syslook(name)
	switch n {
	default:
		Fatalf("chanfn %d", n)
	case 1:
		fn = substArgTypes(fn, t.Elem())
	case 2:
		fn = substArgTypes(fn, t.Elem(), t.Elem())
	}
	return fn
}

func mapfn(name string, t *types.Type) *Node {
	if !t.IsMap() {
		Fatalf("mapfn %v", t)
	}
	fn := syslook(name)
	fn = substArgTypes(fn, t.Key(), t.Elem(), t.Key(), t.Elem())
	return fn
}

func mapfndel(name string, t *types.Type) *Node {
	if !t.IsMap() {
		Fatalf("mapfn %v", t)
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
		if !t.Key().HasHeapPointer() {
			return mapfast32
		}
		if Widthptr == 4 {
			return mapfast32ptr
		}
		Fatalf("small pointer %v", t.Key())
	case AMEM64:
		if !t.Key().HasHeapPointer() {
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

func writebarrierfn(name string, l *types.Type, r *types.Type) *Node {
	fn := syslook(name)
	fn = substArgTypes(fn, l, r)
	return fn
}

func addstr(n *Node, init *Nodes) *Node {
	// orderexpr rewrote OADDSTR to have a list of strings.
	c := n.List.Len()

	if c < 2 {
		Fatalf("addstr count %d too small", c)
	}

	buf := nodnil()
	if n.Esc == EscNone {
		sz := int64(0)
		for _, n1 := range n.List.Slice() {
			if n1.Op == OLITERAL {
				sz += int64(len(n1.Val().U.(string)))
			}
		}

		// Don't allocate the buffer if the result won't fit.
		if sz < tmpstringbufsize {
			// Create temporary buffer for result string on stack.
			t := types.NewArray(types.Types[TUINT8], tmpstringbufsize)
			buf = nod(OADDR, temp(t), nil)
		}
	}

	// build list of string arguments
	args := []*Node{buf}
	for _, n2 := range n.List.Slice() {
		args = append(args, conv(n2, types.Types[TSTRING]))
	}

	var fn string
	if c <= 5 {
		// small numbers of strings use direct runtime helpers.
		// note: orderexpr knows this cutoff too.
		fn = fmt.Sprintf("concatstring%d", c)
	} else {
		// large numbers of strings are passed to the runtime as a slice.
		fn = "concatstrings"

		t := types.NewSlice(types.Types[TSTRING])
		slice := nod(OCOMPLIT, nil, typenod(t))
		if prealloc[n] != nil {
			prealloc[slice] = prealloc[n]
		}
		slice.List.Set(args[1:]) // skip buf arg
		args = []*Node{buf, slice}
		slice.Esc = EscNone
	}

	cat := syslook(fn)
	r := nod(OCALL, cat, nil)
	r.List.Set(args)
	r = typecheck(r, ctxExpr)
	r = walkexpr(r, init)
	r.Type = n.Type

	return r
}

func walkAppendArgs(n *Node, init *Nodes) {
	walkexprlistsafe(n.List.Slice(), init)

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	ls := n.List.Slice()
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
func appendslice(n *Node, init *Nodes) *Node {
	walkAppendArgs(n, init)

	l1 := n.List.First()
	l2 := n.List.Second()

	var nodes Nodes

	// var s []T
	s := temp(l1.Type)
	nodes.Append(nod(OAS, s, l1)) // s = l1

	elemtype := s.Type.Elem()

	// n := len(s) + len(l2)
	nn := temp(types.Types[TINT])
	nodes.Append(nod(OAS, nn, nod(OADD, nod(OLEN, s, nil), nod(OLEN, l2, nil))))

	// if uint(n) > uint(cap(s))
	nif := nod(OIF, nil, nil)
	nuint := conv(nn, types.Types[TUINT])
	scapuint := conv(nod(OCAP, s, nil), types.Types[TUINT])
	nif.Left = nod(OGT, nuint, scapuint)

	// instantiate growslice(typ *type, []any, int) []any
	fn := syslook("growslice")
	fn = substArgTypes(fn, elemtype, elemtype)

	// s = growslice(T, s, n)
	nif.Nbody.Set1(nod(OAS, s, mkcall1(fn, s.Type, &nif.Ninit, typename(elemtype), s, nn)))
	nodes.Append(nif)

	// s = s[:n]
	nt := nod(OSLICE, s, nil)
	nt.SetSliceBounds(nil, nn, nil)
	nt.SetBounded(true)
	nodes.Append(nod(OAS, s, nt))

	var ncopy *Node
	if elemtype.HasHeapPointer() {
		// copy(s[len(l1):], l2)
		nptr1 := nod(OSLICE, s, nil)
		nptr1.SetSliceBounds(nod(OLEN, l1, nil), nil, nil)

		nptr2 := l2

		Curfn.Func.setWBPos(n.Pos)

		// instantiate typedslicecopy(typ *type, dst any, src any) int
		fn := syslook("typedslicecopy")
		fn = substArgTypes(fn, l1.Type, l2.Type)
		ncopy = mkcall1(fn, types.Types[TINT], &nodes, typename(elemtype), nptr1, nptr2)

	} else if instrumenting && !compiling_runtime {
		// rely on runtime to instrument copy.
		// copy(s[len(l1):], l2)
		nptr1 := nod(OSLICE, s, nil)
		nptr1.SetSliceBounds(nod(OLEN, l1, nil), nil, nil)

		nptr2 := l2

		if l2.Type.IsString() {
			// instantiate func slicestringcopy(to any, fr any) int
			fn := syslook("slicestringcopy")
			fn = substArgTypes(fn, l1.Type, l2.Type)
			ncopy = mkcall1(fn, types.Types[TINT], &nodes, nptr1, nptr2)
		} else {
			// instantiate func slicecopy(to any, fr any, wid uintptr) int
			fn := syslook("slicecopy")
			fn = substArgTypes(fn, l1.Type, l2.Type)
			ncopy = mkcall1(fn, types.Types[TINT], &nodes, nptr1, nptr2, nodintconst(elemtype.Width))
		}

	} else {
		// memmove(&s[len(l1)], &l2[0], len(l2)*sizeof(T))
		nptr1 := nod(OINDEX, s, nod(OLEN, l1, nil))
		nptr1.SetBounded(true)
		nptr1 = nod(OADDR, nptr1, nil)

		nptr2 := nod(OSPTR, l2, nil)

		nwid := cheapexpr(conv(nod(OLEN, l2, nil), types.Types[TUINTPTR]), &nodes)
		nwid = nod(OMUL, nwid, nodintconst(elemtype.Width))

		// instantiate func memmove(to *any, frm *any, length uintptr)
		fn := syslook("memmove")
		fn = substArgTypes(fn, elemtype, elemtype)
		ncopy = mkcall1(fn, nil, &nodes, nptr1, nptr2, nwid)
	}
	ln := append(nodes.Slice(), ncopy)

	typecheckslice(ln, ctxStmt)
	walkstmtlist(ln)
	init.Append(ln...)
	return s
}

// isAppendOfMake reports whether n is of the form append(x , make([]T, y)...).
// isAppendOfMake assumes n has already been typechecked.
func isAppendOfMake(n *Node) bool {
	if Debug['N'] != 0 || instrumenting {
		return false
	}

	if n.Typecheck() == 0 {
		Fatalf("missing typecheck: %+v", n)
	}

	if n.Op != OAPPEND || !n.IsDDD() || n.List.Len() != 2 {
		return false
	}

	second := n.List.Second()
	if second.Op != OMAKESLICE || second.Right != nil {
		return false
	}

	// y must be either an integer constant or a variable of type int.
	// typecheck checks that constant arguments to make are not negative and
	// fit into an int.
	// runtime.growslice uses int as type for the newcap argument.
	// Constraining variables to be type int avoids the need for runtime checks
	// that e.g. check if an int64 value fits into an int.
	// TODO(moehrmann): support other integer types that always fit in an int
	y := second.Left
	if !Isconst(y, CTINT) && y.Type.Etype != TINT {
		return false
	}

	return true
}

// extendslice rewrites append(l1, make([]T, l2)...) to
//   init {
//     if l2 < 0 {
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
//     if lptr == sptr || !hasPointers(T) {
//       // growslice did not clear the whole underlying array (or did not get called)
//       hp := &s[len(l1)]
//       hn := l2 * sizeof(T)
//       memclr(hp, hn)
//     }
//   }
//   s
func extendslice(n *Node, init *Nodes) *Node {
	// isAppendOfMake made sure l2 fits in an int.
	l2 := conv(n.List.Second().Left, types.Types[TINT])
	l2 = typecheck(l2, ctxExpr)
	n.List.SetSecond(l2) // walkAppendArgs expects l2 in n.List.Second().

	walkAppendArgs(n, init)

	l1 := n.List.First()
	l2 = n.List.Second() // re-read l2, as it may have been updated by walkAppendArgs

	var nodes []*Node

	// if l2 < 0
	nifneg := nod(OIF, nod(OLT, l2, nodintconst(0)), nil)
	nifneg.SetLikely(false)

	// panicmakeslicelen()
	nifneg.Nbody.Set1(mkcall("panicmakeslicelen", nil, init))
	nodes = append(nodes, nifneg)

	// s := l1
	s := temp(l1.Type)
	nodes = append(nodes, nod(OAS, s, l1))

	elemtype := s.Type.Elem()

	// n := len(s) + l2
	nn := temp(types.Types[TINT])
	nodes = append(nodes, nod(OAS, nn, nod(OADD, nod(OLEN, s, nil), l2)))

	// if uint(n) > uint(cap(s))
	nuint := conv(nn, types.Types[TUINT])
	capuint := conv(nod(OCAP, s, nil), types.Types[TUINT])
	nif := nod(OIF, nod(OGT, nuint, capuint), nil)

	// instantiate growslice(typ *type, old []any, newcap int) []any
	fn := syslook("growslice")
	fn = substArgTypes(fn, elemtype, elemtype)

	// s = growslice(T, s, n)
	nif.Nbody.Set1(nod(OAS, s, mkcall1(fn, s.Type, &nif.Ninit, typename(elemtype), s, nn)))
	nodes = append(nodes, nif)

	// s = s[:n]
	nt := nod(OSLICE, s, nil)
	nt.SetSliceBounds(nil, nn, nil)
	nt.SetBounded(true)
	nodes = append(nodes, nod(OAS, s, nt))

	// lptr := &l1[0]
	l1ptr := temp(l1.Type.Elem().PtrTo())
	tmp := nod(OSPTR, l1, nil)
	nodes = append(nodes, nod(OAS, l1ptr, tmp))

	// sptr := &s[0]
	sptr := temp(elemtype.PtrTo())
	tmp = nod(OSPTR, s, nil)
	nodes = append(nodes, nod(OAS, sptr, tmp))

	// hp := &s[len(l1)]
	hp := nod(OINDEX, s, nod(OLEN, l1, nil))
	hp.SetBounded(true)
	hp = nod(OADDR, hp, nil)
	hp = convnop(hp, types.Types[TUNSAFEPTR])

	// hn := l2 * sizeof(elem(s))
	hn := nod(OMUL, l2, nodintconst(elemtype.Width))
	hn = conv(hn, types.Types[TUINTPTR])

	clrname := "memclrNoHeapPointers"
	hasPointers := types.Haspointers(elemtype)
	if hasPointers {
		clrname = "memclrHasPointers"
		Curfn.Func.setWBPos(n.Pos)
	}

	var clr Nodes
	clrfn := mkcall(clrname, nil, &clr, hp, hn)
	clr.Append(clrfn)

	if hasPointers {
		// if l1ptr == sptr
		nifclr := nod(OIF, nod(OEQ, l1ptr, sptr), nil)
		nifclr.Nbody = clr
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
func walkappend(n *Node, init *Nodes, dst *Node) *Node {
	if !samesafeexpr(dst, n.List.First()) {
		n.List.SetFirst(safeexpr(n.List.First(), init))
		n.List.SetFirst(walkexpr(n.List.First(), init))
	}
	walkexprlistsafe(n.List.Slice()[1:], init)

	nsrc := n.List.First()

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	// Using cheapexpr also makes sure that the evaluation
	// of all arguments (and especially any panics) happen
	// before we begin to modify the slice in a visible way.
	ls := n.List.Slice()[1:]
	for i, n := range ls {
		n = cheapexpr(n, init)
		if !types.Identical(n.Type, nsrc.Type.Elem()) {
			n = assignconv(n, nsrc.Type.Elem(), "append")
			n = walkexpr(n, init)
		}
		ls[i] = n
	}

	argc := n.List.Len() - 1
	if argc < 1 {
		return nsrc
	}

	// General case, with no function calls left as arguments.
	// Leave for gen, except that instrumentation requires old form.
	if !instrumenting || compiling_runtime {
		return n
	}

	var l []*Node

	ns := temp(nsrc.Type)
	l = append(l, nod(OAS, ns, nsrc)) // s = src

	na := nodintconst(int64(argc)) // const argc
	nx := nod(OIF, nil, nil)       // if cap(s) - len(s) < argc
	nx.Left = nod(OLT, nod(OSUB, nod(OCAP, ns, nil), nod(OLEN, ns, nil)), na)

	fn := syslook("growslice") //   growslice(<type>, old []T, mincap int) (ret []T)
	fn = substArgTypes(fn, ns.Type.Elem(), ns.Type.Elem())

	nx.Nbody.Set1(nod(OAS, ns,
		mkcall1(fn, ns.Type, &nx.Ninit, typename(ns.Type.Elem()), ns,
			nod(OADD, nod(OLEN, ns, nil), na))))

	l = append(l, nx)

	nn := temp(types.Types[TINT])
	l = append(l, nod(OAS, nn, nod(OLEN, ns, nil))) // n = len(s)

	nx = nod(OSLICE, ns, nil) // ...s[:n+argc]
	nx.SetSliceBounds(nil, nod(OADD, nn, na), nil)
	nx.SetBounded(true)
	l = append(l, nod(OAS, ns, nx)) // s = s[:n+argc]

	ls = n.List.Slice()[1:]
	for i, n := range ls {
		nx = nod(OINDEX, ns, nn) // s[n] ...
		nx.SetBounded(true)
		l = append(l, nod(OAS, nx, n)) // s[n] = arg
		if i+1 < len(ls) {
			l = append(l, nod(OAS, nn, nod(OADD, nn, nodintconst(1)))) // n = n + 1
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
func copyany(n *Node, init *Nodes, runtimecall bool) *Node {
	if n.Left.Type.Elem().HasHeapPointer() {
		Curfn.Func.setWBPos(n.Pos)
		fn := writebarrierfn("typedslicecopy", n.Left.Type, n.Right.Type)
		return mkcall1(fn, n.Type, init, typename(n.Left.Type.Elem()), n.Left, n.Right)
	}

	if runtimecall {
		if n.Right.Type.IsString() {
			fn := syslook("slicestringcopy")
			fn = substArgTypes(fn, n.Left.Type, n.Right.Type)
			return mkcall1(fn, n.Type, init, n.Left, n.Right)
		}

		fn := syslook("slicecopy")
		fn = substArgTypes(fn, n.Left.Type, n.Right.Type)
		return mkcall1(fn, n.Type, init, n.Left, n.Right, nodintconst(n.Left.Type.Elem().Width))
	}

	n.Left = walkexpr(n.Left, init)
	n.Right = walkexpr(n.Right, init)
	nl := temp(n.Left.Type)
	nr := temp(n.Right.Type)
	var l []*Node
	l = append(l, nod(OAS, nl, n.Left))
	l = append(l, nod(OAS, nr, n.Right))

	nfrm := nod(OSPTR, nr, nil)
	nto := nod(OSPTR, nl, nil)

	nlen := temp(types.Types[TINT])

	// n = len(to)
	l = append(l, nod(OAS, nlen, nod(OLEN, nl, nil)))

	// if n > len(frm) { n = len(frm) }
	nif := nod(OIF, nil, nil)

	nif.Left = nod(OGT, nlen, nod(OLEN, nr, nil))
	nif.Nbody.Append(nod(OAS, nlen, nod(OLEN, nr, nil)))
	l = append(l, nif)

	// if to.ptr != frm.ptr { memmove( ... ) }
	ne := nod(OIF, nod(ONE, nto, nfrm), nil)
	ne.SetLikely(true)
	l = append(l, ne)

	fn := syslook("memmove")
	fn = substArgTypes(fn, nl.Type.Elem(), nl.Type.Elem())
	nwid := temp(types.Types[TUINTPTR])
	setwid := nod(OAS, nwid, conv(nlen, types.Types[TUINTPTR]))
	ne.Nbody.Append(setwid)
	nwid = nod(OMUL, nwid, nodintconst(nl.Type.Elem().Width))
	call := mkcall1(fn, nil, init, nto, nfrm, nwid)
	ne.Nbody.Append(call)

	typecheckslice(l, ctxStmt)
	walkstmtlist(l)
	init.Append(l...)
	return nlen
}

func eqfor(t *types.Type) (n *Node, needsize bool) {
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
		n := newname(sym)
		n.SetClass(PFUNC)
		n.Sym.SetFunc(true)
		n.Type = functype(nil, []*Node{
			anonfield(types.NewPtr(t)),
			anonfield(types.NewPtr(t)),
		}, []*Node{
			anonfield(types.Types[TBOOL]),
		})
		return n, false
	}
	Fatalf("eqfor %v", t)
	return nil, false
}

// The result of walkcompare MUST be assigned back to n, e.g.
// 	n.Left = walkcompare(n.Left, init)
func walkcompare(n *Node, init *Nodes) *Node {
	if n.Left.Type.IsInterface() && n.Right.Type.IsInterface() && n.Left.Op != OLITERAL && n.Right.Op != OLITERAL {
		return walkcompareInterface(n, init)
	}

	if n.Left.Type.IsString() && n.Right.Type.IsString() {
		return walkcompareString(n, init)
	}

	n.Left = walkexpr(n.Left, init)
	n.Right = walkexpr(n.Right, init)

	// Given interface value l and concrete value r, rewrite
	//   l == r
	// into types-equal && data-equal.
	// This is efficient, avoids allocations, and avoids runtime calls.
	var l, r *Node
	if n.Left.Type.IsInterface() && !n.Right.Type.IsInterface() {
		l = n.Left
		r = n.Right
	} else if !n.Left.Type.IsInterface() && n.Right.Type.IsInterface() {
		l = n.Right
		r = n.Left
	}

	if l != nil {
		// Handle both == and !=.
		eq := n.Op
		andor := OOROR
		if eq == OEQ {
			andor = OANDAND
		}
		// Check for types equal.
		// For empty interface, this is:
		//   l.tab == type(r)
		// For non-empty interface, this is:
		//   l.tab != nil && l.tab._type == type(r)
		var eqtype *Node
		tab := nod(OITAB, l, nil)
		rtyp := typename(r.Type)
		if l.Type.IsEmptyInterface() {
			tab.Type = types.NewPtr(types.Types[TUINT8])
			tab.SetTypecheck(1)
			eqtype = nod(eq, tab, rtyp)
		} else {
			nonnil := nod(brcom(eq), nodnil(), tab)
			match := nod(eq, itabType(tab), rtyp)
			eqtype = nod(andor, nonnil, match)
		}
		// Check for data equal.
		eqdata := nod(eq, ifaceData(l, r.Type), r)
		// Put it all together.
		expr := nod(andor, eqtype, eqdata)
		n = finishcompare(n, expr, init)
		return n
	}

	// Must be comparison of array or struct.
	// Otherwise back end handles it.
	// While we're here, decide whether to
	// inline or call an eq alg.
	t := n.Left.Type
	var inline bool

	maxcmpsize := int64(4)
	unalignedLoad := canMergeLoads()
	if unalignedLoad {
		// Keep this low enough to generate less code than a function call.
		maxcmpsize = 2 * int64(thearch.LinkArch.RegSize)
	}

	switch t.Etype {
	default:
		return n
	case TARRAY:
		// We can compare several elements at once with 2/4/8 byte integer compares
		inline = t.NumElem() <= 1 || (issimple[t.Elem().Etype] && (t.NumElem() <= 4 || t.Elem().Width*t.NumElem() <= maxcmpsize))
	case TSTRUCT:
		inline = t.NumComponents(types.IgnoreBlankFields) <= 4
	}

	cmpl := n.Left
	for cmpl != nil && cmpl.Op == OCONVNOP {
		cmpl = cmpl.Left
	}
	cmpr := n.Right
	for cmpr != nil && cmpr.Op == OCONVNOP {
		cmpr = cmpr.Left
	}

	// Chose not to inline. Call equality function directly.
	if !inline {
		if isvaluelit(cmpl) {
			var_ := temp(cmpl.Type)
			anylit(cmpl, var_, init)
			cmpl = var_
		}
		if isvaluelit(cmpr) {
			var_ := temp(cmpr.Type)
			anylit(cmpr, var_, init)
			cmpr = var_
		}
		if !islvalue(cmpl) || !islvalue(cmpr) {
			Fatalf("arguments of comparison must be lvalues - %v %v", cmpl, cmpr)
		}

		// eq algs take pointers
		pl := temp(types.NewPtr(t))
		al := nod(OAS, pl, nod(OADDR, cmpl, nil))
		al = typecheck(al, ctxStmt)
		init.Append(al)

		pr := temp(types.NewPtr(t))
		ar := nod(OAS, pr, nod(OADDR, cmpr, nil))
		ar = typecheck(ar, ctxStmt)
		init.Append(ar)

		fn, needsize := eqfor(t)
		call := nod(OCALL, fn, nil)
		call.List.Append(pl)
		call.List.Append(pr)
		if needsize {
			call.List.Append(nodintconst(t.Width))
		}
		res := call
		if n.Op != OEQ {
			res = nod(ONOT, res, nil)
		}
		n = finishcompare(n, res, init)
		return n
	}

	// inline: build boolean expression comparing element by element
	andor := OANDAND
	if n.Op == ONE {
		andor = OOROR
	}
	var expr *Node
	compare := func(el, er *Node) {
		a := nod(n.Op, el, er)
		if expr == nil {
			expr = a
		} else {
			expr = nod(andor, expr, a)
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
				nodSym(OXDOT, cmpl, sym),
				nodSym(OXDOT, cmpr, sym),
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
				convType = types.Types[TINT64]
				step = 8 / t.Elem().Width
			case remains >= 4 && combine32bit:
				convType = types.Types[TUINT32]
				step = 4 / t.Elem().Width
			case remains >= 2 && combine16bit:
				convType = types.Types[TUINT16]
				step = 2 / t.Elem().Width
			default:
				step = 1
			}
			if step == 1 {
				compare(
					nod(OINDEX, cmpl, nodintconst(i)),
					nod(OINDEX, cmpr, nodintconst(i)),
				)
				i++
				remains -= t.Elem().Width
			} else {
				elemType := t.Elem().ToUnsigned()
				cmplw := nod(OINDEX, cmpl, nodintconst(i))
				cmplw = conv(cmplw, elemType) // convert to unsigned
				cmplw = conv(cmplw, convType) // widen
				cmprw := nod(OINDEX, cmpr, nodintconst(i))
				cmprw = conv(cmprw, elemType)
				cmprw = conv(cmprw, convType)
				// For code like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will generate a single large load.
				for offset := int64(1); offset < step; offset++ {
					lb := nod(OINDEX, cmpl, nodintconst(i+offset))
					lb = conv(lb, elemType)
					lb = conv(lb, convType)
					lb = nod(OLSH, lb, nodintconst(8*t.Elem().Width*offset))
					cmplw = nod(OOR, cmplw, lb)
					rb := nod(OINDEX, cmpr, nodintconst(i+offset))
					rb = conv(rb, elemType)
					rb = conv(rb, convType)
					rb = nod(OLSH, rb, nodintconst(8*t.Elem().Width*offset))
					cmprw = nod(OOR, cmprw, rb)
				}
				compare(cmplw, cmprw)
				i += step
				remains -= step * t.Elem().Width
			}
		}
	}
	if expr == nil {
		expr = nodbool(n.Op == OEQ)
		// We still need to use cmpl and cmpr, in case they contain
		// an expression which might panic. See issue 23837.
		t := temp(cmpl.Type)
		a1 := nod(OAS, t, cmpl)
		a1 = typecheck(a1, ctxStmt)
		a2 := nod(OAS, t, cmpr)
		a2 = typecheck(a2, ctxStmt)
		init.Append(a1, a2)
	}
	n = finishcompare(n, expr, init)
	return n
}

func walkcompareInterface(n *Node, init *Nodes) *Node {
	// ifaceeq(i1 any-1, i2 any-2) (ret bool);
	if !types.Identical(n.Left.Type, n.Right.Type) {
		Fatalf("ifaceeq %v %v %v", n.Op, n.Left.Type, n.Right.Type)
	}
	var fn *Node
	if n.Left.Type.IsEmptyInterface() {
		fn = syslook("efaceeq")
	} else {
		fn = syslook("ifaceeq")
	}

	n.Right = cheapexpr(n.Right, init)
	n.Left = cheapexpr(n.Left, init)
	lt := nod(OITAB, n.Left, nil)
	rt := nod(OITAB, n.Right, nil)
	ld := nod(OIDATA, n.Left, nil)
	rd := nod(OIDATA, n.Right, nil)
	ld.Type = types.Types[TUNSAFEPTR]
	rd.Type = types.Types[TUNSAFEPTR]
	ld.SetTypecheck(1)
	rd.SetTypecheck(1)
	call := mkcall1(fn, n.Type, init, lt, ld, rd)

	// Check itable/type before full compare.
	// Note: short-circuited because order matters.
	var cmp *Node
	if n.Op == OEQ {
		cmp = nod(OANDAND, nod(OEQ, lt, rt), call)
	} else {
		cmp = nod(OOROR, nod(ONE, lt, rt), nod(ONOT, call, nil))
	}
	return finishcompare(n, cmp, init)
}

func walkcompareString(n *Node, init *Nodes) *Node {
	// Rewrite comparisons to short constant strings as length+byte-wise comparisons.
	var cs, ncs *Node // const string, non-const string
	switch {
	case Isconst(n.Left, CTSTR) && Isconst(n.Right, CTSTR):
		// ignore; will be constant evaluated
	case Isconst(n.Left, CTSTR):
		cs = n.Left
		ncs = n.Right
	case Isconst(n.Right, CTSTR):
		cs = n.Right
		ncs = n.Left
	}
	if cs != nil {
		cmp := n.Op
		// Our comparison below assumes that the non-constant string
		// is on the left hand side, so rewrite "" cmp x to x cmp "".
		// See issue 24817.
		if Isconst(n.Left, CTSTR) {
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

		var and Op
		switch cmp {
		case OEQ:
			and = OANDAND
		case ONE:
			and = OOROR
		default:
			// Don't do byte-wise comparisons for <, <=, etc.
			// They're fairly complicated.
			// Length-only checks are ok, though.
			maxRewriteLen = 0
		}
		if s := cs.Val().U.(string); len(s) <= maxRewriteLen {
			if len(s) > 0 {
				ncs = safeexpr(ncs, init)
			}
			r := nod(cmp, nod(OLEN, ncs, nil), nodintconst(int64(len(s))))
			remains := len(s)
			for i := 0; remains > 0; {
				if remains == 1 || !canCombineLoads {
					cb := nodintconst(int64(s[i]))
					ncb := nod(OINDEX, ncs, nodintconst(int64(i)))
					r = nod(and, r, nod(cmp, ncb, cb))
					remains--
					i++
					continue
				}
				var step int
				var convType *types.Type
				switch {
				case remains >= 8 && combine64bit:
					convType = types.Types[TINT64]
					step = 8
				case remains >= 4:
					convType = types.Types[TUINT32]
					step = 4
				case remains >= 2:
					convType = types.Types[TUINT16]
					step = 2
				}
				ncsubstr := nod(OINDEX, ncs, nodintconst(int64(i)))
				ncsubstr = conv(ncsubstr, convType)
				csubstr := int64(s[i])
				// Calculate large constant from bytes as sequence of shifts and ors.
				// Like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will combine this into a single large load.
				for offset := 1; offset < step; offset++ {
					b := nod(OINDEX, ncs, nodintconst(int64(i+offset)))
					b = conv(b, convType)
					b = nod(OLSH, b, nodintconst(int64(8*offset)))
					ncsubstr = nod(OOR, ncsubstr, b)
					csubstr |= int64(s[i+offset]) << uint8(8*offset)
				}
				csubstrPart := nodintconst(csubstr)
				// Compare "step" bytes as once
				r = nod(and, r, nod(cmp, csubstrPart, ncsubstr))
				remains -= step
				i += step
			}
			return finishcompare(n, r, init)
		}
	}

	var r *Node
	if n.Op == OEQ || n.Op == ONE {
		// prepare for rewrite below
		n.Left = cheapexpr(n.Left, init)
		n.Right = cheapexpr(n.Right, init)

		lstr := conv(n.Left, types.Types[TSTRING])
		rstr := conv(n.Right, types.Types[TSTRING])
		lptr := nod(OSPTR, lstr, nil)
		rptr := nod(OSPTR, rstr, nil)
		llen := conv(nod(OLEN, lstr, nil), types.Types[TUINTPTR])
		rlen := conv(nod(OLEN, rstr, nil), types.Types[TUINTPTR])

		fn := syslook("memequal")
		fn = substArgTypes(fn, types.Types[TUINT8], types.Types[TUINT8])
		r = mkcall1(fn, types.Types[TBOOL], init, lptr, rptr, llen)

		// quick check of len before full compare for == or !=.
		// memequal then tests equality up to length len.
		if n.Op == OEQ {
			// len(left) == len(right) && memequal(left, right, len)
			r = nod(OANDAND, nod(OEQ, llen, rlen), r)
		} else {
			// len(left) != len(right) || !memequal(left, right, len)
			r = nod(ONOT, r, nil)
			r = nod(OOROR, nod(ONE, llen, rlen), r)
		}
	} else {
		// sys_cmpstring(s1, s2) :: 0
		r = mkcall("cmpstring", types.Types[TINT], init, conv(n.Left, types.Types[TSTRING]), conv(n.Right, types.Types[TSTRING]))
		r = nod(n.Op, r, nodintconst(0))
	}

	return finishcompare(n, r, init)
}

// The result of finishcompare MUST be assigned back to n, e.g.
// 	n.Left = finishcompare(n.Left, x, r, init)
func finishcompare(n, r *Node, init *Nodes) *Node {
	r = typecheck(r, ctxExpr)
	r = conv(r, n.Type)
	r = walkexpr(r, init)
	return r
}

// isIntOrdering reports whether n is a <, ≤, >, or ≥ ordering between integers.
func (n *Node) isIntOrdering() bool {
	switch n.Op {
	case OLE, OLT, OGE, OGT:
	default:
		return false
	}
	return n.Left.Type.IsInteger() && n.Right.Type.IsInteger()
}

// walkinrange optimizes integer-in-range checks, such as 4 <= x && x < 10.
// n must be an OANDAND or OOROR node.
// The result of walkinrange MUST be assigned back to n, e.g.
// 	n.Left = walkinrange(n.Left)
func walkinrange(n *Node, init *Nodes) *Node {
	// We are looking for something equivalent to a opl b OP b opr c, where:
	// * a, b, and c have integer type
	// * b is side-effect-free
	// * opl and opr are each < or ≤
	// * OP is &&
	l := n.Left
	r := n.Right
	if !l.isIntOrdering() || !r.isIntOrdering() {
		return n
	}

	// Find b, if it exists, and rename appropriately.
	// Input is: l.Left l.Op l.Right ANDAND/OROR r.Left r.Op r.Right
	// Output is: a opl b(==x) ANDAND/OROR b(==x) opr c
	a, opl, b := l.Left, l.Op, l.Right
	x, opr, c := r.Left, r.Op, r.Right
	for i := 0; ; i++ {
		if samesafeexpr(b, x) {
			break
		}
		if i == 3 {
			// Tried all permutations and couldn't find an appropriate b == x.
			return n
		}
		if i&1 == 0 {
			a, opl, b = b, brrev(opl), a
		} else {
			x, opr, c = c, brrev(opr), x
		}
	}

	// If n.Op is ||, apply de Morgan.
	// Negate the internal ops now; we'll negate the top level op at the end.
	// Henceforth assume &&.
	negateResult := n.Op == OOROR
	if negateResult {
		opl = brcom(opl)
		opr = brcom(opr)
	}

	cmpdir := func(o Op) int {
		switch o {
		case OLE, OLT:
			return -1
		case OGE, OGT:
			return +1
		}
		Fatalf("walkinrange cmpdir %v", o)
		return 0
	}
	if cmpdir(opl) != cmpdir(opr) {
		// Not a range check; something like b < a && b < c.
		return n
	}

	switch opl {
	case OGE, OGT:
		// We have something like a > b && b ≥ c.
		// Switch and reverse ops and rename constants,
		// to make it look like a ≤ b && b < c.
		a, c = c, a
		opl, opr = brrev(opr), brrev(opl)
	}

	// We must ensure that c-a is non-negative.
	// For now, require a and c to be constants.
	// In the future, we could also support a == 0 and c == len/cap(...).
	// Unfortunately, by this point, most len/cap expressions have been
	// stored into temporary variables.
	if !Isconst(a, CTINT) || !Isconst(c, CTINT) {
		return n
	}

	// Ensure that Int64() does not overflow on a and c (it'll happen
	// for any const above 2**63; see issue #27143).
	if !a.CanInt64() || !c.CanInt64() {
		return n
	}

	if opl == OLT {
		// We have a < b && ...
		// We need a ≤ b && ... to safely use unsigned comparison tricks.
		// If a is not the maximum constant for b's type,
		// we can increment a and switch to ≤.
		if a.Int64() >= maxintval[b.Type.Etype].Int64() {
			return n
		}
		a = nodintconst(a.Int64() + 1)
		opl = OLE
	}

	bound := c.Int64() - a.Int64()
	if bound < 0 {
		// Bad news. Something like 5 <= x && x < 3.
		// Rare in practice, and we still need to generate side-effects,
		// so just leave it alone.
		return n
	}

	// We have a ≤ b && b < c (or a ≤ b && b ≤ c).
	// This is equivalent to (a-a) ≤ (b-a) && (b-a) < (c-a),
	// which is equivalent to 0 ≤ (b-a) && (b-a) < (c-a),
	// which is equivalent to uint(b-a) < uint(c-a).
	ut := b.Type.ToUnsigned()
	lhs := conv(nod(OSUB, b, a), ut)
	rhs := nodintconst(bound)
	if negateResult {
		// Negate top level.
		opr = brcom(opr)
	}
	cmp := nod(opr, lhs, rhs)
	cmp.Pos = n.Pos
	cmp = addinit(cmp, l.Ninit.Slice())
	cmp = addinit(cmp, r.Ninit.Slice())
	// Typecheck the AST rooted at cmp...
	cmp = typecheck(cmp, ctxExpr)
	// ...but then reset cmp's type to match n's type.
	cmp.Type = n.Type
	cmp = walkexpr(cmp, init)
	return cmp
}

// return 1 if integer n must be in range [0, max), 0 otherwise
func bounded(n *Node, max int64) bool {
	if n.Type == nil || !n.Type.IsInteger() {
		return false
	}

	sign := n.Type.IsSigned()
	bits := int32(8 * n.Type.Width)

	if smallintconst(n) {
		v := n.Int64()
		return 0 <= v && v < max
	}

	switch n.Op {
	case OAND:
		v := int64(-1)
		if smallintconst(n.Left) {
			v = n.Left.Int64()
		} else if smallintconst(n.Right) {
			v = n.Right.Int64()
		}

		if 0 <= v && v < max {
			return true
		}

	case OMOD:
		if !sign && smallintconst(n.Right) {
			v := n.Right.Int64()
			if 0 <= v && v <= max {
				return true
			}
		}

	case ODIV:
		if !sign && smallintconst(n.Right) {
			v := n.Right.Int64()
			for bits > 0 && v >= 2 {
				bits--
				v >>= 1
			}
		}

	case ORSH:
		if !sign && smallintconst(n.Right) {
			v := n.Right.Int64()
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
func usemethod(n *Node) {
	t := n.Left.Type

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
		if p0.Type.Etype != TINT {
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
	if s := res0.Type.Sym; s != nil && s.Name == "Method" && s.Pkg != nil && s.Pkg.Path == "reflect" {
		Curfn.Func.SetReflectMethod(true)
	}
}

func usefield(n *Node) {
	if objabi.Fieldtrack_enabled == 0 {
		return
	}

	switch n.Op {
	default:
		Fatalf("usefield %v", n.Op)

	case ODOT, ODOTPTR:
		break
	}
	if n.Sym == nil {
		// No field name.  This DOTPTR was built by the compiler for access
		// to runtime data structures.  Ignore.
		return
	}

	t := n.Left.Type
	if t.IsPtr() {
		t = t.Elem()
	}
	field := dotField[typeSymKey{t.Orig, n.Sym}]
	if field == nil {
		Fatalf("usefield %v %v without paramfld", n.Left.Type, n.Sym)
	}
	if !strings.Contains(field.Note, "go:\"track\"") {
		return
	}

	outer := n.Left.Type
	if outer.IsPtr() {
		outer = outer.Elem()
	}
	if outer.Sym == nil {
		yyerror("tracked field must be in named struct type")
	}
	if !types.IsExported(field.Sym.Name) {
		yyerror("tracked field must be exported (upper case)")
	}

	sym := tracksym(outer, field)
	if Curfn.Func.FieldTrack == nil {
		Curfn.Func.FieldTrack = make(map[*types.Sym]struct{})
	}
	Curfn.Func.FieldTrack[sym] = struct{}{}
}

func candiscardlist(l Nodes) bool {
	for _, n := range l.Slice() {
		if !candiscard(n) {
			return false
		}
	}
	return true
}

func candiscard(n *Node) bool {
	if n == nil {
		return true
	}

	switch n.Op {
	default:
		return false

		// Discardable as long as the subpieces are.
	case ONAME,
		ONONAME,
		OTYPE,
		OPACK,
		OLITERAL,
		OADD,
		OSUB,
		OOR,
		OXOR,
		OADDSTR,
		OADDR,
		OANDAND,
		OBYTES2STR,
		ORUNES2STR,
		OSTR2BYTES,
		OSTR2RUNES,
		OCAP,
		OCOMPLIT,
		OMAPLIT,
		OSTRUCTLIT,
		OARRAYLIT,
		OSLICELIT,
		OPTRLIT,
		OCONV,
		OCONVIFACE,
		OCONVNOP,
		ODOT,
		OEQ,
		ONE,
		OLT,
		OLE,
		OGT,
		OGE,
		OKEY,
		OSTRUCTKEY,
		OLEN,
		OMUL,
		OLSH,
		ORSH,
		OAND,
		OANDNOT,
		ONEW,
		ONOT,
		OBITNOT,
		OPLUS,
		ONEG,
		OOROR,
		OPAREN,
		ORUNESTR,
		OREAL,
		OIMAG,
		OCOMPLEX:
		break

		// Discardable as long as we know it's not division by zero.
	case ODIV, OMOD:
		if Isconst(n.Right, CTINT) && n.Right.Val().U.(*Mpint).CmpInt64(0) != 0 {
			break
		}
		if Isconst(n.Right, CTFLT) && n.Right.Val().U.(*Mpflt).CmpFloat64(0) != 0 {
			break
		}
		return false

		// Discardable as long as we know it won't fail because of a bad size.
	case OMAKECHAN, OMAKEMAP:
		if Isconst(n.Left, CTINT) && n.Left.Val().U.(*Mpint).CmpInt64(0) == 0 {
			break
		}
		return false

		// Difficult to tell what sizes are okay.
	case OMAKESLICE:
		return false
	}

	if !candiscard(n.Left) || !candiscard(n.Right) || !candiscardlist(n.Ninit) || !candiscardlist(n.Nbody) || !candiscardlist(n.List) || !candiscardlist(n.Rlist) {
		return false
	}

	return true
}

// Rewrite
//	go builtin(x, y, z)
// into
//	go func(a1, a2, a3) {
//		builtin(a1, a2, a3)
//	}(x, y, z)
// for print, println, and delete.

var wrapCall_prgen int

// The result of wrapCall MUST be assigned back to n, e.g.
// 	n.Left = wrapCall(n.Left, init)
func wrapCall(n *Node, init *Nodes) *Node {
	if n.Ninit.Len() != 0 {
		walkstmtlist(n.Ninit.Slice())
		init.AppendNodes(&n.Ninit)
	}

	t := nod(OTFUNC, nil, nil)
	for i, arg := range n.List.Slice() {
		s := lookupN("a", i)
		t.List.Append(symfield(s, arg.Type))
	}

	wrapCall_prgen++
	sym := lookupN("wrap·", wrapCall_prgen)
	fn := dclfunc(sym, t)

	a := nod(n.Op, nil, nil)
	a.List.Set(paramNnames(t.Type))
	a = typecheck(a, ctxStmt)
	fn.Nbody.Set1(a)

	funcbody()

	fn = typecheck(fn, ctxStmt)
	typecheckslice(fn.Nbody.Slice(), ctxStmt)
	xtop = append(xtop, fn)

	a = nod(OCALL, nil, nil)
	a.Left = fn.Func.Nname
	a.List.Set(n.List.Slice())
	a = typecheck(a, ctxStmt)
	a = walkexpr(a, init)
	return a
}

// substArgTypes substitutes the given list of types for
// successive occurrences of the "any" placeholder in the
// type syntax expression n.Type.
// The result of substArgTypes MUST be assigned back to old, e.g.
// 	n.Left = substArgTypes(n.Left, t1, t2)
func substArgTypes(old *Node, types_ ...*types.Type) *Node {
	n := old.copy()

	for _, t := range types_ {
		dowidth(t)
	}
	n.Type = types.SubstAny(n.Type, &types_)
	if len(types_) > 0 {
		Fatalf("substArgTypes: too many argument types")
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
func isRuneCount(n *Node) bool {
	return Debug['N'] == 0 && !instrumenting && n.Op == OLEN && n.Left.Op == OSTR2RUNES
}
