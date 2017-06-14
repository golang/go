// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
	"strings"
)

// The constant is known to runtime.
const (
	tmpstringbufsize = 32
)

func walk(fn *Node) {
	Curfn = fn

	if Debug['W'] != 0 {
		s := fmt.Sprintf("\nbefore %v", Curfn.Func.Nname.Sym)
		dumplist(s, Curfn.Nbody)
	}

	lno := lineno

	// Final typecheck for any unused variables.
	for i, ln := range fn.Func.Dcl {
		if ln.Op == ONAME && (ln.Class() == PAUTO || ln.Class() == PAUTOHEAP) {
			ln = typecheck(ln, Erv|Easgn)
			fn.Func.Dcl[i] = ln
		}
	}

	// Propagate the used flag for typeswitch variables up to the NONAME in it's definition.
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

// adds "adjust" to all the argument locations for the call n.
// n must be a defer or go node that has already been walked.
func adjustargs(n *Node, adjust int) {
	var arg *Node
	var lhs *Node

	callfunc := n.Left
	for _, arg = range callfunc.List.Slice() {
		if arg.Op != OAS {
			Fatalf("call arg not assignment")
		}
		lhs = arg.Left
		if lhs.Op == ONAME {
			// This is a temporary introduced by reorder1.
			// The real store to the stack appears later in the arg list.
			continue
		}

		if lhs.Op != OINDREGSP {
			Fatalf("call argument store does not use OINDREGSP")
		}

		// can't really check this in machine-indep code.
		//if(lhs->val.u.reg != D_SP)
		//      Fatalf("call arg assign not indreg(SP)")
		lhs.Xoffset += int64(adjust)
	}
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
			nn = typecheck(nn, Etop)
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
		switch n.Left.Op {
		case OPRINT, OPRINTN:
			n.Left = walkprintfunc(n.Left, &n.Ninit)

		case OCOPY:
			n.Left = copyany(n.Left, &n.Ninit, true)

		default:
			n.Left = walkexpr(n.Left, &n.Ninit)
		}

		// make room for size & fn arguments.
		adjustargs(n, 2*Widthptr)

	case OFOR, OFORUNTIL:
		if n.Left != nil {
			walkstmtlist(n.Left.Ninit.Slice())
			init := n.Left.Ninit
			n.Left.Ninit.Set(nil)
			n.Left = walkexpr(n.Left, &init)
			n.Left = addinit(n.Left, init.Slice())
		}

		n.Right = walkstmt(n.Right)
		walkstmtlist(n.Nbody.Slice())

	case OIF:
		n.Left = walkexpr(n.Left, &n.Ninit)
		walkstmtlist(n.Nbody.Slice())
		walkstmtlist(n.Rlist.Slice())

	case OPROC:
		switch n.Left.Op {
		case OPRINT, OPRINTN:
			n.Left = walkprintfunc(n.Left, &n.Ninit)

		case OCOPY:
			n.Left = copyany(n.Left, &n.Ninit, true)

		default:
			n.Left = walkexpr(n.Left, &n.Ninit)
		}

		// make room for size & fn arguments.
		adjustargs(n, 2*Widthptr)

	case ORETURN:
		walkexprlist(n.List.Slice(), &n.Ninit)
		if n.List.Len() == 0 {
			break
		}
		if (Curfn.Type.FuncType().Outnamed && n.List.Len() > 1) || paramoutheap(Curfn) {
			// assign to the function out parameters,
			// so that reorder3 can fix up conflicts
			var rl []*Node

			var cl Class
			for _, ln := range Curfn.Func.Dcl {
				cl = ln.Class()
				if cl == PAUTO || cl == PAUTOHEAP {
					break
				}
				if cl == PPARAMOUT {
					if ln.isParamStackCopy() {
						ln = walkexpr(typecheck(nod(OIND, ln.Name.Param.Heapaddr, nil), Erv), nil)
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
				n.List.Set(nil)

				break
			}

			// move function calls out, to make reorder3's job easier.
			walkexprlistsafe(n.List.Slice(), &n.Ninit)

			ll := ascompatee(n.Op, rl, n.List.Slice(), &n.Ninit)
			n.List.Set(reorder3(ll))
			break
		}

		ll := ascompatte(nil, false, Curfn.Type.Results(), n.List.Slice(), 1, &n.Ninit)
		n.List.Set(ll)

	case ORETJMP:
		break

	case OSELECT:
		walkselect(n)

	case OSWITCH:
		walkswitch(n)

	case ORANGE:
		n = walkrange(n)

	case OXFALL:
		yyerror("fallthrough statement out of place")
		n.Op = OFALL
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

	return smallintconst(l) && smallintconst(r) && (t.Elem().Width == 0 || r.Int64() < (1<<16)/t.Elem().Width)
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

// Build name of function for interface conversion.
// Not all names are possible
// (e.g., we'll never generate convE2E or convE2I or convI2E).
func convFuncName(from, to *types.Type) string {
	tkind := to.Tie()
	switch from.Tie() {
	case 'I':
		switch tkind {
		case 'I':
			return "convI2I"
		}
	case 'T':
		switch tkind {
		case 'E':
			switch {
			case from.Size() == 2 && from.Align == 2:
				return "convT2E16"
			case from.Size() == 4 && from.Align == 4 && !types.Haspointers(from):
				return "convT2E32"
			case from.Size() == 8 && from.Align == types.Types[TUINT64].Align && !types.Haspointers(from):
				return "convT2E64"
			case from.IsString():
				return "convT2Estring"
			case from.IsSlice():
				return "convT2Eslice"
			case !types.Haspointers(from):
				return "convT2Enoptr"
			}
			return "convT2E"
		case 'I':
			switch {
			case from.Size() == 2 && from.Align == 2:
				return "convT2I16"
			case from.Size() == 4 && from.Align == 4 && !types.Haspointers(from):
				return "convT2I32"
			case from.Size() == 8 && from.Align == types.Types[TUINT64].Align && !types.Haspointers(from):
				return "convT2I64"
			case from.IsString():
				return "convT2Istring"
			case from.IsSlice():
				return "convT2Islice"
			case !types.Haspointers(from):
				return "convT2Inoptr"
			}
			return "convT2I"
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
		Dump("walk-before", n)
	}

	if n.Typecheck() != 1 {
		Fatalf("missed typecheck: %+v", n)
	}

	if n.Op == ONAME && n.Class() == PAUTOHEAP {
		nn := nod(OIND, n.Name.Param.Heapaddr, nil)
		nn = typecheck(nn, Erv)
		nn = walkexpr(nn, init)
		nn.Left.SetNonNil(true)
		return nn
	}

opswitch:
	switch n.Op {
	default:
		Dump("walk", n)
		Fatalf("walkexpr: switch 1 unknown op %+S", n)

	case ONONAME, OINDREGSP, OEMPTY, OGETG:

	case OTYPE, ONAME, OLITERAL:
		// TODO(mdempsky): Just return n; see discussion on CL 38655.
		// Perhaps refactor to use Node.mayBeShared for these instead.
		// If these return early, make sure to still call
		// stringsym for constant strings.

	case ONOT, OMINUS, OPLUS, OCOM, OREAL, OIMAG, ODOTMETH, ODOTINTER,
		OIND, OSPTR, OITAB, OIDATA, OADDR:
		n.Left = walkexpr(n.Left, init)

	case OEFACE, OAND, OSUB, OMUL, OLT, OLE, OGE, OGT, OADD, OOR, OXOR:
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)

	case ODOT:
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

	case ODOTPTR:
		usefield(n)
		if n.Op == ODOTPTR && n.Left.Type.Elem().Width == 0 {
			// No actual copy will be generated, so emit an explicit nil check.
			n.Left = cheapexpr(n.Left, init)

			checknil(n.Left, init)
		}

		n.Left = walkexpr(n.Left, init)

	case OLEN, OCAP:
		n.Left = walkexpr(n.Left, init)

		// replace len(*[10]int) with 10.
		// delayed until now to preserve side effects.
		t := n.Left.Type

		if t.IsPtr() {
			t = t.Elem()
		}
		if t.IsArray() {
			safeexpr(n.Left, init)
			nodconst(n, n.Type, t.NumElem())
			n.SetTypecheck(1)
		}

	case OLSH, ORSH:
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)
		t := n.Left.Type
		n.SetBounded(bounded(n.Right, 8*t.Width))
		if Debug['m'] != 0 && n.Etype != 0 && !Isconst(n.Right, CTINT) {
			Warn("shift bounds check elided")
		}

	case OCOMPLEX:
		// Use results from call expression as arguments for complex.
		if n.Left == nil && n.Right == nil {
			n.Left = n.List.First()
			n.Right = n.List.Second()
		}
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)

	case OEQ, ONE:
		n.Left = walkexpr(n.Left, init)
		n.Right = walkexpr(n.Right, init)

		// Disable safemode while compiling this code: the code we
		// generate internally can refer to unsafe.Pointer.
		// In this case it can happen if we need to generate an ==
		// for a struct containing a reflect.Value, which itself has
		// an unexported field of type unsafe.Pointer.
		old_safemode := safemode
		safemode = false
		n = walkcompare(n, init)
		safemode = old_safemode

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
		walkexprlist(n.List.Slice(), init)
		n = walkprint(n, init)

	case OPANIC:
		n = mkcall("gopanic", nil, init, n.Left)

	case ORECOVER:
		n = mkcall("gorecover", n.Type, init, nod(OADDR, nodfp, nil))

	case OCLOSUREVAR, OCFUNC:
		n.SetAddable(true)

	case OCALLINTER:
		usemethod(n)
		t := n.Left.Type
		if n.List.Len() != 0 && n.List.First().Op == OAS {
			break
		}
		n.Left = walkexpr(n.Left, init)
		walkexprlist(n.List.Slice(), init)
		ll := ascompatte(n, n.Isddd(), t.Params(), n.List.Slice(), 0, init)
		n.List.Set(reorder1(ll))

	case OCALLFUNC:
		if n.Left.Op == OCLOSURE {
			// Transform direct call of a closure to call of a normal function.
			// transformclosure already did all preparation work.

			// Prepend captured variables to argument list.
			n.List.Prepend(n.Left.Func.Enter.Slice()...)

			n.Left.Func.Enter.Set(nil)

			// Replace OCLOSURE with ONAME/PFUNC.
			n.Left = n.Left.Func.Closure.Func.Nname

			// Update type of OCALLFUNC node.
			// Output arguments had not changed, but their offsets could.
			if n.Left.Type.Results().NumFields() == 1 {
				n.Type = n.Left.Type.Results().Field(0).Type
			} else {
				n.Type = n.Left.Type.Results()
			}
		}

		t := n.Left.Type
		if n.List.Len() != 0 && n.List.First().Op == OAS {
			break
		}

		n.Left = walkexpr(n.Left, init)
		walkexprlist(n.List.Slice(), init)

		ll := ascompatte(n, n.Isddd(), t.Params(), n.List.Slice(), 0, init)
		n.List.Set(reorder1(ll))

	case OCALLMETH:
		t := n.Left.Type
		if n.List.Len() != 0 && n.List.First().Op == OAS {
			break
		}
		n.Left = walkexpr(n.Left, init)
		walkexprlist(n.List.Slice(), init)
		ll := ascompatte(n, false, t.Recvs(), []*Node{n.Left.Left}, 0, init)
		lr := ascompatte(n, n.Isddd(), t.Params(), n.List.Slice(), 0, init)
		ll = append(ll, lr...)
		n.Left.Left = nil
		updateHasCall(n.Left)
		n.List.Set(reorder1(ll))

	case OAS:
		init.AppendNodes(&n.Ninit)

		n.Left = walkexpr(n.Left, init)
		n.Left = safeexpr(n.Left, init)

		if oaslit(n, init) {
			break
		}

		if n.Right == nil {
			// TODO(austin): Check all "implicit zeroing"
			break
		}

		if !instrumenting && iszero(n.Right) {
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
			if r.Isddd() {
				r = appendslice(r, init) // also works for append(slice, string).
			} else {
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

		ll := ascompatet(n.Op, n.List, r.Type)
		n = liststmt(ll)

	// x, y = <-c
	// orderstmt made sure x is addressable.
	case OAS2RECV:
		init.AppendNodes(&n.Ninit)

		r := n.Rlist.First()
		walkexprlistsafe(n.List.Slice(), init)
		r.Left = walkexpr(r.Left, init)
		var n1 *Node
		if isblank(n.List.First()) {
			n1 = nodnil()
		} else {
			n1 = nod(OADDR, n.List.First(), nil)
		}
		n1.Etype = 1 // addr does not escape
		fn := chanfn("chanrecv2", 2, r.Left.Type)
		ok := n.List.Second()
		call := mkcall1(fn, ok.Type, init, r.Left, n1)
		n = nod(OAS, ok, call)
		n = typecheck(n, Etop)

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

		if w := t.Val().Width; w <= 1024 { // 1024 must match ../../../../runtime/hashmap.go:maxZero
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
		if ok := n.List.Second(); !isblank(ok) && ok.Type.IsBoolean() {
			r.Type.Field(1).Type = ok.Type
		}
		n.Rlist.Set1(r)
		n.Op = OAS2FUNC

		// don't generate a = *var if a is _
		if !isblank(a) {
			var_ := temp(types.NewPtr(t.Val()))
			var_.SetTypecheck(1)
			var_.SetNonNil(true) // mapaccess always returns a non-nil pointer
			n.List.SetFirst(var_)
			n = walkexpr(n, init)
			init.Append(n)
			n = nod(OAS, a, nod(OIND, var_, nil))
		}

		n = typecheck(n, Etop)
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

		// Optimize convT2E or convT2I as a two-word copy when T is pointer-shaped.
		if isdirectiface(n.Left.Type) {
			var t *Node
			if n.Type.IsEmptyInterface() {
				t = typename(n.Left.Type)
			} else {
				t = itabname(n.Left.Type, n.Type)
			}
			l := nod(OEFACE, t, n.Left)
			l.Type = n.Type
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
		case n.Left.Type.Size() == 0:
			// n.Left is zero-sized. Use zerobase.
			cheapexpr(n.Left, init) // Evaluate n.Left for side-effects. See issue 19246.
			value = zerobase
		case n.Left.Type.IsBoolean() || (n.Left.Type.Size() == 1 && n.Left.Type.IsInteger()):
			// n.Left is a bool/byte. Use staticbytes[n.Left].
			n.Left = cheapexpr(n.Left, init)
			value = nod(OINDEX, staticbytes, byteindex(n.Left))
			value.SetBounded(true)
		case n.Left.Class() == PEXTERN && n.Left.Name != nil && n.Left.Name.Readonly():
			// n.Left is a readonly global; use it directly.
			value = n.Left
		case !n.Left.Type.IsInterface() && n.Esc == EscNone && n.Left.Type.Width <= 1024:
			// n.Left does not escape. Use a stack temporary initialized to n.Left.
			value = temp(n.Left.Type)
			init.Append(typecheck(nod(OAS, value, n.Left), Etop))
		}

		if value != nil {
			// Value is identical to n.Left.
			// Construct the interface directly: {type/itab, &value}.
			var t *Node
			if n.Type.IsEmptyInterface() {
				t = typename(n.Left.Type)
			} else {
				t = itabname(n.Left.Type, n.Type)
			}
			l := nod(OEFACE, t, typecheck(nod(OADDR, value, nil), Erv))
			l.Type = n.Type
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
		if n.Type.IsEmptyInterface() && n.Left.Type.IsInterface() && !n.Left.Type.IsEmptyInterface() {
			// Evaluate the input interface.
			c := temp(n.Left.Type)
			init.Append(nod(OAS, c, n.Left))

			// Get the itab out of the interface.
			tmp := temp(types.NewPtr(types.Types[TUINT8]))
			init.Append(nod(OAS, tmp, typecheck(nod(OITAB, c, nil), Erv)))

			// Get the type out of the itab.
			nif := nod(OIF, typecheck(nod(ONE, tmp, nodnil()), Erv), nil)
			nif.Nbody.Set1(nod(OAS, tmp, itabType(tmp)))
			init.Append(nif)

			// Build the result.
			e := nod(OEFACE, tmp, ifaceData(c, types.NewPtr(types.Types[TUINT8])))
			e.Type = n.Type // assign type manually, typecheck doesn't understand OEFACE.
			e.SetTypecheck(1)
			n = e
			break
		}

		var ll []*Node
		if n.Type.IsEmptyInterface() {
			if !n.Left.Type.IsInterface() {
				ll = append(ll, typename(n.Left.Type))
			}
		} else {
			if n.Left.Type.IsInterface() {
				ll = append(ll, typename(n.Type))
			} else {
				ll = append(ll, itabname(n.Left.Type, n.Type))
			}
		}

		if n.Left.Type.IsInterface() {
			ll = append(ll, n.Left)
		} else {
			// regular types are passed by reference to avoid C vararg calls
			// orderexpr arranged for n.Left to be a temporary for all
			// the conversions it could see. comparison of an interface
			// with a non-interface, especially in a switch on interface value
			// with non-interface cases, is not visible to orderstmt, so we
			// have to fall back on allocating a temp here.
			if islvalue(n.Left) {
				ll = append(ll, nod(OADDR, n.Left, nil))
			} else {
				ll = append(ll, nod(OADDR, copyexpr(n.Left, n.Left.Type, init), nil))
			}
			dowidth(n.Left.Type)
		}

		fn := syslook(convFuncName(n.Left.Type, n.Type))
		fn = substArgTypes(fn, n.Left.Type, n.Type)
		dowidth(fn.Type)
		n = nod(OCALL, fn, nil)
		n.List.Set(ll)
		n = typecheck(n, Erv)
		n = walkexpr(n, init)

	case OCONV, OCONVNOP:
		if thearch.LinkArch.Family == sys.ARM || thearch.LinkArch.Family == sys.MIPS {
			if n.Left.Type.IsFloat() {
				if n.Type.Etype == TINT64 {
					n = mkcall("float64toint64", n.Type, init, conv(n.Left, types.Types[TFLOAT64]))
					break
				}

				if n.Type.Etype == TUINT64 {
					n = mkcall("float64touint64", n.Type, init, conv(n.Left, types.Types[TFLOAT64]))
					break
				}
			}

			if n.Type.IsFloat() {
				if n.Left.Type.Etype == TINT64 {
					n = conv(mkcall("int64tofloat64", types.Types[TFLOAT64], init, conv(n.Left, types.Types[TINT64])), n.Type)
					break
				}

				if n.Left.Type.Etype == TUINT64 {
					n = conv(mkcall("uint64tofloat64", types.Types[TFLOAT64], init, conv(n.Left, types.Types[TUINT64])), n.Type)
					break
				}
			}
		}

		if thearch.LinkArch.Family == sys.I386 {
			if n.Left.Type.IsFloat() {
				if n.Type.Etype == TINT64 {
					n = mkcall("float64toint64", n.Type, init, conv(n.Left, types.Types[TFLOAT64]))
					break
				}

				if n.Type.Etype == TUINT64 {
					n = mkcall("float64touint64", n.Type, init, conv(n.Left, types.Types[TFLOAT64]))
					break
				}
				if n.Type.Etype == TUINT32 || n.Type.Etype == TUINT || n.Type.Etype == TUINTPTR {
					n = mkcall("float64touint32", n.Type, init, conv(n.Left, types.Types[TFLOAT64]))
					break
				}
			}
			if n.Type.IsFloat() {
				if n.Left.Type.Etype == TINT64 {
					n = conv(mkcall("int64tofloat64", types.Types[TFLOAT64], init, conv(n.Left, types.Types[TINT64])), n.Type)
					break
				}

				if n.Left.Type.Etype == TUINT64 {
					n = conv(mkcall("uint64tofloat64", types.Types[TFLOAT64], init, conv(n.Left, types.Types[TUINT64])), n.Type)
					break
				}
				if n.Left.Type.Etype == TUINT32 || n.Left.Type.Etype == TUINT || n.Left.Type.Etype == TUINTPTR {
					n = conv(mkcall("uint32tofloat64", types.Types[TFLOAT64], init, conv(n.Left, types.Types[TUINT32])), n.Type)
					break
				}
			}
		}

		n.Left = walkexpr(n.Left, init)

	case OANDNOT:
		n.Left = walkexpr(n.Left, init)
		n.Op = OAND
		n.Right = nod(OCOM, n.Right, nil)
		n.Right = typecheck(n.Right, Erv)
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
		if n.Etype == 1 {
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

			if w := t.Val().Width; w <= 1024 { // 1024 must match ../../../../runtime/hashmap.go:maxZero
				n = mkcall1(mapfn(mapaccess1[fast], t), types.NewPtr(t.Val()), init, typename(t), map_, key)
			} else {
				z := zeroaddr(w)
				n = mkcall1(mapfn("mapaccess1_fat", t), types.NewPtr(t.Val()), init, typename(t), map_, key, z)
			}
		}
		n.Type = types.NewPtr(t.Val())
		n.SetNonNil(true) // mapaccess1* and mapassign always return non-nil pointers.
		n = nod(OIND, n, nil)
		n.Type = t.Val()
		n.SetTypecheck(1)

	case ORECV:
		Fatalf("walkexpr ORECV") // should see inside OAS only

	case OSLICE, OSLICEARR, OSLICESTR, OSLICE3, OSLICE3ARR:
		n.Left = walkexpr(n.Left, init)
		low, high, max := n.SliceBounds()
		low = walkexpr(low, init)
		if low != nil && iszero(low) {
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
			if n.Type.Elem().Width >= 1<<16 {
				Fatalf("large ONEW with EscNone: %v", n)
			}
			r := temp(n.Type.Elem())
			r = nod(OAS, r, nil) // zero temp
			r = typecheck(r, Etop)
			init.Append(r)
			r = nod(OADDR, r.Left, nil)
			r = typecheck(r, Erv)
			n = r
		} else {
			n = callnew(n.Type.Elem())
		}

	case OCMPSTR:
		// s + "badgerbadgerbadger" == "badgerbadgerbadger"
		if (Op(n.Etype) == OEQ || Op(n.Etype) == ONE) && Isconst(n.Right, CTSTR) && n.Left.Op == OADDSTR && n.Left.List.Len() == 2 && Isconst(n.Left.List.Second(), CTSTR) && strlit(n.Right) == strlit(n.Left.List.Second()) {
			// TODO(marvin): Fix Node.EType type union.
			r := nod(Op(n.Etype), nod(OLEN, n.Left.List.First(), nil), nodintconst(0))
			r = typecheck(r, Erv)
			r = walkexpr(r, init)
			r.Type = n.Type
			n = r
			break
		}

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
			cmp := Op(n.Etype)
			// maxRewriteLen was chosen empirically.
			// It is the value that minimizes cmd/go file size
			// across most architectures.
			// See the commit description for CL 26758 for details.
			maxRewriteLen := 6
			// Some architectures can load unaligned byte sequence as 1 word.
			// So we can cover longer strings with the same amount of code.
			canCombineLoads := false
			combine64bit := false
			// TODO: does this improve performance on any other architectures?
			switch thearch.LinkArch.Family {
			case sys.AMD64:
				// Larger compare require longer instructions, so keep this reasonably low.
				// Data from CL 26758 shows that longer strings are rare.
				// If we really want we can do 16 byte SSE comparisons in the future.
				maxRewriteLen = 16
				canCombineLoads = true
				combine64bit = true
			case sys.I386:
				maxRewriteLen = 8
				canCombineLoads = true
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
				// TODO(marvin): Fix Node.EType type union.
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
						csubstr = csubstr | int64(s[i+offset])<<uint8(8*offset)
					}
					csubstrPart := nodintconst(csubstr)
					// Compare "step" bytes as once
					r = nod(and, r, nod(cmp, csubstrPart, ncsubstr))
					remains -= step
					i += step
				}
				r = typecheck(r, Erv)
				r = walkexpr(r, init)
				r.Type = n.Type
				n = r
				break
			}
		}

		var r *Node
		// TODO(marvin): Fix Node.EType type union.
		if Op(n.Etype) == OEQ || Op(n.Etype) == ONE {
			// prepare for rewrite below
			n.Left = cheapexpr(n.Left, init)
			n.Right = cheapexpr(n.Right, init)

			r = mkcall("eqstring", types.Types[TBOOL], init, conv(n.Left, types.Types[TSTRING]), conv(n.Right, types.Types[TSTRING]))

			// quick check of len before full compare for == or !=
			// eqstring assumes that the lengths are equal
			// TODO(marvin): Fix Node.EType type union.
			if Op(n.Etype) == OEQ {
				// len(left) == len(right) && eqstring(left, right)
				r = nod(OANDAND, nod(OEQ, nod(OLEN, n.Left, nil), nod(OLEN, n.Right, nil)), r)
			} else {
				// len(left) != len(right) || !eqstring(left, right)
				r = nod(ONOT, r, nil)
				r = nod(OOROR, nod(ONE, nod(OLEN, n.Left, nil), nod(OLEN, n.Right, nil)), r)
			}

			r = typecheck(r, Erv)
			r = walkexpr(r, nil)
		} else {
			// sys_cmpstring(s1, s2) :: 0
			r = mkcall("cmpstring", types.Types[TINT], init, conv(n.Left, types.Types[TSTRING]), conv(n.Right, types.Types[TSTRING]))
			// TODO(marvin): Fix Node.EType type union.
			r = nod(Op(n.Etype), r, nodintconst(0))
		}

		r = typecheck(r, Erv)
		if !n.Type.IsBoolean() {
			Fatalf("cmp %v", n.Type)
		}
		r.Type = n.Type
		n = r

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
		n = mkcall1(chanfn("makechan", 1, n.Type), n.Type, init, typename(n.Type), conv(n.Left, types.Types[TINT64]))

	case OMAKEMAP:
		t := n.Type

		a := nodnil() // hmap buffer
		r := nodnil() // bucket buffer
		if n.Esc == EscNone {
			// Allocate hmap buffer on stack.
			var_ := temp(hmap(t))

			a = nod(OAS, var_, nil) // zero temp
			a = typecheck(a, Etop)
			init.Append(a)
			a = nod(OADDR, var_, nil)

			// Allocate one bucket on stack.
			// Maximum key/value size is 128 bytes, larger objects
			// are stored with an indirection. So max bucket size is 2048+eps.
			var_ = temp(mapbucket(t))

			r = nod(OAS, var_, nil) // zero temp
			r = typecheck(r, Etop)
			init.Append(r)
			r = nod(OADDR, var_, nil)
		}

		fn := syslook("makemap")
		fn = substArgTypes(fn, hmap(t), mapbucket(t), t.Key(), t.Val())
		n = mkcall1(fn, n.Type, init, typename(n.Type), conv(n.Left, types.Types[TINT64]), a, r)

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
			t = types.NewArray(t.Elem(), nonnegintconst(r)) // [r]T
			var_ := temp(t)
			a := nod(OAS, var_, nil) // zero temp
			a = typecheck(a, Etop)
			init.Append(a)
			r := nod(OSLICE, var_, nil) // arr[:l]
			r.SetSliceBounds(nil, l, nil)
			r = conv(r, n.Type) // in case n.Type is named.
			r = typecheck(r, Erv)
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

			// typechecking guarantees that TIDEAL len/cap are positive and fit in an int.
			// The case of len or cap overflow when converting TUINT or TUINTPTR to TINT
			// will be handled by the negative range checks in makeslice during runtime.
			if (len.Type.IsKind(TIDEAL) || maxintval[len.Type.Etype].Cmp(maxintval[TUINT]) <= 0) &&
				(cap.Type.IsKind(TIDEAL) || maxintval[cap.Type.Etype].Cmp(maxintval[TUINT]) <= 0) {
				fnname = "makeslice"
				argtype = types.Types[TINT]
			}

			fn := syslook(fnname)
			fn = substArgTypes(fn, t.Elem()) // any-1
			n = mkcall1(fn, t, init, typename(t.Elem()), conv(len, argtype), conv(cap, argtype))
		}

	case ORUNESTR:
		a := nodnil()
		if n.Esc == EscNone {
			t := types.NewArray(types.Types[TUINT8], 4)
			var_ := temp(t)
			a = nod(OADDR, var_, nil)
		}

		// intstring(*[4]byte, rune)
		n = mkcall("intstring", n.Type, init, a, conv(n.Left, types.Types[TINT64]))

	case OARRAYBYTESTR:
		a := nodnil()
		if n.Esc == EscNone {
			// Create temporary buffer for string on stack.
			t := types.NewArray(types.Types[TUINT8], tmpstringbufsize)

			a = nod(OADDR, temp(t), nil)
		}

		// slicebytetostring(*[32]byte, []byte) string;
		n = mkcall("slicebytetostring", n.Type, init, a, n.Left)

		// slicebytetostringtmp([]byte) string;
	case OARRAYBYTESTRTMP:
		n.Left = walkexpr(n.Left, init)

		if !instrumenting {
			// Let the backend handle OARRAYBYTESTRTMP directly
			// to avoid a function call to slicebytetostringtmp.
			break
		}

		n = mkcall("slicebytetostringtmp", n.Type, init, n.Left)

		// slicerunetostring(*[32]byte, []rune) string;
	case OARRAYRUNESTR:
		a := nodnil()

		if n.Esc == EscNone {
			// Create temporary buffer for string on stack.
			t := types.NewArray(types.Types[TUINT8], tmpstringbufsize)

			a = nod(OADDR, temp(t), nil)
		}

		n = mkcall("slicerunetostring", n.Type, init, a, n.Left)

		// stringtoslicebyte(*32[byte], string) []byte;
	case OSTRARRAYBYTE:
		a := nodnil()

		if n.Esc == EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[TUINT8], tmpstringbufsize)

			a = nod(OADDR, temp(t), nil)
		}

		n = mkcall("stringtoslicebyte", n.Type, init, a, conv(n.Left, types.Types[TSTRING]))

	case OSTRARRAYBYTETMP:
		// []byte(string) conversion that creates a slice
		// referring to the actual string bytes.
		// This conversion is handled later by the backend and
		// is only for use by internal compiler optimizations
		// that know that the slice won't be mutated.
		// The only such case today is:
		// for i, c := range []byte(string)
		n.Left = walkexpr(n.Left, init)

		// stringtoslicerune(*[32]rune, string) []rune
	case OSTRARRAYRUNE:
		a := nodnil()

		if n.Esc == EscNone {
			// Create temporary buffer for slice on stack.
			t := types.NewArray(types.Types[TINT32], tmpstringbufsize)

			a = nod(OADDR, temp(t), nil)
		}

		n = mkcall("stringtoslicerune", n.Type, init, a, n.Left)

		// ifaceeq(i1 any-1, i2 any-2) (ret bool);
	case OCMPIFACE:
		if !eqtype(n.Left.Type, n.Right.Type) {
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
		// TODO(marvin): Fix Node.EType type union.
		var cmp *Node
		if Op(n.Etype) == OEQ {
			cmp = nod(OANDAND, nod(OEQ, lt, rt), call)
		} else {
			cmp = nod(OOROR, nod(ONE, lt, rt), nod(ONOT, call, nil))
		}
		cmp = typecheck(cmp, Erv)
		cmp = walkexpr(cmp, init)
		cmp.Type = n.Type
		n = cmp

	case OARRAYLIT, OSLICELIT, OMAPLIT, OSTRUCTLIT, OPTRLIT:
		if isStaticCompositeLiteral(n) && !canSSAType(n.Type) {
			// n can be directly represented in the read-only data section.
			// Make direct reference to the static data. See issue 12841.
			vstat := staticname(n.Type)
			vstat.Name.SetReadonly(true)
			fixedlit(inInitFunction, initKindStatic, n, vstat, init)
			n = vstat
			n = typecheck(n, Erv)
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
		n = typecheck(n, Erv)
		// Emit string symbol now to avoid emitting
		// any concurrently during the backend.
		if s, ok := n.Val().U.(string); ok {
			_ = stringsym(s)
		}
	}

	updateHasCall(n)

	if Debug['w'] != 0 && n != nil {
		Dump("walk", n)
	}

	lineno = lno
	return n
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

func ascompatee1(op Op, l *Node, r *Node, init *Nodes) *Node {
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
	// a expression list. called in
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
		nn = append(nn, ascompatee1(op, nl[i], nr[i], init))
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

// l is an lv and rt is the type of an rv
// return 1 if this implies a function call
// evaluating the lv or a function call
// in the conversion of the types
func fncall(l *Node, rt *types.Type) bool {
	if l.HasCall() || l.Op == OINDEXMAP {
		return true
	}
	if needwritebarrier(l) {
		return true
	}
	if eqtype(l.Type, rt) {
		return false
	}
	return true
}

// check assign type list to
// a expression list. called in
//	expr-list = func()
func ascompatet(op Op, nl Nodes, nr *types.Type) []*Node {
	if nl.Len() != nr.NumFields() {
		Fatalf("ascompatet: assignment count mismatch: %d = %d", nl.Len(), nr.NumFields())
	}

	var nn, mm Nodes
	for i, l := range nl.Slice() {
		if isblank(l) {
			continue
		}
		r := nr.Field(i)

		// any lv that causes a fn call must be
		// deferred until all the return arguments
		// have been pulled from the output arguments
		if fncall(l, r.Type) {
			tmp := temp(r.Type)
			tmp = typecheck(tmp, Erv)
			a := nod(OAS, l, tmp)
			a = convas(a, &mm)
			mm.Append(a)
			l = tmp
		}

		a := nod(OAS, l, nodarg(r, 0))
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

// nodarg returns a Node for the function argument denoted by t,
// which is either the entire function argument or result struct (t is a  struct *types.Type)
// or a specific argument (t is a *types.Field within a struct *types.Type).
//
// If fp is 0, the node is for use by a caller invoking the given
// function, preparing the arguments before the call
// or retrieving the results after the call.
// In this case, the node will correspond to an outgoing argument
// slot like 8(SP).
//
// If fp is 1, the node is for use by the function itself
// (the callee), to retrieve its arguments or write its results.
// In this case the node will be an ONAME with an appropriate
// type and offset.
func nodarg(t interface{}, fp int) *Node {
	var n *Node

	var funarg types.Funarg
	switch t := t.(type) {
	default:
		Fatalf("bad nodarg %T(%v)", t, t)

	case *types.Type:
		// Entire argument struct, not just one arg
		if !t.IsFuncArgStruct() {
			Fatalf("nodarg: bad type %v", t)
		}
		funarg = t.StructType().Funarg

		// Build fake variable name for whole arg struct.
		n = newname(lookup(".args"))
		n.Type = t
		first := t.Field(0)
		if first == nil {
			Fatalf("nodarg: bad struct")
		}
		if first.Offset == BADWIDTH {
			Fatalf("nodarg: offset not computed for %v", t)
		}
		n.Xoffset = first.Offset

	case *types.Field:
		funarg = t.Funarg
		if fp == 1 {
			// NOTE(rsc): This should be using t.Nname directly,
			// except in the case where t.Nname.Sym is the blank symbol and
			// so the assignment would be discarded during code generation.
			// In that case we need to make a new node, and there is no harm
			// in optimization passes to doing so. But otherwise we should
			// definitely be using the actual declaration and not a newly built node.
			// The extra Fatalf checks here are verifying that this is the case,
			// without changing the actual logic (at time of writing, it's getting
			// toward time for the Go 1.7 beta).
			// At some quieter time (assuming we've never seen these Fatalfs happen)
			// we could change this code to use "expect" directly.
			expect := asNode(t.Nname)
			if expect.isParamHeapCopy() {
				expect = expect.Name.Param.Stackcopy
			}

			for _, n := range Curfn.Func.Dcl {
				if (n.Class() == PPARAM || n.Class() == PPARAMOUT) && !t.Sym.IsBlank() && n.Sym == t.Sym {
					if n != expect {
						Fatalf("nodarg: unexpected node: %v (%p %v) vs %v (%p %v)", n, n, n.Op, asNode(t.Nname), asNode(t.Nname), asNode(t.Nname).Op)
					}
					return n
				}
			}

			if !expect.Sym.IsBlank() {
				Fatalf("nodarg: did not find node in dcl list: %v", expect)
			}
		}

		// Build fake name for individual variable.
		// This is safe because if there was a real declared name
		// we'd have used it above.
		n = newname(lookup("__"))
		n.Type = t.Type
		if t.Offset == BADWIDTH {
			Fatalf("nodarg: offset not computed for %v", t)
		}
		n.Xoffset = t.Offset
		n.Orig = asNode(t.Nname)
	}

	// Rewrite argument named _ to __,
	// or else the assignment to _ will be
	// discarded during code generation.
	if isblank(n) {
		n.Sym = lookup("__")
	}

	switch fp {
	default:
		Fatalf("bad fp")

	case 0: // preparing arguments for call
		n.Op = OINDREGSP
		n.Xoffset += Ctxt.FixedFrameSize()

	case 1: // reading arguments inside call
		n.SetClass(PPARAM)
		if funarg == types.FunargResults {
			n.SetClass(PPARAMOUT)
		}
	}

	n.SetTypecheck(1)
	n.SetAddrtaken(true) // keep optimizers at bay
	return n
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
	n = typecheck(n, Erv)
	if n.Type == nil {
		Fatalf("mkdotargslice: typecheck failed")
	}
	n = walkexpr(n, init)
	return n
}

// check assign expression list to
// a type list. called in
//	return expr-list
//	func(expr-list)
func ascompatte(call *Node, isddd bool, lhs *types.Type, rhs []*Node, fp int, init *Nodes) []*Node {
	var nn []*Node

	// f(g()) where g has multiple return values
	if len(rhs) == 1 && rhs[0].Type.IsFuncArgStruct() {
		// optimization - can do block copy
		if eqtypenoname(rhs[0].Type, lhs) {
			nl := nodarg(lhs, fp)
			nr := nod(OCONVNOP, rhs[0], nil)
			nr.Type = nl.Type
			nn = []*Node{convas(nod(OAS, nl, nr), init)}
			goto ret
		}

		// conversions involved.
		// copy into temporaries.
		var tmps []*Node
		for _, nr := range rhs[0].Type.FieldSlice() {
			tmps = append(tmps, temp(nr.Type))
		}

		a := nod(OAS2, nil, nil)
		a.List.Set(tmps)
		a.Rlist.Set(rhs)
		a = typecheck(a, Etop)
		a = walkstmt(a)
		init.Append(a)

		rhs = tmps
	}

	// For each parameter (LHS), assign its corresponding argument (RHS).
	// If there's a ... parameter (which is only valid as the final
	// parameter) and this is not a ... call expression,
	// then assign the remaining arguments as a slice.
	for i, nl := range lhs.FieldSlice() {
		var nr *Node
		if nl.Isddd() && !isddd {
			nr = mkdotargslice(nl.Type, rhs[i:], init, call.Right)
		} else {
			nr = rhs[i]
		}

		a := nod(OAS, nodarg(nl, fp), nr)
		a = convas(a, init)
		nn = append(nn, a)
	}

ret:
	for _, n := range nn {
		n.SetTypecheck(1)
	}
	return nn
}

// generate code for print
func walkprint(nn *Node, init *Nodes) *Node {
	var r *Node
	var n *Node
	var on *Node
	var t *types.Type
	var et types.EType

	op := nn.Op
	all := nn.List
	var calls []*Node
	notfirst := false

	// Hoist all the argument evaluation up before the lock.
	walkexprlistcheap(all.Slice(), init)

	calls = append(calls, mkcall("printlock", nil, init))
	for i1, n1 := range all.Slice() {
		if notfirst {
			calls = append(calls, mkcall("printsp", nil, init))
		}

		notfirst = op == OPRINTN

		n = n1
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
		all.SetIndex(i1, n)
		if n.Type == nil || n.Type.Etype == TFORW {
			continue
		}

		t = n.Type
		et = n.Type.Etype
		if n.Type.IsInterface() {
			if n.Type.IsEmptyInterface() {
				on = syslook("printeface")
			} else {
				on = syslook("printiface")
			}
			on = substArgTypes(on, n.Type) // any-1
		} else if n.Type.IsPtr() || et == TCHAN || et == TMAP || et == TFUNC || et == TUNSAFEPTR {
			on = syslook("printpointer")
			on = substArgTypes(on, n.Type) // any-1
		} else if n.Type.IsSlice() {
			on = syslook("printslice")
			on = substArgTypes(on, n.Type) // any-1
		} else if isInt[et] {
			if et == TUINT64 {
				if isRuntimePkg(t.Sym.Pkg) && t.Sym.Name == "hex" {
					on = syslook("printhex")
				} else {
					on = syslook("printuint")
				}
			} else {
				on = syslook("printint")
			}
		} else if isFloat[et] {
			on = syslook("printfloat")
		} else if isComplex[et] {
			on = syslook("printcomplex")
		} else if et == TBOOL {
			on = syslook("printbool")
		} else if et == TSTRING {
			on = syslook("printstring")
		} else {
			badtype(OPRINT, n.Type, nil)
			continue
		}

		t = on.Type.Params().Field(0).Type

		if !eqtype(t, n.Type) {
			n = nod(OCONV, n, nil)
			n.Type = t
		}

		r = nod(OCALL, on, nil)
		r.List.Append(n)
		calls = append(calls, r)
	}

	if op == OPRINTN {
		calls = append(calls, mkcall("printnl", nil, nil))
	}

	calls = append(calls, mkcall("printunlock", nil, init))

	typecheckslice(calls, Etop)
	walkexprlist(calls, init)

	r = nod(OEMPTY, nil, nil)
	r = typecheck(r, Etop)
	r = walkexpr(r, init)
	r.Ninit.Set(calls)
	return r
}

func callnew(t *types.Type) *Node {
	if t.NotInHeap() {
		yyerror("%v is go:notinheap; heap allocation disallowed", t)
	}
	dowidth(t)
	fn := syslook("newobject")
	fn = substArgTypes(fn, t)
	v := mkcall1(fn, types.NewPtr(t), nil, typename(t))
	v.SetNonNil(true)
	return v
}

func iscallret(n *Node) bool {
	n = outervalue(n)
	return n.Op == OINDREGSP
}

func isstack(n *Node) bool {
	n = outervalue(n)

	// If n is *autotmp and autotmp = &foo, replace n with foo.
	// We introduce such temps when initializing struct literals.
	if n.Op == OIND && n.Left.Op == ONAME && n.Left.IsAutoTmp() {
		defn := n.Left.Name.Defn
		if defn != nil && defn.Op == OAS && defn.Right.Op == OADDR {
			n = defn.Right.Left
		}
	}

	switch n.Op {
	case OINDREGSP:
		return true

	case ONAME:
		switch n.Class() {
		case PAUTO, PPARAM, PPARAMOUT:
			return true
		}
	}

	return false
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

// Do we need a write barrier for assigning to l?
func needwritebarrier(l *Node) bool {
	if !use_writebarrier {
		return false
	}

	if l == nil || isblank(l) {
		return false
	}

	// No write barrier for write to stack.
	if isstack(l) {
		return false
	}

	// Package unsafe's documentation says storing pointers into
	// reflect.SliceHeader and reflect.StringHeader's Data fields
	// is valid, even though they have type uintptr (#19168).
	if isReflectHeaderDataField(l) {
		return true
	}

	// No write barrier for write of non-pointers.
	dowidth(l.Type)
	if !types.Haspointers(l.Type) {
		return false
	}

	// No write barrier if this is a pointer to a go:notinheap
	// type, since the write barrier's inheap(ptr) check will fail.
	if l.Type.IsPtr() && l.Type.Elem().NotInHeap() {
		return false
	}

	// TODO: We can eliminate write barriers if we know *both* the
	// current and new content of the slot must already be shaded.
	// We know a pointer is shaded if it's nil, or points to
	// static data, a global (variable or function), or the stack.
	// The nil optimization could be particularly useful for
	// writes to just-allocated objects. Unfortunately, knowing
	// the "current" value of the slot requires flow analysis.

	// Otherwise, be conservative and use write barrier.
	return true
}

func convas(n *Node, init *Nodes) *Node {
	if n.Op != OAS {
		Fatalf("convas: not OAS %v", n.Op)
	}

	n.SetTypecheck(1)

	var lt *types.Type
	var rt *types.Type
	if n.Left == nil || n.Right == nil {
		goto out
	}

	lt = n.Left.Type
	rt = n.Right.Type
	if lt == nil || rt == nil {
		goto out
	}

	if isblank(n.Left) {
		n.Right = defaultlit(n.Right, nil)
		goto out
	}

	if !eqtype(lt, rt) {
		n.Right = assignconv(n.Right, lt, "assignment")
		n.Right = walkexpr(n.Right, init)
	}
	dowidth(n.Right.Type)

out:
	updateHasCall(n)
	return n
}

// from ascompat[te]
// evaluating actual function arguments.
//	f(a,b)
// if there is exactly one function expr,
// then it is done first. otherwise must
// make temp variables
func reorder1(all []*Node) []*Node {
	c := 0 // function calls
	t := 0 // total parameters

	for _, n := range all {
		t++
		updateHasCall(n)
		if n.HasCall() {
			c++
		}
	}

	if c == 0 || t == 1 {
		return all
	}

	var g []*Node // fncalls assigned to tempnames
	var f *Node   // last fncall assigned to stack
	var r []*Node // non fncalls and tempnames assigned to stack
	d := 0
	var a *Node
	for _, n := range all {
		if !n.HasCall() {
			r = append(r, n)
			continue
		}

		d++
		if d == c {
			f = n
			continue
		}

		// make assignment of fncall to tempname
		a = temp(n.Right.Type)

		a = nod(OAS, a, n.Right)
		g = append(g, a)

		// put normal arg assignment on list
		// with fncall replaced by tempname
		n.Right = a.Left

		r = append(r, n)
	}

	if f != nil {
		g = append(g, f)
	}
	return append(g, r...)
}

// from ascompat[ee]
//	a,b = c,d
// simultaneous assignment. there cannot
// be later use of an earlier lvalue.
//
// function calls have been removed.
func reorder3(all []*Node) []*Node {
	var l *Node

	// If a needed expression may be affected by an
	// earlier assignment, make an early copy of that
	// expression and use the copy instead.
	var early []*Node

	var mapinit Nodes
	for i, n := range all {
		l = n.Left

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

		case OIND, ODOTPTR:
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
	q = typecheck(q, Etop)
	*early = append(*early, q)
	return q.Left
}

// what's the outer value that a write to n affects?
// outer value means containing struct or array.
func outervalue(n *Node) *Node {
	for {
		if n.Op == OXDOT {
			Fatalf("OXDOT in walk")
		}
		if n.Op == ODOT || n.Op == OPAREN || n.Op == OCONVNOP {
			n = n.Left
			continue
		}

		if n.Op == OINDEX && n.Left.Type != nil && n.Left.Type.IsArray() {
			n = n.Left
			continue
		}

		break
	}

	return n
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
	memwrite := 0

	varwrite := 0
	var a *Node
	for _, an := range all[:i] {
		a = outervalue(an.Left)

		for a.Op == ODOT {
			a = a.Left
		}

		if a.Op != ONAME {
			memwrite = 1
			continue
		}

		switch n.Class() {
		default:
			varwrite = 1
			continue

		case PAUTO, PPARAM, PPARAMOUT:
			if n.Addrtaken() {
				varwrite = 1
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
	if memwrite == 0 && varwrite == 0 {
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
		OMINUS,
		OCOM,
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
				nn = append(nn, walkstmt(typecheck(nod(OAS, v, stackcopy), Etop)))
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
	lno := lineno
	lineno = Curfn.Pos
	for _, f := range Curfn.Type.Results().Fields().Slice() {
		if v := asNode(f.Nname); v != nil && v.Name.Param.Heapaddr != nil {
			// The local which points to the return value is the
			// thing that needs zeroing. This is already handled
			// by a Needzero annotation in plive.go:livenessepilogue.
			continue
		}
		// Zero the stack location containing f.
		Curfn.Func.Enter.Append(nod(OAS, nodarg(f, 1), nil))
	}
	lineno = lno
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
			nn = append(nn, walkstmt(typecheck(nod(OAS, stackcopy, v), Etop)))
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

	n := fn.Type.Params().NumFields()

	r := nod(OCALL, fn, nil)
	r.List.Set(va[:n])
	if fn.Type.Results().NumFields() > 0 {
		r = typecheck(r, Erv|Efnstruct)
	} else {
		r = typecheck(r, Etop)
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
	if eqtype(n.Type, t) {
		return n
	}
	n = nod(OCONV, n, nil)
	n.Type = t
	n = typecheck(n, Erv)
	return n
}

// byteindex converts n, which is byte-sized, to a uint8.
// We cannot use conv, because we allow converting bool to uint8 here,
// which is forbidden in user code.
func byteindex(n *Node) *Node {
	if eqtype(n.Type, types.Types[TUINT8]) {
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
	fn = substArgTypes(fn, t.Key(), t.Val(), t.Key(), t.Val())
	return fn
}

func mapfndel(name string, t *types.Type) *Node {
	if !t.IsMap() {
		Fatalf("mapfn %v", t)
	}
	fn := syslook(name)
	fn = substArgTypes(fn, t.Key(), t.Val(), t.Key())
	return fn
}

const (
	mapslow = iota
	mapfast32
	mapfast64
	mapfaststr
	nmapfast
)

type mapnames [nmapfast]string

func mkmapnames(base string) mapnames {
	return mapnames{base, base + "_fast32", base + "_fast64", base + "_faststr"}
}

var mapaccess1 mapnames = mkmapnames("mapaccess1")
var mapaccess2 mapnames = mkmapnames("mapaccess2")
var mapassign mapnames = mkmapnames("mapassign")
var mapdelete mapnames = mkmapnames("mapdelete")

func mapfast(t *types.Type) int {
	// Check ../../runtime/hashmap.go:maxValueSize before changing.
	if t.Val().Width > 128 {
		return mapslow
	}
	switch algtype(t.Key()) {
	case AMEM32:
		return mapfast32
	case AMEM64:
		return mapfast64
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
	r = typecheck(r, Erv)
	r = walkexpr(r, init)
	r.Type = n.Type

	return r
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
	walkexprlistsafe(n.List.Slice(), init)

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	ls := n.List.Slice()
	for i1, n1 := range ls {
		ls[i1] = cheapexpr(n1, init)
	}

	l1 := n.List.First()
	l2 := n.List.Second()

	var l []*Node

	// var s []T
	s := temp(l1.Type)
	l = append(l, nod(OAS, s, l1)) // s = l1

	// n := len(s) + len(l2)
	nn := temp(types.Types[TINT])
	l = append(l, nod(OAS, nn, nod(OADD, nod(OLEN, s, nil), nod(OLEN, l2, nil))))

	// if uint(n) > uint(cap(s))
	nif := nod(OIF, nil, nil)
	nif.Left = nod(OGT, nod(OCONV, nn, nil), nod(OCONV, nod(OCAP, s, nil), nil))
	nif.Left.Left.Type = types.Types[TUINT]
	nif.Left.Right.Type = types.Types[TUINT]

	// instantiate growslice(Type*, []any, int) []any
	fn := syslook("growslice")
	fn = substArgTypes(fn, s.Type.Elem(), s.Type.Elem())

	// s = growslice(T, s, n)
	nif.Nbody.Set1(nod(OAS, s, mkcall1(fn, s.Type, &nif.Ninit, typename(s.Type.Elem()), s, nn)))
	l = append(l, nif)

	// s = s[:n]
	nt := nod(OSLICE, s, nil)
	nt.SetSliceBounds(nil, nn, nil)
	nt.Etype = 1
	l = append(l, nod(OAS, s, nt))

	if types.Haspointers(l1.Type.Elem()) {
		// copy(s[len(l1):], l2)
		nptr1 := nod(OSLICE, s, nil)
		nptr1.SetSliceBounds(nod(OLEN, l1, nil), nil, nil)
		nptr1.Etype = 1
		nptr2 := l2
		fn := syslook("typedslicecopy")
		fn = substArgTypes(fn, l1.Type, l2.Type)
		var ln Nodes
		ln.Set(l)
		nt := mkcall1(fn, types.Types[TINT], &ln, typename(l1.Type.Elem()), nptr1, nptr2)
		l = append(ln.Slice(), nt)
	} else if instrumenting && !compiling_runtime {
		// rely on runtime to instrument copy.
		// copy(s[len(l1):], l2)
		nptr1 := nod(OSLICE, s, nil)
		nptr1.SetSliceBounds(nod(OLEN, l1, nil), nil, nil)
		nptr1.Etype = 1
		nptr2 := l2
		var fn *Node
		if l2.Type.IsString() {
			fn = syslook("slicestringcopy")
		} else {
			fn = syslook("slicecopy")
		}
		fn = substArgTypes(fn, l1.Type, l2.Type)
		var ln Nodes
		ln.Set(l)
		nt := mkcall1(fn, types.Types[TINT], &ln, nptr1, nptr2, nodintconst(s.Type.Elem().Width))
		l = append(ln.Slice(), nt)
	} else {
		// memmove(&s[len(l1)], &l2[0], len(l2)*sizeof(T))
		nptr1 := nod(OINDEX, s, nod(OLEN, l1, nil))
		nptr1.SetBounded(true)

		nptr1 = nod(OADDR, nptr1, nil)

		nptr2 := nod(OSPTR, l2, nil)

		fn := syslook("memmove")
		fn = substArgTypes(fn, s.Type.Elem(), s.Type.Elem())

		var ln Nodes
		ln.Set(l)
		nwid := cheapexpr(conv(nod(OLEN, l2, nil), types.Types[TUINTPTR]), &ln)

		nwid = nod(OMUL, nwid, nodintconst(s.Type.Elem().Width))
		nt := mkcall1(fn, nil, &ln, nptr1, nptr2, nwid)
		l = append(ln.Slice(), nt)
	}

	typecheckslice(l, Etop)
	walkstmtlist(l)
	init.Append(l...)
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

	// walkexprlistsafe will leave OINDEX (s[n]) alone if both s
	// and n are name or literal, but those may index the slice we're
	// modifying here. Fix explicitly.
	// Using cheapexpr also makes sure that the evaluation
	// of all arguments (and especially any panics) happen
	// before we begin to modify the slice in a visible way.
	ls := n.List.Slice()[1:]
	for i, n := range ls {
		ls[i] = cheapexpr(n, init)
	}

	nsrc := n.List.First()

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
	nx.Etype = 1
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

	typecheckslice(l, Etop)
	walkstmtlist(l)
	init.Append(l...)
	return ns
}

// Lower copy(a, b) to a memmove call or a runtime call.
//
// init {
//   n := len(a)
//   if n > len(b) { n = len(b) }
//   memmove(a.ptr, b.ptr, n*sizeof(elem(a)))
// }
// n;
//
// Also works if b is a string.
//
func copyany(n *Node, init *Nodes, runtimecall bool) *Node {
	if types.Haspointers(n.Left.Type.Elem()) {
		fn := writebarrierfn("typedslicecopy", n.Left.Type, n.Right.Type)
		return mkcall1(fn, n.Type, init, typename(n.Left.Type.Elem()), n.Left, n.Right)
	}

	if runtimecall {
		var fn *Node
		if n.Right.Type.IsString() {
			fn = syslook("slicestringcopy")
		} else {
			fn = syslook("slicecopy")
		}
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

	// Call memmove.
	fn := syslook("memmove")

	fn = substArgTypes(fn, nl.Type.Elem(), nl.Type.Elem())
	nwid := temp(types.Types[TUINTPTR])
	l = append(l, nod(OAS, nwid, conv(nlen, types.Types[TUINTPTR])))
	nwid = nod(OMUL, nwid, nodintconst(nl.Type.Elem().Width))
	l = append(l, mkcall1(fn, nil, init, nto, nfrm, nwid))

	typecheckslice(l, Etop)
	walkstmtlist(l)
	init.Append(l...)
	return nlen
}

func eqfor(t *types.Type, needsize *int) *Node {
	// Should only arrive here with large memory or
	// a struct/array containing a non-memory field/element.
	// Small memory is handled inline, and single non-memory
	// is handled during type check (OCMPSTR etc).
	switch a, _ := algtype1(t); a {
	case AMEM:
		n := syslook("memequal")
		n = substArgTypes(n, t, t)
		*needsize = 1
		return n
	case ASPECIAL:
		sym := typesymprefix(".eq", t)
		n := newname(sym)
		n.SetClass(PFUNC)
		ntype := nod(OTFUNC, nil, nil)
		ntype.List.Append(anonfield(types.NewPtr(t)))
		ntype.List.Append(anonfield(types.NewPtr(t)))
		ntype.Rlist.Append(anonfield(types.Types[TBOOL]))
		ntype = typecheck(ntype, Etype)
		n.Type = ntype.Type
		*needsize = 0
		return n
	}
	Fatalf("eqfor %v", t)
	return nil
}

// The result of walkcompare MUST be assigned back to n, e.g.
// 	n.Left = walkcompare(n.Left, init)
func walkcompare(n *Node, init *Nodes) *Node {
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
		var andor Op
		if eq == OEQ {
			andor = OANDAND
		} else {
			andor = OOROR
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
	unalignedLoad := false
	switch thearch.LinkArch.Family {
	case sys.AMD64, sys.ARM64, sys.S390X:
		// Keep this low enough, to generate less code than function call.
		maxcmpsize = 16
		unalignedLoad = true
	case sys.I386:
		maxcmpsize = 8
		unalignedLoad = true
	}

	switch t.Etype {
	default:
		return n
	case TARRAY:
		// We can compare several elements at once with 2/4/8 byte integer compares
		inline = t.NumElem() <= 1 || (issimple[t.Elem().Etype] && (t.NumElem() <= 4 || t.Elem().Width*t.NumElem() <= maxcmpsize))
	case TSTRUCT:
		inline = t.NumFields() <= 4
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
		al.Right.Etype = 1 // addr does not escape
		al = typecheck(al, Etop)
		init.Append(al)

		pr := temp(types.NewPtr(t))
		ar := nod(OAS, pr, nod(OADDR, cmpr, nil))
		ar.Right.Etype = 1 // addr does not escape
		ar = typecheck(ar, Etop)
		init.Append(ar)

		var needsize int
		call := nod(OCALL, eqfor(t, &needsize), nil)
		call.List.Append(pl)
		call.List.Append(pr)
		if needsize != 0 {
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
					nod(OINDEX, cmpl, nodintconst(int64(i))),
					nod(OINDEX, cmpr, nodintconst(int64(i))),
				)
				i++
				remains -= t.Elem().Width
			} else {
				cmplw := nod(OINDEX, cmpl, nodintconst(int64(i)))
				cmplw = conv(cmplw, convType)
				cmprw := nod(OINDEX, cmpr, nodintconst(int64(i)))
				cmprw = conv(cmprw, convType)
				// For code like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will generate a single large load.
				for offset := int64(1); offset < step; offset++ {
					lb := nod(OINDEX, cmpl, nodintconst(int64(i+offset)))
					lb = conv(lb, convType)
					lb = nod(OLSH, lb, nodintconst(int64(8*t.Elem().Width*offset)))
					cmplw = nod(OOR, cmplw, lb)
					rb := nod(OINDEX, cmpr, nodintconst(int64(i+offset)))
					rb = conv(rb, convType)
					rb = nod(OLSH, rb, nodintconst(int64(8*t.Elem().Width*offset)))
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
	}
	n = finishcompare(n, expr, init)
	return n
}

// The result of finishcompare MUST be assigned back to n, e.g.
// 	n.Left = finishcompare(n.Left, x, r, init)
func finishcompare(n, r *Node, init *Nodes) *Node {
	// Use nn here to avoid passing r to typecheck.
	nn := r
	nn = typecheck(nn, Erv)
	nn = walkexpr(nn, init)
	r = nn
	if r.Type != n.Type {
		r = nod(OCONVNOP, r, nil)
		r.Type = n.Type
		r.SetTypecheck(1)
		nn = r
	}
	return nn
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
	cmp = typecheck(cmp, Erv)
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
	if n := t.Params().NumFields(); n != 1 {
		return
	}
	if n := t.Results().NumFields(); n != 1 && n != 2 {
		return
	}
	p0 := t.Params().Field(0)
	res0 := t.Results().Field(0)
	var res1 *types.Field
	if t.Results().NumFields() == 2 {
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
	if !exportname(field.Sym.Name) {
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
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		OCAP,
		OCMPIFACE,
		OCMPSTR,
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
		OCOM,
		OPLUS,
		OMINUS,
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

// rewrite
//	print(x, y, z)
// into
//	func(a1, a2, a3) {
//		print(a1, a2, a3)
//	}(x, y, z)
// and same for println.

var walkprintfunc_prgen int

// The result of walkprintfunc MUST be assigned back to n, e.g.
// 	n.Left = walkprintfunc(n.Left, init)
func walkprintfunc(n *Node, init *Nodes) *Node {
	if n.Ninit.Len() != 0 {
		walkstmtlist(n.Ninit.Slice())
		init.AppendNodes(&n.Ninit)
	}

	t := nod(OTFUNC, nil, nil)
	num := 0
	var printargs []*Node
	var a *Node
	var buf string
	for _, n1 := range n.List.Slice() {
		buf = fmt.Sprintf("a%d", num)
		num++
		a = namedfield(buf, n1.Type)
		t.List.Append(a)
		printargs = append(printargs, a.Left)
	}

	oldfn := Curfn
	Curfn = nil

	walkprintfunc_prgen++
	sym := lookupN("print·%d", walkprintfunc_prgen)
	fn := dclfunc(sym, t)

	a = nod(n.Op, nil, nil)
	a.List.Set(printargs)
	a = typecheck(a, Etop)
	a = walkstmt(a)

	fn.Nbody.Set1(a)

	funcbody(fn)

	fn = typecheck(fn, Etop)
	typecheckslice(fn.Nbody.Slice(), Etop)
	xtop = append(xtop, fn)
	Curfn = oldfn

	a = nod(OCALL, nil, nil)
	a.Left = fn.Func.Nname
	a.List.Set(n.List.Slice())
	a = typecheck(a, Etop)
	a = walkexpr(a, init)
	return a
}

// substArgTypes substitutes the given list of types for
// successive occurrences of the "any" placeholder in the
// type syntax expression n.Type.
// The result of substArgTypes MUST be assigned back to old, e.g.
// 	n.Left = substArgTypes(n.Left, t1, t2)
func substArgTypes(old *Node, types_ ...*types.Type) *Node {
	n := *old // make shallow copy

	for _, t := range types_ {
		dowidth(t)
	}
	n.Type = types.SubstAny(n.Type, &types_)
	if len(types_) > 0 {
		Fatalf("substArgTypes: too many argument types")
	}
	return &n
}
