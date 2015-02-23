// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The inlining facility makes 2 passes: first caninl determines which
// functions are suitable for inlining, and for those that are it
// saves a copy of the body. Then inlcalls walks each function body to
// expand calls to inlinable functions.
//
// The debug['l'] flag controls the agressiveness. Note that main() swaps level 0 and 1,
// making 1 the default and -l disable.  -ll and more is useful to flush out bugs.
// These additional levels (beyond -l) may be buggy and are not supported.
//      0: disabled
//      1: 40-nodes leaf functions, oneliners, lazy typechecking (default)
//      2: early typechecking of all imported bodies
//      3: allow variadic functions
//      4: allow non-leaf functions , (breaks runtime.Caller)
//      5: transitive inlining
//
//  At some point this may get another default and become switch-offable with -N.
//
//  The debug['m'] flag enables diagnostic output.  a single -m is useful for verifying
//  which calls get inlined or not, more is for debugging, and may go away at any point.
//
// TODO:
//   - inline functions with ... args
//   - handle T.meth(f()) with func f() (t T, arg, arg, )

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

// Used by caninl.

// Used by inlcalls

// Used during inlsubst[list]
var inlfn *Node // function currently being inlined

var inlretlabel *Node // target of the goto substituted in place of a return

var inlretvars *NodeList // temp out variables

// Get the function's package.  For ordinary functions it's on the ->sym, but for imported methods
// the ->sym can be re-used in the local package, so peel it off the receiver's type.
func fnpkg(fn *Node) *Pkg {
	if fn.Type.Thistuple != 0 {
		// method
		rcvr := getthisx(fn.Type).Type.Type

		if Isptr[rcvr.Etype] != 0 {
			rcvr = rcvr.Type
		}
		if rcvr.Sym == nil {
			Fatal("receiver with no sym: [%v] %v  (%v)", Sconv(fn.Sym, 0), Nconv(fn, obj.FmtLong), Tconv(rcvr, 0))
		}
		return rcvr.Sym.Pkg
	}

	// non-method
	return fn.Sym.Pkg
}

// Lazy typechecking of imported bodies.  For local functions, caninl will set ->typecheck
// because they're a copy of an already checked body.
func typecheckinl(fn *Node) {
	lno := int(setlineno(fn))

	// typecheckinl is only for imported functions;
	// their bodies may refer to unsafe as long as the package
	// was marked safe during import (which was checked then).
	// the ->inl of a local function has been typechecked before caninl copied it.
	pkg := fnpkg(fn)

	if pkg == localpkg || pkg == nil {
		return // typecheckinl on local function
	}

	if Debug['m'] > 2 {
		fmt.Printf("typecheck import [%v] %v { %v }\n", Sconv(fn.Sym, 0), Nconv(fn, obj.FmtLong), Hconv(fn.Inl, obj.FmtSharp))
	}

	save_safemode := safemode
	safemode = 0

	savefn := Curfn
	Curfn = fn
	typechecklist(fn.Inl, Etop)
	Curfn = savefn

	safemode = save_safemode

	lineno = int32(lno)
}

// Caninl determines whether fn is inlineable.
// If so, caninl saves fn->nbody in fn->inl and substitutes it with a copy.
// fn and ->nbody will already have been typechecked.
func caninl(fn *Node) {
	if fn.Op != ODCLFUNC {
		Fatal("caninl %v", Nconv(fn, 0))
	}
	if fn.Nname == nil {
		Fatal("caninl no nname %v", Nconv(fn, obj.FmtSign))
	}

	// If fn has no body (is defined outside of Go), cannot inline it.
	if fn.Nbody == nil {
		return
	}

	if fn.Typecheck == 0 {
		Fatal("caninl on non-typechecked function %v", Nconv(fn, 0))
	}

	// can't handle ... args yet
	if Debug['l'] < 3 {
		for t := fn.Type.Type.Down.Down.Type; t != nil; t = t.Down {
			if t.Isddd != 0 {
				return
			}
		}
	}

	budget := 40 // allowed hairyness
	if ishairylist(fn.Nbody, &budget) {
		return
	}

	savefn := Curfn
	Curfn = fn

	fn.Nname.Inl = fn.Nbody
	fn.Nbody = inlcopylist(fn.Nname.Inl)
	fn.Nname.Inldcl = inlcopylist(fn.Nname.Defn.Dcl)

	// hack, TODO, check for better way to link method nodes back to the thing with the ->inl
	// this is so export can find the body of a method
	fn.Type.Nname = fn.Nname

	if Debug['m'] > 1 {
		fmt.Printf("%v: can inline %v as: %v { %v }\n", fn.Line(), Nconv(fn.Nname, obj.FmtSharp), Tconv(fn.Type, obj.FmtSharp), Hconv(fn.Nname.Inl, obj.FmtSharp))
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: can inline %v\n", fn.Line(), Nconv(fn.Nname, 0))
	}

	Curfn = savefn
}

// Look for anything we want to punt on.
func ishairylist(ll *NodeList, budget *int) bool {
	for ; ll != nil; ll = ll.Next {
		if ishairy(ll.N, budget) {
			return true
		}
	}
	return false
}

func ishairy(n *Node, budget *int) bool {
	if n == nil {
		return false
	}

	// Things that are too hairy, irrespective of the budget
	switch n.Op {
	case OCALL,
		OCALLFUNC,
		OCALLINTER,
		OCALLMETH,
		OPANIC,
		ORECOVER:
		if Debug['l'] < 4 {
			return true
		}

	case OCLOSURE,
		OCALLPART,
		ORANGE,
		OFOR,
		OSELECT,
		OSWITCH,
		OPROC,
		ODEFER,
		ODCLTYPE,  // can't print yet
		ODCLCONST, // can't print yet
		ORETJMP:
		return true
	}

	(*budget)--

	return *budget < 0 || ishairy(n.Left, budget) || ishairy(n.Right, budget) || ishairylist(n.List, budget) || ishairylist(n.Rlist, budget) || ishairylist(n.Ninit, budget) || ishairy(n.Ntest, budget) || ishairy(n.Nincr, budget) || ishairylist(n.Nbody, budget) || ishairylist(n.Nelse, budget)
}

// Inlcopy and inlcopylist recursively copy the body of a function.
// Any name-like node of non-local class is marked for re-export by adding it to
// the exportlist.
func inlcopylist(ll *NodeList) *NodeList {
	l := (*NodeList)(nil)
	for ; ll != nil; ll = ll.Next {
		l = list(l, inlcopy(ll.N))
	}
	return l
}

func inlcopy(n *Node) *Node {
	if n == nil {
		return nil
	}

	switch n.Op {
	case ONAME,
		OTYPE,
		OLITERAL:
		return n
	}

	m := Nod(OXXX, nil, nil)
	*m = *n
	m.Inl = nil
	m.Left = inlcopy(n.Left)
	m.Right = inlcopy(n.Right)
	m.List = inlcopylist(n.List)
	m.Rlist = inlcopylist(n.Rlist)
	m.Ninit = inlcopylist(n.Ninit)
	m.Ntest = inlcopy(n.Ntest)
	m.Nincr = inlcopy(n.Nincr)
	m.Nbody = inlcopylist(n.Nbody)
	m.Nelse = inlcopylist(n.Nelse)

	return m
}

// Inlcalls/nodelist/node walks fn's statements and expressions and substitutes any
// calls made to inlineable functions.  This is the external entry point.
func inlcalls(fn *Node) {
	savefn := Curfn
	Curfn = fn
	inlnode(&fn)
	if fn != Curfn {
		Fatal("inlnode replaced curfn")
	}
	Curfn = savefn
}

// Turn an OINLCALL into a statement.
func inlconv2stmt(n *Node) {
	n.Op = OBLOCK

	// n->ninit stays
	n.List = n.Nbody

	n.Nbody = nil
	n.Rlist = nil
}

// Turn an OINLCALL into a single valued expression.
func inlconv2expr(np **Node) {
	n := *np
	r := n.Rlist.N
	addinit(&r, concat(n.Ninit, n.Nbody))
	*np = r
}

// Turn the rlist (with the return values) of the OINLCALL in
// n into an expression list lumping the ninit and body
// containing the inlined statements on the first list element so
// order will be preserved Used in return, oas2func and call
// statements.
func inlconv2list(n *Node) *NodeList {
	if n.Op != OINLCALL || n.Rlist == nil {
		Fatal("inlconv2list %v\n", Nconv(n, obj.FmtSign))
	}

	l := n.Rlist
	addinit(&l.N, concat(n.Ninit, n.Nbody))
	return l
}

func inlnodelist(l *NodeList) {
	for ; l != nil; l = l.Next {
		inlnode(&l.N)
	}
}

// inlnode recurses over the tree to find inlineable calls, which will
// be turned into OINLCALLs by mkinlcall.  When the recursion comes
// back up will examine left, right, list, rlist, ninit, ntest, nincr,
// nbody and nelse and use one of the 4 inlconv/glue functions above
// to turn the OINLCALL into an expression, a statement, or patch it
// in to this nodes list or rlist as appropriate.
// NOTE it makes no sense to pass the glue functions down the
// recursion to the level where the OINLCALL gets created because they
// have to edit /this/ n, so you'd have to push that one down as well,
// but then you may as well do it here.  so this is cleaner and
// shorter and less complicated.
func inlnode(np **Node) {
	if *np == nil {
		return
	}

	n := *np

	switch n.Op {
	// inhibit inlining of their argument
	case ODEFER,
		OPROC:
		switch n.Left.Op {
		case OCALLFUNC,
			OCALLMETH:
			n.Left.Etype = n.Op
		}
		fallthrough

		// TODO do them here (or earlier),
	// so escape analysis can avoid more heapmoves.
	case OCLOSURE:
		return
	}

	lno := int(setlineno(n))

	inlnodelist(n.Ninit)
	for l := n.Ninit; l != nil; l = l.Next {
		if l.N.Op == OINLCALL {
			inlconv2stmt(l.N)
		}
	}

	inlnode(&n.Left)
	if n.Left != nil && n.Left.Op == OINLCALL {
		inlconv2expr(&n.Left)
	}

	inlnode(&n.Right)
	if n.Right != nil && n.Right.Op == OINLCALL {
		inlconv2expr(&n.Right)
	}

	inlnodelist(n.List)
	switch n.Op {
	case OBLOCK:
		for l := n.List; l != nil; l = l.Next {
			if l.N.Op == OINLCALL {
				inlconv2stmt(l.N)
			}
		}

		// if we just replaced arg in f(arg()) or return arg with an inlined call
	// and arg returns multiple values, glue as list
	case ORETURN,
		OCALLFUNC,
		OCALLMETH,
		OCALLINTER,
		OAPPEND,
		OCOMPLEX:
		if count(n.List) == 1 && n.List.N.Op == OINLCALL && count(n.List.N.Rlist) > 1 {
			n.List = inlconv2list(n.List.N)
			break
		}
		fallthrough

		// fallthrough
	default:
		for l := n.List; l != nil; l = l.Next {
			if l.N.Op == OINLCALL {
				inlconv2expr(&l.N)
			}
		}
	}

	inlnodelist(n.Rlist)
	switch n.Op {
	case OAS2FUNC:
		if n.Rlist.N.Op == OINLCALL {
			n.Rlist = inlconv2list(n.Rlist.N)
			n.Op = OAS2
			n.Typecheck = 0
			typecheck(np, Etop)
			break
		}
		fallthrough

		// fallthrough
	default:
		for l := n.Rlist; l != nil; l = l.Next {
			if l.N.Op == OINLCALL {
				inlconv2expr(&l.N)
			}
		}
	}

	inlnode(&n.Ntest)
	if n.Ntest != nil && n.Ntest.Op == OINLCALL {
		inlconv2expr(&n.Ntest)
	}

	inlnode(&n.Nincr)
	if n.Nincr != nil && n.Nincr.Op == OINLCALL {
		inlconv2stmt(n.Nincr)
	}

	inlnodelist(n.Nbody)
	for l := n.Nbody; l != nil; l = l.Next {
		if l.N.Op == OINLCALL {
			inlconv2stmt(l.N)
		}
	}

	inlnodelist(n.Nelse)
	for l := n.Nelse; l != nil; l = l.Next {
		if l.N.Op == OINLCALL {
			inlconv2stmt(l.N)
		}
	}

	// with all the branches out of the way, it is now time to
	// transmogrify this node itself unless inhibited by the
	// switch at the top of this function.
	switch n.Op {
	case OCALLFUNC,
		OCALLMETH:
		if n.Etype == OPROC || n.Etype == ODEFER {
			return
		}
	}

	switch n.Op {
	case OCALLFUNC:
		if Debug['m'] > 3 {
			fmt.Printf("%v:call to func %v\n", n.Line(), Nconv(n.Left, obj.FmtSign))
		}
		if n.Left.Inl != nil { // normal case
			mkinlcall(np, n.Left, int(n.Isddd))
		} else if n.Left.Op == ONAME && n.Left.Left != nil && n.Left.Left.Op == OTYPE && n.Left.Right != nil && n.Left.Right.Op == ONAME { // methods called as functions
			if n.Left.Sym.Def != nil {
				mkinlcall(np, n.Left.Sym.Def, int(n.Isddd))
			}
		}

	case OCALLMETH:
		if Debug['m'] > 3 {
			fmt.Printf("%v:call to meth %v\n", n.Line(), Nconv(n.Left.Right, obj.FmtLong))
		}

		// typecheck should have resolved ODOTMETH->type, whose nname points to the actual function.
		if n.Left.Type == nil {
			Fatal("no function type for [%p] %v\n", n.Left, Nconv(n.Left, obj.FmtSign))
		}

		if n.Left.Type.Nname == nil {
			Fatal("no function definition for [%p] %v\n", n.Left.Type, Tconv(n.Left.Type, obj.FmtSign))
		}

		mkinlcall(np, n.Left.Type.Nname, int(n.Isddd))
	}

	lineno = int32(lno)
}

func mkinlcall(np **Node, fn *Node, isddd int) {
	save_safemode := safemode

	// imported functions may refer to unsafe as long as the
	// package was marked safe during import (already checked).
	pkg := fnpkg(fn)

	if pkg != localpkg && pkg != nil {
		safemode = 0
	}
	mkinlcall1(np, fn, isddd)
	safemode = save_safemode
}

func tinlvar(t *Type) *Node {
	if t.Nname != nil && !isblank(t.Nname) {
		if t.Nname.Inlvar == nil {
			Fatal("missing inlvar for %v\n", Nconv(t.Nname, 0))
		}
		return t.Nname.Inlvar
	}

	typecheck(&nblank, Erv|Easgn)
	return nblank
}

var inlgen int

// if *np is a call, and fn is a function with an inlinable body, substitute *np with an OINLCALL.
// On return ninit has the parameter assignments, the nbody is the
// inlined function body and list, rlist contain the input, output
// parameters.
func mkinlcall1(np **Node, fn *Node, isddd int) {
	// For variadic fn.
	if fn.Inl == nil {
		return
	}

	if fn == Curfn || fn.Defn == Curfn {
		return
	}

	if Debug['l'] < 2 {
		typecheckinl(fn)
	}

	n := *np

	// Bingo, we have a function node, and it has an inlineable body
	if Debug['m'] > 1 {
		fmt.Printf("%v: inlining call to %v %v { %v }\n", n.Line(), Sconv(fn.Sym, 0), Tconv(fn.Type, obj.FmtSharp), Hconv(fn.Inl, obj.FmtSharp))
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: inlining call to %v\n", n.Line(), Nconv(fn, 0))
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v: Before inlining: %v\n", n.Line(), Nconv(n, obj.FmtSign))
	}

	saveinlfn := inlfn
	inlfn = fn

	ninit := n.Ninit

	//dumplist("ninit pre", ninit);

	var dcl *NodeList
	if fn.Defn != nil { // local function
		dcl = fn.Inldcl // imported function
	} else {
		dcl = fn.Dcl
	}

	inlretvars = nil
	i := 0

	// Make temp names to use instead of the originals
	for ll := dcl; ll != nil; ll = ll.Next {
		if ll.N.Class == PPARAMOUT { // return values handled below.
			continue
		}
		if ll.N.Op == ONAME {
			ll.N.Inlvar = inlvar(ll.N)

			// Typecheck because inlvar is not necessarily a function parameter.
			typecheck(&ll.N.Inlvar, Erv)

			if ll.N.Class&^PHEAP != PAUTO {
				ninit = list(ninit, Nod(ODCL, ll.N.Inlvar, nil)) // otherwise gen won't emit the allocations for heapallocs
			}
		}
	}

	// temporaries for return values.
	var m *Node
	for t := getoutargx(fn.Type).Type; t != nil; t = t.Down {
		if t != nil && t.Nname != nil && !isblank(t.Nname) {
			m = inlvar(t.Nname)
			typecheck(&m, Erv)
			t.Nname.Inlvar = m
		} else {
			// anonymous return values, synthesize names for use in assignment that replaces return
			m = retvar(t, i)
			i++
		}

		ninit = list(ninit, Nod(ODCL, m, nil))
		inlretvars = list(inlretvars, m)
	}

	// assign receiver.
	var as *Node
	if fn.Type.Thistuple != 0 && n.Left.Op == ODOTMETH {
		// method call with a receiver.
		t := getthisx(fn.Type).Type

		if t != nil && t.Nname != nil && !isblank(t.Nname) && t.Nname.Inlvar == nil {
			Fatal("missing inlvar for %v\n", Nconv(t.Nname, 0))
		}
		if n.Left.Left == nil {
			Fatal("method call without receiver: %v", Nconv(n, obj.FmtSign))
		}
		if t == nil {
			Fatal("method call unknown receiver type: %v", Nconv(n, obj.FmtSign))
		}
		as = Nod(OAS, tinlvar(t), n.Left.Left)
		if as != nil {
			typecheck(&as, Etop)
			ninit = list(ninit, as)
		}
	}

	// check if inlined function is variadic.
	variadic := false

	varargtype := (*Type)(nil)
	varargcount := 0
	for t := fn.Type.Type.Down.Down.Type; t != nil; t = t.Down {
		if t.Isddd != 0 {
			variadic = true
			varargtype = t.Type
		}
	}

	// but if argument is dotted too forget about variadicity.
	if variadic && isddd != 0 {
		variadic = false
	}

	// check if argument is actually a returned tuple from call.
	multiret := 0

	if n.List != nil && n.List.Next == nil {
		switch n.List.N.Op {
		case OCALL,
			OCALLFUNC,
			OCALLINTER,
			OCALLMETH:
			if n.List.N.Left.Type.Outtuple > 1 {
				multiret = n.List.N.Left.Type.Outtuple - 1
			}
		}
	}

	if variadic {
		varargcount = count(n.List) + multiret
		if n.Left.Op != ODOTMETH {
			varargcount -= fn.Type.Thistuple
		}
		varargcount -= fn.Type.Intuple - 1
	}

	// assign arguments to the parameters' temp names
	as = Nod(OAS2, nil, nil)

	as.Rlist = n.List
	ll := n.List

	// TODO: if len(nlist) == 1 but multiple args, check that n->list->n is a call?
	if fn.Type.Thistuple != 0 && n.Left.Op != ODOTMETH {
		// non-method call to method
		if n.List == nil {
			Fatal("non-method call to method without first arg: %v", Nconv(n, obj.FmtSign))
		}

		// append receiver inlvar to LHS.
		t := getthisx(fn.Type).Type

		if t != nil && t.Nname != nil && !isblank(t.Nname) && t.Nname.Inlvar == nil {
			Fatal("missing inlvar for %v\n", Nconv(t.Nname, 0))
		}
		if t == nil {
			Fatal("method call unknown receiver type: %v", Nconv(n, obj.FmtSign))
		}
		as.List = list(as.List, tinlvar(t))
		ll = ll.Next // track argument count.
	}

	// append ordinary arguments to LHS.
	chkargcount := n.List != nil && n.List.Next != nil

	vararg := (*Node)(nil)      // the slice argument to a variadic call
	varargs := (*NodeList)(nil) // the list of LHS names to put in vararg.
	if !chkargcount {
		// 0 or 1 expression on RHS.
		var i int
		for t := getinargx(fn.Type).Type; t != nil; t = t.Down {
			if variadic && t.Isddd != 0 {
				vararg = tinlvar(t)
				for i = 0; i < varargcount && ll != nil; i++ {
					m = argvar(varargtype, i)
					varargs = list(varargs, m)
					as.List = list(as.List, m)
				}

				break
			}

			as.List = list(as.List, tinlvar(t))
		}
	} else {
		// match arguments except final variadic (unless the call is dotted itself)
		var t *Type
		for t = getinargx(fn.Type).Type; t != nil; {
			if ll == nil {
				break
			}
			if variadic && t.Isddd != 0 {
				break
			}
			as.List = list(as.List, tinlvar(t))
			t = t.Down
			ll = ll.Next
		}

		// match varargcount arguments with variadic parameters.
		if variadic && t != nil && t.Isddd != 0 {
			vararg = tinlvar(t)
			var i int
			for i = 0; i < varargcount && ll != nil; i++ {
				m = argvar(varargtype, i)
				varargs = list(varargs, m)
				as.List = list(as.List, m)
				ll = ll.Next
			}

			if i == varargcount {
				t = t.Down
			}
		}

		if ll != nil || t != nil {
			Fatal("arg count mismatch: %v  vs %v\n", Tconv(getinargx(fn.Type), obj.FmtSharp), Hconv(n.List, obj.FmtComma))
		}
	}

	if as.Rlist != nil {
		typecheck(&as, Etop)
		ninit = list(ninit, as)
	}

	// turn the variadic args into a slice.
	if variadic {
		as = Nod(OAS, vararg, nil)
		if varargcount == 0 {
			as.Right = nodnil()
			as.Right.Type = varargtype
		} else {
			vararrtype := typ(TARRAY)
			vararrtype.Type = varargtype.Type
			vararrtype.Bound = int64(varargcount)

			as.Right = Nod(OCOMPLIT, nil, typenod(varargtype))
			as.Right.List = varargs
			as.Right = Nod(OSLICE, as.Right, Nod(OKEY, nil, nil))
		}

		typecheck(&as, Etop)
		ninit = list(ninit, as)
	}

	// zero the outparams
	for ll := inlretvars; ll != nil; ll = ll.Next {
		as = Nod(OAS, ll.N, nil)
		typecheck(&as, Etop)
		ninit = list(ninit, as)
	}

	inlretlabel = newlabel_inl()
	inlgen++
	body := inlsubstlist(fn.Inl)

	body = list(body, Nod(OGOTO, inlretlabel, nil)) // avoid 'not used' when function doesnt have return
	body = list(body, Nod(OLABEL, inlretlabel, nil))

	typechecklist(body, Etop)

	//dumplist("ninit post", ninit);

	call := Nod(OINLCALL, nil, nil)

	call.Ninit = ninit
	call.Nbody = body
	call.Rlist = inlretvars
	call.Type = n.Type
	call.Typecheck = 1

	setlno(call, int(n.Lineno))

	//dumplist("call body", body);

	*np = call

	inlfn = saveinlfn

	// transitive inlining
	// TODO do this pre-expansion on fn->inl directly.  requires
	// either supporting exporting statemetns with complex ninits
	// or saving inl and making inlinl
	if Debug['l'] >= 5 {
		body := fn.Inl
		fn.Inl = nil // prevent infinite recursion
		inlnodelist(call.Nbody)
		for ll := call.Nbody; ll != nil; ll = ll.Next {
			if ll.N.Op == OINLCALL {
				inlconv2stmt(ll.N)
			}
		}
		fn.Inl = body
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v: After inlining %v\n\n", n.Line(), Nconv(*np, obj.FmtSign))
	}
}

// Every time we expand a function we generate a new set of tmpnames,
// PAUTO's in the calling functions, and link them off of the
// PPARAM's, PAUTOS and PPARAMOUTs of the called function.
func inlvar(var_ *Node) *Node {
	if Debug['m'] > 3 {
		fmt.Printf("inlvar %v\n", Nconv(var_, obj.FmtSign))
	}

	n := newname(var_.Sym)
	n.Type = var_.Type
	n.Class = PAUTO
	n.Used = 1
	n.Curfn = Curfn // the calling function, not the called one
	n.Addrtaken = var_.Addrtaken

	// Esc pass wont run if we're inlining into a iface wrapper.
	// Luckily, we can steal the results from the target func.
	// If inlining a function defined in another package after
	// escape analysis is done, treat all local vars as escaping.
	// See issue 9537.
	if var_.Esc == EscHeap || (inl_nonlocal != 0 && var_.Op == ONAME) {
		addrescapes(n)
	}

	Curfn.Dcl = list(Curfn.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *Type, i int) *Node {
	namebuf = fmt.Sprintf("~r%d", i)
	n := newname(Lookup(namebuf))
	n.Type = t.Type
	n.Class = PAUTO
	n.Used = 1
	n.Curfn = Curfn // the calling function, not the called one
	Curfn.Dcl = list(Curfn.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's arguments
// when they come from a multiple return call.
func argvar(t *Type, i int) *Node {
	namebuf = fmt.Sprintf("~arg%d", i)
	n := newname(Lookup(namebuf))
	n.Type = t.Type
	n.Class = PAUTO
	n.Used = 1
	n.Curfn = Curfn // the calling function, not the called one
	Curfn.Dcl = list(Curfn.Dcl, n)
	return n
}

var newlabel_inl_label int

func newlabel_inl() *Node {
	newlabel_inl_label++
	namebuf = fmt.Sprintf(".inlret%.6d", newlabel_inl_label)
	n := newname(Lookup(namebuf))
	n.Etype = 1 // flag 'safe' for escape analysis (no backjumps)
	return n
}

// inlsubst and inlsubstlist recursively copy the body of the saved
// pristine ->inl body of the function while substituting references
// to input/output parameters with ones to the tmpnames, and
// substituting returns with assignments to the output.
func inlsubstlist(ll *NodeList) *NodeList {
	l := (*NodeList)(nil)
	for ; ll != nil; ll = ll.Next {
		l = list(l, inlsubst(ll.N))
	}
	return l
}

func inlsubst(n *Node) *Node {
	if n == nil {
		return nil
	}

	switch n.Op {
	case ONAME:
		if n.Inlvar != nil { // These will be set during inlnode
			if Debug['m'] > 2 {
				fmt.Printf("substituting name %v  ->  %v\n", Nconv(n, obj.FmtSign), Nconv(n.Inlvar, obj.FmtSign))
			}
			return n.Inlvar
		}

		if Debug['m'] > 2 {
			fmt.Printf("not substituting name %v\n", Nconv(n, obj.FmtSign))
		}
		return n

	case OLITERAL,
		OTYPE:
		return n

		// Since we don't handle bodies with closures, this return is guaranteed to belong to the current inlined function.

	//		dump("Return before substitution", n);
	case ORETURN:
		m := Nod(OGOTO, inlretlabel, nil)

		m.Ninit = inlsubstlist(n.Ninit)

		if inlretvars != nil && n.List != nil {
			as := Nod(OAS2, nil, nil)

			// shallow copy or OINLCALL->rlist will be the same list, and later walk and typecheck may clobber that.
			for ll := inlretvars; ll != nil; ll = ll.Next {
				as.List = list(as.List, ll.N)
			}
			as.Rlist = inlsubstlist(n.List)
			typecheck(&as, Etop)
			m.Ninit = list(m.Ninit, as)
		}

		typechecklist(m.Ninit, Etop)
		typecheck(&m, Etop)

		//		dump("Return after substitution", m);
		return m

	case OGOTO,
		OLABEL:
		m := Nod(OXXX, nil, nil)
		*m = *n
		m.Ninit = nil
		p := fmt.Sprintf("%s·%d", n.Left.Sym.Name, inlgen)
		m.Left = newname(Lookup(p))

		return m
	}

	m := Nod(OXXX, nil, nil)
	*m = *n
	m.Ninit = nil

	if n.Op == OCLOSURE {
		Fatal("cannot inline function containing closure: %v", Nconv(n, obj.FmtSign))
	}

	m.Left = inlsubst(n.Left)
	m.Right = inlsubst(n.Right)
	m.List = inlsubstlist(n.List)
	m.Rlist = inlsubstlist(n.Rlist)
	m.Ninit = concat(m.Ninit, inlsubstlist(n.Ninit))
	m.Ntest = inlsubst(n.Ntest)
	m.Nincr = inlsubst(n.Nincr)
	m.Nbody = inlsubstlist(n.Nbody)
	m.Nelse = inlsubstlist(n.Nelse)

	return m
}

// Plaster over linenumbers
func setlnolist(ll *NodeList, lno int) {
	for ; ll != nil; ll = ll.Next {
		setlno(ll.N, lno)
	}
}

func setlno(n *Node, lno int) {
	if n == nil {
		return
	}

	// don't clobber names, unless they're freshly synthesized
	if n.Op != ONAME || n.Lineno == 0 {
		n.Lineno = int32(lno)
	}

	setlno(n.Left, lno)
	setlno(n.Right, lno)
	setlnolist(n.List, lno)
	setlnolist(n.Rlist, lno)
	setlnolist(n.Ninit, lno)
	setlno(n.Ntest, lno)
	setlno(n.Nincr, lno)
	setlnolist(n.Nbody, lno)
	setlnolist(n.Nelse, lno)
}
