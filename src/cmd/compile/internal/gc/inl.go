// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The inlining facility makes 2 passes: first caninl determines which
// functions are suitable for inlining, and for those that are it
// saves a copy of the body. Then inlcalls walks each function body to
// expand calls to inlinable functions.
//
// The debug['l'] flag controls the aggressiveness. Note that main() swaps level 0 and 1,
// making 1 the default and -l disable.  -ll and more is useful to flush out bugs.
// These additional levels (beyond -l) may be buggy and are not supported.
//      0: disabled
//      1: 40-nodes leaf functions, oneliners, lazy typechecking (default)
//      2: early typechecking of all imported bodies
//      3: allow variadic functions
//      4: allow non-leaf functions , (breaks runtime.Caller)
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

import "fmt"

// Get the function's package. For ordinary functions it's on the ->sym, but for imported methods
// the ->sym can be re-used in the local package, so peel it off the receiver's type.
func fnpkg(fn *Node) *Pkg {
	if fn.Type.Recv() != nil {
		// method
		rcvr := fn.Type.Recv().Type

		if rcvr.IsPtr() {
			rcvr = rcvr.Elem()
		}
		if rcvr.Sym == nil {
			Fatalf("receiver with no sym: [%v] %v  (%v)", fn.Sym, Nconv(fn, FmtLong), rcvr)
		}
		return rcvr.Sym.Pkg
	}

	// non-method
	return fn.Sym.Pkg
}

// Lazy typechecking of imported bodies. For local functions, caninl will set ->typecheck
// because they're a copy of an already checked body.
func typecheckinl(fn *Node) {
	lno := setlineno(fn)

	// typecheckinl is only for imported functions;
	// their bodies may refer to unsafe as long as the package
	// was marked safe during import (which was checked then).
	// the ->inl of a local function has been typechecked before caninl copied it.
	pkg := fnpkg(fn)

	if pkg == localpkg || pkg == nil {
		return // typecheckinl on local function
	}

	if Debug['m'] > 2 || Debug_export != 0 {
		fmt.Printf("typecheck import [%v] %v { %v }\n", fn.Sym, Nconv(fn, FmtLong), hconv(fn.Func.Inl, FmtSharp))
	}

	save_safemode := safemode
	safemode = false

	savefn := Curfn
	Curfn = fn
	typecheckslice(fn.Func.Inl.Slice(), Etop)
	Curfn = savefn

	safemode = save_safemode

	lineno = lno
}

// Caninl determines whether fn is inlineable.
// If so, caninl saves fn->nbody in fn->inl and substitutes it with a copy.
// fn and ->nbody will already have been typechecked.
func caninl(fn *Node) {
	if fn.Op != ODCLFUNC {
		Fatalf("caninl %v", fn)
	}
	if fn.Func.Nname == nil {
		Fatalf("caninl no nname %v", Nconv(fn, FmtSign))
	}

	// If marked "go:noinline", don't inline
	if fn.Func.Pragma&Noinline != 0 {
		return
	}

	// If fn has no body (is defined outside of Go), cannot inline it.
	if fn.Nbody.Len() == 0 {
		return
	}

	if fn.Typecheck == 0 {
		Fatalf("caninl on non-typechecked function %v", fn)
	}

	// can't handle ... args yet
	if Debug['l'] < 3 {
		f := fn.Type.Params().Fields()
		if len := f.Len(); len > 0 {
			if t := f.Index(len - 1); t.Isddd {
				return
			}
		}
	}

	// Runtime package must not be instrumented.
	// Instrument skips runtime package. However, some runtime code can be
	// inlined into other packages and instrumented there. To avoid this,
	// we disable inlining of runtime functions when instrumenting.
	// The example that we observed is inlining of LockOSThread,
	// which lead to false race reports on m contents.
	if instrumenting && myimportpath == "runtime" {
		return
	}

	const maxBudget = 80
	budget := int32(maxBudget) // allowed hairyness
	if ishairylist(fn.Nbody, &budget) || budget < 0 {
		return
	}

	savefn := Curfn
	Curfn = fn

	n := fn.Func.Nname

	n.Func.Inl.Set(fn.Nbody.Slice())
	fn.Nbody.Set(inlcopylist(n.Func.Inl.Slice()))
	inldcl := inlcopylist(n.Name.Defn.Func.Dcl)
	n.Func.Inldcl.Set(inldcl)
	n.Func.InlCost = maxBudget - budget

	// hack, TODO, check for better way to link method nodes back to the thing with the ->inl
	// this is so export can find the body of a method
	fn.Type.SetNname(n)

	if Debug['m'] > 1 {
		fmt.Printf("%v: can inline %v as: %v { %v }\n", fn.Line(), Nconv(n, FmtSharp), Tconv(fn.Type, FmtSharp), hconv(n.Func.Inl, FmtSharp))
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: can inline %v\n", fn.Line(), n)
	}

	Curfn = savefn
}

// Look for anything we want to punt on.
func ishairylist(ll Nodes, budget *int32) bool {
	for _, n := range ll.Slice() {
		if ishairy(n, budget) {
			return true
		}
	}
	return false
}

func ishairy(n *Node, budget *int32) bool {
	if n == nil {
		return false
	}

	switch n.Op {
	// Call is okay if inlinable and we have the budget for the body.
	case OCALLFUNC:
		if fn := n.Left.Func; fn != nil && fn.Inl.Len() != 0 {
			*budget -= fn.InlCost
			break
		}

		if n.Left.Op == ONAME && n.Left.Left != nil && n.Left.Left.Op == OTYPE && n.Left.Right != nil && n.Left.Right.Op == ONAME { // methods called as functions
			if d := n.Left.Sym.Def; d != nil && d.Func.Inl.Len() != 0 {
				*budget -= d.Func.InlCost
				break
			}
		}
		if Debug['l'] < 4 {
			return true
		}

	// Call is okay if inlinable and we have the budget for the body.
	case OCALLMETH:
		t := n.Left.Type
		if t == nil {
			Fatalf("no function type for [%p] %v\n", n.Left, Nconv(n.Left, FmtSign))
		}
		if t.Nname() == nil {
			Fatalf("no function definition for [%p] %v\n", t, Tconv(t, FmtSign))
		}
		if inlfn := t.Nname().Func; inlfn.Inl.Len() != 0 {
			*budget -= inlfn.InlCost
			break
		}
		if Debug['l'] < 4 {
			return true
		}

	// Things that are too hairy, irrespective of the budget
	case OCALL, OCALLINTER, OPANIC, ORECOVER:
		if Debug['l'] < 4 {
			return true
		}

	case OCLOSURE,
		OCALLPART,
		ORANGE,
		OFOR,
		OSELECT,
		OTYPESW,
		OPROC,
		ODEFER,
		ODCLTYPE, // can't print yet
		OBREAK,
		ORETJMP:
		return true
	}

	(*budget)--

	return *budget < 0 || ishairy(n.Left, budget) || ishairy(n.Right, budget) || ishairylist(n.List, budget) || ishairylist(n.Rlist, budget) || ishairylist(n.Ninit, budget) || ishairylist(n.Nbody, budget)
}

// Inlcopy and inlcopylist recursively copy the body of a function.
// Any name-like node of non-local class is marked for re-export by adding it to
// the exportlist.
func inlcopylist(ll []*Node) []*Node {
	s := make([]*Node, 0, len(ll))
	for _, n := range ll {
		s = append(s, inlcopy(n))
	}
	return s
}

func inlcopy(n *Node) *Node {
	if n == nil {
		return nil
	}

	switch n.Op {
	case ONAME, OTYPE, OLITERAL:
		return n
	}

	m := *n
	if m.Func != nil {
		m.Func.Inl.Set(nil)
	}
	m.Left = inlcopy(n.Left)
	m.Right = inlcopy(n.Right)
	m.List.Set(inlcopylist(n.List.Slice()))
	m.Rlist.Set(inlcopylist(n.Rlist.Slice()))
	m.Ninit.Set(inlcopylist(n.Ninit.Slice()))
	m.Nbody.Set(inlcopylist(n.Nbody.Slice()))

	return &m
}

// Inlcalls/nodelist/node walks fn's statements and expressions and substitutes any
// calls made to inlineable functions. This is the external entry point.
func inlcalls(fn *Node) {
	savefn := Curfn
	Curfn = fn
	fn = inlnode(fn)
	if fn != Curfn {
		Fatalf("inlnode replaced curfn")
	}
	Curfn = savefn
}

// Turn an OINLCALL into a statement.
func inlconv2stmt(n *Node) {
	n.Op = OBLOCK

	// n->ninit stays
	n.List.Set(n.Nbody.Slice())

	n.Nbody.Set(nil)
	n.Rlist.Set(nil)
}

// Turn an OINLCALL into a single valued expression.
// The result of inlconv2expr MUST be assigned back to n, e.g.
// 	n.Left = inlconv2expr(n.Left)
func inlconv2expr(n *Node) *Node {
	r := n.Rlist.First()
	return addinit(r, append(n.Ninit.Slice(), n.Nbody.Slice()...))
}

// Turn the rlist (with the return values) of the OINLCALL in
// n into an expression list lumping the ninit and body
// containing the inlined statements on the first list element so
// order will be preserved Used in return, oas2func and call
// statements.
func inlconv2list(n *Node) []*Node {
	if n.Op != OINLCALL || n.Rlist.Len() == 0 {
		Fatalf("inlconv2list %v\n", Nconv(n, FmtSign))
	}

	s := n.Rlist.Slice()
	s[0] = addinit(s[0], append(n.Ninit.Slice(), n.Nbody.Slice()...))
	return s
}

func inlnodelist(l Nodes) {
	s := l.Slice()
	for i := range s {
		s[i] = inlnode(s[i])
	}
}

// inlnode recurses over the tree to find inlineable calls, which will
// be turned into OINLCALLs by mkinlcall. When the recursion comes
// back up will examine left, right, list, rlist, ninit, ntest, nincr,
// nbody and nelse and use one of the 4 inlconv/glue functions above
// to turn the OINLCALL into an expression, a statement, or patch it
// in to this nodes list or rlist as appropriate.
// NOTE it makes no sense to pass the glue functions down the
// recursion to the level where the OINLCALL gets created because they
// have to edit /this/ n, so you'd have to push that one down as well,
// but then you may as well do it here.  so this is cleaner and
// shorter and less complicated.
// The result of inlnode MUST be assigned back to n, e.g.
// 	n.Left = inlnode(n.Left)
func inlnode(n *Node) *Node {
	if n == nil {
		return n
	}

	switch n.Op {
	// inhibit inlining of their argument
	case ODEFER, OPROC:
		switch n.Left.Op {
		case OCALLFUNC, OCALLMETH:
			// TODO(marvin): Fix Node.EType type union.
			n.Left.Etype = EType(n.Op)
		}
		fallthrough

		// TODO do them here (or earlier),
	// so escape analysis can avoid more heapmoves.
	case OCLOSURE:
		return n
	}

	lno := setlineno(n)

	inlnodelist(n.Ninit)
	for _, n1 := range n.Ninit.Slice() {
		if n1.Op == OINLCALL {
			inlconv2stmt(n1)
		}
	}

	n.Left = inlnode(n.Left)
	if n.Left != nil && n.Left.Op == OINLCALL {
		n.Left = inlconv2expr(n.Left)
	}

	n.Right = inlnode(n.Right)
	if n.Right != nil && n.Right.Op == OINLCALL {
		if n.Op == OFOR {
			inlconv2stmt(n.Right)
		} else {
			n.Right = inlconv2expr(n.Right)
		}
	}

	inlnodelist(n.List)
	switch n.Op {
	case OBLOCK:
		for _, n2 := range n.List.Slice() {
			if n2.Op == OINLCALL {
				inlconv2stmt(n2)
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
		if n.List.Len() == 1 && n.List.First().Op == OINLCALL && n.List.First().Rlist.Len() > 1 {
			n.List.Set(inlconv2list(n.List.First()))
			break
		}
		fallthrough

	default:
		s := n.List.Slice()
		for i1, n1 := range s {
			if n1.Op == OINLCALL {
				s[i1] = inlconv2expr(s[i1])
			}
		}
	}

	inlnodelist(n.Rlist)
	switch n.Op {
	case OAS2FUNC:
		if n.Rlist.First().Op == OINLCALL {
			n.Rlist.Set(inlconv2list(n.Rlist.First()))
			n.Op = OAS2
			n.Typecheck = 0
			n = typecheck(n, Etop)
			break
		}
		fallthrough

	default:
		s := n.Rlist.Slice()
		for i1, n1 := range s {
			if n1.Op == OINLCALL {
				if n.Op == OIF {
					inlconv2stmt(n1)
				} else {
					s[i1] = inlconv2expr(s[i1])
				}
			}
		}
	}

	inlnodelist(n.Nbody)
	for _, n := range n.Nbody.Slice() {
		if n.Op == OINLCALL {
			inlconv2stmt(n)
		}
	}

	// with all the branches out of the way, it is now time to
	// transmogrify this node itself unless inhibited by the
	// switch at the top of this function.
	switch n.Op {
	case OCALLFUNC, OCALLMETH:
		// TODO(marvin): Fix Node.EType type union.
		if n.Etype == EType(OPROC) || n.Etype == EType(ODEFER) {
			return n
		}
	}

	switch n.Op {
	case OCALLFUNC:
		if Debug['m'] > 3 {
			fmt.Printf("%v:call to func %v\n", n.Line(), Nconv(n.Left, FmtSign))
		}
		if n.Left.Func != nil && n.Left.Func.Inl.Len() != 0 && !isIntrinsicCall1(n) { // normal case
			n = mkinlcall(n, n.Left, n.Isddd)
		} else if n.Left.Op == ONAME && n.Left.Left != nil && n.Left.Left.Op == OTYPE && n.Left.Right != nil && n.Left.Right.Op == ONAME { // methods called as functions
			if n.Left.Sym.Def != nil {
				n = mkinlcall(n, n.Left.Sym.Def, n.Isddd)
			}
		}

	case OCALLMETH:
		if Debug['m'] > 3 {
			fmt.Printf("%v:call to meth %v\n", n.Line(), Nconv(n.Left.Right, FmtLong))
		}

		// typecheck should have resolved ODOTMETH->type, whose nname points to the actual function.
		if n.Left.Type == nil {
			Fatalf("no function type for [%p] %v\n", n.Left, Nconv(n.Left, FmtSign))
		}

		if n.Left.Type.Nname() == nil {
			Fatalf("no function definition for [%p] %v\n", n.Left.Type, Tconv(n.Left.Type, FmtSign))
		}

		n = mkinlcall(n, n.Left.Type.Nname(), n.Isddd)
	}

	lineno = lno
	return n
}

// The result of mkinlcall MUST be assigned back to n, e.g.
// 	n.Left = mkinlcall(n.Left, fn, isddd)
func mkinlcall(n *Node, fn *Node, isddd bool) *Node {
	save_safemode := safemode

	// imported functions may refer to unsafe as long as the
	// package was marked safe during import (already checked).
	pkg := fnpkg(fn)

	if pkg != localpkg && pkg != nil {
		safemode = false
	}
	n = mkinlcall1(n, fn, isddd)
	safemode = save_safemode
	return n
}

func tinlvar(t *Field) *Node {
	if t.Nname != nil && !isblank(t.Nname) {
		if t.Nname.Name.Inlvar == nil {
			Fatalf("missing inlvar for %v\n", t.Nname)
		}
		return t.Nname.Name.Inlvar
	}

	return typecheck(nblank, Erv|Easgn)
}

var inlgen int

// if *np is a call, and fn is a function with an inlinable body, substitute *np with an OINLCALL.
// On return ninit has the parameter assignments, the nbody is the
// inlined function body and list, rlist contain the input, output
// parameters.
// The result of mkinlcall1 MUST be assigned back to n, e.g.
// 	n.Left = mkinlcall1(n.Left, fn, isddd)
func mkinlcall1(n *Node, fn *Node, isddd bool) *Node {
	// For variadic fn.
	if fn.Func.Inl.Len() == 0 {
		return n
	}

	if fn == Curfn || fn.Name.Defn == Curfn {
		return n
	}

	if Debug['l'] < 2 {
		typecheckinl(fn)
	}

	// Bingo, we have a function node, and it has an inlineable body
	if Debug['m'] > 1 {
		fmt.Printf("%v: inlining call to %v %v { %v }\n", n.Line(), fn.Sym, Tconv(fn.Type, FmtSharp), hconv(fn.Func.Inl, FmtSharp))
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: inlining call to %v\n", n.Line(), fn)
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v: Before inlining: %v\n", n.Line(), Nconv(n, FmtSign))
	}

	ninit := n.Ninit

	//dumplist("ninit pre", ninit);

	var dcl []*Node
	if fn.Name.Defn != nil {
		// local function
		dcl = fn.Func.Inldcl.Slice()
	} else {
		// imported function
		dcl = fn.Func.Dcl
	}

	var retvars []*Node
	i := 0

	// Make temp names to use instead of the originals
	for _, ln := range dcl {
		if ln.Class == PPARAMOUT { // return values handled below.
			continue
		}
		if ln.isParamStackCopy() { // ignore the on-stack copy of a parameter that moved to the heap
			continue
		}
		if ln.Op == ONAME {
			ln.Name.Inlvar = typecheck(inlvar(ln), Erv)
			if ln.Class == PPARAM || ln.Name.Param.Stackcopy != nil && ln.Name.Param.Stackcopy.Class == PPARAM {
				ninit.Append(Nod(ODCL, ln.Name.Inlvar, nil))
			}
		}
	}

	// temporaries for return values.
	var m *Node
	for _, t := range fn.Type.Results().Fields().Slice() {
		if t != nil && t.Nname != nil && !isblank(t.Nname) {
			m = inlvar(t.Nname)
			m = typecheck(m, Erv)
			t.Nname.Name.Inlvar = m
		} else {
			// anonymous return values, synthesize names for use in assignment that replaces return
			m = retvar(t, i)
			i++
		}

		ninit.Append(Nod(ODCL, m, nil))
		retvars = append(retvars, m)
	}

	// assign receiver.
	if fn.Type.Recv() != nil && n.Left.Op == ODOTMETH {
		// method call with a receiver.
		t := fn.Type.Recv()

		if t != nil && t.Nname != nil && !isblank(t.Nname) && t.Nname.Name.Inlvar == nil {
			Fatalf("missing inlvar for %v\n", t.Nname)
		}
		if n.Left.Left == nil {
			Fatalf("method call without receiver: %v", Nconv(n, FmtSign))
		}
		if t == nil {
			Fatalf("method call unknown receiver type: %v", Nconv(n, FmtSign))
		}
		as := Nod(OAS, tinlvar(t), n.Left.Left)
		if as != nil {
			as = typecheck(as, Etop)
			ninit.Append(as)
		}
	}

	// check if inlined function is variadic.
	variadic := false

	var varargtype *Type
	varargcount := 0
	for _, t := range fn.Type.Params().Fields().Slice() {
		if t.Isddd {
			variadic = true
			varargtype = t.Type
		}
	}

	// but if argument is dotted too forget about variadicity.
	if variadic && isddd {
		variadic = false
	}

	// check if argument is actually a returned tuple from call.
	multiret := 0

	if n.List.Len() == 1 {
		switch n.List.First().Op {
		case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH:
			if n.List.First().Left.Type.Results().NumFields() > 1 {
				multiret = n.List.First().Left.Type.Results().NumFields() - 1
			}
		}
	}

	if variadic {
		varargcount = n.List.Len() + multiret
		if n.Left.Op != ODOTMETH {
			varargcount -= fn.Type.Recvs().NumFields()
		}
		varargcount -= fn.Type.Params().NumFields() - 1
	}

	// assign arguments to the parameters' temp names
	as := Nod(OAS2, nil, nil)

	as.Rlist.Set(n.List.Slice())
	li := 0

	// TODO: if len(nlist) == 1 but multiple args, check that n->list->n is a call?
	if fn.Type.Recv() != nil && n.Left.Op != ODOTMETH {
		// non-method call to method
		if n.List.Len() == 0 {
			Fatalf("non-method call to method without first arg: %v", Nconv(n, FmtSign))
		}

		// append receiver inlvar to LHS.
		t := fn.Type.Recv()

		if t != nil && t.Nname != nil && !isblank(t.Nname) && t.Nname.Name.Inlvar == nil {
			Fatalf("missing inlvar for %v\n", t.Nname)
		}
		if t == nil {
			Fatalf("method call unknown receiver type: %v", Nconv(n, FmtSign))
		}
		as.List.Append(tinlvar(t))
		li++
	}

	// append ordinary arguments to LHS.
	chkargcount := n.List.Len() > 1

	var vararg *Node    // the slice argument to a variadic call
	var varargs []*Node // the list of LHS names to put in vararg.
	if !chkargcount {
		// 0 or 1 expression on RHS.
		var i int
		for _, t := range fn.Type.Params().Fields().Slice() {
			if variadic && t.Isddd {
				vararg = tinlvar(t)
				for i = 0; i < varargcount && li < n.List.Len(); i++ {
					m = argvar(varargtype, i)
					varargs = append(varargs, m)
					as.List.Append(m)
				}

				break
			}

			as.List.Append(tinlvar(t))
		}
	} else {
		// match arguments except final variadic (unless the call is dotted itself)
		t, it := IterFields(fn.Type.Params())
		for t != nil {
			if li >= n.List.Len() {
				break
			}
			if variadic && t.Isddd {
				break
			}
			as.List.Append(tinlvar(t))
			t = it.Next()
			li++
		}

		// match varargcount arguments with variadic parameters.
		if variadic && t != nil && t.Isddd {
			vararg = tinlvar(t)
			var i int
			for i = 0; i < varargcount && li < n.List.Len(); i++ {
				m = argvar(varargtype, i)
				varargs = append(varargs, m)
				as.List.Append(m)
				li++
			}

			if i == varargcount {
				t = it.Next()
			}
		}

		if li < n.List.Len() || t != nil {
			Fatalf("arg count mismatch: %v  vs %v\n", Tconv(fn.Type.Params(), FmtSharp), hconv(n.List, FmtComma))
		}
	}

	if as.Rlist.Len() != 0 {
		as = typecheck(as, Etop)
		ninit.Append(as)
	}

	// turn the variadic args into a slice.
	if variadic {
		as = Nod(OAS, vararg, nil)
		if varargcount == 0 {
			as.Right = nodnil()
			as.Right.Type = varargtype
		} else {
			vararrtype := typArray(varargtype.Elem(), int64(varargcount))
			as.Right = Nod(OCOMPLIT, nil, typenod(vararrtype))
			as.Right.List.Set(varargs)
			as.Right = Nod(OSLICE, as.Right, nil)
		}

		as = typecheck(as, Etop)
		ninit.Append(as)
	}

	// zero the outparams
	for _, n := range retvars {
		as = Nod(OAS, n, nil)
		as = typecheck(as, Etop)
		ninit.Append(as)
	}

	retlabel := newlabel_inl()
	inlgen++

	subst := inlsubst{
		retlabel: retlabel,
		retvars:  retvars,
	}

	body := subst.list(fn.Func.Inl)

	body = append(body, Nod(OGOTO, retlabel, nil)) // avoid 'not used' when function doesn't have return
	body = append(body, Nod(OLABEL, retlabel, nil))

	typecheckslice(body, Etop)

	//dumplist("ninit post", ninit);

	call := Nod(OINLCALL, nil, nil)

	call.Ninit.Set(ninit.Slice())
	call.Nbody.Set(body)
	call.Rlist.Set(retvars)
	call.Type = n.Type
	call.Typecheck = 1

	// Hide the args from setlno -- the parameters to the inlined
	// call already have good line numbers that should be preserved.
	args := as.Rlist
	as.Rlist.Set(nil)

	setlno(call, n.Lineno)

	as.Rlist.Set(args.Slice())

	//dumplist("call body", body);

	n = call

	// transitive inlining
	// might be nice to do this before exporting the body,
	// but can't emit the body with inlining expanded.
	// instead we emit the things that the body needs
	// and each use must redo the inlining.
	// luckily these are small.
	body = fn.Func.Inl.Slice()
	fn.Func.Inl.Set(nil) // prevent infinite recursion (shouldn't happen anyway)
	inlnodelist(call.Nbody)
	for _, n := range call.Nbody.Slice() {
		if n.Op == OINLCALL {
			inlconv2stmt(n)
		}
	}
	fn.Func.Inl.Set(body)

	if Debug['m'] > 2 {
		fmt.Printf("%v: After inlining %v\n\n", n.Line(), Nconv(n, FmtSign))
	}

	return n
}

// Every time we expand a function we generate a new set of tmpnames,
// PAUTO's in the calling functions, and link them off of the
// PPARAM's, PAUTOS and PPARAMOUTs of the called function.
func inlvar(var_ *Node) *Node {
	if Debug['m'] > 3 {
		fmt.Printf("inlvar %v\n", Nconv(var_, FmtSign))
	}

	n := newname(var_.Sym)
	n.Type = var_.Type
	n.Class = PAUTO
	n.Used = true
	n.Name.Curfn = Curfn // the calling function, not the called one
	n.Addrtaken = var_.Addrtaken

	// This may no longer be necessary now that we run escape analysis
	// after wrapper generation, but for 1.5 this is conservatively left
	// unchanged. See bugs 11053 and 9537.
	if var_.Esc == EscHeap {
		addrescapes(n)
	}

	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *Field, i int) *Node {
	n := newname(LookupN("~r", i))
	n.Type = t.Type
	n.Class = PAUTO
	n.Used = true
	n.Name.Curfn = Curfn // the calling function, not the called one
	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's arguments
// when they come from a multiple return call.
func argvar(t *Type, i int) *Node {
	n := newname(LookupN("~arg", i))
	n.Type = t.Elem()
	n.Class = PAUTO
	n.Used = true
	n.Name.Curfn = Curfn // the calling function, not the called one
	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	return n
}

var newlabel_inl_label int

func newlabel_inl() *Node {
	newlabel_inl_label++
	n := newname(LookupN(".inlret", newlabel_inl_label))
	n.Etype = 1 // flag 'safe' for escape analysis (no backjumps)
	return n
}

// The inlsubst type implements the actual inlining of a single
// function call.
type inlsubst struct {
	// Target of the goto substituted in place of a return.
	retlabel *Node

	// Temporary result variables.
	retvars []*Node
}

// list inlines a list of nodes.
func (subst *inlsubst) list(ll Nodes) []*Node {
	s := make([]*Node, 0, ll.Len())
	for _, n := range ll.Slice() {
		s = append(s, subst.node(n))
	}
	return s
}

// node recursively copies a node from the saved pristine body of the
// inlined function, substituting references to input/output
// parameters with ones to the tmpnames, and substituting returns with
// assignments to the output.
func (subst *inlsubst) node(n *Node) *Node {
	if n == nil {
		return nil
	}

	switch n.Op {
	case ONAME:
		if n.Name.Inlvar != nil { // These will be set during inlnode
			if Debug['m'] > 2 {
				fmt.Printf("substituting name %v  ->  %v\n", Nconv(n, FmtSign), Nconv(n.Name.Inlvar, FmtSign))
			}
			return n.Name.Inlvar
		}

		if Debug['m'] > 2 {
			fmt.Printf("not substituting name %v\n", Nconv(n, FmtSign))
		}
		return n

	case OLITERAL, OTYPE:
		return n

		// Since we don't handle bodies with closures, this return is guaranteed to belong to the current inlined function.

	//		dump("Return before substitution", n);
	case ORETURN:
		m := Nod(OGOTO, subst.retlabel, nil)

		m.Ninit.Set(subst.list(n.Ninit))

		if len(subst.retvars) != 0 && n.List.Len() != 0 {
			as := Nod(OAS2, nil, nil)

			// Make a shallow copy of retvars.
			// Otherwise OINLCALL.Rlist will be the same list,
			// and later walk and typecheck may clobber it.
			for _, n := range subst.retvars {
				as.List.Append(n)
			}
			as.Rlist.Set(subst.list(n.List))
			as = typecheck(as, Etop)
			m.Ninit.Append(as)
		}

		typecheckslice(m.Ninit.Slice(), Etop)
		m = typecheck(m, Etop)

		//		dump("Return after substitution", m);
		return m

	case OGOTO, OLABEL:
		m := Nod(OXXX, nil, nil)
		*m = *n
		m.Ninit.Set(nil)
		p := fmt.Sprintf("%s·%d", n.Left.Sym.Name, inlgen)
		m.Left = newname(Lookup(p))

		return m
	default:
		m := Nod(OXXX, nil, nil)
		*m = *n
		m.Ninit.Set(nil)

		if n.Op == OCLOSURE {
			Fatalf("cannot inline function containing closure: %v", Nconv(n, FmtSign))
		}

		m.Left = subst.node(n.Left)
		m.Right = subst.node(n.Right)
		m.List.Set(subst.list(n.List))
		m.Rlist.Set(subst.list(n.Rlist))
		m.Ninit.Set(append(m.Ninit.Slice(), subst.list(n.Ninit)...))
		m.Nbody.Set(subst.list(n.Nbody))

		return m
	}
}

// Plaster over linenumbers
func setlnolist(ll Nodes, lno int32) {
	for _, n := range ll.Slice() {
		setlno(n, lno)
	}
}

func setlno(n *Node, lno int32) {
	if n == nil {
		return
	}

	// don't clobber names, unless they're freshly synthesized
	if n.Op != ONAME || n.Lineno == 0 {
		n.Lineno = lno
	}

	setlno(n.Left, lno)
	setlno(n.Right, lno)
	setlnolist(n.List, lno)
	setlnolist(n.Rlist, lno)
	setlnolist(n.Ninit, lno)
	setlnolist(n.Nbody, lno)
}
