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
//      1: 80-nodes leaf functions, oneliners, lazy typechecking (default)
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

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

// Get the function's package. For ordinary functions it's on the ->sym, but for imported methods
// the ->sym can be re-used in the local package, so peel it off the receiver's type.
func fnpkg(fn *Node) *types.Pkg {
	if fn.IsMethod() {
		// method
		rcvr := fn.Type.Recv().Type

		if rcvr.IsPtr() {
			rcvr = rcvr.Elem()
		}
		if rcvr.Sym == nil {
			Fatalf("receiver with no sym: [%v] %L  (%v)", fn.Sym, fn, rcvr)
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
		fmt.Printf("typecheck import [%v] %L { %#v }\n", fn.Sym, fn, fn.Func.Inl)
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
		Fatalf("caninl no nname %+v", fn)
	}

	var reason string // reason, if any, that the function was not inlined
	if Debug['m'] > 1 {
		defer func() {
			if reason != "" {
				fmt.Printf("%v: cannot inline %v: %s\n", fn.Line(), fn.Func.Nname, reason)
			}
		}()
	}

	// If marked "go:noinline", don't inline
	if fn.Func.Pragma&Noinline != 0 {
		reason = "marked go:noinline"
		return
	}

	// If marked "go:cgo_unsafe_args", don't inline, since the
	// function makes assumptions about its argument frame layout.
	if fn.Func.Pragma&CgoUnsafeArgs != 0 {
		reason = "marked go:cgo_unsafe_args"
		return
	}

	// If fn has no body (is defined outside of Go), cannot inline it.
	if fn.Nbody.Len() == 0 {
		reason = "no function body"
		return
	}

	if fn.Typecheck() == 0 {
		Fatalf("caninl on non-typechecked function %v", fn)
	}

	// can't handle ... args yet
	if Debug['l'] < 3 {
		f := fn.Type.Params().Fields()
		if len := f.Len(); len > 0 {
			if t := f.Index(len - 1); t.Isddd() {
				reason = "has ... args"
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
		reason = "instrumenting and is runtime function"
		return
	}

	const maxBudget = 80
	visitor := hairyVisitor{budget: maxBudget}
	if visitor.visitList(fn.Nbody) {
		reason = visitor.reason
		return
	}
	if visitor.budget < 0 {
		reason = "function too complex"
		return
	}

	savefn := Curfn
	Curfn = fn

	n := fn.Func.Nname

	n.Func.Inl.Set(fn.Nbody.Slice())
	fn.Nbody.Set(inlcopylist(n.Func.Inl.Slice()))
	inldcl := inlcopylist(n.Name.Defn.Func.Dcl)
	n.Func.Inldcl.Set(inldcl)
	n.Func.InlCost = maxBudget - visitor.budget

	// hack, TODO, check for better way to link method nodes back to the thing with the ->inl
	// this is so export can find the body of a method
	fn.Type.FuncType().Nname = asTypesNode(n)

	if Debug['m'] > 1 {
		fmt.Printf("%v: can inline %#v as: %#v { %#v }\n", fn.Line(), n, fn.Type, n.Func.Inl)
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: can inline %v\n", fn.Line(), n)
	}

	Curfn = savefn
}

// hairyVisitor visits a function body to determine its inlining
// hairiness and whether or not it can be inlined.
type hairyVisitor struct {
	budget int32
	reason string
}

// Look for anything we want to punt on.
func (v *hairyVisitor) visitList(ll Nodes) bool {
	for _, n := range ll.Slice() {
		if v.visit(n) {
			return true
		}
	}
	return false
}

func (v *hairyVisitor) visit(n *Node) bool {
	if n == nil {
		return false
	}

	switch n.Op {
	// Call is okay if inlinable and we have the budget for the body.
	case OCALLFUNC:
		if isIntrinsicCall(n) {
			v.budget--
			break
		}
		// Functions that call runtime.getcaller{pc,sp} can not be inlined
		// because getcaller{pc,sp} expect a pointer to the caller's first argument.
		if n.Left.Op == ONAME && n.Left.Class() == PFUNC && isRuntimePkg(n.Left.Sym.Pkg) {
			fn := n.Left.Sym.Name
			if fn == "getcallerpc" || fn == "getcallersp" {
				v.reason = "call to " + fn
				return true
			}
		}

		if fn := n.Left.Func; fn != nil && fn.Inl.Len() != 0 {
			v.budget -= fn.InlCost
			break
		}

		if n.isMethodCalledAsFunction() {
			if d := asNode(n.Left.Sym.Def); d != nil && d.Func.Inl.Len() != 0 {
				v.budget -= d.Func.InlCost
				break
			}
		}
		if Debug['l'] < 4 {
			v.reason = "non-leaf function"
			return true
		}

	// Call is okay if inlinable and we have the budget for the body.
	case OCALLMETH:
		t := n.Left.Type
		if t == nil {
			Fatalf("no function type for [%p] %+v\n", n.Left, n.Left)
		}
		if t.Nname() == nil {
			Fatalf("no function definition for [%p] %+v\n", t, t)
		}
		if inlfn := asNode(t.FuncType().Nname).Func; inlfn.Inl.Len() != 0 {
			v.budget -= inlfn.InlCost
			break
		}
		if Debug['l'] < 4 {
			v.reason = "non-leaf method"
			return true
		}

	// Things that are too hairy, irrespective of the budget
	case OCALL, OCALLINTER, OPANIC, ORECOVER:
		if Debug['l'] < 4 {
			v.reason = "non-leaf op " + n.Op.String()
			return true
		}

	case OCLOSURE,
		OCALLPART,
		ORANGE,
		OFOR,
		OFORUNTIL,
		OSELECT,
		OTYPESW,
		OPROC,
		ODEFER,
		ODCLTYPE, // can't print yet
		OBREAK,
		ORETJMP:
		v.reason = "unhandled op " + n.Op.String()
		return true
	}

	v.budget--
	// TODO(mdempsky/josharian): Hacks to appease toolstash; remove.
	// See issue 17566 and CL 31674 for discussion.
	switch n.Op {
	case OSTRUCTKEY:
		v.budget--
	case OSLICE, OSLICEARR, OSLICESTR:
		v.budget--
	case OSLICE3, OSLICE3ARR:
		v.budget -= 2
	}

	if v.budget < 0 {
		v.reason = "function too complex"
		return true
	}

	return v.visit(n.Left) || v.visit(n.Right) ||
		v.visitList(n.List) || v.visitList(n.Rlist) ||
		v.visitList(n.Ninit) || v.visitList(n.Nbody)
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
		Fatalf("inlconv2list %+v\n", n)
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
			n.Left.SetNoInline(true)
		}
		return n

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
		if n.Op == OFOR || n.Op == OFORUNTIL {
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

	case ORETURN, OCALLFUNC, OCALLMETH, OCALLINTER, OAPPEND, OCOMPLEX:
		// if we just replaced arg in f(arg()) or return arg with an inlined call
		// and arg returns multiple values, glue as list
		if n.List.Len() == 1 && n.List.First().Op == OINLCALL && n.List.First().Rlist.Len() > 1 {
			n.List.Set(inlconv2list(n.List.First()))
			break
		}
		fallthrough

	default:
		s := n.List.Slice()
		for i1, n1 := range s {
			if n1 != nil && n1.Op == OINLCALL {
				s[i1] = inlconv2expr(s[i1])
			}
		}
	}

	inlnodelist(n.Rlist)
	if n.Op == OAS2FUNC && n.Rlist.First().Op == OINLCALL {
		n.Rlist.Set(inlconv2list(n.Rlist.First()))
		n.Op = OAS2
		n.SetTypecheck(0)
		n = typecheck(n, Etop)
	} else {
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
		if n.NoInline() {
			return n
		}
	}

	switch n.Op {
	case OCALLFUNC:
		if Debug['m'] > 3 {
			fmt.Printf("%v:call to func %+v\n", n.Line(), n.Left)
		}
		if n.Left.Func != nil && n.Left.Func.Inl.Len() != 0 && !isIntrinsicCall(n) { // normal case
			n = mkinlcall(n, n.Left, n.Isddd())
		} else if n.isMethodCalledAsFunction() && asNode(n.Left.Sym.Def) != nil {
			n = mkinlcall(n, asNode(n.Left.Sym.Def), n.Isddd())
		}

	case OCALLMETH:
		if Debug['m'] > 3 {
			fmt.Printf("%v:call to meth %L\n", n.Line(), n.Left.Right)
		}

		// typecheck should have resolved ODOTMETH->type, whose nname points to the actual function.
		if n.Left.Type == nil {
			Fatalf("no function type for [%p] %+v\n", n.Left, n.Left)
		}

		if n.Left.Type.Nname() == nil {
			Fatalf("no function definition for [%p] %+v\n", n.Left.Type, n.Left.Type)
		}

		n = mkinlcall(n, asNode(n.Left.Type.FuncType().Nname), n.Isddd())
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

func tinlvar(t *types.Field, inlvars map[*Node]*Node) *Node {
	if asNode(t.Nname) != nil && !isblank(asNode(t.Nname)) {
		inlvar := inlvars[asNode(t.Nname)]
		if inlvar == nil {
			Fatalf("missing inlvar for %v\n", asNode(t.Nname))
		}
		return inlvar
	}

	return typecheck(nblank, Erv|Easgn)
}

var inlgen int

// If n is a call, and fn is a function with an inlinable body,
// return an OINLCALL.
// On return ninit has the parameter assignments, the nbody is the
// inlined function body and list, rlist contain the input, output
// parameters.
// The result of mkinlcall1 MUST be assigned back to n, e.g.
// 	n.Left = mkinlcall1(n.Left, fn, isddd)
func mkinlcall1(n *Node, fn *Node, isddd bool) *Node {
	if fn.Func.Inl.Len() == 0 {
		// No inlinable body.
		return n
	}

	if fn == Curfn || fn.Name.Defn == Curfn {
		// Can't recursively inline a function into itself.
		return n
	}

	if Debug['l'] < 2 {
		typecheckinl(fn)
	}

	// We have a function node, and it has an inlineable body.
	if Debug['m'] > 1 {
		fmt.Printf("%v: inlining call to %v %#v { %#v }\n", n.Line(), fn.Sym, fn.Type, fn.Func.Inl)
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: inlining call to %v\n", n.Line(), fn)
	}
	if Debug['m'] > 2 {
		fmt.Printf("%v: Before inlining: %+v\n", n.Line(), n)
	}

	ninit := n.Ninit

	// Find declarations corresponding to inlineable body.
	var dcl []*Node
	if fn.Name.Defn != nil {
		dcl = fn.Func.Inldcl.Slice() // local function
	} else {
		dcl = fn.Func.Dcl // imported function
	}

	// Make temp names to use instead of the originals.
	inlvars := make(map[*Node]*Node)
	for _, ln := range dcl {
		if ln.Op != ONAME {
			continue
		}
		if ln.Class() == PPARAMOUT { // return values handled below.
			continue
		}
		if ln.isParamStackCopy() { // ignore the on-stack copy of a parameter that moved to the heap
			continue
		}
		inlvars[ln] = typecheck(inlvar(ln), Erv)
		if ln.Class() == PPARAM || ln.Name.Param.Stackcopy != nil && ln.Name.Param.Stackcopy.Class() == PPARAM {
			ninit.Append(nod(ODCL, inlvars[ln], nil))
		}
	}

	// temporaries for return values.
	var retvars []*Node
	for i, t := range fn.Type.Results().Fields().Slice() {
		var m *Node
		if t != nil && asNode(t.Nname) != nil && !isblank(asNode(t.Nname)) {
			m = inlvar(asNode(t.Nname))
			m = typecheck(m, Erv)
			inlvars[asNode(t.Nname)] = m
		} else {
			// anonymous return values, synthesize names for use in assignment that replaces return
			m = retvar(t, i)
		}

		ninit.Append(nod(ODCL, m, nil))
		retvars = append(retvars, m)
	}

	// Assign arguments to the parameters' temp names.
	as := nod(OAS2, nil, nil)
	as.Rlist.Set(n.List.Slice())

	// For non-dotted calls to variadic functions, we assign the
	// variadic parameter's temp name separately.
	var vas *Node

	if fn.IsMethod() {
		rcv := fn.Type.Recv()

		if n.Left.Op == ODOTMETH {
			// For x.M(...), assign x directly to the
			// receiver parameter.
			if n.Left.Left == nil {
				Fatalf("method call without receiver: %+v", n)
			}
			ras := nod(OAS, tinlvar(rcv, inlvars), n.Left.Left)
			ras = typecheck(ras, Etop)
			ninit.Append(ras)
		} else {
			// For T.M(...), add the receiver parameter to
			// as.List, so it's assigned by the normal
			// arguments.
			if as.Rlist.Len() == 0 {
				Fatalf("non-method call to method without first arg: %+v", n)
			}
			as.List.Append(tinlvar(rcv, inlvars))
		}
	}

	for _, param := range fn.Type.Params().Fields().Slice() {
		// For ordinary parameters or variadic parameters in
		// dotted calls, just add the variable to the
		// assignment list, and we're done.
		if !param.Isddd() || isddd {
			as.List.Append(tinlvar(param, inlvars))
			continue
		}

		// Otherwise, we need to collect the remaining values
		// to pass as a slice.

		numvals := n.List.Len()
		if numvals == 1 && n.List.First().Type.IsFuncArgStruct() {
			numvals = n.List.First().Type.NumFields()
		}

		x := as.List.Len()
		for as.List.Len() < numvals {
			as.List.Append(argvar(param.Type, as.List.Len()))
		}
		varargs := as.List.Slice()[x:]

		vas = nod(OAS, tinlvar(param, inlvars), nil)
		if len(varargs) == 0 {
			vas.Right = nodnil()
			vas.Right.Type = param.Type
		} else {
			vas.Right = nod(OCOMPLIT, nil, typenod(param.Type))
			vas.Right.List.Set(varargs)
		}
	}

	if as.Rlist.Len() != 0 {
		as = typecheck(as, Etop)
		ninit.Append(as)
	}

	if vas != nil {
		vas = typecheck(vas, Etop)
		ninit.Append(vas)
	}

	// Zero the return parameters.
	for _, n := range retvars {
		ras := nod(OAS, n, nil)
		ras = typecheck(ras, Etop)
		ninit.Append(ras)
	}

	retlabel := autolabel(".i")
	retlabel.Etype = 1 // flag 'safe' for escape analysis (no backjumps)

	inlgen++

	subst := inlsubst{
		retlabel: retlabel,
		retvars:  retvars,
		inlvars:  inlvars,
	}

	body := subst.list(fn.Func.Inl)

	lab := nod(OLABEL, retlabel, nil)
	body = append(body, lab)

	typecheckslice(body, Etop)

	//dumplist("ninit post", ninit);

	call := nod(OINLCALL, nil, nil)
	call.Ninit.Set(ninit.Slice())
	call.Nbody.Set(body)
	call.Rlist.Set(retvars)
	call.Type = n.Type
	call.SetTypecheck(1)

	// Hide the args from setPos -- the parameters to the inlined
	// call already have good line numbers that should be preserved.
	args := as.Rlist
	as.Rlist.Set(nil)

	// Rewrite the line information for the inlined AST.
	parent := -1
	callBase := Ctxt.PosTable.Pos(n.Pos).Base()
	if callBase != nil {
		parent = callBase.InliningIndex()
	}
	newIndex := Ctxt.InlTree.Add(parent, n.Pos, fn.Sym.Linksym())
	setpos := &setPos{
		bases:       make(map[*src.PosBase]*src.PosBase),
		newInlIndex: newIndex,
	}
	setpos.node(call)

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
		fmt.Printf("%v: After inlining %+v\n\n", n.Line(), n)
	}

	return n
}

// Every time we expand a function we generate a new set of tmpnames,
// PAUTO's in the calling functions, and link them off of the
// PPARAM's, PAUTOS and PPARAMOUTs of the called function.
func inlvar(var_ *Node) *Node {
	if Debug['m'] > 3 {
		fmt.Printf("inlvar %+v\n", var_)
	}

	n := newname(var_.Sym)
	n.Type = var_.Type
	n.SetClass(PAUTO)
	n.Name.SetUsed(true)
	n.Name.Curfn = Curfn // the calling function, not the called one
	n.SetAddrtaken(var_.Addrtaken())

	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *types.Field, i int) *Node {
	n := newname(lookupN("~r", i))
	n.Type = t.Type
	n.SetClass(PAUTO)
	n.Name.SetUsed(true)
	n.Name.Curfn = Curfn // the calling function, not the called one
	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's arguments
// when they come from a multiple return call.
func argvar(t *types.Type, i int) *Node {
	n := newname(lookupN("~arg", i))
	n.Type = t.Elem()
	n.SetClass(PAUTO)
	n.Name.SetUsed(true)
	n.Name.Curfn = Curfn // the calling function, not the called one
	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	return n
}

// The inlsubst type implements the actual inlining of a single
// function call.
type inlsubst struct {
	// Target of the goto substituted in place of a return.
	retlabel *Node

	// Temporary result variables.
	retvars []*Node

	inlvars map[*Node]*Node
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
		if inlvar := subst.inlvars[n]; inlvar != nil { // These will be set during inlnode
			if Debug['m'] > 2 {
				fmt.Printf("substituting name %+v  ->  %+v\n", n, inlvar)
			}
			return inlvar
		}

		if Debug['m'] > 2 {
			fmt.Printf("not substituting name %+v\n", n)
		}
		return n

	case OLITERAL, OTYPE:
		// If n is a named constant or type, we can continue
		// using it in the inline copy. Otherwise, make a copy
		// so we can update the line number.
		if n.Sym != nil {
			return n
		}

		// Since we don't handle bodies with closures, this return is guaranteed to belong to the current inlined function.

	//		dump("Return before substitution", n);
	case ORETURN:
		m := nod(OGOTO, subst.retlabel, nil)

		m.Ninit.Set(subst.list(n.Ninit))

		if len(subst.retvars) != 0 && n.List.Len() != 0 {
			as := nod(OAS2, nil, nil)

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
		m := nod(OXXX, nil, nil)
		*m = *n
		m.Ninit.Set(nil)
		p := fmt.Sprintf("%s·%d", n.Left.Sym.Name, inlgen)
		m.Left = newname(lookup(p))

		return m
	}

	m := nod(OXXX, nil, nil)
	*m = *n
	m.Ninit.Set(nil)

	if n.Op == OCLOSURE {
		Fatalf("cannot inline function containing closure: %+v", n)
	}

	m.Left = subst.node(n.Left)
	m.Right = subst.node(n.Right)
	m.List.Set(subst.list(n.List))
	m.Rlist.Set(subst.list(n.Rlist))
	m.Ninit.Set(append(m.Ninit.Slice(), subst.list(n.Ninit)...))
	m.Nbody.Set(subst.list(n.Nbody))

	return m
}

// setPos is a visitor to update position info with a new inlining index.
type setPos struct {
	bases       map[*src.PosBase]*src.PosBase
	newInlIndex int
}

func (s *setPos) nodelist(ll Nodes) {
	for _, n := range ll.Slice() {
		s.node(n)
	}
}

func (s *setPos) node(n *Node) {
	if n == nil {
		return
	}
	if n.Op == OLITERAL || n.Op == OTYPE {
		if n.Sym != nil {
			// This node is not a copy, so don't clobber position.
			return
		}
	}

	// don't clobber names, unless they're freshly synthesized
	if n.Op != ONAME || !n.Pos.IsKnown() {
		n.Pos = s.updatedPos(n)
	}

	s.node(n.Left)
	s.node(n.Right)
	s.nodelist(n.List)
	s.nodelist(n.Rlist)
	s.nodelist(n.Ninit)
	s.nodelist(n.Nbody)
}

func (s *setPos) updatedPos(n *Node) src.XPos {
	pos := Ctxt.PosTable.Pos(n.Pos)
	oldbase := pos.Base() // can be nil
	newbase := s.bases[oldbase]
	if newbase == nil {
		newbase = src.NewInliningBase(oldbase, s.newInlIndex)
		pos.SetBase(newbase)
		s.bases[oldbase] = newbase
	}
	pos.SetBase(newbase)
	return Ctxt.PosTable.XPos(pos)
}

func (n *Node) isMethodCalledAsFunction() bool {
	return n.Left.Op == ONAME && n.Left.Left != nil && n.Left.Left.Op == OTYPE && n.Left.Right != nil && n.Left.Right.Op == ONAME
}
