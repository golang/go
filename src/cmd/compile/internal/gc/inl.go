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
// making 1 the default and -l disable. Additional levels (beyond -l) may be buggy and
// are not supported.
//      0: disabled
//      1: 80-nodes leaf functions, oneliners, panic, lazy typechecking (default)
//      2: (unassigned)
//      3: (unassigned)
//      4: allow non-leaf functions
//
// At some point this may get another default and become switch-offable with -N.
//
// The -d typcheckinl flag enables early typechecking of all imported bodies,
// which is useful to flush out bugs.
//
// The debug['m'] flag enables diagnostic output.  a single -m is useful for verifying
// which calls get inlined or not, more is for debugging, and may go away at any point.

package gc

import (
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"strings"
)

// Inlining budget parameters, gathered in one place
const (
	inlineMaxBudget       = 80
	inlineExtraAppendCost = 0
	// default is to inline if there's at most one call. -l=4 overrides this by using 1 instead.
	inlineExtraCallCost  = 57              // 57 was benchmarked to provided most benefit with no bad surprises; see https://github.com/golang/go/issues/19348#issuecomment-439370742
	inlineExtraPanicCost = 1               // do not penalize inlining panics.
	inlineExtraThrowCost = inlineMaxBudget // with current (2018-05/1.11) code, inlining runtime.throw does not help.

	inlineBigFunctionNodes   = 5000 // Functions with this many nodes are considered "big".
	inlineBigFunctionMaxCost = 20   // Max cost of inlinee when inlining into a "big" function.
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

	expandInline(fn)

	// typecheckinl is only for imported functions;
	// their bodies may refer to unsafe as long as the package
	// was marked safe during import (which was checked then).
	// the ->inl of a local function has been typechecked before caninl copied it.
	pkg := fnpkg(fn)

	if pkg == localpkg || pkg == nil {
		return // typecheckinl on local function
	}

	if Debug['m'] > 2 || Debug_export != 0 {
		fmt.Printf("typecheck import [%v] %L { %#v }\n", fn.Sym, fn, asNodes(fn.Func.Inl.Body))
	}

	savefn := Curfn
	Curfn = fn
	typecheckslice(fn.Func.Inl.Body, ctxStmt)
	Curfn = savefn

	// During typechecking, declarations are added to
	// Curfn.Func.Dcl. Move them to Inl.Dcl for consistency with
	// how local functions behave. (Append because typecheckinl
	// may be called multiple times.)
	fn.Func.Inl.Dcl = append(fn.Func.Inl.Dcl, fn.Func.Dcl...)
	fn.Func.Dcl = nil

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
	if Debug['m'] > 1 || logopt.Enabled() {
		defer func() {
			if reason != "" {
				if Debug['m'] > 1 {
					fmt.Printf("%v: cannot inline %v: %s\n", fn.Line(), fn.Func.Nname, reason)
				}
				if logopt.Enabled() {
					logopt.LogOpt(fn.Pos, "cannotInlineFunction", "inline", fn.funcname(), reason)
				}
			}
		}()
	}

	// If marked "go:noinline", don't inline
	if fn.Func.Pragma&Noinline != 0 {
		reason = "marked go:noinline"
		return
	}

	// If marked "go:norace" and -race compilation, don't inline.
	if flag_race && fn.Func.Pragma&Norace != 0 {
		reason = "marked go:norace with -race compilation"
		return
	}

	// If marked "go:nocheckptr" and -d checkptr compilation, don't inline.
	if Debug_checkptr != 0 && fn.Func.Pragma&NoCheckPtr != 0 {
		reason = "marked go:nocheckptr"
		return
	}

	// If marked "go:cgo_unsafe_args", don't inline, since the
	// function makes assumptions about its argument frame layout.
	if fn.Func.Pragma&CgoUnsafeArgs != 0 {
		reason = "marked go:cgo_unsafe_args"
		return
	}

	// If marked as "go:uintptrescapes", don't inline, since the
	// escape information is lost during inlining.
	if fn.Func.Pragma&UintptrEscapes != 0 {
		reason = "marked as having an escaping uintptr argument"
		return
	}

	// The nowritebarrierrec checker currently works at function
	// granularity, so inlining yeswritebarrierrec functions can
	// confuse it (#22342). As a workaround, disallow inlining
	// them for now.
	if fn.Func.Pragma&Yeswritebarrierrec != 0 {
		reason = "marked go:yeswritebarrierrec"
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

	n := fn.Func.Nname
	if n.Func.InlinabilityChecked() {
		return
	}
	defer n.Func.SetInlinabilityChecked(true)

	cc := int32(inlineExtraCallCost)
	if Debug['l'] == 4 {
		cc = 1 // this appears to yield better performance than 0.
	}

	// At this point in the game the function we're looking at may
	// have "stale" autos, vars that still appear in the Dcl list, but
	// which no longer have any uses in the function body (due to
	// elimination by deadcode). We'd like to exclude these dead vars
	// when creating the "Inline.Dcl" field below; to accomplish this,
	// the hairyVisitor below builds up a map of used/referenced
	// locals, and we use this map to produce a pruned Inline.Dcl
	// list. See issue 25249 for more context.

	visitor := hairyVisitor{
		budget:        inlineMaxBudget,
		extraCallCost: cc,
		usedLocals:    make(map[*Node]bool),
	}
	if visitor.visitList(fn.Nbody) {
		reason = visitor.reason
		return
	}
	if visitor.budget < 0 {
		reason = fmt.Sprintf("function too complex: cost %d exceeds budget %d", inlineMaxBudget-visitor.budget, inlineMaxBudget)
		return
	}

	n.Func.Inl = &Inline{
		Cost: inlineMaxBudget - visitor.budget,
		Dcl:  inlcopylist(pruneUnusedAutos(n.Name.Defn.Func.Dcl, &visitor)),
		Body: inlcopylist(fn.Nbody.Slice()),
	}

	// hack, TODO, check for better way to link method nodes back to the thing with the ->inl
	// this is so export can find the body of a method
	fn.Type.FuncType().Nname = asTypesNode(n)

	if Debug['m'] > 1 {
		fmt.Printf("%v: can inline %#v with cost %d as: %#v { %#v }\n", fn.Line(), n, inlineMaxBudget-visitor.budget, fn.Type, asNodes(n.Func.Inl.Body))
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: can inline %v\n", fn.Line(), n)
	}
	if logopt.Enabled() {
		logopt.LogOpt(fn.Pos, "canInlineFunction", "inline", fn.funcname(), fmt.Sprintf("cost: %d", inlineMaxBudget-visitor.budget))
	}
}

// inlFlood marks n's inline body for export and recursively ensures
// all called functions are marked too.
func inlFlood(n *Node) {
	if n == nil {
		return
	}
	if n.Op != ONAME || n.Class() != PFUNC {
		Fatalf("inlFlood: unexpected %v, %v, %v", n, n.Op, n.Class())
	}
	if n.Func == nil {
		Fatalf("inlFlood: missing Func on %v", n)
	}
	if n.Func.Inl == nil {
		return
	}

	if n.Func.ExportInline() {
		return
	}
	n.Func.SetExportInline(true)

	typecheckinl(n)

	inspectList(asNodes(n.Func.Inl.Body), func(n *Node) bool {
		switch n.Op {
		case ONAME:
			// Mark any referenced global variables or
			// functions for reexport. Skip methods,
			// because they're reexported alongside their
			// receiver type.
			if n.Class() == PEXTERN || n.Class() == PFUNC && !n.isMethodExpression() {
				exportsym(n)
			}

		case OCALLFUNC, OCALLMETH:
			// Recursively flood any functions called by
			// this one.
			inlFlood(asNode(n.Left.Type.Nname()))
		}
		return true
	})
}

// hairyVisitor visits a function body to determine its inlining
// hairiness and whether or not it can be inlined.
type hairyVisitor struct {
	budget        int32
	reason        string
	extraCallCost int32
	usedLocals    map[*Node]bool
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
		// Functions that call runtime.getcaller{pc,sp} can not be inlined
		// because getcaller{pc,sp} expect a pointer to the caller's first argument.
		//
		// runtime.throw is a "cheap call" like panic in normal code.
		if n.Left.Op == ONAME && n.Left.Class() == PFUNC && isRuntimePkg(n.Left.Sym.Pkg) {
			fn := n.Left.Sym.Name
			if fn == "getcallerpc" || fn == "getcallersp" {
				v.reason = "call to " + fn
				return true
			}
			if fn == "throw" {
				v.budget -= inlineExtraThrowCost
				break
			}
		}

		if isIntrinsicCall(n) {
			// Treat like any other node.
			break
		}

		if fn := n.Left.Func; fn != nil && fn.Inl != nil {
			v.budget -= fn.Inl.Cost
			break
		}
		if n.Left.isMethodExpression() {
			if d := asNode(n.Left.Sym.Def); d != nil && d.Func.Inl != nil {
				v.budget -= d.Func.Inl.Cost
				break
			}
		}
		// TODO(mdempsky): Budget for OCLOSURE calls if we
		// ever allow that. See #15561 and #23093.

		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	// Call is okay if inlinable and we have the budget for the body.
	case OCALLMETH:
		t := n.Left.Type
		if t == nil {
			Fatalf("no function type for [%p] %+v\n", n.Left, n.Left)
		}
		if t.Nname() == nil {
			Fatalf("no function definition for [%p] %+v\n", t, t)
		}
		if isRuntimePkg(n.Left.Sym.Pkg) {
			fn := n.Left.Sym.Name
			if fn == "heapBits.nextArena" {
				// Special case: explicitly allow
				// mid-stack inlining of
				// runtime.heapBits.next even though
				// it calls slow-path
				// runtime.heapBits.nextArena.
				break
			}
		}
		if inlfn := asNode(t.FuncType().Nname).Func; inlfn.Inl != nil {
			v.budget -= inlfn.Inl.Cost
			break
		}
		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	// Things that are too hairy, irrespective of the budget
	case OCALL, OCALLINTER:
		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	case OPANIC:
		v.budget -= inlineExtraPanicCost

	case ORECOVER:
		// recover matches the argument frame pointer to find
		// the right panic value, so it needs an argument frame.
		v.reason = "call to recover"
		return true

	case OCLOSURE,
		OCALLPART,
		ORANGE,
		OFOR,
		OFORUNTIL,
		OSELECT,
		OTYPESW,
		OGO,
		ODEFER,
		ODCLTYPE, // can't print yet
		OBREAK,
		ORETJMP:
		v.reason = "unhandled op " + n.Op.String()
		return true

	case OAPPEND:
		v.budget -= inlineExtraAppendCost

	case ODCLCONST, OEMPTY, OFALL, OLABEL:
		// These nodes don't produce code; omit from inlining budget.
		return false

	case OIF:
		if Isconst(n.Left, CTBOOL) {
			// This if and the condition cost nothing.
			return v.visitList(n.Ninit) || v.visitList(n.Nbody) ||
				v.visitList(n.Rlist)
		}

	case ONAME:
		if n.Class() == PAUTO {
			v.usedLocals[n] = true
		}

	}

	v.budget--

	// When debugging, don't stop early, to get full cost of inlining this function
	if v.budget < 0 && Debug['m'] < 2 && !logopt.Enabled() {
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

	m := n.copy()
	if m.Func != nil {
		Fatalf("unexpected Func: %v", m)
	}
	m.Left = inlcopy(n.Left)
	m.Right = inlcopy(n.Right)
	m.List.Set(inlcopylist(n.List.Slice()))
	m.Rlist.Set(inlcopylist(n.Rlist.Slice()))
	m.Ninit.Set(inlcopylist(n.Ninit.Slice()))
	m.Nbody.Set(inlcopylist(n.Nbody.Slice()))

	return m
}

func countNodes(n *Node) int {
	if n == nil {
		return 0
	}
	cnt := 1
	cnt += countNodes(n.Left)
	cnt += countNodes(n.Right)
	for _, n1 := range n.Ninit.Slice() {
		cnt += countNodes(n1)
	}
	for _, n1 := range n.Nbody.Slice() {
		cnt += countNodes(n1)
	}
	for _, n1 := range n.List.Slice() {
		cnt += countNodes(n1)
	}
	for _, n1 := range n.Rlist.Slice() {
		cnt += countNodes(n1)
	}
	return cnt
}

// Inlcalls/nodelist/node walks fn's statements and expressions and substitutes any
// calls made to inlineable functions. This is the external entry point.
func inlcalls(fn *Node) {
	savefn := Curfn
	Curfn = fn
	maxCost := int32(inlineMaxBudget)
	if countNodes(fn) >= inlineBigFunctionNodes {
		maxCost = inlineBigFunctionMaxCost
	}
	fn = inlnode(fn, maxCost)
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

func inlnodelist(l Nodes, maxCost int32) {
	s := l.Slice()
	for i := range s {
		s[i] = inlnode(s[i], maxCost)
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
func inlnode(n *Node, maxCost int32) *Node {
	if n == nil {
		return n
	}

	switch n.Op {
	// inhibit inlining of their argument
	case ODEFER, OGO:
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

	inlnodelist(n.Ninit, maxCost)
	for _, n1 := range n.Ninit.Slice() {
		if n1.Op == OINLCALL {
			inlconv2stmt(n1)
		}
	}

	n.Left = inlnode(n.Left, maxCost)
	if n.Left != nil && n.Left.Op == OINLCALL {
		n.Left = inlconv2expr(n.Left)
	}

	n.Right = inlnode(n.Right, maxCost)
	if n.Right != nil && n.Right.Op == OINLCALL {
		if n.Op == OFOR || n.Op == OFORUNTIL {
			inlconv2stmt(n.Right)
		} else if n.Op == OAS2FUNC {
			n.Rlist.Set(inlconv2list(n.Right))
			n.Right = nil
			n.Op = OAS2
			n.SetTypecheck(0)
			n = typecheck(n, ctxStmt)
		} else {
			n.Right = inlconv2expr(n.Right)
		}
	}

	inlnodelist(n.List, maxCost)
	if n.Op == OBLOCK {
		for _, n2 := range n.List.Slice() {
			if n2.Op == OINLCALL {
				inlconv2stmt(n2)
			}
		}
	} else {
		s := n.List.Slice()
		for i1, n1 := range s {
			if n1 != nil && n1.Op == OINLCALL {
				s[i1] = inlconv2expr(s[i1])
			}
		}
	}

	inlnodelist(n.Rlist, maxCost)
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

	inlnodelist(n.Nbody, maxCost)
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
		if n.Left.Func != nil && n.Left.Func.Inl != nil && !isIntrinsicCall(n) { // normal case
			n = mkinlcall(n, n.Left, maxCost)
		} else if n.Left.isMethodExpression() && asNode(n.Left.Sym.Def) != nil {
			n = mkinlcall(n, asNode(n.Left.Sym.Def), maxCost)
		} else if n.Left.Op == OCLOSURE {
			if f := inlinableClosure(n.Left); f != nil {
				n = mkinlcall(n, f, maxCost)
			}
		} else if n.Left.Op == ONAME && n.Left.Name != nil && n.Left.Name.Defn != nil {
			if d := n.Left.Name.Defn; d.Op == OAS && d.Right.Op == OCLOSURE {
				if f := inlinableClosure(d.Right); f != nil {
					// NB: this check is necessary to prevent indirect re-assignment of the variable
					// having the address taken after the invocation or only used for reads is actually fine
					// but we have no easy way to distinguish the safe cases
					if d.Left.Name.Addrtaken() {
						if Debug['m'] > 1 {
							fmt.Printf("%v: cannot inline escaping closure variable %v\n", n.Line(), n.Left)
						}
						break
					}

					// ensure the variable is never re-assigned
					if unsafe, a := reassigned(n.Left); unsafe {
						if Debug['m'] > 1 {
							if a != nil {
								fmt.Printf("%v: cannot inline re-assigned closure variable at %v: %v\n", n.Line(), a.Line(), a)
							} else {
								fmt.Printf("%v: cannot inline global closure variable %v\n", n.Line(), n.Left)
							}
						}
						break
					}
					n = mkinlcall(n, f, maxCost)
				}
			}
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

		n = mkinlcall(n, asNode(n.Left.Type.FuncType().Nname), maxCost)
	}

	lineno = lno
	return n
}

// inlinableClosure takes an OCLOSURE node and follows linkage to the matching ONAME with
// the inlinable body. Returns nil if the function is not inlinable.
func inlinableClosure(n *Node) *Node {
	c := n.Func.Closure
	caninl(c)
	f := c.Func.Nname
	if f == nil || f.Func.Inl == nil {
		return nil
	}
	return f
}

// reassigned takes an ONAME node, walks the function in which it is defined, and returns a boolean
// indicating whether the name has any assignments other than its declaration.
// The second return value is the first such assignment encountered in the walk, if any. It is mostly
// useful for -m output documenting the reason for inhibited optimizations.
// NB: global variables are always considered to be re-assigned.
// TODO: handle initial declaration not including an assignment and followed by a single assignment?
func reassigned(n *Node) (bool, *Node) {
	if n.Op != ONAME {
		Fatalf("reassigned %v", n)
	}
	// no way to reliably check for no-reassignment of globals, assume it can be
	if n.Name.Curfn == nil {
		return true, nil
	}
	f := n.Name.Curfn
	// There just might be a good reason for this although this can be pretty surprising:
	// local variables inside a closure have Curfn pointing to the OCLOSURE node instead
	// of the corresponding ODCLFUNC.
	// We need to walk the function body to check for reassignments so we follow the
	// linkage to the ODCLFUNC node as that is where body is held.
	if f.Op == OCLOSURE {
		f = f.Func.Closure
	}
	v := reassignVisitor{name: n}
	a := v.visitList(f.Nbody)
	return a != nil, a
}

type reassignVisitor struct {
	name *Node
}

func (v *reassignVisitor) visit(n *Node) *Node {
	if n == nil {
		return nil
	}
	switch n.Op {
	case OAS:
		if n.Left == v.name && n != v.name.Name.Defn {
			return n
		}
		return nil
	case OAS2, OAS2FUNC, OAS2MAPR, OAS2DOTTYPE:
		for _, p := range n.List.Slice() {
			if p == v.name && n != v.name.Name.Defn {
				return n
			}
		}
		return nil
	}
	if a := v.visit(n.Left); a != nil {
		return a
	}
	if a := v.visit(n.Right); a != nil {
		return a
	}
	if a := v.visitList(n.List); a != nil {
		return a
	}
	if a := v.visitList(n.Rlist); a != nil {
		return a
	}
	if a := v.visitList(n.Ninit); a != nil {
		return a
	}
	if a := v.visitList(n.Nbody); a != nil {
		return a
	}
	return nil
}

func (v *reassignVisitor) visitList(l Nodes) *Node {
	for _, n := range l.Slice() {
		if a := v.visit(n); a != nil {
			return a
		}
	}
	return nil
}

func tinlvar(t *types.Field, inlvars map[*Node]*Node) *Node {
	if n := asNode(t.Nname); n != nil && !n.isBlank() {
		inlvar := inlvars[n]
		if inlvar == nil {
			Fatalf("missing inlvar for %v\n", n)
		}
		return inlvar
	}

	return typecheck(nblank, ctxExpr|ctxAssign)
}

var inlgen int

// If n is a call, and fn is a function with an inlinable body,
// return an OINLCALL.
// On return ninit has the parameter assignments, the nbody is the
// inlined function body and list, rlist contain the input, output
// parameters.
// The result of mkinlcall MUST be assigned back to n, e.g.
// 	n.Left = mkinlcall(n.Left, fn, isddd)
func mkinlcall(n, fn *Node, maxCost int32) *Node {
	if fn.Func.Inl == nil {
		// No inlinable body.
		return n
	}
	if fn.Func.Inl.Cost > maxCost {
		// The inlined function body is too big. Typically we use this check to restrict
		// inlining into very big functions.  See issue 26546 and 17566.
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos, "cannotInlineCall", "inline", Curfn.funcname(),
				fmt.Sprintf("cost %d of %s exceeds max large caller cost %d", fn.Func.Inl.Cost, fn.pkgFuncName(), maxCost))
		}
		return n
	}

	if fn == Curfn || fn.Name.Defn == Curfn {
		// Can't recursively inline a function into itself.
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos, "cannotInlineCall", "inline", fmt.Sprintf("recursive call to %s", Curfn.funcname()))
		}
		return n
	}

	if instrumenting && isRuntimePkg(fn.Sym.Pkg) {
		// Runtime package must not be instrumented.
		// Instrument skips runtime package. However, some runtime code can be
		// inlined into other packages and instrumented there. To avoid this,
		// we disable inlining of runtime functions when instrumenting.
		// The example that we observed is inlining of LockOSThread,
		// which lead to false race reports on m contents.
		return n
	}

	if Debug_typecheckinl == 0 {
		typecheckinl(fn)
	}

	// We have a function node, and it has an inlineable body.
	if Debug['m'] > 1 {
		fmt.Printf("%v: inlining call to %v %#v { %#v }\n", n.Line(), fn.Sym, fn.Type, asNodes(fn.Func.Inl.Body))
	} else if Debug['m'] != 0 {
		fmt.Printf("%v: inlining call to %v\n", n.Line(), fn)
	}
	if Debug['m'] > 2 {
		fmt.Printf("%v: Before inlining: %+v\n", n.Line(), n)
	}
	if logopt.Enabled() {
		logopt.LogOpt(n.Pos, "inlineCall", "inline", Curfn.funcname(), fn.pkgFuncName())
	}

	if ssaDump != "" && ssaDump == Curfn.funcname() {
		ssaDumpInlined = append(ssaDumpInlined, fn)
	}

	ninit := n.Ninit

	// Make temp names to use instead of the originals.
	inlvars := make(map[*Node]*Node)

	// record formals/locals for later post-processing
	var inlfvars []*Node

	// Handle captured variables when inlining closures.
	if fn.Name.Defn != nil {
		if c := fn.Name.Defn.Func.Closure; c != nil {
			for _, v := range c.Func.Closure.Func.Cvars.Slice() {
				if v.Op == OXXX {
					continue
				}

				o := v.Name.Param.Outer
				// make sure the outer param matches the inlining location
				// NB: if we enabled inlining of functions containing OCLOSURE or refined
				// the reassigned check via some sort of copy propagation this would most
				// likely need to be changed to a loop to walk up to the correct Param
				if o == nil || (o.Name.Curfn != Curfn && o.Name.Curfn.Func.Closure != Curfn) {
					Fatalf("%v: unresolvable capture %v %v\n", n.Line(), fn, v)
				}

				if v.Name.Byval() {
					iv := typecheck(inlvar(v), ctxExpr)
					ninit.Append(nod(ODCL, iv, nil))
					ninit.Append(typecheck(nod(OAS, iv, o), ctxStmt))
					inlvars[v] = iv
				} else {
					addr := newname(lookup("&" + v.Sym.Name))
					addr.Type = types.NewPtr(v.Type)
					ia := typecheck(inlvar(addr), ctxExpr)
					ninit.Append(nod(ODCL, ia, nil))
					ninit.Append(typecheck(nod(OAS, ia, nod(OADDR, o, nil)), ctxStmt))
					inlvars[addr] = ia

					// When capturing by reference, all occurrence of the captured var
					// must be substituted with dereference of the temporary address
					inlvars[v] = typecheck(nod(ODEREF, ia, nil), ctxExpr)
				}
			}
		}
	}

	for _, ln := range fn.Func.Inl.Dcl {
		if ln.Op != ONAME {
			continue
		}
		if ln.Class() == PPARAMOUT { // return values handled below.
			continue
		}
		if ln.isParamStackCopy() { // ignore the on-stack copy of a parameter that moved to the heap
			continue
		}
		inlvars[ln] = typecheck(inlvar(ln), ctxExpr)
		if ln.Class() == PPARAM || ln.Name.Param.Stackcopy != nil && ln.Name.Param.Stackcopy.Class() == PPARAM {
			ninit.Append(nod(ODCL, inlvars[ln], nil))
		}
		if genDwarfInline > 0 {
			inlf := inlvars[ln]
			if ln.Class() == PPARAM {
				inlf.Name.SetInlFormal(true)
			} else {
				inlf.Name.SetInlLocal(true)
			}
			inlf.Pos = ln.Pos
			inlfvars = append(inlfvars, inlf)
		}
	}

	// temporaries for return values.
	var retvars []*Node
	for i, t := range fn.Type.Results().Fields().Slice() {
		var m *Node
		mpos := t.Pos
		if n := asNode(t.Nname); n != nil && !n.isBlank() {
			m = inlvar(n)
			m = typecheck(m, ctxExpr)
			inlvars[n] = m
		} else {
			// anonymous return values, synthesize names for use in assignment that replaces return
			m = retvar(t, i)
		}

		if genDwarfInline > 0 {
			// Don't update the src.Pos on a return variable if it
			// was manufactured by the inliner (e.g. "~R2"); such vars
			// were not part of the original callee.
			if !strings.HasPrefix(m.Sym.Name, "~R") {
				m.Name.SetInlFormal(true)
				m.Pos = mpos
				inlfvars = append(inlfvars, m)
			}
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
			ras = typecheck(ras, ctxStmt)
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
		if !param.IsDDD() || n.IsDDD() {
			as.List.Append(tinlvar(param, inlvars))
			continue
		}

		// Otherwise, we need to collect the remaining values
		// to pass as a slice.

		numvals := n.List.Len()

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
		as = typecheck(as, ctxStmt)
		ninit.Append(as)
	}

	if vas != nil {
		vas = typecheck(vas, ctxStmt)
		ninit.Append(vas)
	}

	// Zero the return parameters.
	for _, n := range retvars {
		ras := nod(OAS, n, nil)
		ras = typecheck(ras, ctxStmt)
		ninit.Append(ras)
	}

	retlabel := autolabel(".i")

	inlgen++

	parent := -1
	if b := Ctxt.PosTable.Pos(n.Pos).Base(); b != nil {
		parent = b.InliningIndex()
	}
	newIndex := Ctxt.InlTree.Add(parent, n.Pos, fn.Sym.Linksym())

	// Add an inline mark just before the inlined body.
	// This mark is inline in the code so that it's a reasonable spot
	// to put a breakpoint. Not sure if that's really necessary or not
	// (in which case it could go at the end of the function instead).
	// Note issue 28603.
	inlMark := nod(OINLMARK, nil, nil)
	inlMark.Pos = n.Pos.WithIsStmt()
	inlMark.Xoffset = int64(newIndex)
	ninit.Append(inlMark)

	if genDwarfInline > 0 {
		if !fn.Sym.Linksym().WasInlined() {
			Ctxt.DwFixups.SetPrecursorFunc(fn.Sym.Linksym(), fn)
			fn.Sym.Linksym().Set(obj.AttrWasInlined, true)
		}
	}

	subst := inlsubst{
		retlabel:    retlabel,
		retvars:     retvars,
		inlvars:     inlvars,
		bases:       make(map[*src.PosBase]*src.PosBase),
		newInlIndex: newIndex,
	}

	body := subst.list(asNodes(fn.Func.Inl.Body))

	lab := nodSym(OLABEL, nil, retlabel)
	body = append(body, lab)

	typecheckslice(body, ctxStmt)

	if genDwarfInline > 0 {
		for _, v := range inlfvars {
			v.Pos = subst.updatedPos(v.Pos)
		}
	}

	//dumplist("ninit post", ninit);

	call := nod(OINLCALL, nil, nil)
	call.Ninit.Set(ninit.Slice())
	call.Nbody.Set(body)
	call.Rlist.Set(retvars)
	call.Type = n.Type
	call.SetTypecheck(1)

	// transitive inlining
	// might be nice to do this before exporting the body,
	// but can't emit the body with inlining expanded.
	// instead we emit the things that the body needs
	// and each use must redo the inlining.
	// luckily these are small.
	inlnodelist(call.Nbody, maxCost)
	for _, n := range call.Nbody.Slice() {
		if n.Op == OINLCALL {
			inlconv2stmt(n)
		}
	}

	if Debug['m'] > 2 {
		fmt.Printf("%v: After inlining %+v\n\n", call.Line(), call)
	}

	return call
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
	n.Name.SetAddrtaken(var_.Name.Addrtaken())

	Curfn.Func.Dcl = append(Curfn.Func.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *types.Field, i int) *Node {
	n := newname(lookupN("~R", i))
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
	retlabel *types.Sym

	// Temporary result variables.
	retvars []*Node

	inlvars map[*Node]*Node

	// bases maps from original PosBase to PosBase with an extra
	// inlined call frame.
	bases map[*src.PosBase]*src.PosBase

	// newInlIndex is the index of the inlined call frame to
	// insert for inlined nodes.
	newInlIndex int
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
		m := nodSym(OGOTO, nil, subst.retlabel)
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
			as = typecheck(as, ctxStmt)
			m.Ninit.Append(as)
		}

		typecheckslice(m.Ninit.Slice(), ctxStmt)
		m = typecheck(m, ctxStmt)

		//		dump("Return after substitution", m);
		return m

	case OGOTO, OLABEL:
		m := n.copy()
		m.Pos = subst.updatedPos(m.Pos)
		m.Ninit.Set(nil)
		p := fmt.Sprintf("%s·%d", n.Sym.Name, inlgen)
		m.Sym = lookup(p)

		return m
	}

	m := n.copy()
	m.Pos = subst.updatedPos(m.Pos)
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

func (subst *inlsubst) updatedPos(xpos src.XPos) src.XPos {
	pos := Ctxt.PosTable.Pos(xpos)
	oldbase := pos.Base() // can be nil
	newbase := subst.bases[oldbase]
	if newbase == nil {
		newbase = src.NewInliningBase(oldbase, subst.newInlIndex)
		subst.bases[oldbase] = newbase
	}
	pos.SetBase(newbase)
	return Ctxt.PosTable.XPos(pos)
}

func pruneUnusedAutos(ll []*Node, vis *hairyVisitor) []*Node {
	s := make([]*Node, 0, len(ll))
	for _, n := range ll {
		if n.Class() == PAUTO {
			if _, found := vis.usedLocals[n]; !found {
				continue
			}
		}
		s = append(s, n)
	}
	return s
}
