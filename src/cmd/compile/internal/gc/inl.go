// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The inlining facility makes 2 passes: first caninl determines which
// functions are suitable for inlining, and for those that are it
// saves a copy of the body. Then inlcalls walks each function body to
// expand calls to inlinable functions.
//
// The Debug.l flag controls the aggressiveness. Note that main() swaps level 0 and 1,
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
// The Debug.m flag enables diagnostic output.  a single -m is useful for verifying
// which calls get inlined or not, more is for debugging, and may go away at any point.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"go/constant"
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
func fnpkg(fn *ir.Name) *types.Pkg {
	if ir.IsMethod(fn) {
		// method
		rcvr := fn.Type().Recv().Type

		if rcvr.IsPtr() {
			rcvr = rcvr.Elem()
		}
		if rcvr.Sym == nil {
			base.Fatalf("receiver with no sym: [%v] %L  (%v)", fn.Sym(), fn, rcvr)
		}
		return rcvr.Sym.Pkg
	}

	// non-method
	return fn.Sym().Pkg
}

// Lazy typechecking of imported bodies. For local functions, caninl will set ->typecheck
// because they're a copy of an already checked body.
func typecheckinl(fn *ir.Func) {
	lno := setlineno(fn.Nname)

	expandInline(fn)

	// typecheckinl is only for imported functions;
	// their bodies may refer to unsafe as long as the package
	// was marked safe during import (which was checked then).
	// the ->inl of a local function has been typechecked before caninl copied it.
	pkg := fnpkg(fn.Nname)

	if pkg == ir.LocalPkg || pkg == nil {
		return // typecheckinl on local function
	}

	if base.Flag.LowerM > 2 || base.Debug.Export != 0 {
		fmt.Printf("typecheck import [%v] %L { %#v }\n", fn.Sym(), fn, ir.AsNodes(fn.Inl.Body))
	}

	savefn := Curfn
	Curfn = fn
	typecheckslice(fn.Inl.Body, ctxStmt)
	Curfn = savefn

	// During expandInline (which imports fn.Func.Inl.Body),
	// declarations are added to fn.Func.Dcl by funcHdr(). Move them
	// to fn.Func.Inl.Dcl for consistency with how local functions
	// behave. (Append because typecheckinl may be called multiple
	// times.)
	fn.Inl.Dcl = append(fn.Inl.Dcl, fn.Dcl...)
	fn.Dcl = nil

	base.Pos = lno
}

// Caninl determines whether fn is inlineable.
// If so, caninl saves fn->nbody in fn->inl and substitutes it with a copy.
// fn and ->nbody will already have been typechecked.
func caninl(fn *ir.Func) {
	if fn.Nname == nil {
		base.Fatalf("caninl no nname %+v", fn)
	}

	var reason string // reason, if any, that the function was not inlined
	if base.Flag.LowerM > 1 || logopt.Enabled() {
		defer func() {
			if reason != "" {
				if base.Flag.LowerM > 1 {
					fmt.Printf("%v: cannot inline %v: %s\n", ir.Line(fn), fn.Nname, reason)
				}
				if logopt.Enabled() {
					logopt.LogOpt(fn.Pos(), "cannotInlineFunction", "inline", ir.FuncName(fn), reason)
				}
			}
		}()
	}

	// If marked "go:noinline", don't inline
	if fn.Pragma&ir.Noinline != 0 {
		reason = "marked go:noinline"
		return
	}

	// If marked "go:norace" and -race compilation, don't inline.
	if base.Flag.Race && fn.Pragma&ir.Norace != 0 {
		reason = "marked go:norace with -race compilation"
		return
	}

	// If marked "go:nocheckptr" and -d checkptr compilation, don't inline.
	if base.Debug.Checkptr != 0 && fn.Pragma&ir.NoCheckPtr != 0 {
		reason = "marked go:nocheckptr"
		return
	}

	// If marked "go:cgo_unsafe_args", don't inline, since the
	// function makes assumptions about its argument frame layout.
	if fn.Pragma&ir.CgoUnsafeArgs != 0 {
		reason = "marked go:cgo_unsafe_args"
		return
	}

	// If marked as "go:uintptrescapes", don't inline, since the
	// escape information is lost during inlining.
	if fn.Pragma&ir.UintptrEscapes != 0 {
		reason = "marked as having an escaping uintptr argument"
		return
	}

	// The nowritebarrierrec checker currently works at function
	// granularity, so inlining yeswritebarrierrec functions can
	// confuse it (#22342). As a workaround, disallow inlining
	// them for now.
	if fn.Pragma&ir.Yeswritebarrierrec != 0 {
		reason = "marked go:yeswritebarrierrec"
		return
	}

	// If fn has no body (is defined outside of Go), cannot inline it.
	if fn.Body().Len() == 0 {
		reason = "no function body"
		return
	}

	if fn.Typecheck() == 0 {
		base.Fatalf("caninl on non-typechecked function %v", fn)
	}

	n := fn.Nname
	if n.Func().InlinabilityChecked() {
		return
	}
	defer n.Func().SetInlinabilityChecked(true)

	cc := int32(inlineExtraCallCost)
	if base.Flag.LowerL == 4 {
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
		usedLocals:    make(map[ir.Node]bool),
	}
	if visitor.visitList(fn.Body()) {
		reason = visitor.reason
		return
	}
	if visitor.budget < 0 {
		reason = fmt.Sprintf("function too complex: cost %d exceeds budget %d", inlineMaxBudget-visitor.budget, inlineMaxBudget)
		return
	}

	n.Func().Inl = &ir.Inline{
		Cost: inlineMaxBudget - visitor.budget,
		Dcl:  pruneUnusedAutos(n.Defn.Func().Dcl, &visitor),
		Body: inlcopylist(fn.Body().Slice()),
	}

	if base.Flag.LowerM > 1 {
		fmt.Printf("%v: can inline %#v with cost %d as: %#v { %#v }\n", ir.Line(fn), n, inlineMaxBudget-visitor.budget, fn.Type(), ir.AsNodes(n.Func().Inl.Body))
	} else if base.Flag.LowerM != 0 {
		fmt.Printf("%v: can inline %v\n", ir.Line(fn), n)
	}
	if logopt.Enabled() {
		logopt.LogOpt(fn.Pos(), "canInlineFunction", "inline", ir.FuncName(fn), fmt.Sprintf("cost: %d", inlineMaxBudget-visitor.budget))
	}
}

// inlFlood marks n's inline body for export and recursively ensures
// all called functions are marked too.
func inlFlood(n *ir.Name) {
	if n == nil {
		return
	}
	if n.Op() != ir.ONAME || n.Class() != ir.PFUNC {
		base.Fatalf("inlFlood: unexpected %v, %v, %v", n, n.Op(), n.Class())
	}
	fn := n.Func()
	if fn == nil {
		base.Fatalf("inlFlood: missing Func on %v", n)
	}
	if fn.Inl == nil {
		return
	}

	if fn.ExportInline() {
		return
	}
	fn.SetExportInline(true)

	typecheckinl(fn)

	// Recursively identify all referenced functions for
	// reexport. We want to include even non-called functions,
	// because after inlining they might be callable.
	ir.InspectList(ir.AsNodes(fn.Inl.Body), func(n ir.Node) bool {
		switch n.Op() {
		case ir.OMETHEXPR, ir.ODOTMETH:
			inlFlood(methodExprName(n))

		case ir.ONAME:
			n := n.(*ir.Name)
			switch n.Class() {
			case ir.PFUNC:
				inlFlood(n)
				exportsym(n)
			case ir.PEXTERN:
				exportsym(n)
			}

		case ir.OCALLPART:
			// Okay, because we don't yet inline indirect
			// calls to method values.
		case ir.OCLOSURE:
			// If the closure is inlinable, we'll need to
			// flood it too. But today we don't support
			// inlining functions that contain closures.
			//
			// When we do, we'll probably want:
			//     inlFlood(n.Func.Closure.Func.Nname)
			base.Fatalf("unexpected closure in inlinable function")
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
	usedLocals    map[ir.Node]bool
}

// Look for anything we want to punt on.
func (v *hairyVisitor) visitList(ll ir.Nodes) bool {
	for _, n := range ll.Slice() {
		if v.visit(n) {
			return true
		}
	}
	return false
}

func (v *hairyVisitor) visit(n ir.Node) bool {
	if n == nil {
		return false
	}

	switch n.Op() {
	// Call is okay if inlinable and we have the budget for the body.
	case ir.OCALLFUNC:
		// Functions that call runtime.getcaller{pc,sp} can not be inlined
		// because getcaller{pc,sp} expect a pointer to the caller's first argument.
		//
		// runtime.throw is a "cheap call" like panic in normal code.
		if n.Left().Op() == ir.ONAME && n.Left().Class() == ir.PFUNC && isRuntimePkg(n.Left().Sym().Pkg) {
			fn := n.Left().Sym().Name
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

		if fn := inlCallee(n.Left()); fn != nil && fn.Inl != nil {
			v.budget -= fn.Inl.Cost
			break
		}

		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	// Call is okay if inlinable and we have the budget for the body.
	case ir.OCALLMETH:
		t := n.Left().Type()
		if t == nil {
			base.Fatalf("no function type for [%p] %+v\n", n.Left(), n.Left())
		}
		if isRuntimePkg(n.Left().Sym().Pkg) {
			fn := n.Left().Sym().Name
			if fn == "heapBits.nextArena" {
				// Special case: explicitly allow
				// mid-stack inlining of
				// runtime.heapBits.next even though
				// it calls slow-path
				// runtime.heapBits.nextArena.
				break
			}
		}
		if inlfn := methodExprName(n.Left()).Func(); inlfn.Inl != nil {
			v.budget -= inlfn.Inl.Cost
			break
		}
		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	// Things that are too hairy, irrespective of the budget
	case ir.OCALL, ir.OCALLINTER:
		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	case ir.OPANIC:
		v.budget -= inlineExtraPanicCost

	case ir.ORECOVER:
		// recover matches the argument frame pointer to find
		// the right panic value, so it needs an argument frame.
		v.reason = "call to recover"
		return true

	case ir.OCLOSURE,
		ir.ORANGE,
		ir.OSELECT,
		ir.OGO,
		ir.ODEFER,
		ir.ODCLTYPE, // can't print yet
		ir.ORETJMP:
		v.reason = "unhandled op " + n.Op().String()
		return true

	case ir.OAPPEND:
		v.budget -= inlineExtraAppendCost

	case ir.ODCLCONST, ir.OEMPTY, ir.OFALL:
		// These nodes don't produce code; omit from inlining budget.
		return false

	case ir.OFOR, ir.OFORUNTIL, ir.OSWITCH:
		// ORANGE, OSELECT in "unhandled" above
		if n.Sym() != nil {
			v.reason = "labeled control"
			return true
		}

	case ir.OBREAK, ir.OCONTINUE:
		if n.Sym() != nil {
			// Should have short-circuited due to labeled control error above.
			base.Fatalf("unexpected labeled break/continue: %v", n)
		}

	case ir.OIF:
		if ir.IsConst(n.Left(), constant.Bool) {
			// This if and the condition cost nothing.
			return v.visitList(n.Init()) || v.visitList(n.Body()) ||
				v.visitList(n.Rlist())
		}

	case ir.ONAME:
		if n.Class() == ir.PAUTO {
			v.usedLocals[n] = true
		}

	}

	v.budget--

	// When debugging, don't stop early, to get full cost of inlining this function
	if v.budget < 0 && base.Flag.LowerM < 2 && !logopt.Enabled() {
		return true
	}

	return v.visit(n.Left()) || v.visit(n.Right()) ||
		v.visitList(n.List()) || v.visitList(n.Rlist()) ||
		v.visitList(n.Init()) || v.visitList(n.Body())
}

// inlcopylist (together with inlcopy) recursively copies a list of nodes, except
// that it keeps the same ONAME, OTYPE, and OLITERAL nodes. It is used for copying
// the body and dcls of an inlineable function.
func inlcopylist(ll []ir.Node) []ir.Node {
	s := make([]ir.Node, 0, len(ll))
	for _, n := range ll {
		s = append(s, inlcopy(n))
	}
	return s
}

func inlcopy(n ir.Node) ir.Node {
	if n == nil {
		return nil
	}

	switch n.Op() {
	case ir.ONAME, ir.OTYPE, ir.OLITERAL, ir.ONIL:
		return n
	}

	m := ir.Copy(n)
	if n.Op() != ir.OCALLPART && m.Func() != nil {
		base.Fatalf("unexpected Func: %v", m)
	}
	m.SetLeft(inlcopy(n.Left()))
	m.SetRight(inlcopy(n.Right()))
	m.PtrList().Set(inlcopylist(n.List().Slice()))
	m.PtrRlist().Set(inlcopylist(n.Rlist().Slice()))
	m.PtrInit().Set(inlcopylist(n.Init().Slice()))
	m.PtrBody().Set(inlcopylist(n.Body().Slice()))

	return m
}

func countNodes(n ir.Node) int {
	if n == nil {
		return 0
	}
	cnt := 1
	cnt += countNodes(n.Left())
	cnt += countNodes(n.Right())
	for _, n1 := range n.Init().Slice() {
		cnt += countNodes(n1)
	}
	for _, n1 := range n.Body().Slice() {
		cnt += countNodes(n1)
	}
	for _, n1 := range n.List().Slice() {
		cnt += countNodes(n1)
	}
	for _, n1 := range n.Rlist().Slice() {
		cnt += countNodes(n1)
	}
	return cnt
}

// Inlcalls/nodelist/node walks fn's statements and expressions and substitutes any
// calls made to inlineable functions. This is the external entry point.
func inlcalls(fn *ir.Func) {
	savefn := Curfn
	Curfn = fn
	maxCost := int32(inlineMaxBudget)
	if countNodes(fn) >= inlineBigFunctionNodes {
		maxCost = inlineBigFunctionMaxCost
	}
	// Map to keep track of functions that have been inlined at a particular
	// call site, in order to stop inlining when we reach the beginning of a
	// recursion cycle again. We don't inline immediately recursive functions,
	// but allow inlining if there is a recursion cycle of many functions.
	// Most likely, the inlining will stop before we even hit the beginning of
	// the cycle again, but the map catches the unusual case.
	inlMap := make(map[*ir.Func]bool)
	fn = inlnode(fn, maxCost, inlMap).(*ir.Func)
	if fn != Curfn {
		base.Fatalf("inlnode replaced curfn")
	}
	Curfn = savefn
}

// Turn an OINLCALL into a statement.
func inlconv2stmt(inlcall ir.Node) ir.Node {
	n := ir.NodAt(inlcall.Pos(), ir.OBLOCK, nil, nil)
	n.SetList(inlcall.Body())
	n.SetInit(inlcall.Init())
	return n
}

// Turn an OINLCALL into a single valued expression.
// The result of inlconv2expr MUST be assigned back to n, e.g.
// 	n.Left = inlconv2expr(n.Left)
func inlconv2expr(n ir.Node) ir.Node {
	r := n.Rlist().First()
	return addinit(r, append(n.Init().Slice(), n.Body().Slice()...))
}

// Turn the rlist (with the return values) of the OINLCALL in
// n into an expression list lumping the ninit and body
// containing the inlined statements on the first list element so
// order will be preserved Used in return, oas2func and call
// statements.
func inlconv2list(n ir.Node) []ir.Node {
	if n.Op() != ir.OINLCALL || n.Rlist().Len() == 0 {
		base.Fatalf("inlconv2list %+v\n", n)
	}

	s := n.Rlist().Slice()
	s[0] = addinit(s[0], append(n.Init().Slice(), n.Body().Slice()...))
	return s
}

func inlnodelist(l ir.Nodes, maxCost int32, inlMap map[*ir.Func]bool) {
	s := l.Slice()
	for i := range s {
		s[i] = inlnode(s[i], maxCost, inlMap)
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
func inlnode(n ir.Node, maxCost int32, inlMap map[*ir.Func]bool) ir.Node {
	if n == nil {
		return n
	}

	switch n.Op() {
	case ir.ODEFER, ir.OGO:
		switch n.Left().Op() {
		case ir.OCALLFUNC, ir.OCALLMETH:
			n.Left().SetNoInline(true)
		}

	// TODO do them here (or earlier),
	// so escape analysis can avoid more heapmoves.
	case ir.OCLOSURE:
		return n
	case ir.OCALLMETH:
		// Prevent inlining some reflect.Value methods when using checkptr,
		// even when package reflect was compiled without it (#35073).
		if s := n.Left().Sym(); base.Debug.Checkptr != 0 && isReflectPkg(s.Pkg) && (s.Name == "Value.UnsafeAddr" || s.Name == "Value.Pointer") {
			return n
		}
	}

	lno := setlineno(n)

	inlnodelist(n.Init(), maxCost, inlMap)
	init := n.Init().Slice()
	for i, n1 := range init {
		if n1.Op() == ir.OINLCALL {
			init[i] = inlconv2stmt(n1)
		}
	}

	n.SetLeft(inlnode(n.Left(), maxCost, inlMap))
	if n.Left() != nil && n.Left().Op() == ir.OINLCALL {
		n.SetLeft(inlconv2expr(n.Left()))
	}

	n.SetRight(inlnode(n.Right(), maxCost, inlMap))
	if n.Right() != nil && n.Right().Op() == ir.OINLCALL {
		if n.Op() == ir.OFOR || n.Op() == ir.OFORUNTIL {
			n.SetRight(inlconv2stmt(n.Right()))
		} else {
			n.SetRight(inlconv2expr(n.Right()))
		}
	}

	inlnodelist(n.List(), maxCost, inlMap)
	s := n.List().Slice()
	convert := inlconv2expr
	if n.Op() == ir.OBLOCK {
		convert = inlconv2stmt
	}
	for i, n1 := range s {
		if n1 != nil && n1.Op() == ir.OINLCALL {
			s[i] = convert(n1)
		}
	}

	inlnodelist(n.Rlist(), maxCost, inlMap)

	if n.Op() == ir.OAS2FUNC && n.Rlist().First().Op() == ir.OINLCALL {
		n.PtrRlist().Set(inlconv2list(n.Rlist().First()))
		n.SetOp(ir.OAS2)
		n.SetTypecheck(0)
		n = typecheck(n, ctxStmt)
	}

	s = n.Rlist().Slice()
	for i, n1 := range s {
		if n1.Op() == ir.OINLCALL {
			if n.Op() == ir.OIF {
				s[i] = inlconv2stmt(n1)
			} else {
				s[i] = inlconv2expr(n1)
			}
		}
	}

	inlnodelist(n.Body(), maxCost, inlMap)
	s = n.Body().Slice()
	for i, n1 := range s {
		if n1.Op() == ir.OINLCALL {
			s[i] = inlconv2stmt(n1)
		}
	}

	// with all the branches out of the way, it is now time to
	// transmogrify this node itself unless inhibited by the
	// switch at the top of this function.
	switch n.Op() {
	case ir.OCALLFUNC, ir.OCALLMETH:
		if n.NoInline() {
			return n
		}
	}

	switch n.Op() {
	case ir.OCALLFUNC:
		if base.Flag.LowerM > 3 {
			fmt.Printf("%v:call to func %+v\n", ir.Line(n), n.Left())
		}
		if isIntrinsicCall(n) {
			break
		}
		if fn := inlCallee(n.Left()); fn != nil && fn.Inl != nil {
			n = mkinlcall(n, fn, maxCost, inlMap)
		}

	case ir.OCALLMETH:
		if base.Flag.LowerM > 3 {
			fmt.Printf("%v:call to meth %L\n", ir.Line(n), n.Left().Right())
		}

		// typecheck should have resolved ODOTMETH->type, whose nname points to the actual function.
		if n.Left().Type() == nil {
			base.Fatalf("no function type for [%p] %+v\n", n.Left(), n.Left())
		}

		n = mkinlcall(n, methodExprName(n.Left()).Func(), maxCost, inlMap)
	}

	base.Pos = lno
	return n
}

// inlCallee takes a function-typed expression and returns the underlying function ONAME
// that it refers to if statically known. Otherwise, it returns nil.
func inlCallee(fn ir.Node) *ir.Func {
	fn = staticValue(fn)
	switch {
	case fn.Op() == ir.OMETHEXPR:
		n := methodExprName(fn)
		// Check that receiver type matches fn.Left.
		// TODO(mdempsky): Handle implicit dereference
		// of pointer receiver argument?
		if n == nil || !types.Identical(n.Type().Recv().Type, fn.Left().Type()) {
			return nil
		}
		return n.Func()
	case fn.Op() == ir.ONAME && fn.Class() == ir.PFUNC:
		return fn.Func()
	case fn.Op() == ir.OCLOSURE:
		c := fn.Func()
		caninl(c)
		return c
	}
	return nil
}

func staticValue(n ir.Node) ir.Node {
	for {
		if n.Op() == ir.OCONVNOP {
			n = n.Left()
			continue
		}

		n1 := staticValue1(n)
		if n1 == nil {
			return n
		}
		n = n1
	}
}

// staticValue1 implements a simple SSA-like optimization. If n is a local variable
// that is initialized and never reassigned, staticValue1 returns the initializer
// expression. Otherwise, it returns nil.
func staticValue1(n ir.Node) ir.Node {
	if n.Op() != ir.ONAME || n.Class() != ir.PAUTO || n.Name().Addrtaken() {
		return nil
	}

	defn := n.Name().Defn
	if defn == nil {
		return nil
	}

	var rhs ir.Node
FindRHS:
	switch defn.Op() {
	case ir.OAS:
		rhs = defn.Right()
	case ir.OAS2:
		for i, lhs := range defn.List().Slice() {
			if lhs == n {
				rhs = defn.Rlist().Index(i)
				break FindRHS
			}
		}
		base.Fatalf("%v missing from LHS of %v", n, defn)
	default:
		return nil
	}
	if rhs == nil {
		base.Fatalf("RHS is nil: %v", defn)
	}

	unsafe, _ := reassigned(n.(*ir.Name))
	if unsafe {
		return nil
	}

	return rhs
}

// reassigned takes an ONAME node, walks the function in which it is defined, and returns a boolean
// indicating whether the name has any assignments other than its declaration.
// The second return value is the first such assignment encountered in the walk, if any. It is mostly
// useful for -m output documenting the reason for inhibited optimizations.
// NB: global variables are always considered to be re-assigned.
// TODO: handle initial declaration not including an assignment and followed by a single assignment?
func reassigned(n *ir.Name) (bool, ir.Node) {
	if n.Op() != ir.ONAME {
		base.Fatalf("reassigned %v", n)
	}
	// no way to reliably check for no-reassignment of globals, assume it can be
	if n.Curfn == nil {
		return true, nil
	}
	f := n.Curfn
	v := reassignVisitor{name: n}
	a := v.visitList(f.Body())
	return a != nil, a
}

type reassignVisitor struct {
	name ir.Node
}

func (v *reassignVisitor) visit(n ir.Node) ir.Node {
	if n == nil {
		return nil
	}
	switch n.Op() {
	case ir.OAS:
		if n.Left() == v.name && n != v.name.Name().Defn {
			return n
		}
	case ir.OAS2, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2DOTTYPE:
		for _, p := range n.List().Slice() {
			if p == v.name && n != v.name.Name().Defn {
				return n
			}
		}
	}
	if a := v.visit(n.Left()); a != nil {
		return a
	}
	if a := v.visit(n.Right()); a != nil {
		return a
	}
	if a := v.visitList(n.List()); a != nil {
		return a
	}
	if a := v.visitList(n.Rlist()); a != nil {
		return a
	}
	if a := v.visitList(n.Init()); a != nil {
		return a
	}
	if a := v.visitList(n.Body()); a != nil {
		return a
	}
	return nil
}

func (v *reassignVisitor) visitList(l ir.Nodes) ir.Node {
	for _, n := range l.Slice() {
		if a := v.visit(n); a != nil {
			return a
		}
	}
	return nil
}

func inlParam(t *types.Field, as ir.Node, inlvars map[*ir.Name]ir.Node) ir.Node {
	n := ir.AsNode(t.Nname)
	if n == nil || ir.IsBlank(n) {
		return ir.BlankNode
	}

	inlvar := inlvars[n.(*ir.Name)]
	if inlvar == nil {
		base.Fatalf("missing inlvar for %v", n)
	}
	as.PtrInit().Append(ir.Nod(ir.ODCL, inlvar, nil))
	inlvar.Name().Defn = as
	return inlvar
}

var inlgen int

// If n is a call node (OCALLFUNC or OCALLMETH), and fn is an ONAME node for a
// function with an inlinable body, return an OINLCALL node that can replace n.
// The returned node's Ninit has the parameter assignments, the Nbody is the
// inlined function body, and (List, Rlist) contain the (input, output)
// parameters.
// The result of mkinlcall MUST be assigned back to n, e.g.
// 	n.Left = mkinlcall(n.Left, fn, isddd)
func mkinlcall(n ir.Node, fn *ir.Func, maxCost int32, inlMap map[*ir.Func]bool) ir.Node {
	if fn.Inl == nil {
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(Curfn),
				fmt.Sprintf("%s cannot be inlined", ir.PkgFuncName(fn)))
		}
		return n
	}
	if fn.Inl.Cost > maxCost {
		// The inlined function body is too big. Typically we use this check to restrict
		// inlining into very big functions.  See issue 26546 and 17566.
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(Curfn),
				fmt.Sprintf("cost %d of %s exceeds max large caller cost %d", fn.Inl.Cost, ir.PkgFuncName(fn), maxCost))
		}
		return n
	}

	if fn == Curfn {
		// Can't recursively inline a function into itself.
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", fmt.Sprintf("recursive call to %s", ir.FuncName(Curfn)))
		}
		return n
	}

	if instrumenting && isRuntimePkg(fn.Sym().Pkg) {
		// Runtime package must not be instrumented.
		// Instrument skips runtime package. However, some runtime code can be
		// inlined into other packages and instrumented there. To avoid this,
		// we disable inlining of runtime functions when instrumenting.
		// The example that we observed is inlining of LockOSThread,
		// which lead to false race reports on m contents.
		return n
	}

	if inlMap[fn] {
		if base.Flag.LowerM > 1 {
			fmt.Printf("%v: cannot inline %v into %v: repeated recursive cycle\n", ir.Line(n), fn, ir.FuncName(Curfn))
		}
		return n
	}
	inlMap[fn] = true
	defer func() {
		inlMap[fn] = false
	}()
	if base.Debug.TypecheckInl == 0 {
		typecheckinl(fn)
	}

	// We have a function node, and it has an inlineable body.
	if base.Flag.LowerM > 1 {
		fmt.Printf("%v: inlining call to %v %#v { %#v }\n", ir.Line(n), fn.Sym(), fn.Type(), ir.AsNodes(fn.Inl.Body))
	} else if base.Flag.LowerM != 0 {
		fmt.Printf("%v: inlining call to %v\n", ir.Line(n), fn)
	}
	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: Before inlining: %+v\n", ir.Line(n), n)
	}

	if ssaDump != "" && ssaDump == ir.FuncName(Curfn) {
		ssaDumpInlined = append(ssaDumpInlined, fn)
	}

	ninit := n.Init()

	// For normal function calls, the function callee expression
	// may contain side effects (e.g., added by addinit during
	// inlconv2expr or inlconv2list). Make sure to preserve these,
	// if necessary (#42703).
	if n.Op() == ir.OCALLFUNC {
		callee := n.Left()
		for callee.Op() == ir.OCONVNOP {
			ninit.AppendNodes(callee.PtrInit())
			callee = callee.Left()
		}
		if callee.Op() != ir.ONAME && callee.Op() != ir.OCLOSURE && callee.Op() != ir.OMETHEXPR {
			base.Fatalf("unexpected callee expression: %v", callee)
		}
	}

	// Make temp names to use instead of the originals.
	inlvars := make(map[*ir.Name]ir.Node)

	// record formals/locals for later post-processing
	var inlfvars []ir.Node

	// Handle captured variables when inlining closures.
	if c := fn.OClosure; c != nil {
		for _, v := range fn.ClosureVars {
			if v.Op() == ir.OXXX {
				continue
			}

			o := v.Outer
			// make sure the outer param matches the inlining location
			// NB: if we enabled inlining of functions containing OCLOSURE or refined
			// the reassigned check via some sort of copy propagation this would most
			// likely need to be changed to a loop to walk up to the correct Param
			if o == nil || o.Curfn != Curfn {
				base.Fatalf("%v: unresolvable capture %v %v\n", ir.Line(n), fn, v)
			}

			if v.Byval() {
				iv := typecheck(inlvar(v), ctxExpr)
				ninit.Append(ir.Nod(ir.ODCL, iv, nil))
				ninit.Append(typecheck(ir.Nod(ir.OAS, iv, o), ctxStmt))
				inlvars[v] = iv
			} else {
				addr := NewName(lookup("&" + v.Sym().Name))
				addr.SetType(types.NewPtr(v.Type()))
				ia := typecheck(inlvar(addr), ctxExpr)
				ninit.Append(ir.Nod(ir.ODCL, ia, nil))
				ninit.Append(typecheck(ir.Nod(ir.OAS, ia, ir.Nod(ir.OADDR, o, nil)), ctxStmt))
				inlvars[addr] = ia

				// When capturing by reference, all occurrence of the captured var
				// must be substituted with dereference of the temporary address
				inlvars[v] = typecheck(ir.Nod(ir.ODEREF, ia, nil), ctxExpr)
			}
		}
	}

	for _, ln := range fn.Inl.Dcl {
		if ln.Op() != ir.ONAME {
			continue
		}
		if ln.Class() == ir.PPARAMOUT { // return values handled below.
			continue
		}
		if isParamStackCopy(ln) { // ignore the on-stack copy of a parameter that moved to the heap
			// TODO(mdempsky): Remove once I'm confident
			// this never actually happens. We currently
			// perform inlining before escape analysis, so
			// nothing should have moved to the heap yet.
			base.Fatalf("impossible: %v", ln)
		}
		inlf := typecheck(inlvar(ln), ctxExpr)
		inlvars[ln] = inlf
		if base.Flag.GenDwarfInl > 0 {
			if ln.Class() == ir.PPARAM {
				inlf.Name().SetInlFormal(true)
			} else {
				inlf.Name().SetInlLocal(true)
			}
			inlf.SetPos(ln.Pos())
			inlfvars = append(inlfvars, inlf)
		}
	}

	nreturns := 0
	ir.InspectList(ir.AsNodes(fn.Inl.Body), func(n ir.Node) bool {
		if n != nil && n.Op() == ir.ORETURN {
			nreturns++
		}
		return true
	})

	// We can delay declaring+initializing result parameters if:
	// (1) there's only one "return" statement in the inlined
	// function, and (2) the result parameters aren't named.
	delayretvars := nreturns == 1

	// temporaries for return values.
	var retvars []ir.Node
	for i, t := range fn.Type().Results().Fields().Slice() {
		var m ir.Node
		if n := ir.AsNode(t.Nname); n != nil && !ir.IsBlank(n) && !strings.HasPrefix(n.Sym().Name, "~r") {
			n := n.(*ir.Name)
			m = inlvar(n)
			m = typecheck(m, ctxExpr)
			inlvars[n] = m
			delayretvars = false // found a named result parameter
		} else {
			// anonymous return values, synthesize names for use in assignment that replaces return
			m = retvar(t, i)
		}

		if base.Flag.GenDwarfInl > 0 {
			// Don't update the src.Pos on a return variable if it
			// was manufactured by the inliner (e.g. "~R2"); such vars
			// were not part of the original callee.
			if !strings.HasPrefix(m.Sym().Name, "~R") {
				m.Name().SetInlFormal(true)
				m.SetPos(t.Pos)
				inlfvars = append(inlfvars, m)
			}
		}

		retvars = append(retvars, m)
	}

	// Assign arguments to the parameters' temp names.
	as := ir.Nod(ir.OAS2, nil, nil)
	as.SetColas(true)
	if n.Op() == ir.OCALLMETH {
		if n.Left().Left() == nil {
			base.Fatalf("method call without receiver: %+v", n)
		}
		as.PtrRlist().Append(n.Left().Left())
	}
	as.PtrRlist().Append(n.List().Slice()...)

	// For non-dotted calls to variadic functions, we assign the
	// variadic parameter's temp name separately.
	var vas ir.Node

	if recv := fn.Type().Recv(); recv != nil {
		as.PtrList().Append(inlParam(recv, as, inlvars))
	}
	for _, param := range fn.Type().Params().Fields().Slice() {
		// For ordinary parameters or variadic parameters in
		// dotted calls, just add the variable to the
		// assignment list, and we're done.
		if !param.IsDDD() || n.IsDDD() {
			as.PtrList().Append(inlParam(param, as, inlvars))
			continue
		}

		// Otherwise, we need to collect the remaining values
		// to pass as a slice.

		x := as.List().Len()
		for as.List().Len() < as.Rlist().Len() {
			as.PtrList().Append(argvar(param.Type, as.List().Len()))
		}
		varargs := as.List().Slice()[x:]

		vas = ir.Nod(ir.OAS, nil, nil)
		vas.SetLeft(inlParam(param, vas, inlvars))
		if len(varargs) == 0 {
			vas.SetRight(nodnil())
			vas.Right().SetType(param.Type)
		} else {
			vas.SetRight(ir.Nod(ir.OCOMPLIT, nil, ir.TypeNode(param.Type)))
			vas.Right().PtrList().Set(varargs)
		}
	}

	if as.Rlist().Len() != 0 {
		as = typecheck(as, ctxStmt)
		ninit.Append(as)
	}

	if vas != nil {
		vas = typecheck(vas, ctxStmt)
		ninit.Append(vas)
	}

	if !delayretvars {
		// Zero the return parameters.
		for _, n := range retvars {
			ninit.Append(ir.Nod(ir.ODCL, n, nil))
			ras := ir.Nod(ir.OAS, n, nil)
			ras = typecheck(ras, ctxStmt)
			ninit.Append(ras)
		}
	}

	retlabel := autolabel(".i")

	inlgen++

	parent := -1
	if b := base.Ctxt.PosTable.Pos(n.Pos()).Base(); b != nil {
		parent = b.InliningIndex()
	}

	sym := fn.Sym().Linksym()
	newIndex := base.Ctxt.InlTree.Add(parent, n.Pos(), sym)

	// Add an inline mark just before the inlined body.
	// This mark is inline in the code so that it's a reasonable spot
	// to put a breakpoint. Not sure if that's really necessary or not
	// (in which case it could go at the end of the function instead).
	// Note issue 28603.
	inlMark := ir.Nod(ir.OINLMARK, nil, nil)
	inlMark.SetPos(n.Pos().WithIsStmt())
	inlMark.SetOffset(int64(newIndex))
	ninit.Append(inlMark)

	if base.Flag.GenDwarfInl > 0 {
		if !sym.WasInlined() {
			base.Ctxt.DwFixups.SetPrecursorFunc(sym, fn)
			sym.Set(obj.AttrWasInlined, true)
		}
	}

	subst := inlsubst{
		retlabel:     retlabel,
		retvars:      retvars,
		delayretvars: delayretvars,
		inlvars:      inlvars,
		bases:        make(map[*src.PosBase]*src.PosBase),
		newInlIndex:  newIndex,
	}

	body := subst.list(ir.AsNodes(fn.Inl.Body))

	lab := nodSym(ir.OLABEL, nil, retlabel)
	body = append(body, lab)

	typecheckslice(body, ctxStmt)

	if base.Flag.GenDwarfInl > 0 {
		for _, v := range inlfvars {
			v.SetPos(subst.updatedPos(v.Pos()))
		}
	}

	//dumplist("ninit post", ninit);

	call := ir.Nod(ir.OINLCALL, nil, nil)
	call.PtrInit().Set(ninit.Slice())
	call.PtrBody().Set(body)
	call.PtrRlist().Set(retvars)
	call.SetType(n.Type())
	call.SetTypecheck(1)

	// transitive inlining
	// might be nice to do this before exporting the body,
	// but can't emit the body with inlining expanded.
	// instead we emit the things that the body needs
	// and each use must redo the inlining.
	// luckily these are small.
	inlnodelist(call.Body(), maxCost, inlMap)
	s := call.Body().Slice()
	for i, n1 := range s {
		if n1.Op() == ir.OINLCALL {
			s[i] = inlconv2stmt(n1)
		}
	}

	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: After inlining %+v\n\n", ir.Line(call), call)
	}

	return call
}

// Every time we expand a function we generate a new set of tmpnames,
// PAUTO's in the calling functions, and link them off of the
// PPARAM's, PAUTOS and PPARAMOUTs of the called function.
func inlvar(var_ ir.Node) ir.Node {
	if base.Flag.LowerM > 3 {
		fmt.Printf("inlvar %+v\n", var_)
	}

	n := NewName(var_.Sym())
	n.SetType(var_.Type())
	n.SetClass(ir.PAUTO)
	n.SetUsed(true)
	n.Curfn = Curfn // the calling function, not the called one
	n.SetAddrtaken(var_.Name().Addrtaken())

	Curfn.Dcl = append(Curfn.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *types.Field, i int) ir.Node {
	n := NewName(lookupN("~R", i))
	n.SetType(t.Type)
	n.SetClass(ir.PAUTO)
	n.SetUsed(true)
	n.Curfn = Curfn // the calling function, not the called one
	Curfn.Dcl = append(Curfn.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's arguments
// when they come from a multiple return call.
func argvar(t *types.Type, i int) ir.Node {
	n := NewName(lookupN("~arg", i))
	n.SetType(t.Elem())
	n.SetClass(ir.PAUTO)
	n.SetUsed(true)
	n.Curfn = Curfn // the calling function, not the called one
	Curfn.Dcl = append(Curfn.Dcl, n)
	return n
}

// The inlsubst type implements the actual inlining of a single
// function call.
type inlsubst struct {
	// Target of the goto substituted in place of a return.
	retlabel *types.Sym

	// Temporary result variables.
	retvars []ir.Node

	// Whether result variables should be initialized at the
	// "return" statement.
	delayretvars bool

	inlvars map[*ir.Name]ir.Node

	// bases maps from original PosBase to PosBase with an extra
	// inlined call frame.
	bases map[*src.PosBase]*src.PosBase

	// newInlIndex is the index of the inlined call frame to
	// insert for inlined nodes.
	newInlIndex int
}

// list inlines a list of nodes.
func (subst *inlsubst) list(ll ir.Nodes) []ir.Node {
	s := make([]ir.Node, 0, ll.Len())
	for _, n := range ll.Slice() {
		s = append(s, subst.node(n))
	}
	return s
}

// node recursively copies a node from the saved pristine body of the
// inlined function, substituting references to input/output
// parameters with ones to the tmpnames, and substituting returns with
// assignments to the output.
func (subst *inlsubst) node(n ir.Node) ir.Node {
	if n == nil {
		return nil
	}

	switch n.Op() {
	case ir.ONAME:
		n := n.(*ir.Name)
		if inlvar := subst.inlvars[n]; inlvar != nil { // These will be set during inlnode
			if base.Flag.LowerM > 2 {
				fmt.Printf("substituting name %+v  ->  %+v\n", n, inlvar)
			}
			return inlvar
		}

		if base.Flag.LowerM > 2 {
			fmt.Printf("not substituting name %+v\n", n)
		}
		return n

	case ir.OMETHEXPR:
		return n

	case ir.OLITERAL, ir.ONIL, ir.OTYPE:
		// If n is a named constant or type, we can continue
		// using it in the inline copy. Otherwise, make a copy
		// so we can update the line number.
		if n.Sym() != nil {
			return n
		}

		// Since we don't handle bodies with closures, this return is guaranteed to belong to the current inlined function.

	//		dump("Return before substitution", n);
	case ir.ORETURN:
		m := nodSym(ir.OGOTO, nil, subst.retlabel)
		m.PtrInit().Set(subst.list(n.Init()))

		if len(subst.retvars) != 0 && n.List().Len() != 0 {
			as := ir.Nod(ir.OAS2, nil, nil)

			// Make a shallow copy of retvars.
			// Otherwise OINLCALL.Rlist will be the same list,
			// and later walk and typecheck may clobber it.
			for _, n := range subst.retvars {
				as.PtrList().Append(n)
			}
			as.PtrRlist().Set(subst.list(n.List()))

			if subst.delayretvars {
				for _, n := range as.List().Slice() {
					as.PtrInit().Append(ir.Nod(ir.ODCL, n, nil))
					n.Name().Defn = as
				}
			}

			as = typecheck(as, ctxStmt)
			m.PtrInit().Append(as)
		}

		typecheckslice(m.Init().Slice(), ctxStmt)
		m = typecheck(m, ctxStmt)

		//		dump("Return after substitution", m);
		return m

	case ir.OGOTO, ir.OLABEL:
		m := ir.Copy(n)
		m.SetPos(subst.updatedPos(m.Pos()))
		m.PtrInit().Set(nil)
		p := fmt.Sprintf("%s·%d", n.Sym().Name, inlgen)
		m.SetSym(lookup(p))

		return m
	}

	m := ir.Copy(n)
	m.SetPos(subst.updatedPos(m.Pos()))
	m.PtrInit().Set(nil)

	if n.Op() == ir.OCLOSURE {
		base.Fatalf("cannot inline function containing closure: %+v", n)
	}

	m.SetLeft(subst.node(n.Left()))
	m.SetRight(subst.node(n.Right()))
	m.PtrList().Set(subst.list(n.List()))
	m.PtrRlist().Set(subst.list(n.Rlist()))
	m.PtrInit().Set(append(m.Init().Slice(), subst.list(n.Init())...))
	m.PtrBody().Set(subst.list(n.Body()))

	return m
}

func (subst *inlsubst) updatedPos(xpos src.XPos) src.XPos {
	pos := base.Ctxt.PosTable.Pos(xpos)
	oldbase := pos.Base() // can be nil
	newbase := subst.bases[oldbase]
	if newbase == nil {
		newbase = src.NewInliningBase(oldbase, subst.newInlIndex)
		subst.bases[oldbase] = newbase
	}
	pos.SetBase(newbase)
	return base.Ctxt.PosTable.XPos(pos)
}

func pruneUnusedAutos(ll []*ir.Name, vis *hairyVisitor) []*ir.Name {
	s := make([]*ir.Name, 0, len(ll))
	for _, n := range ll {
		if n.Class() == ir.PAUTO {
			if _, found := vis.usedLocals[n]; !found {
				continue
			}
		}
		s = append(s, n)
	}
	return s
}

// devirtualize replaces interface method calls within fn with direct
// concrete-type method calls where applicable.
func devirtualize(fn *ir.Func) {
	Curfn = fn
	ir.InspectList(fn.Body(), func(n ir.Node) bool {
		if n.Op() == ir.OCALLINTER {
			devirtualizeCall(n)
		}
		return true
	})
}

func devirtualizeCall(call ir.Node) {
	recv := staticValue(call.Left().Left())
	if recv.Op() != ir.OCONVIFACE {
		return
	}

	typ := recv.Left().Type()
	if typ.IsInterface() {
		return
	}

	x := ir.NodAt(call.Left().Pos(), ir.ODOTTYPE, call.Left().Left(), nil)
	x.SetType(typ)
	x = nodlSym(call.Left().Pos(), ir.OXDOT, x, call.Left().Sym())
	x = typecheck(x, ctxExpr|ctxCallee)
	switch x.Op() {
	case ir.ODOTMETH:
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "devirtualizing %v to %v", call.Left(), typ)
		}
		call.SetOp(ir.OCALLMETH)
		call.SetLeft(x)
	case ir.ODOTINTER:
		// Promoted method from embedded interface-typed field (#42279).
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "partially devirtualizing %v to %v", call.Left(), typ)
		}
		call.SetOp(ir.OCALLINTER)
		call.SetLeft(x)
	default:
		// TODO(mdempsky): Turn back into Fatalf after more testing.
		if base.Flag.LowerM != 0 {
			base.WarnfAt(call.Pos(), "failed to devirtualize %v (%v)", x, x.Op())
		}
		return
	}

	// Duplicated logic from typecheck for function call return
	// value types.
	//
	// Receiver parameter size may have changed; need to update
	// call.Type to get correct stack offsets for result
	// parameters.
	checkwidth(x.Type())
	switch ft := x.Type(); ft.NumResults() {
	case 0:
	case 1:
		call.SetType(ft.Results().Field(0).Type)
	default:
		call.SetType(ft.Results())
	}
}
