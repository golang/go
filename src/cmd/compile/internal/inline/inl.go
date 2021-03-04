// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The inlining facility makes 2 passes: first caninl determines which
// functions are suitable for inlining, and for those that are it
// saves a copy of the body. Then InlineCalls walks each function body to
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

package inline

import (
	"fmt"
	"go/constant"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
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

func InlinePackage() {
	// Find functions that can be inlined and clone them before walk expands them.
	ir.VisitFuncsBottomUp(typecheck.Target.Decls, func(list []*ir.Func, recursive bool) {
		numfns := numNonClosures(list)
		for _, n := range list {
			if !recursive || numfns > 1 {
				// We allow inlining if there is no
				// recursion, or the recursion cycle is
				// across more than one function.
				CanInline(n)
			} else {
				if base.Flag.LowerM > 1 {
					fmt.Printf("%v: cannot inline %v: recursive\n", ir.Line(n), n.Nname)
				}
			}
			InlineCalls(n)
		}
	})
}

// CanInline determines whether fn is inlineable.
// If so, CanInline saves fn->nbody in fn->inl and substitutes it with a copy.
// fn and ->nbody will already have been typechecked.
func CanInline(fn *ir.Func) {
	if fn.Nname == nil {
		base.Fatalf("CanInline no nname %+v", fn)
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
	if len(fn.Body) == 0 {
		reason = "no function body"
		return
	}

	if fn.Typecheck() == 0 {
		base.Fatalf("CanInline on non-typechecked function %v", fn)
	}

	n := fn.Nname
	if n.Func.InlinabilityChecked() {
		return
	}
	defer n.Func.SetInlinabilityChecked(true)

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
	}
	if visitor.tooHairy(fn) {
		reason = visitor.reason
		return
	}

	n.Func.Inl = &ir.Inline{
		Cost: inlineMaxBudget - visitor.budget,
		Dcl:  pruneUnusedAutos(n.Defn.(*ir.Func).Dcl, &visitor),
		Body: inlcopylist(fn.Body),
	}

	if base.Flag.LowerM > 1 {
		fmt.Printf("%v: can inline %v with cost %d as: %v { %v }\n", ir.Line(fn), n, inlineMaxBudget-visitor.budget, fn.Type(), ir.Nodes(n.Func.Inl.Body))
	} else if base.Flag.LowerM != 0 {
		fmt.Printf("%v: can inline %v\n", ir.Line(fn), n)
	}
	if logopt.Enabled() {
		logopt.LogOpt(fn.Pos(), "canInlineFunction", "inline", ir.FuncName(fn), fmt.Sprintf("cost: %d", inlineMaxBudget-visitor.budget))
	}
}

// Inline_Flood marks n's inline body for export and recursively ensures
// all called functions are marked too.
func Inline_Flood(n *ir.Name, exportsym func(*ir.Name)) {
	if n == nil {
		return
	}
	if n.Op() != ir.ONAME || n.Class != ir.PFUNC {
		base.Fatalf("Inline_Flood: unexpected %v, %v, %v", n, n.Op(), n.Class)
	}
	fn := n.Func
	if fn == nil {
		base.Fatalf("Inline_Flood: missing Func on %v", n)
	}
	if fn.Inl == nil {
		return
	}

	if fn.ExportInline() {
		return
	}
	fn.SetExportInline(true)

	typecheck.ImportedBody(fn)

	var doFlood func(n ir.Node)
	doFlood = func(n ir.Node) {
		switch n.Op() {
		case ir.OMETHEXPR, ir.ODOTMETH:
			Inline_Flood(ir.MethodExprName(n), exportsym)

		case ir.ONAME:
			n := n.(*ir.Name)
			switch n.Class {
			case ir.PFUNC:
				Inline_Flood(n, exportsym)
				exportsym(n)
			case ir.PEXTERN:
				exportsym(n)
			}

		case ir.OCALLPART:
			// Okay, because we don't yet inline indirect
			// calls to method values.
		case ir.OCLOSURE:
			// VisitList doesn't visit closure bodies, so force a
			// recursive call to VisitList on the body of the closure.
			ir.VisitList(n.(*ir.ClosureExpr).Func.Body, doFlood)
		}
	}

	// Recursively identify all referenced functions for
	// reexport. We want to include even non-called functions,
	// because after inlining they might be callable.
	ir.VisitList(ir.Nodes(fn.Inl.Body), doFlood)
}

// hairyVisitor visits a function body to determine its inlining
// hairiness and whether or not it can be inlined.
type hairyVisitor struct {
	budget        int32
	reason        string
	extraCallCost int32
	usedLocals    ir.NameSet
	do            func(ir.Node) bool
}

func (v *hairyVisitor) tooHairy(fn *ir.Func) bool {
	v.do = v.doNode // cache closure
	if ir.DoChildren(fn, v.do) {
		return true
	}
	if v.budget < 0 {
		v.reason = fmt.Sprintf("function too complex: cost %d exceeds budget %d", inlineMaxBudget-v.budget, inlineMaxBudget)
		return true
	}
	return false
}

func (v *hairyVisitor) doNode(n ir.Node) bool {
	if n == nil {
		return false
	}
	switch n.Op() {
	// Call is okay if inlinable and we have the budget for the body.
	case ir.OCALLFUNC:
		n := n.(*ir.CallExpr)
		// Functions that call runtime.getcaller{pc,sp} can not be inlined
		// because getcaller{pc,sp} expect a pointer to the caller's first argument.
		//
		// runtime.throw is a "cheap call" like panic in normal code.
		if n.X.Op() == ir.ONAME {
			name := n.X.(*ir.Name)
			if name.Class == ir.PFUNC && types.IsRuntimePkg(name.Sym().Pkg) {
				fn := name.Sym().Name
				if fn == "getcallerpc" || fn == "getcallersp" {
					v.reason = "call to " + fn
					return true
				}
				if fn == "throw" {
					v.budget -= inlineExtraThrowCost
					break
				}
			}
		}

		if ir.IsIntrinsicCall(n) {
			// Treat like any other node.
			break
		}

		if fn := inlCallee(n.X); fn != nil && fn.Inl != nil {
			v.budget -= fn.Inl.Cost
			break
		}

		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	// Call is okay if inlinable and we have the budget for the body.
	case ir.OCALLMETH:
		n := n.(*ir.CallExpr)
		t := n.X.Type()
		if t == nil {
			base.Fatalf("no function type for [%p] %+v\n", n.X, n.X)
		}
		fn := ir.MethodExprName(n.X).Func
		if types.IsRuntimePkg(fn.Sym().Pkg) && fn.Sym().Name == "heapBits.nextArena" {
			// Special case: explicitly allow
			// mid-stack inlining of
			// runtime.heapBits.next even though
			// it calls slow-path
			// runtime.heapBits.nextArena.
			break
		}
		if fn.Inl != nil {
			v.budget -= fn.Inl.Cost
			break
		}
		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	// Things that are too hairy, irrespective of the budget
	case ir.OCALL, ir.OCALLINTER:
		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	case ir.OPANIC:
		n := n.(*ir.UnaryExpr)
		if n.X.Op() == ir.OCONVIFACE && n.X.(*ir.ConvExpr).Implicit() {
			// Hack to keep reflect.flag.mustBe inlinable for TestIntendedInlining.
			// Before CL 284412, these conversions were introduced later in the
			// compiler, so they didn't count against inlining budget.
			v.budget++
		}
		v.budget -= inlineExtraPanicCost

	case ir.ORECOVER:
		// recover matches the argument frame pointer to find
		// the right panic value, so it needs an argument frame.
		v.reason = "call to recover"
		return true

	case ir.OCLOSURE:
		if base.Debug.InlFuncsWithClosures == 0 {
			v.reason = "not inlining functions with closures"
			return true
		}

		// TODO(danscales): Maybe make budget proportional to number of closure
		// variables, e.g.:
		//v.budget -= int32(len(n.(*ir.ClosureExpr).Func.ClosureVars) * 3)
		v.budget -= 15
		// Scan body of closure (which DoChildren doesn't automatically
		// do) to check for disallowed ops in the body and include the
		// body in the budget.
		if doList(n.(*ir.ClosureExpr).Func.Body, v.do) {
			return true
		}

	case ir.ORANGE,
		ir.OSELECT,
		ir.OGO,
		ir.ODEFER,
		ir.ODCLTYPE, // can't print yet
		ir.OTAILCALL:
		v.reason = "unhandled op " + n.Op().String()
		return true

	case ir.OAPPEND:
		v.budget -= inlineExtraAppendCost

	case ir.ODEREF:
		// *(*X)(unsafe.Pointer(&x)) is low-cost
		n := n.(*ir.StarExpr)

		ptr := n.X
		for ptr.Op() == ir.OCONVNOP {
			ptr = ptr.(*ir.ConvExpr).X
		}
		if ptr.Op() == ir.OADDR {
			v.budget += 1 // undo half of default cost of ir.ODEREF+ir.OADDR
		}

	case ir.OCONVNOP:
		// This doesn't produce code, but the children might.
		v.budget++ // undo default cost

	case ir.ODCLCONST, ir.OFALL:
		// These nodes don't produce code; omit from inlining budget.
		return false

	case ir.OFOR, ir.OFORUNTIL:
		n := n.(*ir.ForStmt)
		if n.Label != nil {
			v.reason = "labeled control"
			return true
		}
	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		if n.Label != nil {
			v.reason = "labeled control"
			return true
		}
	// case ir.ORANGE, ir.OSELECT in "unhandled" above

	case ir.OBREAK, ir.OCONTINUE:
		n := n.(*ir.BranchStmt)
		if n.Label != nil {
			// Should have short-circuited due to labeled control error above.
			base.Fatalf("unexpected labeled break/continue: %v", n)
		}

	case ir.OIF:
		n := n.(*ir.IfStmt)
		if ir.IsConst(n.Cond, constant.Bool) {
			// This if and the condition cost nothing.
			// TODO(rsc): It seems strange that we visit the dead branch.
			return doList(n.Init(), v.do) ||
				doList(n.Body, v.do) ||
				doList(n.Else, v.do)
		}

	case ir.ONAME:
		n := n.(*ir.Name)
		if n.Class == ir.PAUTO {
			v.usedLocals.Add(n)
		}

	case ir.OBLOCK:
		// The only OBLOCK we should see at this point is an empty one.
		// In any event, let the visitList(n.List()) below take care of the statements,
		// and don't charge for the OBLOCK itself. The ++ undoes the -- below.
		v.budget++

	case ir.OCALLPART, ir.OSLICELIT:
		v.budget-- // Hack for toolstash -cmp.

	case ir.OMETHEXPR:
		v.budget++ // Hack for toolstash -cmp.
	}

	v.budget--

	// When debugging, don't stop early, to get full cost of inlining this function
	if v.budget < 0 && base.Flag.LowerM < 2 && !logopt.Enabled() {
		v.reason = "too expensive"
		return true
	}

	return ir.DoChildren(n, v.do)
}

func isBigFunc(fn *ir.Func) bool {
	budget := inlineBigFunctionNodes
	return ir.Any(fn, func(n ir.Node) bool {
		budget--
		return budget <= 0
	})
}

// inlcopylist (together with inlcopy) recursively copies a list of nodes, except
// that it keeps the same ONAME, OTYPE, and OLITERAL nodes. It is used for copying
// the body and dcls of an inlineable function.
func inlcopylist(ll []ir.Node) []ir.Node {
	s := make([]ir.Node, len(ll))
	for i, n := range ll {
		s[i] = inlcopy(n)
	}
	return s
}

// inlcopy is like DeepCopy(), but does extra work to copy closures.
func inlcopy(n ir.Node) ir.Node {
	var edit func(ir.Node) ir.Node
	edit = func(x ir.Node) ir.Node {
		switch x.Op() {
		case ir.ONAME, ir.OTYPE, ir.OLITERAL, ir.ONIL:
			return x
		}
		m := ir.Copy(x)
		ir.EditChildren(m, edit)
		if x.Op() == ir.OCLOSURE {
			x := x.(*ir.ClosureExpr)
			// Need to save/duplicate x.Func.Nname,
			// x.Func.Nname.Ntype, x.Func.Dcl, x.Func.ClosureVars, and
			// x.Func.Body for iexport and local inlining.
			oldfn := x.Func
			newfn := ir.NewFunc(oldfn.Pos())
			if oldfn.ClosureCalled() {
				newfn.SetClosureCalled(true)
			}
			m.(*ir.ClosureExpr).Func = newfn
			newfn.Nname = ir.NewNameAt(oldfn.Nname.Pos(), oldfn.Nname.Sym())
			// XXX OK to share fn.Type() ??
			newfn.Nname.SetType(oldfn.Nname.Type())
			newfn.Nname.Ntype = inlcopy(oldfn.Nname.Ntype).(ir.Ntype)
			newfn.Body = inlcopylist(oldfn.Body)
			// Make shallow copy of the Dcl and ClosureVar slices
			newfn.Dcl = append([]*ir.Name(nil), oldfn.Dcl...)
			newfn.ClosureVars = append([]*ir.Name(nil), oldfn.ClosureVars...)
		}
		return m
	}
	return edit(n)
}

// Inlcalls/nodelist/node walks fn's statements and expressions and substitutes any
// calls made to inlineable functions. This is the external entry point.
func InlineCalls(fn *ir.Func) {
	savefn := ir.CurFunc
	ir.CurFunc = fn
	maxCost := int32(inlineMaxBudget)
	if isBigFunc(fn) {
		maxCost = inlineBigFunctionMaxCost
	}
	// Map to keep track of functions that have been inlined at a particular
	// call site, in order to stop inlining when we reach the beginning of a
	// recursion cycle again. We don't inline immediately recursive functions,
	// but allow inlining if there is a recursion cycle of many functions.
	// Most likely, the inlining will stop before we even hit the beginning of
	// the cycle again, but the map catches the unusual case.
	inlMap := make(map[*ir.Func]bool)
	var edit func(ir.Node) ir.Node
	edit = func(n ir.Node) ir.Node {
		return inlnode(n, maxCost, inlMap, edit)
	}
	ir.EditChildren(fn, edit)
	ir.CurFunc = savefn
}

// Turn an OINLCALL into a statement.
func inlconv2stmt(inlcall *ir.InlinedCallExpr) ir.Node {
	n := ir.NewBlockStmt(inlcall.Pos(), nil)
	n.List = inlcall.Init()
	n.List.Append(inlcall.Body.Take()...)
	return n
}

// Turn an OINLCALL into a single valued expression.
// The result of inlconv2expr MUST be assigned back to n, e.g.
// 	n.Left = inlconv2expr(n.Left)
func inlconv2expr(n *ir.InlinedCallExpr) ir.Node {
	r := n.ReturnVars[0]
	return ir.InitExpr(append(n.Init(), n.Body...), r)
}

// Turn the rlist (with the return values) of the OINLCALL in
// n into an expression list lumping the ninit and body
// containing the inlined statements on the first list element so
// order will be preserved. Used in return, oas2func and call
// statements.
func inlconv2list(n *ir.InlinedCallExpr) []ir.Node {
	if n.Op() != ir.OINLCALL || len(n.ReturnVars) == 0 {
		base.Fatalf("inlconv2list %+v\n", n)
	}

	s := n.ReturnVars
	s[0] = ir.InitExpr(append(n.Init(), n.Body...), s[0])
	return s
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
func inlnode(n ir.Node, maxCost int32, inlMap map[*ir.Func]bool, edit func(ir.Node) ir.Node) ir.Node {
	if n == nil {
		return n
	}

	switch n.Op() {
	case ir.ODEFER, ir.OGO:
		n := n.(*ir.GoDeferStmt)
		switch call := n.Call; call.Op() {
		case ir.OCALLFUNC, ir.OCALLMETH:
			call := call.(*ir.CallExpr)
			call.NoInline = true
		}

	// TODO do them here (or earlier),
	// so escape analysis can avoid more heapmoves.
	case ir.OCLOSURE:
		return n
	case ir.OCALLMETH:
		// Prevent inlining some reflect.Value methods when using checkptr,
		// even when package reflect was compiled without it (#35073).
		n := n.(*ir.CallExpr)
		if s := ir.MethodExprName(n.X).Sym(); base.Debug.Checkptr != 0 && types.IsReflectPkg(s.Pkg) && (s.Name == "Value.UnsafeAddr" || s.Name == "Value.Pointer") {
			return n
		}
	}

	lno := ir.SetPos(n)

	ir.EditChildren(n, edit)

	if as := n; as.Op() == ir.OAS2FUNC {
		as := as.(*ir.AssignListStmt)
		if as.Rhs[0].Op() == ir.OINLCALL {
			as.Rhs = inlconv2list(as.Rhs[0].(*ir.InlinedCallExpr))
			as.SetOp(ir.OAS2)
			as.SetTypecheck(0)
			n = typecheck.Stmt(as)
		}
	}

	// with all the branches out of the way, it is now time to
	// transmogrify this node itself unless inhibited by the
	// switch at the top of this function.
	switch n.Op() {
	case ir.OCALLFUNC, ir.OCALLMETH:
		n := n.(*ir.CallExpr)
		if n.NoInline {
			return n
		}
	}

	var call *ir.CallExpr
	switch n.Op() {
	case ir.OCALLFUNC:
		call = n.(*ir.CallExpr)
		if base.Flag.LowerM > 3 {
			fmt.Printf("%v:call to func %+v\n", ir.Line(n), call.X)
		}
		if ir.IsIntrinsicCall(call) {
			break
		}
		if fn := inlCallee(call.X); fn != nil && fn.Inl != nil {
			n = mkinlcall(call, fn, maxCost, inlMap, edit)
		}

	case ir.OCALLMETH:
		call = n.(*ir.CallExpr)
		if base.Flag.LowerM > 3 {
			fmt.Printf("%v:call to meth %v\n", ir.Line(n), call.X.(*ir.SelectorExpr).Sel)
		}

		// typecheck should have resolved ODOTMETH->type, whose nname points to the actual function.
		if call.X.Type() == nil {
			base.Fatalf("no function type for [%p] %+v\n", call.X, call.X)
		}

		n = mkinlcall(call, ir.MethodExprName(call.X).Func, maxCost, inlMap, edit)
	}

	base.Pos = lno

	if n.Op() == ir.OINLCALL {
		ic := n.(*ir.InlinedCallExpr)
		switch call.Use {
		default:
			ir.Dump("call", call)
			base.Fatalf("call missing use")
		case ir.CallUseExpr:
			n = inlconv2expr(ic)
		case ir.CallUseStmt:
			n = inlconv2stmt(ic)
		case ir.CallUseList:
			// leave for caller to convert
		}
	}

	return n
}

// inlCallee takes a function-typed expression and returns the underlying function ONAME
// that it refers to if statically known. Otherwise, it returns nil.
func inlCallee(fn ir.Node) *ir.Func {
	fn = ir.StaticValue(fn)
	switch fn.Op() {
	case ir.OMETHEXPR:
		fn := fn.(*ir.SelectorExpr)
		n := ir.MethodExprName(fn)
		// Check that receiver type matches fn.X.
		// TODO(mdempsky): Handle implicit dereference
		// of pointer receiver argument?
		if n == nil || !types.Identical(n.Type().Recv().Type, fn.X.Type()) {
			return nil
		}
		return n.Func
	case ir.ONAME:
		fn := fn.(*ir.Name)
		if fn.Class == ir.PFUNC {
			return fn.Func
		}
	case ir.OCLOSURE:
		fn := fn.(*ir.ClosureExpr)
		c := fn.Func
		CanInline(c)
		return c
	}
	return nil
}

func inlParam(t *types.Field, as ir.InitNode, inlvars map[*ir.Name]*ir.Name) ir.Node {
	if t.Nname == nil {
		return ir.BlankNode
	}
	n := t.Nname.(*ir.Name)
	if ir.IsBlank(n) {
		return ir.BlankNode
	}
	inlvar := inlvars[n]
	if inlvar == nil {
		base.Fatalf("missing inlvar for %v", n)
	}
	as.PtrInit().Append(ir.NewDecl(base.Pos, ir.ODCL, inlvar))
	inlvar.Name().Defn = as
	return inlvar
}

var inlgen int

// SSADumpInline gives the SSA back end a chance to dump the function
// when producing output for debugging the compiler itself.
var SSADumpInline = func(*ir.Func) {}

// If n is a call node (OCALLFUNC or OCALLMETH), and fn is an ONAME node for a
// function with an inlinable body, return an OINLCALL node that can replace n.
// The returned node's Ninit has the parameter assignments, the Nbody is the
// inlined function body, and (List, Rlist) contain the (input, output)
// parameters.
// The result of mkinlcall MUST be assigned back to n, e.g.
// 	n.Left = mkinlcall(n.Left, fn, isddd)
func mkinlcall(n *ir.CallExpr, fn *ir.Func, maxCost int32, inlMap map[*ir.Func]bool, edit func(ir.Node) ir.Node) ir.Node {
	if fn.Inl == nil {
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(ir.CurFunc),
				fmt.Sprintf("%s cannot be inlined", ir.PkgFuncName(fn)))
		}
		return n
	}
	if fn.Inl.Cost > maxCost {
		// The inlined function body is too big. Typically we use this check to restrict
		// inlining into very big functions.  See issue 26546 and 17566.
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(ir.CurFunc),
				fmt.Sprintf("cost %d of %s exceeds max large caller cost %d", fn.Inl.Cost, ir.PkgFuncName(fn), maxCost))
		}
		return n
	}

	if fn == ir.CurFunc {
		// Can't recursively inline a function into itself.
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", fmt.Sprintf("recursive call to %s", ir.FuncName(ir.CurFunc)))
		}
		return n
	}

	if base.Flag.Cfg.Instrumenting && types.IsRuntimePkg(fn.Sym().Pkg) {
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
			fmt.Printf("%v: cannot inline %v into %v: repeated recursive cycle\n", ir.Line(n), fn, ir.FuncName(ir.CurFunc))
		}
		return n
	}
	inlMap[fn] = true
	defer func() {
		inlMap[fn] = false
	}()
	if base.Debug.TypecheckInl == 0 {
		typecheck.ImportedBody(fn)
	}

	// We have a function node, and it has an inlineable body.
	if base.Flag.LowerM > 1 {
		fmt.Printf("%v: inlining call to %v %v { %v }\n", ir.Line(n), fn.Sym(), fn.Type(), ir.Nodes(fn.Inl.Body))
	} else if base.Flag.LowerM != 0 {
		fmt.Printf("%v: inlining call to %v\n", ir.Line(n), fn)
	}
	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: Before inlining: %+v\n", ir.Line(n), n)
	}

	SSADumpInline(fn)

	ninit := n.Init()

	// For normal function calls, the function callee expression
	// may contain side effects (e.g., added by addinit during
	// inlconv2expr or inlconv2list). Make sure to preserve these,
	// if necessary (#42703).
	if n.Op() == ir.OCALLFUNC {
		callee := n.X
		for callee.Op() == ir.OCONVNOP {
			conv := callee.(*ir.ConvExpr)
			ninit.Append(ir.TakeInit(conv)...)
			callee = conv.X
		}
		if callee.Op() != ir.ONAME && callee.Op() != ir.OCLOSURE && callee.Op() != ir.OMETHEXPR {
			base.Fatalf("unexpected callee expression: %v", callee)
		}
	}

	// Make temp names to use instead of the originals.
	inlvars := make(map[*ir.Name]*ir.Name)

	// record formals/locals for later post-processing
	var inlfvars []*ir.Name

	for _, ln := range fn.Inl.Dcl {
		if ln.Op() != ir.ONAME {
			continue
		}
		if ln.Class == ir.PPARAMOUT { // return values handled below.
			continue
		}
		inlf := typecheck.Expr(inlvar(ln)).(*ir.Name)
		inlvars[ln] = inlf
		if base.Flag.GenDwarfInl > 0 {
			if ln.Class == ir.PPARAM {
				inlf.Name().SetInlFormal(true)
			} else {
				inlf.Name().SetInlLocal(true)
			}
			inlf.SetPos(ln.Pos())
			inlfvars = append(inlfvars, inlf)
		}
	}

	// We can delay declaring+initializing result parameters if:
	// (1) there's exactly one "return" statement in the inlined function;
	// (2) it's not an empty return statement (#44355); and
	// (3) the result parameters aren't named.
	delayretvars := true

	nreturns := 0
	ir.VisitList(ir.Nodes(fn.Inl.Body), func(n ir.Node) {
		if n, ok := n.(*ir.ReturnStmt); ok {
			nreturns++
			if len(n.Results) == 0 {
				delayretvars = false // empty return statement (case 2)
			}
		}
	})

	if nreturns != 1 {
		delayretvars = false // not exactly one return statement (case 1)
	}

	// temporaries for return values.
	var retvars []ir.Node
	for i, t := range fn.Type().Results().Fields().Slice() {
		var m *ir.Name
		if nn := t.Nname; nn != nil && !ir.IsBlank(nn.(*ir.Name)) && !strings.HasPrefix(nn.Sym().Name, "~r") {
			n := nn.(*ir.Name)
			m = inlvar(n)
			m = typecheck.Expr(m).(*ir.Name)
			inlvars[n] = m
			delayretvars = false // found a named result parameter (case 3)
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
	as := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
	as.Def = true
	if n.Op() == ir.OCALLMETH {
		sel := n.X.(*ir.SelectorExpr)
		if sel.X == nil {
			base.Fatalf("method call without receiver: %+v", n)
		}
		as.Rhs.Append(sel.X)
	}
	as.Rhs.Append(n.Args...)

	// For non-dotted calls to variadic functions, we assign the
	// variadic parameter's temp name separately.
	var vas *ir.AssignStmt

	if recv := fn.Type().Recv(); recv != nil {
		as.Lhs.Append(inlParam(recv, as, inlvars))
	}
	for _, param := range fn.Type().Params().Fields().Slice() {
		// For ordinary parameters or variadic parameters in
		// dotted calls, just add the variable to the
		// assignment list, and we're done.
		if !param.IsDDD() || n.IsDDD {
			as.Lhs.Append(inlParam(param, as, inlvars))
			continue
		}

		// Otherwise, we need to collect the remaining values
		// to pass as a slice.

		x := len(as.Lhs)
		for len(as.Lhs) < len(as.Rhs) {
			as.Lhs.Append(argvar(param.Type, len(as.Lhs)))
		}
		varargs := as.Lhs[x:]

		vas = ir.NewAssignStmt(base.Pos, nil, nil)
		vas.X = inlParam(param, vas, inlvars)
		if len(varargs) == 0 {
			vas.Y = typecheck.NodNil()
			vas.Y.SetType(param.Type)
		} else {
			lit := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(param.Type), nil)
			lit.List = varargs
			vas.Y = lit
		}
	}

	if len(as.Rhs) != 0 {
		ninit.Append(typecheck.Stmt(as))
	}

	if vas != nil {
		ninit.Append(typecheck.Stmt(vas))
	}

	if !delayretvars {
		// Zero the return parameters.
		for _, n := range retvars {
			ninit.Append(ir.NewDecl(base.Pos, ir.ODCL, n.(*ir.Name)))
			ras := ir.NewAssignStmt(base.Pos, n, nil)
			ninit.Append(typecheck.Stmt(ras))
		}
	}

	retlabel := typecheck.AutoLabel(".i")

	inlgen++

	parent := -1
	if b := base.Ctxt.PosTable.Pos(n.Pos()).Base(); b != nil {
		parent = b.InliningIndex()
	}

	sym := fn.Linksym()
	newIndex := base.Ctxt.InlTree.Add(parent, n.Pos(), sym)

	// Add an inline mark just before the inlined body.
	// This mark is inline in the code so that it's a reasonable spot
	// to put a breakpoint. Not sure if that's really necessary or not
	// (in which case it could go at the end of the function instead).
	// Note issue 28603.
	inlMark := ir.NewInlineMarkStmt(base.Pos, types.BADWIDTH)
	inlMark.SetPos(n.Pos().WithIsStmt())
	inlMark.Index = int64(newIndex)
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
		fn:           fn,
	}
	subst.edit = subst.node

	body := subst.list(ir.Nodes(fn.Inl.Body))

	lab := ir.NewLabelStmt(base.Pos, retlabel)
	body = append(body, lab)

	typecheck.Stmts(body)

	if base.Flag.GenDwarfInl > 0 {
		for _, v := range inlfvars {
			v.SetPos(subst.updatedPos(v.Pos()))
		}
	}

	//dumplist("ninit post", ninit);

	call := ir.NewInlinedCallExpr(base.Pos, nil, nil)
	*call.PtrInit() = ninit
	call.Body = body
	call.ReturnVars = retvars
	call.SetType(n.Type())
	call.SetTypecheck(1)

	// transitive inlining
	// might be nice to do this before exporting the body,
	// but can't emit the body with inlining expanded.
	// instead we emit the things that the body needs
	// and each use must redo the inlining.
	// luckily these are small.
	ir.EditChildren(call, edit)

	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: After inlining %+v\n\n", ir.Line(call), call)
	}

	return call
}

// Every time we expand a function we generate a new set of tmpnames,
// PAUTO's in the calling functions, and link them off of the
// PPARAM's, PAUTOS and PPARAMOUTs of the called function.
func inlvar(var_ *ir.Name) *ir.Name {
	if base.Flag.LowerM > 3 {
		fmt.Printf("inlvar %+v\n", var_)
	}

	n := typecheck.NewName(var_.Sym())
	n.SetType(var_.Type())
	n.Class = ir.PAUTO
	n.SetUsed(true)
	n.Curfn = ir.CurFunc // the calling function, not the called one
	n.SetAddrtaken(var_.Addrtaken())

	ir.CurFunc.Dcl = append(ir.CurFunc.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *types.Field, i int) *ir.Name {
	n := typecheck.NewName(typecheck.LookupNum("~R", i))
	n.SetType(t.Type)
	n.Class = ir.PAUTO
	n.SetUsed(true)
	n.Curfn = ir.CurFunc // the calling function, not the called one
	ir.CurFunc.Dcl = append(ir.CurFunc.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's arguments
// when they come from a multiple return call.
func argvar(t *types.Type, i int) ir.Node {
	n := typecheck.NewName(typecheck.LookupNum("~arg", i))
	n.SetType(t.Elem())
	n.Class = ir.PAUTO
	n.SetUsed(true)
	n.Curfn = ir.CurFunc // the calling function, not the called one
	ir.CurFunc.Dcl = append(ir.CurFunc.Dcl, n)
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

	inlvars map[*ir.Name]*ir.Name

	// bases maps from original PosBase to PosBase with an extra
	// inlined call frame.
	bases map[*src.PosBase]*src.PosBase

	// newInlIndex is the index of the inlined call frame to
	// insert for inlined nodes.
	newInlIndex int

	edit func(ir.Node) ir.Node // cached copy of subst.node method value closure

	// If non-nil, we are inside a closure inside the inlined function, and
	// newclofn is the Func of the new inlined closure.
	newclofn *ir.Func

	fn *ir.Func // For debug -- the func that is being inlined
}

// list inlines a list of nodes.
func (subst *inlsubst) list(ll ir.Nodes) []ir.Node {
	s := make([]ir.Node, 0, len(ll))
	for _, n := range ll {
		s = append(s, subst.node(n))
	}
	return s
}

// fields returns a list of the fields of a struct type representing receiver,
// params, or results, after duplicating the field nodes and substituting the
// Nname nodes inside the field nodes.
func (subst *inlsubst) fields(oldt *types.Type) []*types.Field {
	oldfields := oldt.FieldSlice()
	newfields := make([]*types.Field, len(oldfields))
	for i := range oldfields {
		newfields[i] = oldfields[i].Copy()
		if oldfields[i].Nname != nil {
			newfields[i].Nname = subst.node(oldfields[i].Nname.(*ir.Name))
		}
	}
	return newfields
}

// clovar creates a new ONAME node for a local variable or param of a closure
// inside a function being inlined.
func (subst *inlsubst) clovar(n *ir.Name) *ir.Name {
	// TODO(danscales): want to get rid of this shallow copy, with code like the
	// following, but it is hard to copy all the necessary flags in a maintainable way.
	// m := ir.NewNameAt(n.Pos(), n.Sym())
	// m.Class = n.Class
	// m.SetType(n.Type())
	// m.SetTypecheck(1)
	//if n.IsClosureVar() {
	//	m.SetIsClosureVar(true)
	//}
	m := &ir.Name{}
	*m = *n
	m.Curfn = subst.newclofn
	if n.Defn != nil && n.Defn.Op() == ir.ONAME {
		if !n.IsClosureVar() {
			base.FatalfAt(n.Pos(), "want closure variable, got: %+v", n)
		}
		if n.Sym().Pkg != types.LocalPkg {
			// If the closure came from inlining a function from
			// another package, must change package of captured
			// variable to localpkg, so that the fields of the closure
			// struct are local package and can be accessed even if
			// name is not exported. If you disable this code, you can
			// reproduce the problem by running 'go test
			// go/internal/srcimporter'. TODO(mdempsky) - maybe change
			// how we create closure structs?
			m.SetSym(types.LocalPkg.Lookup(n.Sym().Name))
		}
		// Make sure any inlvar which is the Defn
		// of an ONAME closure var is rewritten
		// during inlining. Don't substitute
		// if Defn node is outside inlined function.
		if subst.inlvars[n.Defn.(*ir.Name)] != nil {
			m.Defn = subst.node(n.Defn)
		}
	}
	if n.Outer != nil {
		// Either the outer variable is defined in function being inlined,
		// and we will replace it with the substituted variable, or it is
		// defined outside the function being inlined, and we should just
		// skip the outer variable (the closure variable of the function
		// being inlined).
		s := subst.node(n.Outer).(*ir.Name)
		if s == n.Outer {
			s = n.Outer.Outer
		}
		m.Outer = s
	}
	return m
}

// closure does the necessary substitions for a ClosureExpr n and returns the new
// closure node.
func (subst *inlsubst) closure(n *ir.ClosureExpr) ir.Node {
	m := ir.Copy(n)
	m.SetPos(subst.updatedPos(m.Pos()))
	ir.EditChildren(m, subst.edit)

	//fmt.Printf("Inlining func %v with closure into %v\n", subst.fn, ir.FuncName(ir.CurFunc))

	// The following is similar to funcLit
	oldfn := n.Func
	newfn := ir.NewFunc(oldfn.Pos())
	// These three lines are not strictly necessary, but just to be clear
	// that new function needs to redo typechecking and inlinability.
	newfn.SetTypecheck(0)
	newfn.SetInlinabilityChecked(false)
	newfn.Inl = nil
	newfn.SetIsHiddenClosure(true)
	newfn.Nname = ir.NewNameAt(n.Pos(), ir.BlankNode.Sym())
	newfn.Nname.Func = newfn
	newfn.Nname.Ntype = subst.node(oldfn.Nname.Ntype).(ir.Ntype)
	newfn.Nname.Defn = newfn

	m.(*ir.ClosureExpr).Func = newfn
	newfn.OClosure = m.(*ir.ClosureExpr)

	if subst.newclofn != nil {
		//fmt.Printf("Inlining a closure with a nested closure\n")
	}
	prevxfunc := subst.newclofn

	// Mark that we are now substituting within a closure (within the
	// inlined function), and create new nodes for all the local
	// vars/params inside this closure.
	subst.newclofn = newfn
	newfn.Dcl = nil
	newfn.ClosureVars = nil
	for _, oldv := range oldfn.Dcl {
		newv := subst.clovar(oldv)
		subst.inlvars[oldv] = newv
		newfn.Dcl = append(newfn.Dcl, newv)
	}
	for _, oldv := range oldfn.ClosureVars {
		newv := subst.clovar(oldv)
		subst.inlvars[oldv] = newv
		newfn.ClosureVars = append(newfn.ClosureVars, newv)
	}

	// Need to replace ONAME nodes in
	// newfn.Type().FuncType().Receiver/Params/Results.FieldSlice().Nname
	oldt := oldfn.Type()
	newrecvs := subst.fields(oldt.Recvs())
	var newrecv *types.Field
	if len(newrecvs) > 0 {
		newrecv = newrecvs[0]
	}
	newt := types.NewSignature(oldt.Pkg(), newrecv,
		nil, subst.fields(oldt.Params()), subst.fields(oldt.Results()))

	newfn.Nname.SetType(newt)
	newfn.Body = subst.list(oldfn.Body)

	// Remove the nodes for the current closure from subst.inlvars
	for _, oldv := range oldfn.Dcl {
		delete(subst.inlvars, oldv)
	}
	for _, oldv := range oldfn.ClosureVars {
		delete(subst.inlvars, oldv)
	}
	// Go back to previous closure func
	subst.newclofn = prevxfunc

	// Actually create the named function for the closure, now that
	// the closure is inlined in a specific function.
	m.SetTypecheck(0)
	if oldfn.ClosureCalled() {
		typecheck.Callee(m)
	} else {
		typecheck.Expr(m)
	}
	return m
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

		// Handle captured variables when inlining closures.
		if n.IsClosureVar() && subst.newclofn == nil {
			o := n.Outer

			// Deal with case where sequence of closures are inlined.
			// TODO(danscales) - write test case to see if we need to
			// go up multiple levels.
			if o.Curfn != ir.CurFunc {
				o = o.Outer
			}

			// make sure the outer param matches the inlining location
			if o == nil || o.Curfn != ir.CurFunc {
				base.Fatalf("%v: unresolvable capture %v\n", ir.Line(n), n)
			}

			if base.Flag.LowerM > 2 {
				fmt.Printf("substituting captured name %+v  ->  %+v\n", n, o)
			}
			return o
		}

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
		n := n.(*ir.SelectorExpr)
		return n

	case ir.OLITERAL, ir.ONIL, ir.OTYPE:
		// If n is a named constant or type, we can continue
		// using it in the inline copy. Otherwise, make a copy
		// so we can update the line number.
		if n.Sym() != nil {
			return n
		}

	case ir.ORETURN:
		if subst.newclofn != nil {
			// Don't do special substitutions if inside a closure
			break
		}
		// Since we don't handle bodies with closures,
		// this return is guaranteed to belong to the current inlined function.
		n := n.(*ir.ReturnStmt)
		init := subst.list(n.Init())
		if len(subst.retvars) != 0 && len(n.Results) != 0 {
			as := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)

			// Make a shallow copy of retvars.
			// Otherwise OINLCALL.Rlist will be the same list,
			// and later walk and typecheck may clobber it.
			for _, n := range subst.retvars {
				as.Lhs.Append(n)
			}
			as.Rhs = subst.list(n.Results)

			if subst.delayretvars {
				for _, n := range as.Lhs {
					as.PtrInit().Append(ir.NewDecl(base.Pos, ir.ODCL, n.(*ir.Name)))
					n.Name().Defn = as
				}
			}

			init = append(init, typecheck.Stmt(as))
		}
		init = append(init, ir.NewBranchStmt(base.Pos, ir.OGOTO, subst.retlabel))
		typecheck.Stmts(init)
		return ir.NewBlockStmt(base.Pos, init)

	case ir.OGOTO:
		n := n.(*ir.BranchStmt)
		m := ir.Copy(n).(*ir.BranchStmt)
		m.SetPos(subst.updatedPos(m.Pos()))
		*m.PtrInit() = nil
		p := fmt.Sprintf("%s·%d", n.Label.Name, inlgen)
		m.Label = typecheck.Lookup(p)
		return m

	case ir.OLABEL:
		if subst.newclofn != nil {
			// Don't do special substitutions if inside a closure
			break
		}
		n := n.(*ir.LabelStmt)
		m := ir.Copy(n).(*ir.LabelStmt)
		m.SetPos(subst.updatedPos(m.Pos()))
		*m.PtrInit() = nil
		p := fmt.Sprintf("%s·%d", n.Label.Name, inlgen)
		m.Label = typecheck.Lookup(p)
		return m

	case ir.OCLOSURE:
		return subst.closure(n.(*ir.ClosureExpr))

	}

	m := ir.Copy(n)
	m.SetPos(subst.updatedPos(m.Pos()))
	ir.EditChildren(m, subst.edit)
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
		if n.Class == ir.PAUTO {
			if !vis.usedLocals.Has(n) {
				continue
			}
		}
		s = append(s, n)
	}
	return s
}

// numNonClosures returns the number of functions in list which are not closures.
func numNonClosures(list []*ir.Func) int {
	count := 0
	for _, fn := range list {
		if fn.OClosure == nil {
			count++
		}
	}
	return count
}

func doList(list []ir.Node, do func(ir.Node) bool) bool {
	for _, x := range list {
		if x != nil {
			if do(x) {
				return true
			}
		}
	}
	return false
}
