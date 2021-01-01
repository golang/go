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

package inline

import (
	"errors"
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

// Caninl determines whether fn is inlineable.
// If so, CanInline saves fn->nbody in fn->inl and substitutes it with a copy.
// fn and ->nbody will already have been typechecked.
func CanInline(fn *ir.Func) {
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
	if len(fn.Body) == 0 {
		reason = "no function body"
		return
	}

	if fn.Typecheck() == 0 {
		base.Fatalf("caninl on non-typechecked function %v", fn)
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
		usedLocals:    make(map[*ir.Name]bool),
	}
	if visitor.tooHairy(fn) {
		reason = visitor.reason
		return
	}

	n.Func.Inl = &ir.Inline{
		Cost: inlineMaxBudget - visitor.budget,
		Dcl:  pruneUnusedAutos(n.Defn.(*ir.Func).Dcl, &visitor),
		Body: ir.DeepCopyList(src.NoXPos, fn.Body),
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
	if n.Op() != ir.ONAME || n.Class_ != ir.PFUNC {
		base.Fatalf("inlFlood: unexpected %v, %v, %v", n, n.Op(), n.Class_)
	}
	fn := n.Func
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

	typecheck.ImportedBody(fn)

	// Recursively identify all referenced functions for
	// reexport. We want to include even non-called functions,
	// because after inlining they might be callable.
	ir.VisitList(ir.Nodes(fn.Inl.Body), func(n ir.Node) {
		switch n.Op() {
		case ir.OMETHEXPR, ir.ODOTMETH:
			Inline_Flood(ir.MethodExprName(n), exportsym)

		case ir.ONAME:
			n := n.(*ir.Name)
			switch n.Class_ {
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
			// If the closure is inlinable, we'll need to
			// flood it too. But today we don't support
			// inlining functions that contain closures.
			//
			// When we do, we'll probably want:
			//     inlFlood(n.Func.Closure.Func.Nname)
			base.Fatalf("unexpected closure in inlinable function")
		}
	})
}

// hairyVisitor visits a function body to determine its inlining
// hairiness and whether or not it can be inlined.
type hairyVisitor struct {
	budget        int32
	reason        string
	extraCallCost int32
	usedLocals    map[*ir.Name]bool
	do            func(ir.Node) error
}

var errBudget = errors.New("too expensive")

func (v *hairyVisitor) tooHairy(fn *ir.Func) bool {
	v.do = v.doNode // cache closure

	err := errChildren(fn, v.do)
	if err != nil {
		v.reason = err.Error()
		return true
	}
	if v.budget < 0 {
		v.reason = fmt.Sprintf("function too complex: cost %d exceeds budget %d", inlineMaxBudget-v.budget, inlineMaxBudget)
		return true
	}
	return false
}

func (v *hairyVisitor) doNode(n ir.Node) error {
	if n == nil {
		return nil
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
			if name.Class_ == ir.PFUNC && types.IsRuntimePkg(name.Sym().Pkg) {
				fn := name.Sym().Name
				if fn == "getcallerpc" || fn == "getcallersp" {
					return errors.New("call to " + fn)
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
		v.budget -= inlineExtraPanicCost

	case ir.ORECOVER:
		// recover matches the argument frame pointer to find
		// the right panic value, so it needs an argument frame.
		return errors.New("call to recover")

	case ir.OCLOSURE,
		ir.ORANGE,
		ir.OSELECT,
		ir.OGO,
		ir.ODEFER,
		ir.ODCLTYPE, // can't print yet
		ir.ORETJMP:
		return errors.New("unhandled op " + n.Op().String())

	case ir.OAPPEND:
		v.budget -= inlineExtraAppendCost

	case ir.ODCLCONST, ir.OFALL:
		// These nodes don't produce code; omit from inlining budget.
		return nil

	case ir.OFOR, ir.OFORUNTIL:
		n := n.(*ir.ForStmt)
		if n.Label != nil {
			return errors.New("labeled control")
		}
	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		if n.Label != nil {
			return errors.New("labeled control")
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
			if err := errList(n.Init(), v.do); err != nil {
				return err
			}
			if err := errList(n.Body, v.do); err != nil {
				return err
			}
			if err := errList(n.Else, v.do); err != nil {
				return err
			}
			return nil
		}

	case ir.ONAME:
		n := n.(*ir.Name)
		if n.Class_ == ir.PAUTO {
			v.usedLocals[n] = true
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
		return errBudget
	}

	return errChildren(n, v.do)
}

func isBigFunc(fn *ir.Func) bool {
	budget := inlineBigFunctionNodes
	return ir.Any(fn, func(n ir.Node) bool {
		budget--
		return budget <= 0
	})
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
			as.Rhs.Set(inlconv2list(as.Rhs[0].(*ir.InlinedCallExpr)))
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
		if fn.Class_ == ir.PFUNC {
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

func inlParam(t *types.Field, as ir.Node, inlvars map[*ir.Name]ir.Node) ir.Node {
	n := ir.AsNode(t.Nname)
	if n == nil || ir.IsBlank(n) {
		return ir.BlankNode
	}

	inlvar := inlvars[n.(*ir.Name)]
	if inlvar == nil {
		base.Fatalf("missing inlvar for %v", n)
	}
	as.PtrInit().Append(ir.NewDecl(base.Pos, ir.ODCL, inlvar.(*ir.Name)))
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
			ninit.Append(conv.PtrInit().Take()...)
			callee = conv.X
		}
		if callee.Op() != ir.ONAME && callee.Op() != ir.OCLOSURE && callee.Op() != ir.OMETHEXPR {
			base.Fatalf("unexpected callee expression: %v", callee)
		}
	}

	// Make temp names to use instead of the originals.
	inlvars := make(map[*ir.Name]ir.Node)

	// record formals/locals for later post-processing
	var inlfvars []ir.Node

	for _, ln := range fn.Inl.Dcl {
		if ln.Op() != ir.ONAME {
			continue
		}
		if ln.Class_ == ir.PPARAMOUT { // return values handled below.
			continue
		}
		if ir.IsParamStackCopy(ln) { // ignore the on-stack copy of a parameter that moved to the heap
			// TODO(mdempsky): Remove once I'm confident
			// this never actually happens. We currently
			// perform inlining before escape analysis, so
			// nothing should have moved to the heap yet.
			base.Fatalf("impossible: %v", ln)
		}
		inlf := typecheck.Expr(inlvar(ln))
		inlvars[ln] = inlf
		if base.Flag.GenDwarfInl > 0 {
			if ln.Class_ == ir.PPARAM {
				inlf.Name().SetInlFormal(true)
			} else {
				inlf.Name().SetInlLocal(true)
			}
			inlf.SetPos(ln.Pos())
			inlfvars = append(inlfvars, inlf)
		}
	}

	nreturns := 0
	ir.VisitList(ir.Nodes(fn.Inl.Body), func(n ir.Node) {
		if n != nil && n.Op() == ir.ORETURN {
			nreturns++
		}
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
			m = typecheck.Expr(m)
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
			lit := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(param.Type).(ir.Ntype), nil)
			lit.List.Set(varargs)
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
	call.PtrInit().Set(ninit)
	call.Body.Set(body)
	call.ReturnVars.Set(retvars)
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
func inlvar(var_ ir.Node) ir.Node {
	if base.Flag.LowerM > 3 {
		fmt.Printf("inlvar %+v\n", var_)
	}

	n := typecheck.NewName(var_.Sym())
	n.SetType(var_.Type())
	n.Class_ = ir.PAUTO
	n.SetUsed(true)
	n.Curfn = ir.CurFunc // the calling function, not the called one
	n.SetAddrtaken(var_.Name().Addrtaken())

	ir.CurFunc.Dcl = append(ir.CurFunc.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *types.Field, i int) ir.Node {
	n := typecheck.NewName(typecheck.LookupNum("~R", i))
	n.SetType(t.Type)
	n.Class_ = ir.PAUTO
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
	n.Class_ = ir.PAUTO
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

	inlvars map[*ir.Name]ir.Node

	// bases maps from original PosBase to PosBase with an extra
	// inlined call frame.
	bases map[*src.PosBase]*src.PosBase

	// newInlIndex is the index of the inlined call frame to
	// insert for inlined nodes.
	newInlIndex int

	edit func(ir.Node) ir.Node // cached copy of subst.node method value closure
}

// list inlines a list of nodes.
func (subst *inlsubst) list(ll ir.Nodes) []ir.Node {
	s := make([]ir.Node, 0, len(ll))
	for _, n := range ll {
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

		// Handle captured variables when inlining closures.
		if n.IsClosureVar() {
			o := n.Outer

			// make sure the outer param matches the inlining location
			// NB: if we enabled inlining of functions containing OCLOSURE or refined
			// the reassigned check via some sort of copy propagation this would most
			// likely need to be changed to a loop to walk up to the correct Param
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
		if n, ok := n.(*ir.Name); ok && n.Op() == ir.OLITERAL {
			// This happens for unnamed OLITERAL.
			// which should really not be a *Name, but for now it is.
			// ir.Copy(n) is not allowed generally and would panic below,
			// but it's OK in this situation.
			n = n.CloneName()
			n.SetPos(subst.updatedPos(n.Pos()))
			return n
		}

	case ir.ORETURN:
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
			as.Rhs.Set(subst.list(n.Results))

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
		m.PtrInit().Set(nil)
		p := fmt.Sprintf("%s·%d", n.Label.Name, inlgen)
		m.Label = typecheck.Lookup(p)
		return m

	case ir.OLABEL:
		n := n.(*ir.LabelStmt)
		m := ir.Copy(n).(*ir.LabelStmt)
		m.SetPos(subst.updatedPos(m.Pos()))
		m.PtrInit().Set(nil)
		p := fmt.Sprintf("%s·%d", n.Label.Name, inlgen)
		m.Label = typecheck.Lookup(p)
		return m
	}

	if n.Op() == ir.OCLOSURE {
		base.Fatalf("cannot inline function containing closure: %+v", n)
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
		if n.Class_ == ir.PAUTO {
			if _, found := vis.usedLocals[n]; !found {
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

// TODO(mdempsky): Update inl.go to use ir.DoChildren directly.
func errChildren(n ir.Node, do func(ir.Node) error) (err error) {
	ir.DoChildren(n, func(x ir.Node) bool {
		err = do(x)
		return err != nil
	})
	return
}
func errList(list []ir.Node, do func(ir.Node) error) error {
	for _, x := range list {
		if x != nil {
			if err := do(x); err != nil {
				return err
			}
		}
	}
	return nil
}
