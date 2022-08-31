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

// InlinePackage finds functions that can be inlined and clones them before walk expands them.
func InlinePackage() {
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
// If so, CanInline saves copies of fn.Body and fn.Dcl in fn.Inl.
// fn and fn.Body will already have been typechecked.
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

	// If marked as "go:uintptrkeepalive", don't inline, since the
	// keep alive information is lost during inlining.
	//
	// TODO(prattmic): This is handled on calls during escape analysis,
	// which is after inlining. Move prior to inlining so the keep-alive is
	// maintained after inlining.
	if fn.Pragma&ir.UintptrKeepAlive != 0 {
		reason = "marked as having a keep-alive uintptr argument"
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

		CanDelayResults: canDelayResults(fn),
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

// canDelayResults reports whether inlined calls to fn can delay
// declaring the result parameter until the "return" statement.
func canDelayResults(fn *ir.Func) bool {
	// We can delay declaring+initializing result parameters if:
	// (1) there's exactly one "return" statement in the inlined function;
	// (2) it's not an empty return statement (#44355); and
	// (3) the result parameters aren't named.

	nreturns := 0
	ir.VisitList(fn.Body, func(n ir.Node) {
		if n, ok := n.(*ir.ReturnStmt); ok {
			nreturns++
			if len(n.Results) == 0 {
				nreturns++ // empty return statement (case 2)
			}
		}
	})

	if nreturns != 1 {
		return false // not exactly one return statement (case 1)
	}

	// temporaries for return values.
	for _, param := range fn.Type().Results().FieldSlice() {
		if sym := types.OrigSym(param.Sym); sym != nil && !sym.IsBlank() {
			return false // found a named result parameter (case 3)
		}
	}

	return true
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
		if n.X.Op() == ir.OMETHEXPR {
			if meth := ir.MethodExprName(n.X); meth != nil {
				if fn := meth.Func; fn != nil {
					s := fn.Sym()
					var cheap bool
					if types.IsRuntimePkg(s.Pkg) && s.Name == "heapBits.nextArena" {
						// Special case: explicitly allow mid-stack inlining of
						// runtime.heapBits.next even though it calls slow-path
						// runtime.heapBits.nextArena.
						cheap = true
					}
					// Special case: on architectures that can do unaligned loads,
					// explicitly mark encoding/binary methods as cheap,
					// because in practice they are, even though our inlining
					// budgeting system does not see that. See issue 42958.
					if base.Ctxt.Arch.CanMergeLoads && s.Pkg.Path == "encoding/binary" {
						switch s.Name {
						case "littleEndian.Uint64", "littleEndian.Uint32", "littleEndian.Uint16",
							"bigEndian.Uint64", "bigEndian.Uint32", "bigEndian.Uint16",
							"littleEndian.PutUint64", "littleEndian.PutUint32", "littleEndian.PutUint16",
							"bigEndian.PutUint64", "bigEndian.PutUint32", "bigEndian.PutUint16":
							cheap = true
						}
					}
					if cheap {
						break // treat like any other node, that is, cost of 1
					}
				}
			}
		}

		if ir.IsIntrinsicCall(n) {
			// Treat like any other node.
			break
		}

		if fn := inlCallee(n.X); fn != nil && typecheck.HaveInlineBody(fn) {
			v.budget -= fn.Inl.Cost
			break
		}

		// Call cost for non-leaf inlining.
		v.budget -= v.extraCallCost

	case ir.OCALLMETH:
		base.FatalfAt(n.Pos(), "OCALLMETH missed by typecheck")

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

	case ir.OGO,
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

	case ir.OIF:
		n := n.(*ir.IfStmt)
		if ir.IsConst(n.Cond, constant.Bool) {
			// This if and the condition cost nothing.
			if doList(n.Init(), v.do) {
				return true
			}
			if ir.BoolVal(n.Cond) {
				return doList(n.Body, v.do)
			} else {
				return doList(n.Else, v.do)
			}
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

	case ir.OMETHVALUE, ir.OSLICELIT:
		v.budget-- // Hack for toolstash -cmp.

	case ir.OMETHEXPR:
		v.budget++ // Hack for toolstash -cmp.

	case ir.OAS2:
		n := n.(*ir.AssignListStmt)

		// Unified IR unconditionally rewrites:
		//
		//	a, b = f()
		//
		// into:
		//
		//	DCL tmp1
		//	DCL tmp2
		//	tmp1, tmp2 = f()
		//	a, b = tmp1, tmp2
		//
		// so that it can insert implicit conversions as necessary. To
		// minimize impact to the existing inlining heuristics (in
		// particular, to avoid breaking the existing inlinability regress
		// tests), we need to compensate for this here.
		if base.Debug.Unified != 0 {
			if init := n.Rhs[0].Init(); len(init) == 1 {
				if _, ok := init[0].(*ir.AssignListStmt); ok {
					// 4 for each value, because each temporary variable now
					// appears 3 times (DCL, LHS, RHS), plus an extra DCL node.
					//
					// 1 for the extra "tmp1, tmp2 = f()" assignment statement.
					v.budget += 4*int32(len(n.Lhs)) + 1
				}
			}
		}
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
			m.(*ir.ClosureExpr).Func = newfn
			newfn.Nname = ir.NewNameAt(oldfn.Nname.Pos(), oldfn.Nname.Sym())
			// XXX OK to share fn.Type() ??
			newfn.Nname.SetType(oldfn.Nname.Type())
			newfn.Body = inlcopylist(oldfn.Body)
			// Make shallow copy of the Dcl and ClosureVar slices
			newfn.Dcl = append([]*ir.Name(nil), oldfn.Dcl...)
			newfn.ClosureVars = append([]*ir.Name(nil), oldfn.ClosureVars...)
		}
		return m
	}
	return edit(n)
}

// InlineCalls/inlnode walks fn's statements and expressions and substitutes any
// calls made to inlineable functions. This is the external entry point.
func InlineCalls(fn *ir.Func) {
	savefn := ir.CurFunc
	ir.CurFunc = fn
	maxCost := int32(inlineMaxBudget)
	if isBigFunc(fn) {
		maxCost = inlineBigFunctionMaxCost
	}
	var inlCalls []*ir.InlinedCallExpr
	var edit func(ir.Node) ir.Node
	edit = func(n ir.Node) ir.Node {
		return inlnode(n, maxCost, &inlCalls, edit)
	}
	ir.EditChildren(fn, edit)

	// If we inlined any calls, we want to recursively visit their
	// bodies for further inlining. However, we need to wait until
	// *after* the original function body has been expanded, or else
	// inlCallee can have false positives (e.g., #54632).
	for len(inlCalls) > 0 {
		call := inlCalls[0]
		inlCalls = inlCalls[1:]
		ir.EditChildren(call, edit)
	}

	ir.CurFunc = savefn
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
//
//	n.Left = inlnode(n.Left)
func inlnode(n ir.Node, maxCost int32, inlCalls *[]*ir.InlinedCallExpr, edit func(ir.Node) ir.Node) ir.Node {
	if n == nil {
		return n
	}

	switch n.Op() {
	case ir.ODEFER, ir.OGO:
		n := n.(*ir.GoDeferStmt)
		switch call := n.Call; call.Op() {
		case ir.OCALLMETH:
			base.FatalfAt(call.Pos(), "OCALLMETH missed by typecheck")
		case ir.OCALLFUNC:
			call := call.(*ir.CallExpr)
			call.NoInline = true
		}
	case ir.OTAILCALL:
		n := n.(*ir.TailCallStmt)
		n.Call.NoInline = true // Not inline a tail call for now. Maybe we could inline it just like RETURN fn(arg)?

	// TODO do them here (or earlier),
	// so escape analysis can avoid more heapmoves.
	case ir.OCLOSURE:
		return n
	case ir.OCALLMETH:
		base.FatalfAt(n.Pos(), "OCALLMETH missed by typecheck")
	case ir.OCALLFUNC:
		n := n.(*ir.CallExpr)
		if n.X.Op() == ir.OMETHEXPR {
			// Prevent inlining some reflect.Value methods when using checkptr,
			// even when package reflect was compiled without it (#35073).
			if meth := ir.MethodExprName(n.X); meth != nil {
				s := meth.Sym()
				if base.Debug.Checkptr != 0 && types.IsReflectPkg(s.Pkg) && (s.Name == "Value.UnsafeAddr" || s.Name == "Value.Pointer") {
					return n
				}
			}
		}
	}

	lno := ir.SetPos(n)

	ir.EditChildren(n, edit)

	// with all the branches out of the way, it is now time to
	// transmogrify this node itself unless inhibited by the
	// switch at the top of this function.
	switch n.Op() {
	case ir.OCALLMETH:
		base.FatalfAt(n.Pos(), "OCALLMETH missed by typecheck")

	case ir.OCALLFUNC:
		call := n.(*ir.CallExpr)
		if call.NoInline {
			break
		}
		if base.Flag.LowerM > 3 {
			fmt.Printf("%v:call to func %+v\n", ir.Line(n), call.X)
		}
		if ir.IsIntrinsicCall(call) {
			break
		}
		if fn := inlCallee(call.X); fn != nil && typecheck.HaveInlineBody(fn) {
			n = mkinlcall(call, fn, maxCost, inlCalls, edit)
		}
	}

	base.Pos = lno

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

// InlineCall allows the inliner implementation to be overridden.
// If it returns nil, the function will not be inlined.
var InlineCall = oldInlineCall

// If n is a OCALLFUNC node, and fn is an ONAME node for a
// function with an inlinable body, return an OINLCALL node that can replace n.
// The returned node's Ninit has the parameter assignments, the Nbody is the
// inlined function body, and (List, Rlist) contain the (input, output)
// parameters.
// The result of mkinlcall MUST be assigned back to n, e.g.
//
//	n.Left = mkinlcall(n.Left, fn, isddd)
func mkinlcall(n *ir.CallExpr, fn *ir.Func, maxCost int32, inlCalls *[]*ir.InlinedCallExpr, edit func(ir.Node) ir.Node) ir.Node {
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

	// The non-unified frontend has issues with inlining and shape parameters.
	if base.Debug.Unified == 0 {
		// Don't inline a function fn that has no shape parameters, but is passed at
		// least one shape arg. This means we must be inlining a non-generic function
		// fn that was passed into a generic function, and can be called with a shape
		// arg because it matches an appropriate type parameters. But fn may include
		// an interface conversion (that may be applied to a shape arg) that was not
		// apparent when we first created the instantiation of the generic function.
		// We can't handle this if we actually do the inlining, since we want to know
		// all interface conversions immediately after stenciling. So, we avoid
		// inlining in this case, see issue #49309. (1)
		//
		// See discussion on go.dev/cl/406475 for more background.
		if !fn.Type().Params().HasShape() {
			for _, arg := range n.Args {
				if arg.Type().HasShape() {
					if logopt.Enabled() {
						logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(ir.CurFunc),
							fmt.Sprintf("inlining function %v has no-shape params with shape args", ir.FuncName(fn)))
					}
					return n
				}
			}
		} else {
			// Don't inline a function fn that has shape parameters, but is passed no shape arg.
			// See comments (1) above, and issue #51909.
			inlineable := len(n.Args) == 0 // Function has shape in type, with no arguments can always be inlined.
			for _, arg := range n.Args {
				if arg.Type().HasShape() {
					inlineable = true
					break
				}
			}
			if !inlineable {
				if logopt.Enabled() {
					logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(ir.CurFunc),
						fmt.Sprintf("inlining function %v has shape params with no-shape args", ir.FuncName(fn)))
				}
				return n
			}
		}
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

	parent := base.Ctxt.PosTable.Pos(n.Pos()).Base().InliningIndex()
	sym := fn.Linksym()

	// Check if we've already inlined this function at this particular
	// call site, in order to stop inlining when we reach the beginning
	// of a recursion cycle again. We don't inline immediately recursive
	// functions, but allow inlining if there is a recursion cycle of
	// many functions. Most likely, the inlining will stop before we
	// even hit the beginning of the cycle again, but this catches the
	// unusual case.
	for inlIndex := parent; inlIndex >= 0; inlIndex = base.Ctxt.InlTree.Parent(inlIndex) {
		if base.Ctxt.InlTree.InlinedFunction(inlIndex) == sym {
			if base.Flag.LowerM > 1 {
				fmt.Printf("%v: cannot inline %v into %v: repeated recursive cycle\n", ir.Line(n), fn, ir.FuncName(ir.CurFunc))
			}
			return n
		}
	}

	typecheck.FixVariadicCall(n)

	inlIndex := base.Ctxt.InlTree.Add(parent, n.Pos(), sym)

	if base.Flag.GenDwarfInl > 0 {
		if !sym.WasInlined() {
			base.Ctxt.DwFixups.SetPrecursorFunc(sym, fn)
			sym.Set(obj.AttrWasInlined, true)
		}
	}

	if base.Flag.LowerM != 0 {
		fmt.Printf("%v: inlining call to %v\n", ir.Line(n), fn)
	}
	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: Before inlining: %+v\n", ir.Line(n), n)
	}

	res := InlineCall(n, fn, inlIndex)
	if res == nil {
		base.FatalfAt(n.Pos(), "inlining call to %v failed", fn)
	}

	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: After inlining %+v\n\n", ir.Line(res), res)
	}

	*inlCalls = append(*inlCalls, res)

	return res
}

// CalleeEffects appends any side effects from evaluating callee to init.
func CalleeEffects(init *ir.Nodes, callee ir.Node) {
	for {
		init.Append(ir.TakeInit(callee)...)

		switch callee.Op() {
		case ir.ONAME, ir.OCLOSURE, ir.OMETHEXPR:
			return // done

		case ir.OCONVNOP:
			conv := callee.(*ir.ConvExpr)
			callee = conv.X

		case ir.OINLCALL:
			ic := callee.(*ir.InlinedCallExpr)
			init.Append(ic.Body.Take()...)
			callee = ic.SingleResult()

		default:
			base.FatalfAt(callee.Pos(), "unexpected callee expression: %v", callee)
		}
	}
}

// oldInlineCall creates an InlinedCallExpr to replace the given call
// expression. fn is the callee function to be inlined. inlIndex is
// the inlining tree position index, for use with src.NewInliningBase
// when rewriting positions.
func oldInlineCall(call *ir.CallExpr, fn *ir.Func, inlIndex int) *ir.InlinedCallExpr {
	if base.Debug.TypecheckInl == 0 {
		typecheck.ImportedBody(fn)
	}

	SSADumpInline(fn)

	ninit := call.Init()

	// For normal function calls, the function callee expression
	// may contain side effects. Make sure to preserve these,
	// if necessary (#42703).
	if call.Op() == ir.OCALLFUNC {
		CalleeEffects(&ninit, call.X)
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
	// temporaries for return values.
	var retvars []ir.Node
	for i, t := range fn.Type().Results().Fields().Slice() {
		var m *ir.Name
		if nn := t.Nname; nn != nil && !ir.IsBlank(nn.(*ir.Name)) && !strings.HasPrefix(nn.Sym().Name, "~r") {
			n := nn.(*ir.Name)
			m = inlvar(n)
			m = typecheck.Expr(m).(*ir.Name)
			inlvars[n] = m
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
	if call.Op() == ir.OCALLMETH {
		base.FatalfAt(call.Pos(), "OCALLMETH missed by typecheck")
	}
	as.Rhs.Append(call.Args...)

	if recv := fn.Type().Recv(); recv != nil {
		as.Lhs.Append(inlParam(recv, as, inlvars))
	}
	for _, param := range fn.Type().Params().Fields().Slice() {
		as.Lhs.Append(inlParam(param, as, inlvars))
	}

	if len(as.Rhs) != 0 {
		ninit.Append(typecheck.Stmt(as))
	}

	if !fn.Inl.CanDelayResults {
		// Zero the return parameters.
		for _, n := range retvars {
			ninit.Append(ir.NewDecl(base.Pos, ir.ODCL, n.(*ir.Name)))
			ras := ir.NewAssignStmt(base.Pos, n, nil)
			ninit.Append(typecheck.Stmt(ras))
		}
	}

	retlabel := typecheck.AutoLabel(".i")

	inlgen++

	// Add an inline mark just before the inlined body.
	// This mark is inline in the code so that it's a reasonable spot
	// to put a breakpoint. Not sure if that's really necessary or not
	// (in which case it could go at the end of the function instead).
	// Note issue 28603.
	ninit.Append(ir.NewInlineMarkStmt(call.Pos().WithIsStmt(), int64(inlIndex)))

	subst := inlsubst{
		retlabel:    retlabel,
		retvars:     retvars,
		inlvars:     inlvars,
		defnMarker:  ir.NilExpr{},
		bases:       make(map[*src.PosBase]*src.PosBase),
		newInlIndex: inlIndex,
		fn:          fn,
	}
	subst.edit = subst.node

	body := subst.list(ir.Nodes(fn.Inl.Body))

	lab := ir.NewLabelStmt(base.Pos, retlabel)
	body = append(body, lab)

	if base.Flag.GenDwarfInl > 0 {
		for _, v := range inlfvars {
			v.SetPos(subst.updatedPos(v.Pos()))
		}
	}

	//dumplist("ninit post", ninit);

	res := ir.NewInlinedCallExpr(base.Pos, body, retvars)
	res.SetInit(ninit)
	res.SetType(call.Type())
	res.SetTypecheck(1)
	return res
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
	n.SetTypecheck(1)
	n.Class = ir.PAUTO
	n.SetUsed(true)
	n.SetAutoTemp(var_.AutoTemp())
	n.Curfn = ir.CurFunc // the calling function, not the called one
	n.SetAddrtaken(var_.Addrtaken())

	ir.CurFunc.Dcl = append(ir.CurFunc.Dcl, n)
	return n
}

// Synthesize a variable to store the inlined function's results in.
func retvar(t *types.Field, i int) *ir.Name {
	n := typecheck.NewName(typecheck.LookupNum("~R", i))
	n.SetType(t.Type)
	n.SetTypecheck(1)
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

	inlvars map[*ir.Name]*ir.Name
	// defnMarker is used to mark a Node for reassignment.
	// inlsubst.clovar set this during creating new ONAME.
	// inlsubst.node will set the correct Defn for inlvar.
	defnMarker ir.NilExpr

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

	// If true, then don't update source positions during substitution
	// (retain old source positions).
	noPosUpdate bool
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
	m := ir.NewNameAt(n.Pos(), n.Sym())
	m.Class = n.Class
	m.SetType(n.Type())
	m.SetTypecheck(1)
	if n.IsClosureVar() {
		m.SetIsClosureVar(true)
	}
	if n.Addrtaken() {
		m.SetAddrtaken(true)
	}
	if n.Used() {
		m.SetUsed(true)
	}
	m.Defn = n.Defn

	m.Curfn = subst.newclofn

	switch defn := n.Defn.(type) {
	case nil:
		// ok
	case *ir.Name:
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
	case *ir.AssignStmt, *ir.AssignListStmt:
		// Mark node for reassignment at the end of inlsubst.node.
		m.Defn = &subst.defnMarker
	case *ir.TypeSwitchGuard:
		// TODO(mdempsky): Set m.Defn properly. See discussion on #45743.
	case *ir.RangeStmt:
		// TODO: Set m.Defn properly if we support inlining range statement in the future.
	default:
		base.FatalfAt(n.Pos(), "unexpected Defn: %+v", defn)
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
	// Prior to the subst edit, set a flag in the inlsubst to indicate
	// that we don't want to update the source positions in the new
	// closure function. If we do this, it will appear that the
	// closure itself has things inlined into it, which is not the
	// case. See issue #46234 for more details. At the same time, we
	// do want to update the position in the new ClosureExpr (which is
	// part of the function we're working on). See #49171 for an
	// example of what happens if we miss that update.
	newClosurePos := subst.updatedPos(n.Pos())
	defer func(prev bool) { subst.noPosUpdate = prev }(subst.noPosUpdate)
	subst.noPosUpdate = true

	//fmt.Printf("Inlining func %v with closure into %v\n", subst.fn, ir.FuncName(ir.CurFunc))

	oldfn := n.Func
	newfn := ir.NewClosureFunc(oldfn.Pos(), true)

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
	newclo := newfn.OClosure
	newclo.SetPos(newClosurePos)
	newclo.SetInit(subst.list(n.Init()))
	return typecheck.Expr(newclo)
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
		// Because of the above test for subst.newclofn,
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

			if subst.fn.Inl.CanDelayResults {
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

	case ir.OGOTO, ir.OBREAK, ir.OCONTINUE:
		if subst.newclofn != nil {
			// Don't do special substitutions if inside a closure
			break
		}
		n := n.(*ir.BranchStmt)
		m := ir.Copy(n).(*ir.BranchStmt)
		m.SetPos(subst.updatedPos(m.Pos()))
		m.SetInit(nil)
		m.Label = translateLabel(n.Label)
		return m

	case ir.OLABEL:
		if subst.newclofn != nil {
			// Don't do special substitutions if inside a closure
			break
		}
		n := n.(*ir.LabelStmt)
		m := ir.Copy(n).(*ir.LabelStmt)
		m.SetPos(subst.updatedPos(m.Pos()))
		m.SetInit(nil)
		m.Label = translateLabel(n.Label)
		return m

	case ir.OCLOSURE:
		return subst.closure(n.(*ir.ClosureExpr))

	}

	m := ir.Copy(n)
	m.SetPos(subst.updatedPos(m.Pos()))
	ir.EditChildren(m, subst.edit)

	if subst.newclofn == nil {
		// Translate any label on FOR, RANGE loops, SWITCH or SELECT
		switch m.Op() {
		case ir.OFOR:
			m := m.(*ir.ForStmt)
			m.Label = translateLabel(m.Label)
			return m

		case ir.ORANGE:
			m := m.(*ir.RangeStmt)
			m.Label = translateLabel(m.Label)
			return m

		case ir.OSWITCH:
			m := m.(*ir.SwitchStmt)
			m.Label = translateLabel(m.Label)
			return m

		case ir.OSELECT:
			m := m.(*ir.SelectStmt)
			m.Label = translateLabel(m.Label)
			return m
		}
	}

	switch m := m.(type) {
	case *ir.AssignStmt:
		if lhs, ok := m.X.(*ir.Name); ok && lhs.Defn == &subst.defnMarker {
			lhs.Defn = m
		}
	case *ir.AssignListStmt:
		for _, lhs := range m.Lhs {
			if lhs, ok := lhs.(*ir.Name); ok && lhs.Defn == &subst.defnMarker {
				lhs.Defn = m
			}
		}
	}

	return m
}

// translateLabel makes a label from an inlined function (if non-nil) be unique by
// adding "·inlgen".
func translateLabel(l *types.Sym) *types.Sym {
	if l == nil {
		return nil
	}
	p := fmt.Sprintf("%s·%d", l.Name, inlgen)
	return typecheck.Lookup(p)
}

func (subst *inlsubst) updatedPos(xpos src.XPos) src.XPos {
	if subst.noPosUpdate {
		return xpos
	}
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
