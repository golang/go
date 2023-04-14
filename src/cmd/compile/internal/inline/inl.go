// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// The inlining facility makes 2 passes: first CanInline determines which
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
	"sort"
	"strconv"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/pgo"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
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

var (
	// List of all hot callee nodes.
	// TODO(prattmic): Make this non-global.
	candHotCalleeMap = make(map[*pgo.IRNode]struct{})

	// List of all hot call sites. CallSiteInfo.Callee is always nil.
	// TODO(prattmic): Make this non-global.
	candHotEdgeMap = make(map[pgo.CallSiteInfo]struct{})

	// List of inlined call sites. CallSiteInfo.Callee is always nil.
	// TODO(prattmic): Make this non-global.
	inlinedCallSites = make(map[pgo.CallSiteInfo]struct{})

	// Threshold in percentage for hot callsite inlining.
	inlineHotCallSiteThresholdPercent float64

	// Threshold in CDF percentage for hot callsite inlining,
	// that is, for a threshold of X the hottest callsites that
	// make up the top X% of total edge weight will be
	// considered hot for inlining candidates.
	inlineCDFHotCallSiteThresholdPercent = float64(99)

	// Budget increased due to hotness.
	inlineHotMaxBudget int32 = 2000
)

// pgoInlinePrologue records the hot callsites from ir-graph.
func pgoInlinePrologue(p *pgo.Profile, decls []ir.Node) {
	if base.Debug.PGOInlineCDFThreshold != "" {
		if s, err := strconv.ParseFloat(base.Debug.PGOInlineCDFThreshold, 64); err == nil && s >= 0 && s <= 100 {
			inlineCDFHotCallSiteThresholdPercent = s
		} else {
			base.Fatalf("invalid PGOInlineCDFThreshold, must be between 0 and 100")
		}
	}
	var hotCallsites []pgo.NodeMapKey
	inlineHotCallSiteThresholdPercent, hotCallsites = hotNodesFromCDF(p)
	if base.Debug.PGOInline > 0 {
		fmt.Printf("hot-callsite-thres-from-CDF=%v\n", inlineHotCallSiteThresholdPercent)
	}

	if x := base.Debug.PGOInlineBudget; x != 0 {
		inlineHotMaxBudget = int32(x)
	}

	for _, n := range hotCallsites {
		// mark inlineable callees from hot edges
		if callee := p.WeightedCG.IRNodes[n.CalleeName]; callee != nil {
			candHotCalleeMap[callee] = struct{}{}
		}
		// mark hot call sites
		if caller := p.WeightedCG.IRNodes[n.CallerName]; caller != nil {
			csi := pgo.CallSiteInfo{LineOffset: n.CallSiteOffset, Caller: caller.AST}
			candHotEdgeMap[csi] = struct{}{}
		}
	}

	if base.Debug.PGOInline >= 2 {
		fmt.Printf("hot-cg before inline in dot format:")
		p.PrintWeightedCallGraphDOT(inlineHotCallSiteThresholdPercent)
	}
}

// hotNodesFromCDF computes an edge weight threshold and the list of hot
// nodes that make up the given percentage of the CDF. The threshold, as
// a percent, is the lower bound of weight for nodes to be considered hot
// (currently only used in debug prints) (in case of equal weights,
// comparing with the threshold may not accurately reflect which nodes are
// considiered hot).
func hotNodesFromCDF(p *pgo.Profile) (float64, []pgo.NodeMapKey) {
	nodes := make([]pgo.NodeMapKey, len(p.NodeMap))
	i := 0
	for n := range p.NodeMap {
		nodes[i] = n
		i++
	}
	sort.Slice(nodes, func(i, j int) bool {
		ni, nj := nodes[i], nodes[j]
		if wi, wj := p.NodeMap[ni].EWeight, p.NodeMap[nj].EWeight; wi != wj {
			return wi > wj // want larger weight first
		}
		// same weight, order by name/line number
		if ni.CallerName != nj.CallerName {
			return ni.CallerName < nj.CallerName
		}
		if ni.CalleeName != nj.CalleeName {
			return ni.CalleeName < nj.CalleeName
		}
		return ni.CallSiteOffset < nj.CallSiteOffset
	})
	cum := int64(0)
	for i, n := range nodes {
		w := p.NodeMap[n].EWeight
		cum += w
		if pgo.WeightInPercentage(cum, p.TotalEdgeWeight) > inlineCDFHotCallSiteThresholdPercent {
			// nodes[:i+1] to include the very last node that makes it to go over the threshold.
			// (Say, if the CDF threshold is 50% and one hot node takes 60% of weight, we want to
			// include that node instead of excluding it.)
			return pgo.WeightInPercentage(w, p.TotalEdgeWeight), nodes[:i+1]
		}
	}
	return 0, nodes
}

// pgoInlineEpilogue updates IRGraph after inlining.
func pgoInlineEpilogue(p *pgo.Profile, decls []ir.Node) {
	if base.Debug.PGOInline >= 2 {
		ir.VisitFuncsBottomUp(decls, func(list []*ir.Func, recursive bool) {
			for _, f := range list {
				name := ir.PkgFuncName(f)
				if n, ok := p.WeightedCG.IRNodes[name]; ok {
					p.RedirectEdges(n, inlinedCallSites)
				}
			}
		})
		// Print the call-graph after inlining. This is a debugging feature.
		fmt.Printf("hot-cg after inline in dot:")
		p.PrintWeightedCallGraphDOT(inlineHotCallSiteThresholdPercent)
	}
}

// InlinePackage finds functions that can be inlined and clones them before walk expands them.
func InlinePackage(p *pgo.Profile) {
	InlineDecls(p, typecheck.Target.Decls, true)
}

// InlineDecls applies inlining to the given batch of declarations.
func InlineDecls(p *pgo.Profile, decls []ir.Node, doInline bool) {
	if p != nil {
		pgoInlinePrologue(p, decls)
	}

	doCanInline := func(n *ir.Func, recursive bool, numfns int) {
		if !recursive || numfns > 1 {
			// We allow inlining if there is no
			// recursion, or the recursion cycle is
			// across more than one function.
			CanInline(n, p)
		} else {
			if base.Flag.LowerM > 1 && n.OClosure == nil {
				fmt.Printf("%v: cannot inline %v: recursive\n", ir.Line(n), n.Nname)
			}
		}
	}

	ir.VisitFuncsBottomUp(decls, func(list []*ir.Func, recursive bool) {
		numfns := numNonClosures(list)
		// We visit functions within an SCC in fairly arbitrary order,
		// so by computing inlinability for all functions in the SCC
		// before performing any inlining, the results are less
		// sensitive to the order within the SCC (see #58905 for an
		// example).
		if base.Debug.InlineSCCOnePass == 0 {
			// Compute inlinability for all functions in the SCC ...
			for _, n := range list {
				doCanInline(n, recursive, numfns)
			}
			// ... then make a second pass to do inlining of calls.
			if doInline {
				for _, n := range list {
					InlineCalls(n, p)
				}
			}
		} else {
			// Legacy ordering to make it easier to triage any bugs
			// or compile time issues that might crop up.
			for _, n := range list {
				doCanInline(n, recursive, numfns)
				if doInline {
					InlineCalls(n, p)
				}
			}
		}
	})

	// Rewalk post-inlining functions to check for closures that are
	// still visible but were (over-agressively) marked as dead, and
	// undo that marking here. See #59404 for more context.
	ir.VisitFuncsBottomUp(decls, func(list []*ir.Func, recursive bool) {
		for _, n := range list {
			ir.Visit(n, func(node ir.Node) {
				if clo, ok := node.(*ir.ClosureExpr); ok && clo.Func.IsHiddenClosure() {
					clo.Func.SetIsDeadcodeClosure(false)
				}
			})
		}
	})

	if p != nil {
		pgoInlineEpilogue(p, decls)
	}
}

// CanInline determines whether fn is inlineable.
// If so, CanInline saves copies of fn.Body and fn.Dcl in fn.Inl.
// fn and fn.Body will already have been typechecked.
func CanInline(fn *ir.Func, profile *pgo.Profile) {
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

	// If fn is synthetic hash or eq function, cannot inline it.
	// The function is not generated in Unified IR frontend at this moment.
	if ir.IsEqOrHashFunc(fn) {
		reason = "type eq/hash function"
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

	// Update the budget for profile-guided inlining.
	budget := int32(inlineMaxBudget)
	if profile != nil {
		if n, ok := profile.WeightedCG.IRNodes[ir.PkgFuncName(fn)]; ok {
			if _, ok := candHotCalleeMap[n]; ok {
				budget = int32(inlineHotMaxBudget)
				if base.Debug.PGOInline > 0 {
					fmt.Printf("hot-node enabled increased budget=%v for func=%v\n", budget, ir.PkgFuncName(fn))
				}
			}
		}
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
		curFunc:       fn,
		budget:        budget,
		maxBudget:     budget,
		extraCallCost: cc,
		profile:       profile,
	}
	if visitor.tooHairy(fn) {
		reason = visitor.reason
		return
	}

	n.Func.Inl = &ir.Inline{
		Cost: budget - visitor.budget,
		Dcl:  pruneUnusedAutos(n.Defn.(*ir.Func).Dcl, &visitor),
		Body: inlcopylist(fn.Body),

		CanDelayResults: canDelayResults(fn),
	}

	if base.Flag.LowerM > 1 {
		fmt.Printf("%v: can inline %v with cost %d as: %v { %v }\n", ir.Line(fn), n, budget-visitor.budget, fn.Type(), ir.Nodes(n.Func.Inl.Body))
	} else if base.Flag.LowerM != 0 {
		fmt.Printf("%v: can inline %v\n", ir.Line(fn), n)
	}
	if logopt.Enabled() {
		logopt.LogOpt(fn.Pos(), "canInlineFunction", "inline", ir.FuncName(fn), fmt.Sprintf("cost: %d", budget-visitor.budget))
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
	// This is needed to access the current caller in the doNode function.
	curFunc       *ir.Func
	budget        int32
	maxBudget     int32
	reason        string
	extraCallCost int32
	usedLocals    ir.NameSet
	do            func(ir.Node) bool
	profile       *pgo.Profile
}

func (v *hairyVisitor) tooHairy(fn *ir.Func) bool {
	v.do = v.doNode // cache closure
	if ir.DoChildren(fn, v.do) {
		return true
	}
	if v.budget < 0 {
		v.reason = fmt.Sprintf("function too complex: cost %d exceeds budget %d", v.maxBudget-v.budget, v.maxBudget)
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
			// Special case for coverage counter updates; although
			// these correspond to real operations, we treat them as
			// zero cost for the moment. This is due to the existence
			// of tests that are sensitive to inlining-- if the
			// insertion of coverage instrumentation happens to tip a
			// given function over the threshold and move it from
			// "inlinable" to "not-inlinable", this can cause changes
			// in allocation behavior, which can then result in test
			// failures (a good example is the TestAllocations in
			// crypto/ed25519).
			if isAtomicCoverageCounterUpdate(n) {
				return false
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
							"bigEndian.PutUint64", "bigEndian.PutUint32", "bigEndian.PutUint16",
							"littleEndian.AppendUint64", "littleEndian.AppendUint32", "littleEndian.AppendUint16",
							"bigEndian.AppendUint64", "bigEndian.AppendUint32", "bigEndian.AppendUint16":
							cheap = true
						}
					}
					if cheap {
						break // treat like any other node, that is, cost of 1
					}
				}
			}
		}

		// Determine if the callee edge is for an inlinable hot callee or not.
		if v.profile != nil && v.curFunc != nil {
			if fn := inlCallee(n.X, v.profile); fn != nil && typecheck.HaveInlineBody(fn) {
				lineOffset := pgo.NodeLineOffset(n, fn)
				csi := pgo.CallSiteInfo{LineOffset: lineOffset, Caller: v.curFunc}
				if _, o := candHotEdgeMap[csi]; o {
					if base.Debug.PGOInline > 0 {
						fmt.Printf("hot-callsite identified at line=%v for func=%v\n", ir.Line(n), ir.PkgFuncName(v.curFunc))
					}
				}
			}
		}

		if ir.IsIntrinsicCall(n) {
			// Treat like any other node.
			break
		}

		if fn := inlCallee(n.X, v.profile); fn != nil && typecheck.HaveInlineBody(fn) {
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

	case ir.OADDR:
		n := n.(*ir.AddrExpr)
		// Make "&s.f" cost 0 when f's offset is zero.
		if dot, ok := n.X.(*ir.SelectorExpr); ok && (dot.Op() == ir.ODOT || dot.Op() == ir.ODOTPTR) {
			if _, ok := dot.X.(*ir.Name); ok && dot.Selection.Offset == 0 {
				v.budget += 2 // undo ir.OADDR+ir.ODOT/ir.ODOTPTR
			}
		}

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
		//
		// See also identical logic in isBigFunc.
		if init := n.Rhs[0].Init(); len(init) == 1 {
			if _, ok := init[0].(*ir.AssignListStmt); ok {
				// 4 for each value, because each temporary variable now
				// appears 3 times (DCL, LHS, RHS), plus an extra DCL node.
				//
				// 1 for the extra "tmp1, tmp2 = f()" assignment statement.
				v.budget += 4*int32(len(n.Lhs)) + 1
			}
		}

	case ir.OAS:
		// Special case for coverage counter updates and coverage
		// function registrations. Although these correspond to real
		// operations, we treat them as zero cost for the moment. This
		// is primarily due to the existence of tests that are
		// sensitive to inlining-- if the insertion of coverage
		// instrumentation happens to tip a given function over the
		// threshold and move it from "inlinable" to "not-inlinable",
		// this can cause changes in allocation behavior, which can
		// then result in test failures (a good example is the
		// TestAllocations in crypto/ed25519).
		n := n.(*ir.AssignStmt)
		if n.X.Op() == ir.OINDEX && isIndexingCoverageCounter(n.X) {
			return false
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
		// See logic in hairyVisitor.doNode, explaining unified IR's
		// handling of "a, b = f()" assignments.
		if n, ok := n.(*ir.AssignListStmt); ok && n.Op() == ir.OAS2 {
			if init := n.Rhs[0].Init(); len(init) == 1 {
				if _, ok := init[0].(*ir.AssignListStmt); ok {
					budget += 4*len(n.Lhs) + 1
				}
			}
		}

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
func InlineCalls(fn *ir.Func, profile *pgo.Profile) {
	savefn := ir.CurFunc
	ir.CurFunc = fn
	bigCaller := isBigFunc(fn)
	if bigCaller && base.Flag.LowerM > 1 {
		fmt.Printf("%v: function %v considered 'big'; reducing max cost of inlinees\n", ir.Line(fn), fn)
	}
	var inlCalls []*ir.InlinedCallExpr
	var edit func(ir.Node) ir.Node
	edit = func(n ir.Node) ir.Node {
		return inlnode(n, bigCaller, &inlCalls, edit, profile)
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
func inlnode(n ir.Node, bigCaller bool, inlCalls *[]*ir.InlinedCallExpr, edit func(ir.Node) ir.Node, profile *pgo.Profile) ir.Node {
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
		if fn := inlCallee(call.X, profile); fn != nil && typecheck.HaveInlineBody(fn) {
			n = mkinlcall(call, fn, bigCaller, inlCalls, edit)
			if fn.IsHiddenClosure() {
				// Visit function to pick out any contained hidden
				// closures to mark them as dead, since they will no
				// longer be reachable (if we leave them live, they
				// will get skipped during escape analysis, which
				// could mean that go/defer statements don't get
				// desugared, causing later problems in walk). See
				// #59404 for more context. Note also that the code
				// below can sometimes be too aggressive (marking a closure
				// dead even though it was captured by a local var).
				// In this case we'll undo the dead marking in a cleanup
				// pass that happens at the end of InlineDecls.
				var vis func(node ir.Node)
				vis = func(node ir.Node) {
					if clo, ok := node.(*ir.ClosureExpr); ok && clo.Func.IsHiddenClosure() && !clo.Func.IsDeadcodeClosure() {
						if base.Flag.LowerM > 2 {
							fmt.Printf("%v: closure %v marked as dead\n", ir.Line(clo.Func), clo.Func)
						}
						clo.Func.SetIsDeadcodeClosure(true)
						ir.Visit(clo.Func, vis)
					}
				}
				ir.Visit(fn, vis)
			}
		}
	}

	base.Pos = lno

	return n
}

// inlCallee takes a function-typed expression and returns the underlying function ONAME
// that it refers to if statically known. Otherwise, it returns nil.
func inlCallee(fn ir.Node, profile *pgo.Profile) *ir.Func {
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
		CanInline(c, profile)
		return c
	}
	return nil
}

var inlgen int

// SSADumpInline gives the SSA back end a chance to dump the function
// when producing output for debugging the compiler itself.
var SSADumpInline = func(*ir.Func) {}

// InlineCall allows the inliner implementation to be overridden.
// If it returns nil, the function will not be inlined.
var InlineCall = func(call *ir.CallExpr, fn *ir.Func, inlIndex int) *ir.InlinedCallExpr {
	base.Fatalf("inline.InlineCall not overridden")
	panic("unreachable")
}

// inlineCostOK returns true if call n from caller to callee is cheap enough to
// inline. bigCaller indicates that caller is a big function.
//
// If inlineCostOK returns false, it also returns the max cost that the callee
// exceeded.
func inlineCostOK(n *ir.CallExpr, caller, callee *ir.Func, bigCaller bool) (bool, int32) {
	maxCost := int32(inlineMaxBudget)
	if bigCaller {
		// We use this to restrict inlining into very big functions.
		// See issue 26546 and 17566.
		maxCost = inlineBigFunctionMaxCost
	}

	if callee.Inl.Cost <= maxCost {
		// Simple case. Function is already cheap enough.
		return true, 0
	}

	// We'll also allow inlining of hot functions below inlineHotMaxBudget,
	// but only in small functions.

	lineOffset := pgo.NodeLineOffset(n, caller)
	csi := pgo.CallSiteInfo{LineOffset: lineOffset, Caller: caller}
	if _, ok := candHotEdgeMap[csi]; !ok {
		// Cold
		return false, maxCost
	}

	// Hot

	if bigCaller {
		if base.Debug.PGOInline > 0 {
			fmt.Printf("hot-big check disallows inlining for call %s (cost %d) at %v in big function %s\n", ir.PkgFuncName(callee), callee.Inl.Cost, ir.Line(n), ir.PkgFuncName(caller))
		}
		return false, maxCost
	}

	if callee.Inl.Cost > inlineHotMaxBudget {
		return false, inlineHotMaxBudget
	}

	if base.Debug.PGOInline > 0 {
		fmt.Printf("hot-budget check allows inlining for call %s (cost %d) at %v in function %s\n", ir.PkgFuncName(callee), callee.Inl.Cost, ir.Line(n), ir.PkgFuncName(caller))
	}

	return true, 0
}

// If n is a OCALLFUNC node, and fn is an ONAME node for a
// function with an inlinable body, return an OINLCALL node that can replace n.
// The returned node's Ninit has the parameter assignments, the Nbody is the
// inlined function body, and (List, Rlist) contain the (input, output)
// parameters.
// The result of mkinlcall MUST be assigned back to n, e.g.
//
//	n.Left = mkinlcall(n.Left, fn, isddd)
func mkinlcall(n *ir.CallExpr, fn *ir.Func, bigCaller bool, inlCalls *[]*ir.InlinedCallExpr, edit func(ir.Node) ir.Node) ir.Node {
	if fn.Inl == nil {
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(ir.CurFunc),
				fmt.Sprintf("%s cannot be inlined", ir.PkgFuncName(fn)))
		}
		return n
	}

	if ok, maxCost := inlineCostOK(n, ir.CurFunc, fn, bigCaller); !ok {
		if logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(ir.CurFunc),
			fmt.Sprintf("cost %d of %s exceeds max caller cost %d", fn.Inl.Cost, ir.PkgFuncName(fn), maxCost))
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

	typecheck.AssertFixedCall(n)

	inlIndex := base.Ctxt.InlTree.Add(parent, n.Pos(), sym)

	closureInitLSym := func(n *ir.CallExpr, fn *ir.Func) {
		// The linker needs FuncInfo metadata for all inlined
		// functions. This is typically handled by gc.enqueueFunc
		// calling ir.InitLSym for all function declarations in
		// typecheck.Target.Decls (ir.UseClosure adds all closures to
		// Decls).
		//
		// However, non-trivial closures in Decls are ignored, and are
		// insteaded enqueued when walk of the calling function
		// discovers them.
		//
		// This presents a problem for direct calls to closures.
		// Inlining will replace the entire closure definition with its
		// body, which hides the closure from walk and thus suppresses
		// symbol creation.
		//
		// Explicitly create a symbol early in this edge case to ensure
		// we keep this metadata.
		//
		// TODO: Refactor to keep a reference so this can all be done
		// by enqueueFunc.

		if n.Op() != ir.OCALLFUNC {
			// Not a standard call.
			return
		}
		if n.X.Op() != ir.OCLOSURE {
			// Not a direct closure call.
			return
		}

		clo := n.X.(*ir.ClosureExpr)
		if ir.IsTrivialClosure(clo) {
			// enqueueFunc will handle trivial closures anyways.
			return
		}

		ir.InitLSym(fn, true)
	}

	closureInitLSym(n, fn)

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

	if base.Debug.PGOInline > 0 {
		csi := pgo.CallSiteInfo{LineOffset: pgo.NodeLineOffset(n, fn), Caller: ir.CurFunc}
		if _, ok := inlinedCallSites[csi]; !ok {
			inlinedCallSites[csi] = struct{}{}
		}
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

// isIndexingCoverageCounter returns true if the specified node 'n' is indexing
// into a coverage counter array.
func isIndexingCoverageCounter(n ir.Node) bool {
	if n.Op() != ir.OINDEX {
		return false
	}
	ixn := n.(*ir.IndexExpr)
	if ixn.X.Op() != ir.ONAME || !ixn.X.Type().IsArray() {
		return false
	}
	nn := ixn.X.(*ir.Name)
	return nn.CoverageCounter()
}

// isAtomicCoverageCounterUpdate examines the specified node to
// determine whether it represents a call to sync/atomic.AddUint32 to
// increment a coverage counter.
func isAtomicCoverageCounterUpdate(cn *ir.CallExpr) bool {
	if cn.X.Op() != ir.ONAME {
		return false
	}
	name := cn.X.(*ir.Name)
	if name.Class != ir.PFUNC {
		return false
	}
	fn := name.Sym().Name
	if name.Sym().Pkg.Path != "sync/atomic" ||
		(fn != "AddUint32" && fn != "StoreUint32") {
		return false
	}
	if len(cn.Args) != 2 || cn.Args[0].Op() != ir.OADDR {
		return false
	}
	adn := cn.Args[0].(*ir.AddrExpr)
	v := isIndexingCoverageCounter(adn.X)
	return v
}
