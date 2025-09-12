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
	"internal/buildcfg"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/inline/inlheur"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/pgoir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/pgo"
	"cmd/internal/src"
)

// Inlining budget parameters, gathered in one place
const (
	inlineMaxBudget       = 80
	inlineExtraAppendCost = 0
	// default is to inline if there's at most one call. -l=4 overrides this by using 1 instead.
	inlineExtraCallCost  = 57              // 57 was benchmarked to provided most benefit with no bad surprises; see https://github.com/golang/go/issues/19348#issuecomment-439370742
	inlineParamCallCost  = 17              // calling a parameter only costs this much extra (inlining might expose a constant function)
	inlineExtraPanicCost = 1               // do not penalize inlining panics.
	inlineExtraThrowCost = inlineMaxBudget // with current (2018-05/1.11) code, inlining runtime.throw does not help.

	inlineBigFunctionNodes      = 5000                 // Functions with this many nodes are considered "big".
	inlineBigFunctionMaxCost    = 20                   // Max cost of inlinee when inlining into a "big" function.
	inlineClosureCalledOnceCost = 10 * inlineMaxBudget // if a closure is just called once, inline it.
)

var (
	// List of all hot callee nodes.
	// TODO(prattmic): Make this non-global.
	candHotCalleeMap = make(map[*pgoir.IRNode]struct{})

	// Set of functions that contain hot call sites.
	hasHotCall = make(map[*ir.Func]struct{})

	// List of all hot call sites. CallSiteInfo.Callee is always nil.
	// TODO(prattmic): Make this non-global.
	candHotEdgeMap = make(map[pgoir.CallSiteInfo]struct{})

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

func IsPgoHotFunc(fn *ir.Func, profile *pgoir.Profile) bool {
	if profile == nil {
		return false
	}
	if n, ok := profile.WeightedCG.IRNodes[ir.LinkFuncName(fn)]; ok {
		_, ok := candHotCalleeMap[n]
		return ok
	}
	return false
}

func HasPgoHotInline(fn *ir.Func) bool {
	_, has := hasHotCall[fn]
	return has
}

// PGOInlinePrologue records the hot callsites from ir-graph.
func PGOInlinePrologue(p *pgoir.Profile) {
	if base.Debug.PGOInlineCDFThreshold != "" {
		if s, err := strconv.ParseFloat(base.Debug.PGOInlineCDFThreshold, 64); err == nil && s >= 0 && s <= 100 {
			inlineCDFHotCallSiteThresholdPercent = s
		} else {
			base.Fatalf("invalid PGOInlineCDFThreshold, must be between 0 and 100")
		}
	}
	var hotCallsites []pgo.NamedCallEdge
	inlineHotCallSiteThresholdPercent, hotCallsites = hotNodesFromCDF(p)
	if base.Debug.PGODebug > 0 {
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
		if caller := p.WeightedCG.IRNodes[n.CallerName]; caller != nil && caller.AST != nil {
			csi := pgoir.CallSiteInfo{LineOffset: n.CallSiteOffset, Caller: caller.AST}
			candHotEdgeMap[csi] = struct{}{}
		}
	}

	if base.Debug.PGODebug >= 3 {
		fmt.Printf("hot-cg before inline in dot format:")
		p.PrintWeightedCallGraphDOT(inlineHotCallSiteThresholdPercent)
	}
}

// hotNodesFromCDF computes an edge weight threshold and the list of hot
// nodes that make up the given percentage of the CDF. The threshold, as
// a percent, is the lower bound of weight for nodes to be considered hot
// (currently only used in debug prints) (in case of equal weights,
// comparing with the threshold may not accurately reflect which nodes are
// considered hot).
func hotNodesFromCDF(p *pgoir.Profile) (float64, []pgo.NamedCallEdge) {
	cum := int64(0)
	for i, n := range p.NamedEdgeMap.ByWeight {
		w := p.NamedEdgeMap.Weight[n]
		cum += w
		if pgo.WeightInPercentage(cum, p.TotalWeight) > inlineCDFHotCallSiteThresholdPercent {
			// nodes[:i+1] to include the very last node that makes it to go over the threshold.
			// (Say, if the CDF threshold is 50% and one hot node takes 60% of weight, we want to
			// include that node instead of excluding it.)
			return pgo.WeightInPercentage(w, p.TotalWeight), p.NamedEdgeMap.ByWeight[:i+1]
		}
	}
	return 0, p.NamedEdgeMap.ByWeight
}

// CanInlineFuncs computes whether a batch of functions are inlinable.
func CanInlineFuncs(funcs []*ir.Func, profile *pgoir.Profile) {
	if profile != nil {
		PGOInlinePrologue(profile)
	}

	if base.Flag.LowerL == 0 {
		return
	}

	ir.VisitFuncsBottomUp(funcs, func(funcs []*ir.Func, recursive bool) {
		for _, fn := range funcs {
			CanInline(fn, profile)
			if inlheur.Enabled() {
				analyzeFuncProps(fn, profile)
			}
		}
	})
}

// inlineBudget determines the max budget for function 'fn' prior to
// analyzing the hairiness of the body of 'fn'. We pass in the pgo
// profile if available (which can change the budget), also a
// 'relaxed' flag, which expands the budget slightly to allow for the
// possibility that a call to the function might have its score
// adjusted downwards. If 'verbose' is set, then print a remark where
// we boost the budget due to PGO.
func inlineBudget(fn *ir.Func, profile *pgoir.Profile, relaxed bool, verbose bool) int32 {
	// Update the budget for profile-guided inlining.
	budget := int32(inlineMaxBudget)
	if IsPgoHotFunc(fn, profile) {
		budget = inlineHotMaxBudget
		if verbose {
			fmt.Printf("hot-node enabled increased budget=%v for func=%v\n", budget, ir.PkgFuncName(fn))
		}
	}
	if relaxed {
		budget += inlheur.BudgetExpansion(inlineMaxBudget)
	}
	if fn.ClosureParent != nil {
		// be very liberal here, if the closure is only called once, the budget is large
		budget = max(budget, inlineClosureCalledOnceCost)
	}
	return budget
}

// CanInline determines whether fn is inlineable.
// If so, CanInline saves copies of fn.Body and fn.Dcl in fn.Inl.
// fn and fn.Body will already have been typechecked.
func CanInline(fn *ir.Func, profile *pgoir.Profile) {
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

	reason = InlineImpossible(fn)
	if reason != "" {
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

	// Used a "relaxed" inline budget if the new inliner is enabled.
	relaxed := inlheur.Enabled()

	// Compute the inline budget for this func.
	budget := inlineBudget(fn, profile, relaxed, base.Debug.PGODebug > 0)

	// At this point in the game the function we're looking at may
	// have "stale" autos, vars that still appear in the Dcl list, but
	// which no longer have any uses in the function body (due to
	// elimination by deadcode). We'd like to exclude these dead vars
	// when creating the "Inline.Dcl" field below; to accomplish this,
	// the hairyVisitor below builds up a map of used/referenced
	// locals, and we use this map to produce a pruned Inline.Dcl
	// list. See issue 25459 for more context.

	visitor := hairyVisitor{
		curFunc:       fn,
		isBigFunc:     IsBigFunc(fn),
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
		Cost:            budget - visitor.budget,
		Dcl:             pruneUnusedAutos(n.Func.Dcl, &visitor),
		HaveDcl:         true,
		CanDelayResults: canDelayResults(fn),
	}
	if base.Flag.LowerM != 0 || logopt.Enabled() {
		noteInlinableFunc(n, fn, budget-visitor.budget)
	}
}

// noteInlinableFunc issues a message to the user that the specified
// function is inlinable.
func noteInlinableFunc(n *ir.Name, fn *ir.Func, cost int32) {
	if base.Flag.LowerM > 1 {
		fmt.Printf("%v: can inline %v with cost %d as: %v { %v }\n", ir.Line(fn), n, cost, fn.Type(), ir.Nodes(fn.Body))
	} else if base.Flag.LowerM != 0 {
		fmt.Printf("%v: can inline %v\n", ir.Line(fn), n)
	}
	// JSON optimization log output.
	if logopt.Enabled() {
		logopt.LogOpt(fn.Pos(), "canInlineFunction", "inline", ir.FuncName(fn), fmt.Sprintf("cost: %d", cost))
	}
}

// InlineImpossible returns a non-empty reason string if fn is impossible to
// inline regardless of cost or contents.
func InlineImpossible(fn *ir.Func) string {
	var reason string // reason, if any, that the function can not be inlined.
	if fn.Nname == nil {
		reason = "no name"
		return reason
	}

	// If marked "go:noinline", don't inline.
	if fn.Pragma&ir.Noinline != 0 {
		reason = "marked go:noinline"
		return reason
	}

	// If marked "go:norace" and -race compilation, don't inline.
	if base.Flag.Race && fn.Pragma&ir.Norace != 0 {
		reason = "marked go:norace with -race compilation"
		return reason
	}

	// If marked "go:nocheckptr" and -d checkptr compilation, don't inline.
	if base.Debug.Checkptr != 0 && fn.Pragma&ir.NoCheckPtr != 0 {
		reason = "marked go:nocheckptr"
		return reason
	}

	// If marked "go:cgo_unsafe_args", don't inline, since the function
	// makes assumptions about its argument frame layout.
	if fn.Pragma&ir.CgoUnsafeArgs != 0 {
		reason = "marked go:cgo_unsafe_args"
		return reason
	}

	// If marked as "go:uintptrkeepalive", don't inline, since the keep
	// alive information is lost during inlining.
	//
	// TODO(prattmic): This is handled on calls during escape analysis,
	// which is after inlining. Move prior to inlining so the keep-alive is
	// maintained after inlining.
	if fn.Pragma&ir.UintptrKeepAlive != 0 {
		reason = "marked as having a keep-alive uintptr argument"
		return reason
	}

	// If marked as "go:uintptrescapes", don't inline, since the escape
	// information is lost during inlining.
	if fn.Pragma&ir.UintptrEscapes != 0 {
		reason = "marked as having an escaping uintptr argument"
		return reason
	}

	// The nowritebarrierrec checker currently works at function
	// granularity, so inlining yeswritebarrierrec functions can confuse it
	// (#22342). As a workaround, disallow inlining them for now.
	if fn.Pragma&ir.Yeswritebarrierrec != 0 {
		reason = "marked go:yeswritebarrierrec"
		return reason
	}

	// If a local function has no fn.Body (is defined outside of Go), cannot inline it.
	// Imported functions don't have fn.Body but might have inline body in fn.Inl.
	if len(fn.Body) == 0 && !typecheck.HaveInlineBody(fn) {
		reason = "no function body"
		return reason
	}

	return ""
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
	for _, param := range fn.Type().Results() {
		if sym := param.Sym; sym != nil && !sym.IsBlank() {
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
	isBigFunc     bool
	budget        int32
	maxBudget     int32
	reason        string
	extraCallCost int32
	usedLocals    ir.NameSet
	do            func(ir.Node) bool
	profile       *pgoir.Profile
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

// doNode visits n and its children, updates the state in v, and returns true if
// n makes the current function too hairy for inlining.
func (v *hairyVisitor) doNode(n ir.Node) bool {
	if n == nil {
		return false
	}
opSwitch:
	switch n.Op() {
	// Call is okay if inlinable and we have the budget for the body.
	case ir.OCALLFUNC:
		n := n.(*ir.CallExpr)
		var cheap bool
		if n.Fun.Op() == ir.ONAME {
			name := n.Fun.(*ir.Name)
			if name.Class == ir.PFUNC {
				s := name.Sym()
				fn := s.Name
				switch s.Pkg.Path {
				case "internal/abi":
					switch fn {
					case "NoEscape":
						// Special case for internal/abi.NoEscape. It does just type
						// conversions to appease the escape analysis, and doesn't
						// generate code.
						cheap = true
					}
					if strings.HasPrefix(fn, "EscapeNonString[") {
						// internal/abi.EscapeNonString[T] is a compiler intrinsic
						// implemented in the escape analysis phase.
						cheap = true
					}
				case "internal/runtime/sys":
					switch fn {
					case "GetCallerPC", "GetCallerSP":
						// Functions that call GetCallerPC/SP can not be inlined
						// because users expect the PC/SP of the logical caller,
						// but GetCallerPC/SP returns the physical caller.
						v.reason = "call to " + fn
						return true
					}
				case "go.runtime":
					switch fn {
					case "throw":
						// runtime.throw is a "cheap call" like panic in normal code.
						v.budget -= inlineExtraThrowCost
						break opSwitch
					case "panicrangestate":
						cheap = true
					}
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
		if n.Fun.Op() == ir.OMETHEXPR {
			if meth := ir.MethodExprName(n.Fun); meth != nil {
				if fn := meth.Func; fn != nil {
					s := fn.Sym()
					if types.RuntimeSymName(s) == "heapBits.nextArena" {
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
				}
			}
		}

		// A call to a parameter is optimistically a cheap call, if it's a constant function
		// perhaps it will inline, it also can simplify escape analysis.
		extraCost := v.extraCallCost

		if n.Fun.Op() == ir.ONAME {
			name := n.Fun.(*ir.Name)
			if name.Class == ir.PFUNC {
				// Special case: on architectures that can do unaligned loads,
				// explicitly mark internal/byteorder methods as cheap,
				// because in practice they are, even though our inlining
				// budgeting system does not see that. See issue 42958.
				if base.Ctxt.Arch.CanMergeLoads && name.Sym().Pkg.Path == "internal/byteorder" {
					switch name.Sym().Name {
					case "LEUint64", "LEUint32", "LEUint16",
						"BEUint64", "BEUint32", "BEUint16",
						"LEPutUint64", "LEPutUint32", "LEPutUint16",
						"BEPutUint64", "BEPutUint32", "BEPutUint16",
						"LEAppendUint64", "LEAppendUint32", "LEAppendUint16",
						"BEAppendUint64", "BEAppendUint32", "BEAppendUint16":
						cheap = true
					}
				}
			}
			if name.Class == ir.PPARAM || name.Class == ir.PAUTOHEAP && name.IsClosureVar() {
				extraCost = min(extraCost, inlineParamCallCost)
			}
		}

		if cheap {
			break // treat like any other node, that is, cost of 1
		}

		if ir.IsIntrinsicCall(n) {
			// Treat like any other node.
			break
		}

		if callee := inlCallee(v.curFunc, n.Fun, v.profile, false); callee != nil && typecheck.HaveInlineBody(callee) {
			// Check whether we'd actually inline this call. Set
			// log == false since we aren't actually doing inlining
			// yet.
			if ok, _, _ := canInlineCallExpr(v.curFunc, n, callee, v.isBigFunc, false, false); ok {
				// mkinlcall would inline this call [1], so use
				// the cost of the inline body as the cost of
				// the call, as that is what will actually
				// appear in the code.
				//
				// [1] This is almost a perfect match to the
				// mkinlcall logic, except that
				// canInlineCallExpr considers inlining cycles
				// by looking at what has already been inlined.
				// Since we haven't done any inlining yet we
				// will miss those.
				//
				// TODO: in the case of a single-call closure, the inlining budget here is potentially much, much larger.
				//
				v.budget -= callee.Inl.Cost
				break
			}
		}

		// Call cost for non-leaf inlining.
		v.budget -= extraCost

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
		// TODO: maybe we could allow inlining of recover() now?
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
		// TODO(austin): However, if we're able to inline this closure into
		// v.curFunc, then we actually pay nothing for the closure captures. We
		// should try to account for that if we're going to account for captures.
		v.budget -= 15

	case ir.OGO, ir.ODEFER, ir.OTAILCALL:
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

	case ir.OFALL, ir.OTYPE:
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
		// See also identical logic in IsBigFunc.
		if len(n.Rhs) > 0 {
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

	case ir.OSLICE, ir.OSLICEARR, ir.OSLICESTR, ir.OSLICE3, ir.OSLICE3ARR:
		n := n.(*ir.SliceExpr)

		// Ignore superfluous slicing.
		if n.Low != nil && n.Low.Op() == ir.OLITERAL && ir.Int64Val(n.Low) == 0 {
			v.budget++
		}
		if n.High != nil && n.High.Op() == ir.OLEN && n.High.(*ir.UnaryExpr).X == n.X {
			v.budget += 2
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

// IsBigFunc reports whether fn is a "big" function.
//
// Note: The criteria for "big" is heuristic and subject to change.
func IsBigFunc(fn *ir.Func) bool {
	budget := inlineBigFunctionNodes
	return ir.Any(fn, func(n ir.Node) bool {
		// See logic in hairyVisitor.doNode, explaining unified IR's
		// handling of "a, b = f()" assignments.
		if n, ok := n.(*ir.AssignListStmt); ok && n.Op() == ir.OAS2 && len(n.Rhs) > 0 {
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

// inlineCallCheck returns whether a call will never be inlineable
// for basic reasons, and whether the call is an intrinisic call.
// The intrinsic result singles out intrinsic calls for debug logging.
func inlineCallCheck(callerfn *ir.Func, call *ir.CallExpr) (bool, bool) {
	if base.Flag.LowerL == 0 {
		return false, false
	}
	if call.Op() != ir.OCALLFUNC {
		return false, false
	}
	if call.GoDefer || call.NoInline {
		return false, false
	}

	// Prevent inlining some reflect.Value methods when using checkptr,
	// even when package reflect was compiled without it (#35073).
	if base.Debug.Checkptr != 0 && call.Fun.Op() == ir.OMETHEXPR {
		if method := ir.MethodExprName(call.Fun); method != nil {
			switch types.ReflectSymName(method.Sym()) {
			case "Value.UnsafeAddr", "Value.Pointer":
				return false, false
			}
		}
	}

	// internal/abi.EscapeNonString[T] is a compiler intrinsic implemented
	// in the escape analysis phase.
	if fn := ir.StaticCalleeName(call.Fun); fn != nil && fn.Sym().Pkg.Path == "internal/abi" &&
		strings.HasPrefix(fn.Sym().Name, "EscapeNonString[") {
		return false, true
	}

	if ir.IsIntrinsicCall(call) {
		return false, true
	}
	return true, false
}

// InlineCallTarget returns the resolved-for-inlining target of a call.
// It does not necessarily guarantee that the target can be inlined, though
// obvious exclusions are applied.
func InlineCallTarget(callerfn *ir.Func, call *ir.CallExpr, profile *pgoir.Profile) *ir.Func {
	if mightInline, _ := inlineCallCheck(callerfn, call); !mightInline {
		return nil
	}
	return inlCallee(callerfn, call.Fun, profile, true)
}

// TryInlineCall returns an inlined call expression for call, or nil
// if inlining is not possible.
func TryInlineCall(callerfn *ir.Func, call *ir.CallExpr, bigCaller bool, profile *pgoir.Profile, closureCalledOnce bool) *ir.InlinedCallExpr {
	mightInline, isIntrinsic := inlineCallCheck(callerfn, call)

	// Preserve old logging behavior
	if (mightInline || isIntrinsic) && base.Flag.LowerM > 3 {
		fmt.Printf("%v:call to func %+v\n", ir.Line(call), call.Fun)
	}
	if !mightInline {
		return nil
	}

	if fn := inlCallee(callerfn, call.Fun, profile, false); fn != nil && typecheck.HaveInlineBody(fn) {
		return mkinlcall(callerfn, call, fn, bigCaller, closureCalledOnce)
	}
	return nil
}

// inlCallee takes a function-typed expression and returns the underlying function ONAME
// that it refers to if statically known. Otherwise, it returns nil.
// resolveOnly skips cost-based inlineability checks for closures; the result may not actually be inlineable.
func inlCallee(caller *ir.Func, fn ir.Node, profile *pgoir.Profile, resolveOnly bool) (res *ir.Func) {
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
		if len(c.ClosureVars) != 0 && c.ClosureVars[0].Outer.Curfn != caller {
			return nil // inliner doesn't support inlining across closure frames
		}
		if !resolveOnly {
			CanInline(c, profile)
		}
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
var InlineCall = func(callerfn *ir.Func, call *ir.CallExpr, fn *ir.Func, inlIndex int) *ir.InlinedCallExpr {
	base.Fatalf("inline.InlineCall not overridden")
	panic("unreachable")
}

// inlineCostOK returns true if call n from caller to callee is cheap enough to
// inline. bigCaller indicates that caller is a big function.
//
// In addition to the "cost OK" boolean, it also returns
//   - the "max cost" limit used to make the decision (which may differ depending on func size)
//   - the score assigned to this specific callsite
//   - whether the inlined function is "hot" according to PGO.
func inlineCostOK(n *ir.CallExpr, caller, callee *ir.Func, bigCaller, closureCalledOnce bool) (bool, int32, int32, bool) {
	maxCost := int32(inlineMaxBudget)

	if bigCaller {
		// We use this to restrict inlining into very big functions.
		// See issue 26546 and 17566.
		maxCost = inlineBigFunctionMaxCost
	}

	if callee.ClosureParent != nil {
		maxCost *= 2           // favor inlining closures
		if closureCalledOnce { // really favor inlining the one call to this closure
			maxCost = max(maxCost, inlineClosureCalledOnceCost)
		}
	}

	metric := callee.Inl.Cost
	if inlheur.Enabled() {
		score, ok := inlheur.GetCallSiteScore(caller, n)
		if ok {
			metric = int32(score)
		}
	}

	lineOffset := pgoir.NodeLineOffset(n, caller)
	csi := pgoir.CallSiteInfo{LineOffset: lineOffset, Caller: caller}
	_, hot := candHotEdgeMap[csi]

	if metric <= maxCost {
		// Simple case. Function is already cheap enough.
		return true, 0, metric, hot
	}

	// We'll also allow inlining of hot functions below inlineHotMaxBudget,
	// but only in small functions.

	if !hot {
		// Cold
		return false, maxCost, metric, false
	}

	// Hot

	if bigCaller {
		if base.Debug.PGODebug > 0 {
			fmt.Printf("hot-big check disallows inlining for call %s (cost %d) at %v in big function %s\n", ir.PkgFuncName(callee), callee.Inl.Cost, ir.Line(n), ir.PkgFuncName(caller))
		}
		return false, maxCost, metric, false
	}

	if metric > inlineHotMaxBudget {
		return false, inlineHotMaxBudget, metric, false
	}

	if !base.PGOHash.MatchPosWithInfo(n.Pos(), "inline", nil) {
		// De-selected by PGO Hash.
		return false, maxCost, metric, false
	}

	if base.Debug.PGODebug > 0 {
		fmt.Printf("hot-budget check allows inlining for call %s (cost %d) at %v in function %s\n", ir.PkgFuncName(callee), callee.Inl.Cost, ir.Line(n), ir.PkgFuncName(caller))
	}

	return true, 0, metric, hot
}

// parsePos returns all the inlining positions and the innermost position.
func parsePos(pos src.XPos, posTmp []src.Pos) ([]src.Pos, src.Pos) {
	ctxt := base.Ctxt
	ctxt.AllPos(pos, func(p src.Pos) {
		posTmp = append(posTmp, p)
	})
	l := len(posTmp) - 1
	return posTmp[:l], posTmp[l]
}

// canInlineCallExpr returns true if the call n from caller to callee
// can be inlined, plus the score computed for the call expr in question,
// and whether the callee is hot according to PGO.
// bigCaller indicates that caller is a big function. log
// indicates that the 'cannot inline' reason should be logged.
//
// Preconditions: CanInline(callee) has already been called.
func canInlineCallExpr(callerfn *ir.Func, n *ir.CallExpr, callee *ir.Func, bigCaller, closureCalledOnce bool, log bool) (bool, int32, bool) {
	if callee.Inl == nil {
		// callee is never inlinable.
		if log && logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(callerfn),
				fmt.Sprintf("%s cannot be inlined", ir.PkgFuncName(callee)))
		}
		return false, 0, false
	}

	ok, maxCost, callSiteScore, hot := inlineCostOK(n, callerfn, callee, bigCaller, closureCalledOnce)
	if !ok {
		// callee cost too high for this call site.
		if log && logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(callerfn),
				fmt.Sprintf("cost %d of %s exceeds max caller cost %d", callee.Inl.Cost, ir.PkgFuncName(callee), maxCost))
		}
		return false, 0, false
	}

	callees, calleeInner := parsePos(n.Pos(), make([]src.Pos, 0, 10))

	for _, p := range callees {
		if p.Line() == calleeInner.Line() && p.Col() == calleeInner.Col() && p.AbsFilename() == calleeInner.AbsFilename() {
			if log && logopt.Enabled() {
				logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", fmt.Sprintf("recursive call to %s", ir.FuncName(callerfn)))
			}
			return false, 0, false
		}
	}

	if base.Flag.Cfg.Instrumenting && types.IsNoInstrumentPkg(callee.Sym().Pkg) {
		// Runtime package must not be instrumented.
		// Instrument skips runtime package. However, some runtime code can be
		// inlined into other packages and instrumented there. To avoid this,
		// we disable inlining of runtime functions when instrumenting.
		// The example that we observed is inlining of LockOSThread,
		// which lead to false race reports on m contents.
		if log && logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(callerfn),
				fmt.Sprintf("call to runtime function %s in instrumented build", ir.PkgFuncName(callee)))
		}
		return false, 0, false
	}

	if base.Flag.Race && types.IsNoRacePkg(callee.Sym().Pkg) {
		if log && logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(callerfn),
				fmt.Sprintf(`call to into "no-race" package function %s in race build`, ir.PkgFuncName(callee)))
		}
		return false, 0, false
	}

	if base.Debug.Checkptr != 0 && types.IsRuntimePkg(callee.Sym().Pkg) {
		// We don't instrument runtime packages for checkptr (see base/flag.go).
		if log && logopt.Enabled() {
			logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(callerfn),
				fmt.Sprintf(`call to into runtime package function %s in -d=checkptr build`, ir.PkgFuncName(callee)))
		}
		return false, 0, false
	}

	// Check if we've already inlined this function at this particular
	// call site, in order to stop inlining when we reach the beginning
	// of a recursion cycle again. We don't inline immediately recursive
	// functions, but allow inlining if there is a recursion cycle of
	// many functions. Most likely, the inlining will stop before we
	// even hit the beginning of the cycle again, but this catches the
	// unusual case.
	parent := base.Ctxt.PosTable.Pos(n.Pos()).Base().InliningIndex()
	sym := callee.Linksym()
	for inlIndex := parent; inlIndex >= 0; inlIndex = base.Ctxt.InlTree.Parent(inlIndex) {
		if base.Ctxt.InlTree.InlinedFunction(inlIndex) == sym {
			if log {
				if base.Flag.LowerM > 1 {
					fmt.Printf("%v: cannot inline %v into %v: repeated recursive cycle\n", ir.Line(n), callee, ir.FuncName(callerfn))
				}
				if logopt.Enabled() {
					logopt.LogOpt(n.Pos(), "cannotInlineCall", "inline", ir.FuncName(callerfn),
						fmt.Sprintf("repeated recursive cycle to %s", ir.PkgFuncName(callee)))
				}
			}
			return false, 0, false
		}
	}

	return true, callSiteScore, hot
}

// mkinlcall returns an OINLCALL node that can replace OCALLFUNC n, or
// nil if it cannot be inlined. callerfn is the function that contains
// n, and fn is the function being called.
//
// The result of mkinlcall MUST be assigned back to n, e.g.
//
//	n.Left = mkinlcall(n.Left, fn, isddd)
func mkinlcall(callerfn *ir.Func, n *ir.CallExpr, fn *ir.Func, bigCaller, closureCalledOnce bool) *ir.InlinedCallExpr {
	ok, score, hot := canInlineCallExpr(callerfn, n, fn, bigCaller, closureCalledOnce, true)
	if !ok {
		return nil
	}
	if hot {
		hasHotCall[callerfn] = struct{}{}
	}
	typecheck.AssertFixedCall(n)

	parent := base.Ctxt.PosTable.Pos(n.Pos()).Base().InliningIndex()
	sym := fn.Linksym()
	inlIndex := base.Ctxt.InlTree.Add(parent, n.Pos(), sym, ir.FuncName(fn))

	closureInitLSym := func(n *ir.CallExpr, fn *ir.Func) {
		// The linker needs FuncInfo metadata for all inlined
		// functions. This is typically handled by gc.enqueueFunc
		// calling ir.InitLSym for all function declarations in
		// typecheck.Target.Decls (ir.UseClosure adds all closures to
		// Decls).
		//
		// However, closures in Decls are ignored, and are
		// instead enqueued when walk of the calling function
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

		var nf = n.Fun
		// Skips ir.OCONVNOPs, see issue #73716.
		for nf.Op() == ir.OCONVNOP {
			nf = nf.(*ir.ConvExpr).X
		}
		if nf.Op() != ir.OCLOSURE {
			// Not a direct closure call or one with type conversion.
			return
		}

		clo := nf.(*ir.ClosureExpr)
		if !clo.Func.IsClosure() {
			// enqueueFunc will handle non closures anyways.
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
		if buildcfg.Experiment.NewInliner {
			fmt.Printf("%v: inlining call to %v with score %d\n",
				ir.Line(n), fn, score)
		} else {
			fmt.Printf("%v: inlining call to %v\n", ir.Line(n), fn)
		}
	}
	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: Before inlining: %+v\n", ir.Line(n), n)
	}

	res := InlineCall(callerfn, n, fn, inlIndex)

	if res == nil {
		base.FatalfAt(n.Pos(), "inlining call to %v failed", fn)
	}

	if base.Flag.LowerM > 2 {
		fmt.Printf("%v: After inlining %+v\n\n", ir.Line(res), res)
	}

	if inlheur.Enabled() {
		inlheur.UpdateCallsiteTable(callerfn, n, res)
	}

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
				// TODO(mdempsky): Simplify code after confident that this
				// never happens anymore.
				base.FatalfAt(n.Pos(), "unused auto: %v", n)
				continue
			}
		}
		s = append(s, n)
	}
	return s
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
	// CoverageAuxVar implies either a coverage counter or a package
	// ID; since the cover tool never emits code to index into ID vars
	// this is effectively testing whether nn is a coverage counter.
	return nn.CoverageAuxVar()
}

// isAtomicCoverageCounterUpdate examines the specified node to
// determine whether it represents a call to sync/atomic.AddUint32 to
// increment a coverage counter.
func isAtomicCoverageCounterUpdate(cn *ir.CallExpr) bool {
	if cn.Fun.Op() != ir.ONAME {
		return false
	}
	name := cn.Fun.(*ir.Name)
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

func PostProcessCallSites(profile *pgoir.Profile) {
	if base.Debug.DumpInlCallSiteScores != 0 {
		budgetCallback := func(fn *ir.Func, prof *pgoir.Profile) (int32, bool) {
			v := inlineBudget(fn, prof, false, false)
			return v, v == inlineHotMaxBudget
		}
		inlheur.DumpInlCallSiteScores(profile, budgetCallback)
	}
}

func analyzeFuncProps(fn *ir.Func, p *pgoir.Profile) {
	canInline := func(fn *ir.Func) { CanInline(fn, p) }
	budgetForFunc := func(fn *ir.Func) int32 {
		return inlineBudget(fn, p, true, false)
	}
	inlheur.AnalyzeFunc(fn, canInline, budgetForFunc, inlineMaxBudget)
}
