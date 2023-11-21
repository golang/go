// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgo"
	"cmd/compile/internal/typecheck"
	"fmt"
	"os"
	"strings"
)

type callSiteAnalyzer struct {
	cstab    CallSiteTab
	fn       *ir.Func
	ptab     map[ir.Node]pstate
	nstack   []ir.Node
	loopNest int
	isInit   bool
}

func makeCallSiteAnalyzer(fn *ir.Func, cstab CallSiteTab, ptab map[ir.Node]pstate, loopNestingLevel int) *callSiteAnalyzer {
	isInit := fn.IsPackageInit() || strings.HasPrefix(fn.Sym().Name, "init.")
	return &callSiteAnalyzer{
		fn:       fn,
		cstab:    cstab,
		ptab:     ptab,
		isInit:   isInit,
		loopNest: loopNestingLevel,
		nstack:   []ir.Node{fn},
	}
}

// computeCallSiteTable builds and returns a table of call sites for
// the specified region in function fn. A region here corresponds to a
// specific subtree within the AST for a function. The main intended
// use cases are for 'region' to be either A) an entire function body,
// or B) an inlined call expression.
func computeCallSiteTable(fn *ir.Func, region ir.Nodes, cstab CallSiteTab, ptab map[ir.Node]pstate, loopNestingLevel int) CallSiteTab {
	csa := makeCallSiteAnalyzer(fn, cstab, ptab, loopNestingLevel)
	var doNode func(ir.Node) bool
	doNode = func(n ir.Node) bool {
		csa.nodeVisitPre(n)
		ir.DoChildren(n, doNode)
		csa.nodeVisitPost(n)
		return false
	}
	for _, n := range region {
		doNode(n)
	}
	return csa.cstab
}

func (csa *callSiteAnalyzer) flagsForNode(call *ir.CallExpr) CSPropBits {
	var r CSPropBits

	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= analyzing call at %s\n",
			fmtFullPos(call.Pos()))
	}

	// Set a bit if this call is within a loop.
	if csa.loopNest > 0 {
		r |= CallSiteInLoop
	}

	// Set a bit if the call is within an init function (either
	// compiler-generated or user-written).
	if csa.isInit {
		r |= CallSiteInInitFunc
	}

	// Decide whether to apply the panic path heuristic. Hack: don't
	// apply this heuristic in the function "main.main" (mostly just
	// to avoid annoying users).
	if !isMainMain(csa.fn) {
		r = csa.determinePanicPathBits(call, r)
	}

	return r
}

// determinePanicPathBits updates the CallSiteOnPanicPath bit within
// "r" if we think this call is on an unconditional path to
// panic/exit. Do this by walking back up the node stack to see if we
// can find either A) an enclosing panic, or B) a statement node that
// we've determined leads to a panic/exit.
func (csa *callSiteAnalyzer) determinePanicPathBits(call ir.Node, r CSPropBits) CSPropBits {
	csa.nstack = append(csa.nstack, call)
	defer func() {
		csa.nstack = csa.nstack[:len(csa.nstack)-1]
	}()

	for ri := range csa.nstack[:len(csa.nstack)-1] {
		i := len(csa.nstack) - ri - 1
		n := csa.nstack[i]
		_, isCallExpr := n.(*ir.CallExpr)
		_, isStmt := n.(ir.Stmt)
		if isCallExpr {
			isStmt = false
		}

		if debugTrace&debugTraceCalls != 0 {
			ps, inps := csa.ptab[n]
			fmt.Fprintf(os.Stderr, "=-= callpar %d op=%s ps=%s inptab=%v stmt=%v\n", i, n.Op().String(), ps.String(), inps, isStmt)
		}

		if n.Op() == ir.OPANIC {
			r |= CallSiteOnPanicPath
			break
		}
		if v, ok := csa.ptab[n]; ok {
			if v == psCallsPanic {
				r |= CallSiteOnPanicPath
				break
			}
			if isStmt {
				break
			}
		}
	}
	return r
}

// propsForArg returns property bits for a given call argument expression arg.
func (csa *callSiteAnalyzer) propsForArg(arg ir.Node) ActualExprPropBits {
	_, islit := isLiteral(arg)
	if islit {
		return ActualExprConstant
	}
	if isConcreteConvIface(arg) {
		return ActualExprIsConcreteConvIface
	}
	fname, isfunc, _ := isFuncName(arg)
	if isfunc {
		if fn := fname.Func; fn != nil && typecheck.HaveInlineBody(fn) {
			return ActualExprIsInlinableFunc
		}
		return ActualExprIsFunc
	}
	return 0
}

// argPropsForCall returns a slice of argument properties for the
// expressions being passed to the callee in the specific call
// expression; these will be stored in the CallSite object for a given
// call and then consulted when scoring. If no arg has any interesting
// properties we try to save some space and return a nil slice.
func (csa *callSiteAnalyzer) argPropsForCall(ce *ir.CallExpr) []ActualExprPropBits {
	rv := make([]ActualExprPropBits, len(ce.Args))
	somethingInteresting := false
	for idx := range ce.Args {
		argProp := csa.propsForArg(ce.Args[idx])
		somethingInteresting = somethingInteresting || (argProp != 0)
		rv[idx] = argProp
	}
	if !somethingInteresting {
		return nil
	}
	return rv
}

func (csa *callSiteAnalyzer) addCallSite(callee *ir.Func, call *ir.CallExpr) {
	flags := csa.flagsForNode(call)
	argProps := csa.argPropsForCall(call)
	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= props %+v for call %v\n", argProps, call)
	}
	// FIXME: maybe bulk-allocate these?
	cs := &CallSite{
		Call:     call,
		Callee:   callee,
		Assign:   csa.containingAssignment(call),
		ArgProps: argProps,
		Flags:    flags,
		ID:       uint(len(csa.cstab)),
	}
	if _, ok := csa.cstab[call]; ok {
		fmt.Fprintf(os.Stderr, "*** cstab duplicate entry at: %s\n",
			fmtFullPos(call.Pos()))
		fmt.Fprintf(os.Stderr, "*** call: %+v\n", call)
		panic("bad")
	}
	// Set initial score for callsite to the cost computed
	// by CanInline; this score will be refined later based
	// on heuristics.
	cs.Score = int(callee.Inl.Cost)

	if csa.cstab == nil {
		csa.cstab = make(CallSiteTab)
	}
	csa.cstab[call] = cs
	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= added callsite: caller=%v callee=%v n=%s\n",
			csa.fn, callee, fmtFullPos(call.Pos()))
	}
}

func (csa *callSiteAnalyzer) nodeVisitPre(n ir.Node) {
	switch n.Op() {
	case ir.ORANGE, ir.OFOR:
		if !hasTopLevelLoopBodyReturnOrBreak(loopBody(n)) {
			csa.loopNest++
		}
	case ir.OCALLFUNC:
		ce := n.(*ir.CallExpr)
		callee := pgo.DirectCallee(ce.Fun)
		if callee != nil && callee.Inl != nil {
			csa.addCallSite(callee, ce)
		}
	}
	csa.nstack = append(csa.nstack, n)
}

func (csa *callSiteAnalyzer) nodeVisitPost(n ir.Node) {
	csa.nstack = csa.nstack[:len(csa.nstack)-1]
	switch n.Op() {
	case ir.ORANGE, ir.OFOR:
		if !hasTopLevelLoopBodyReturnOrBreak(loopBody(n)) {
			csa.loopNest--
		}
	}
}

func loopBody(n ir.Node) ir.Nodes {
	if forst, ok := n.(*ir.ForStmt); ok {
		return forst.Body
	}
	if rst, ok := n.(*ir.RangeStmt); ok {
		return rst.Body
	}
	return nil
}

// hasTopLevelLoopBodyReturnOrBreak examines the body of a "for" or
// "range" loop to try to verify that it is a real loop, as opposed to
// a construct that is syntactically loopy but doesn't actually iterate
// multiple times, like:
//
//	for {
//	  blah()
//	  return 1
//	}
//
// [Remark: the pattern above crops up quite a bit in the source code
// for the compiler itself, e.g. the auto-generated rewrite code]
//
// Note that we don't look for GOTO statements here, so it's possible
// we'll get the wrong result for a loop with complicated control
// jumps via gotos.
func hasTopLevelLoopBodyReturnOrBreak(loopBody ir.Nodes) bool {
	for _, n := range loopBody {
		if n.Op() == ir.ORETURN || n.Op() == ir.OBREAK {
			return true
		}
	}
	return false
}

// containingAssignment returns the top-level assignment statement
// for a statement level function call "n". Examples:
//
//	x := foo()
//	x, y := bar(z, baz())
//	if blah() { ...
//
// Here the top-level assignment statement for the foo() call is the
// statement assigning to "x"; the top-level assignment for "bar()"
// call is the assignment to x,y. For the baz() and blah() calls,
// there is no top level assignment statement.
//
// The unstated goal here is that we want to use the containing
// assignment to establish a connection between a given call and the
// variables to which its results/returns are being assigned.
//
// Note that for the "bar" command above, the front end sometimes
// decomposes this into two assignments, the first one assigning the
// call to a pair of auto-temps, then the second one assigning the
// auto-temps to the user-visible vars. This helper will return the
// second (outer) of these two.
func (csa *callSiteAnalyzer) containingAssignment(n ir.Node) ir.Node {
	parent := csa.nstack[len(csa.nstack)-1]

	// assignsOnlyAutoTemps returns TRUE of the specified OAS2FUNC
	// node assigns only auto-temps.
	assignsOnlyAutoTemps := func(x ir.Node) bool {
		alst := x.(*ir.AssignListStmt)
		oa2init := alst.Init()
		if len(oa2init) == 0 {
			return false
		}
		for _, v := range oa2init {
			d := v.(*ir.Decl)
			if !ir.IsAutoTmp(d.X) {
				return false
			}
		}
		return true
	}

	// Simple case: x := foo()
	if parent.Op() == ir.OAS {
		return parent
	}

	// Multi-return case: x, y := bar()
	if parent.Op() == ir.OAS2FUNC {
		// Hack city: if the result vars are auto-temps, try looking
		// for an outer assignment in the tree. The code shape we're
		// looking for here is:
		//
		// OAS1({x,y},OCONVNOP(OAS2FUNC({auto1,auto2},OCALLFUNC(bar))))
		//
		if assignsOnlyAutoTemps(parent) {
			par2 := csa.nstack[len(csa.nstack)-2]
			if par2.Op() == ir.OAS2 {
				return par2
			}
			if par2.Op() == ir.OCONVNOP {
				par3 := csa.nstack[len(csa.nstack)-3]
				if par3.Op() == ir.OAS2 {
					return par3
				}
			}
		}
	}

	return nil
}

// UpdateCallsiteTable handles updating of callerfn's call site table
// after an inlined has been carried out, e.g. the call at 'n' as been
// turned into the inlined call expression 'ic' within function
// callerfn. The chief thing of interest here is to make sure that any
// call nodes within 'ic' are added to the call site table for
// 'callerfn' and scored appropriately.
func UpdateCallsiteTable(callerfn *ir.Func, n *ir.CallExpr, ic *ir.InlinedCallExpr) {
	enableDebugTraceIfEnv()
	defer disableDebugTrace()

	funcInlHeur, ok := fpmap[callerfn]
	if !ok {
		// This can happen for compiler-generated wrappers.
		if debugTrace&debugTraceCalls != 0 {
			fmt.Fprintf(os.Stderr, "=-= early exit, no entry for caller fn %v\n", callerfn)
		}
		return
	}

	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= UpdateCallsiteTable(caller=%v, cs=%s)\n",
			callerfn, fmtFullPos(n.Pos()))
	}

	// Mark the call in question as inlined.
	oldcs, ok := funcInlHeur.cstab[n]
	if !ok {
		// This can happen for compiler-generated wrappers.
		return
	}
	oldcs.aux |= csAuxInlined

	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= marked as inlined: callee=%v %s\n",
			oldcs.Callee, EncodeCallSiteKey(oldcs))
	}

	// Walk the inlined call region to collect new callsites.
	var icp pstate
	if oldcs.Flags&CallSiteOnPanicPath != 0 {
		icp = psCallsPanic
	}
	var loopNestLevel int
	if oldcs.Flags&CallSiteInLoop != 0 {
		loopNestLevel = 1
	}
	ptab := map[ir.Node]pstate{ic: icp}
	icstab := computeCallSiteTable(callerfn, ic.Body, nil, ptab, loopNestLevel)

	// Record parent callsite. This is primarily for debug output.
	for _, cs := range icstab {
		cs.parent = oldcs
	}

	// Score the calls in the inlined body. Note the setting of "doCallResults"
	// to false here: at the moment there isn't any easy way to localize
	// or region-ize the work done by "rescoreBasedOnCallResultUses", which
	// currently does a walk over the entire function to look for uses
	// of a given set of results.
	const doCallResults = false
	scoreCallsRegion(callerfn, ic.Body, icstab, doCallResults, ic)
}
