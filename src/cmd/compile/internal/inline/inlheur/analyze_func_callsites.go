// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgoir"
	"cmd/compile/internal/typecheck"
	"fmt"
	"os"
	"strings"
)

type callSiteAnalyzer struct {
	fn *ir.Func
	*nameFinder
}

type callSiteTableBuilder struct {
	fn *ir.Func
	*nameFinder
	cstab    CallSiteTab
	ptab     map[ir.Node]pstate
	nstack   []ir.Node
	loopNest int
	isInit   bool
}

func makeCallSiteAnalyzer(fn *ir.Func) *callSiteAnalyzer {
	return &callSiteAnalyzer{
		fn:         fn,
		nameFinder: newNameFinder(fn),
	}
}

func makeCallSiteTableBuilder(fn *ir.Func, cstab CallSiteTab, ptab map[ir.Node]pstate, loopNestingLevel int, nf *nameFinder) *callSiteTableBuilder {
	isInit := fn.IsPackageInit() || strings.HasPrefix(fn.Sym().Name, "init.")
	return &callSiteTableBuilder{
		fn:         fn,
		cstab:      cstab,
		ptab:       ptab,
		isInit:     isInit,
		loopNest:   loopNestingLevel,
		nstack:     []ir.Node{fn},
		nameFinder: nf,
	}
}

// computeCallSiteTable builds and returns a table of call sites for
// the specified region in function fn. A region here corresponds to a
// specific subtree within the AST for a function. The main intended
// use cases are for 'region' to be either A) an entire function body,
// or B) an inlined call expression.
func computeCallSiteTable(fn *ir.Func, region ir.Nodes, cstab CallSiteTab, ptab map[ir.Node]pstate, loopNestingLevel int, nf *nameFinder) CallSiteTab {
	cstb := makeCallSiteTableBuilder(fn, cstab, ptab, loopNestingLevel, nf)
	var doNode func(ir.Node) bool
	doNode = func { n ->
		cstb.nodeVisitPre(n)
		ir.DoChildren(n, doNode)
		cstb.nodeVisitPost(n)
		return false
	}
	for _, n := range region {
		doNode(n)
	}
	return cstb.cstab
}

func (cstb *callSiteTableBuilder) flagsForNode(call *ir.CallExpr) CSPropBits {
	var r CSPropBits

	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= analyzing call at %s\n",
			fmtFullPos(call.Pos()))
	}

	// Set a bit if this call is within a loop.
	if cstb.loopNest > 0 {
		r |= CallSiteInLoop
	}

	// Set a bit if the call is within an init function (either
	// compiler-generated or user-written).
	if cstb.isInit {
		r |= CallSiteInInitFunc
	}

	// Decide whether to apply the panic path heuristic. Hack: don't
	// apply this heuristic in the function "main.main" (mostly just
	// to avoid annoying users).
	if !isMainMain(cstb.fn) {
		r = cstb.determinePanicPathBits(call, r)
	}

	return r
}

// determinePanicPathBits updates the CallSiteOnPanicPath bit within
// "r" if we think this call is on an unconditional path to
// panic/exit. Do this by walking back up the node stack to see if we
// can find either A) an enclosing panic, or B) a statement node that
// we've determined leads to a panic/exit.
func (cstb *callSiteTableBuilder) determinePanicPathBits(call ir.Node, r CSPropBits) CSPropBits {
	cstb.nstack = append(cstb.nstack, call)
	defer func() {
		cstb.nstack = cstb.nstack[:len(cstb.nstack)-1]
	}()

	for ri := range cstb.nstack[:len(cstb.nstack)-1] {
		i := len(cstb.nstack) - ri - 1
		n := cstb.nstack[i]
		_, isCallExpr := n.(*ir.CallExpr)
		_, isStmt := n.(ir.Stmt)
		if isCallExpr {
			isStmt = false
		}

		if debugTrace&debugTraceCalls != 0 {
			ps, inps := cstb.ptab[n]
			fmt.Fprintf(os.Stderr, "=-= callpar %d op=%s ps=%s inptab=%v stmt=%v\n", i, n.Op().String(), ps.String(), inps, isStmt)
		}

		if n.Op() == ir.OPANIC {
			r |= CallSiteOnPanicPath
			break
		}
		if v, ok := cstb.ptab[n]; ok {
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
func (cstb *callSiteTableBuilder) propsForArg(arg ir.Node) ActualExprPropBits {
	if cval := cstb.constValue(arg); cval != nil {
		return ActualExprConstant
	}
	if cstb.isConcreteConvIface(arg) {
		return ActualExprIsConcreteConvIface
	}
	fname := cstb.funcName(arg)
	if fname != nil {
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
func (cstb *callSiteTableBuilder) argPropsForCall(ce *ir.CallExpr) []ActualExprPropBits {
	rv := make([]ActualExprPropBits, len(ce.Args))
	somethingInteresting := false
	for idx := range ce.Args {
		argProp := cstb.propsForArg(ce.Args[idx])
		somethingInteresting = somethingInteresting || (argProp != 0)
		rv[idx] = argProp
	}
	if !somethingInteresting {
		return nil
	}
	return rv
}

func (cstb *callSiteTableBuilder) addCallSite(callee *ir.Func, call *ir.CallExpr) {
	flags := cstb.flagsForNode(call)
	argProps := cstb.argPropsForCall(call)
	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= props %+v for call %v\n", argProps, call)
	}
	// FIXME: maybe bulk-allocate these?
	cs := &CallSite{
		Call:     call,
		Callee:   callee,
		Assign:   cstb.containingAssignment(call),
		ArgProps: argProps,
		Flags:    flags,
		ID:       uint(len(cstb.cstab)),
	}
	if _, ok := cstb.cstab[call]; ok {
		fmt.Fprintf(os.Stderr, "*** cstab duplicate entry at: %s\n",
			fmtFullPos(call.Pos()))
		fmt.Fprintf(os.Stderr, "*** call: %+v\n", call)
		panic("bad")
	}
	// Set initial score for callsite to the cost computed
	// by CanInline; this score will be refined later based
	// on heuristics.
	cs.Score = int(callee.Inl.Cost)

	if cstb.cstab == nil {
		cstb.cstab = make(CallSiteTab)
	}
	cstb.cstab[call] = cs
	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= added callsite: caller=%v callee=%v n=%s\n",
			cstb.fn, callee, fmtFullPos(call.Pos()))
	}
}

func (cstb *callSiteTableBuilder) nodeVisitPre(n ir.Node) {
	switch n.Op() {
	case ir.ORANGE, ir.OFOR:
		if !hasTopLevelLoopBodyReturnOrBreak(loopBody(n)) {
			cstb.loopNest++
		}
	case ir.OCALLFUNC:
		ce := n.(*ir.CallExpr)
		callee := pgoir.DirectCallee(ce.Fun)
		if callee != nil && callee.Inl != nil {
			cstb.addCallSite(callee, ce)
		}
	}
	cstb.nstack = append(cstb.nstack, n)
}

func (cstb *callSiteTableBuilder) nodeVisitPost(n ir.Node) {
	cstb.nstack = cstb.nstack[:len(cstb.nstack)-1]
	switch n.Op() {
	case ir.ORANGE, ir.OFOR:
		if !hasTopLevelLoopBodyReturnOrBreak(loopBody(n)) {
			cstb.loopNest--
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
func (cstb *callSiteTableBuilder) containingAssignment(n ir.Node) ir.Node {
	parent := cstb.nstack[len(cstb.nstack)-1]

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
			par2 := cstb.nstack[len(cstb.nstack)-2]
			if par2.Op() == ir.OAS2 {
				return par2
			}
			if par2.Op() == ir.OCONVNOP {
				par3 := cstb.nstack[len(cstb.nstack)-3]
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
	nf := newNameFinder(nil)
	icstab := computeCallSiteTable(callerfn, ic.Body, nil, ptab, loopNestLevel, nf)

	// Record parent callsite. This is primarily for debug output.
	for _, cs := range icstab {
		cs.parent = oldcs
	}

	// Score the calls in the inlined body. Note the setting of
	// "doCallResults" to false here: at the moment there isn't any
	// easy way to localize or region-ize the work done by
	// "rescoreBasedOnCallResultUses", which currently does a walk
	// over the entire function to look for uses of a given set of
	// results. Similarly we're passing nil to makeCallSiteAnalyzer,
	// so as to run name finding without the use of static value &
	// friends.
	csa := makeCallSiteAnalyzer(nil)
	const doCallResults = false
	csa.scoreCallsRegion(callerfn, ic.Body, icstab, doCallResults, ic)
}
