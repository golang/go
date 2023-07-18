// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgo"
	"fmt"
	"os"
	"sort"
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

func makeCallSiteAnalyzer(fn *ir.Func, ptab map[ir.Node]pstate) *callSiteAnalyzer {
	isInit := fn.IsPackageInit() || strings.HasPrefix(fn.Sym().Name, "init.")
	return &callSiteAnalyzer{
		fn:     fn,
		cstab:  make(CallSiteTab),
		ptab:   ptab,
		isInit: isInit,
	}
}

func computeCallSiteTable(fn *ir.Func, ptab map[ir.Node]pstate) CallSiteTab {
	if debugTrace != 0 {
		fmt.Fprintf(os.Stderr, "=-= making callsite table for func %v:\n",
			fn.Sym().Name)
	}
	csa := makeCallSiteAnalyzer(fn, ptab)
	var doNode func(ir.Node) bool
	doNode = func(n ir.Node) bool {
		csa.nodeVisitPre(n)
		ir.DoChildren(n, doNode)
		csa.nodeVisitPost(n)
		return false
	}
	doNode(fn)
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

func (csa *callSiteAnalyzer) addCallSite(callee *ir.Func, call *ir.CallExpr) {
	flags := csa.flagsForNode(call)
	// FIXME: maybe bulk-allocate these?
	cs := &CallSite{
		Call:   call,
		Callee: callee,
		Assign: csa.containingAssignment(call),
		Flags:  flags,
		ID:     uint(len(csa.cstab)),
	}
	if _, ok := csa.cstab[call]; ok {
		fmt.Fprintf(os.Stderr, "*** cstab duplicate entry at: %s\n",
			fmtFullPos(call.Pos()))
		fmt.Fprintf(os.Stderr, "*** call: %+v\n", call)
		panic("bad")
	}
	if callee.Inl != nil {
		// Set initial score for callsite to the cost computed
		// by CanInline; this score will be refined later based
		// on heuristics.
		cs.Score = int(callee.Inl.Cost)
	}

	csa.cstab[call] = cs
	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= added callsite: callee=%s call=%v\n",
			callee.Sym().Name, callee)
	}
}

// ScoreCalls assigns numeric scores to each of the callsites in
// function 'fn'; the lower the score, the more helpful we think it
// will be to inline.
//
// Unlike a lot of the other inline heuristics machinery, callsite
// scoring can't be done as part of the CanInline call for a function,
// due to fact that we may be working on a non-trivial SCC. So for
// example with this SCC:
//
//	func foo(x int) {           func bar(x int, f func()) {
//	  if x != 0 {                  f()
//	    bar(x, func(){})           foo(x-1)
//	  }                         }
//	}
//
// We don't want to perform scoring for the 'foo' call in "bar" until
// after foo has been analyzed, but it's conceivable that CanInline
// might visit bar before foo for this SCC.
func ScoreCalls(fn *ir.Func) {
	enableDebugTraceIfEnv()
	defer disableDebugTrace()
	if debugTrace&debugTraceScoring != 0 {
		fmt.Fprintf(os.Stderr, "=-= ScoreCalls(%v)\n", ir.FuncName(fn))
	}

	fih, ok := fpmap[fn]
	if !ok {
		// TODO: add an assert/panic here.
		return
	}

	// Sort callsites to avoid any surprises with non deterministic
	// map iteration order (this is probably not needed, but here just
	// in case).
	csl := make([]*CallSite, 0, len(fih.cstab))
	for _, cs := range fih.cstab {
		csl = append(csl, cs)
	}
	sort.Slice(csl, func(i, j int) bool {
		return csl[i].ID < csl[j].ID
	})

	// Score each call site.
	for _, cs := range csl {
		var cprops *FuncProps
		fihcprops := false
		desercprops := false
		if fih, ok := fpmap[cs.Callee]; ok {
			cprops = fih.props
			fihcprops = true
		} else if cs.Callee.Inl != nil {
			cprops = DeserializeFromString(cs.Callee.Inl.Properties)
			desercprops = true
		} else {
			if base.Debug.DumpInlFuncProps != "" {
				fmt.Fprintf(os.Stderr, "=-= *** unable to score call to %s from %s\n", cs.Callee.Sym().Name, fmtFullPos(cs.Call.Pos()))
				panic("should never happen")
			} else {
				continue
			}
		}
		cs.Score, cs.ScoreMask = computeCallSiteScore(cs.Callee, cprops, cs.Call, cs.Flags)

		if debugTrace&debugTraceScoring != 0 {
			fmt.Fprintf(os.Stderr, "=-= scoring call at %s: flags=%d score=%d fih=%v deser=%v\n", fmtFullPos(cs.Call.Pos()), cs.Flags, cs.Score, fihcprops, desercprops)
		}
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
		callee := pgo.DirectCallee(ce.X)
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
// call is the assignment to x,y.   For the baz() and blah() calls,
// there is no top level assignment statement.
//
// The unstated goal here is that we want to use the containing assignment
// to establish a connection between a given call and the variables
// to which its results/returns are being assigned.
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
