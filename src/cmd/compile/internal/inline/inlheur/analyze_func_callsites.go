// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/pgo"
	"fmt"
	"os"
)

type callSiteAnalyzer struct {
	cstab  CallSiteTab
	nstack []ir.Node
}

func makeCallSiteAnalyzer(fn *ir.Func) *callSiteAnalyzer {
	return &callSiteAnalyzer{
		cstab: make(CallSiteTab),
	}
}

func computeCallSiteTable(fn *ir.Func) CallSiteTab {
	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= making callsite table for func %v:\n",
			fn.Sym().Name)
	}
	csa := makeCallSiteAnalyzer(fn)
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
	return 0
}

func (csa *callSiteAnalyzer) addCallSite(callee *ir.Func, call *ir.CallExpr) {
	// FIXME: maybe bulk-allocate these?
	cs := &CallSite{
		Call:   call,
		Callee: callee,
		Assign: csa.containingAssignment(call),
		Flags:  csa.flagsForNode(call),
		Id:     uint(len(csa.cstab)),
	}
	if _, ok := csa.cstab[call]; ok {
		fmt.Fprintf(os.Stderr, "*** cstab duplicate entry at: %s\n",
			fmtFullPos(call.Pos()))
		fmt.Fprintf(os.Stderr, "*** call: %+v\n", call)
		panic("bad")
	}
	if debugTrace&debugTraceCalls != 0 {
		fmt.Fprintf(os.Stderr, "=-= added callsite: callee=%s call=%v\n",
			callee.Sym().Name, callee)
	}

	csa.cstab[call] = cs
}

func (csa *callSiteAnalyzer) nodeVisitPre(n ir.Node) {
	switch n.Op() {
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
