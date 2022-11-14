// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"fmt"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

func EnableNoWriteBarrierRecCheck() {
	nowritebarrierrecCheck = newNowritebarrierrecChecker()
}

func NoWriteBarrierRecCheck() {
	// Write barriers are now known. Check the
	// call graph.
	nowritebarrierrecCheck.check()
	nowritebarrierrecCheck = nil
}

var nowritebarrierrecCheck *nowritebarrierrecChecker

type nowritebarrierrecChecker struct {
	// extraCalls contains extra function calls that may not be
	// visible during later analysis. It maps from the ODCLFUNC of
	// the caller to a list of callees.
	extraCalls map[*ir.Func][]nowritebarrierrecCall

	// curfn is the current function during AST walks.
	curfn *ir.Func
}

type nowritebarrierrecCall struct {
	target *ir.Func // caller or callee
	lineno src.XPos // line of call
}

// newNowritebarrierrecChecker creates a nowritebarrierrecChecker. It
// must be called before walk
func newNowritebarrierrecChecker() *nowritebarrierrecChecker {
	c := &nowritebarrierrecChecker{
		extraCalls: make(map[*ir.Func][]nowritebarrierrecCall),
	}

	// Find all systemstack calls and record their targets. In
	// general, flow analysis can't see into systemstack, but it's
	// important to handle it for this check, so we model it
	// directly. This has to happen before transforming closures in walk since
	// it's a lot harder to work out the argument after.
	for _, n := range typecheck.Target.Decls {
		if n.Op() != ir.ODCLFUNC {
			continue
		}
		c.curfn = n.(*ir.Func)
		if c.curfn.ABIWrapper() {
			// We only want "real" calls to these
			// functions, not the generated ones within
			// their own ABI wrappers.
			continue
		}
		ir.Visit(n, c.findExtraCalls)
	}
	c.curfn = nil
	return c
}

func (c *nowritebarrierrecChecker) findExtraCalls(nn ir.Node) {
	if nn.Op() != ir.OCALLFUNC {
		return
	}
	n := nn.(*ir.CallExpr)
	if n.X == nil || n.X.Op() != ir.ONAME {
		return
	}
	fn := n.X.(*ir.Name)
	if fn.Class != ir.PFUNC || fn.Defn == nil {
		return
	}
	if !types.IsRuntimePkg(fn.Sym().Pkg) || fn.Sym().Name != "systemstack" {
		return
	}

	var callee *ir.Func
	arg := n.Args[0]
	switch arg.Op() {
	case ir.ONAME:
		arg := arg.(*ir.Name)
		callee = arg.Defn.(*ir.Func)
	case ir.OCLOSURE:
		arg := arg.(*ir.ClosureExpr)
		callee = arg.Func
	default:
		base.Fatalf("expected ONAME or OCLOSURE node, got %+v", arg)
	}
	if callee.Op() != ir.ODCLFUNC {
		base.Fatalf("expected ODCLFUNC node, got %+v", callee)
	}
	c.extraCalls[c.curfn] = append(c.extraCalls[c.curfn], nowritebarrierrecCall{callee, n.Pos()})
}

// recordCall records a call from ODCLFUNC node "from", to function
// symbol "to" at position pos.
//
// This should be done as late as possible during compilation to
// capture precise call graphs. The target of the call is an LSym
// because that's all we know after we start SSA.
//
// This can be called concurrently for different from Nodes.
func (c *nowritebarrierrecChecker) recordCall(fn *ir.Func, to *obj.LSym, pos src.XPos) {
	// We record this information on the *Func so this is concurrent-safe.
	if fn.NWBRCalls == nil {
		fn.NWBRCalls = new([]ir.SymAndPos)
	}
	*fn.NWBRCalls = append(*fn.NWBRCalls, ir.SymAndPos{Sym: to, Pos: pos})
}

func (c *nowritebarrierrecChecker) check() {
	// We walk the call graph as late as possible so we can
	// capture all calls created by lowering, but this means we
	// only get to see the obj.LSyms of calls. symToFunc lets us
	// get back to the ODCLFUNCs.
	symToFunc := make(map[*obj.LSym]*ir.Func)
	// funcs records the back-edges of the BFS call graph walk. It
	// maps from the ODCLFUNC of each function that must not have
	// write barriers to the call that inhibits them. Functions
	// that are directly marked go:nowritebarrierrec are in this
	// map with a zero-valued nowritebarrierrecCall. This also
	// acts as the set of marks for the BFS of the call graph.
	funcs := make(map[*ir.Func]nowritebarrierrecCall)
	// q is the queue of ODCLFUNC Nodes to visit in BFS order.
	var q ir.NameQueue

	for _, n := range typecheck.Target.Decls {
		if n.Op() != ir.ODCLFUNC {
			continue
		}
		fn := n.(*ir.Func)

		symToFunc[fn.LSym] = fn

		// Make nowritebarrierrec functions BFS roots.
		if fn.Pragma&ir.Nowritebarrierrec != 0 {
			funcs[fn] = nowritebarrierrecCall{}
			q.PushRight(fn.Nname)
		}
		// Check go:nowritebarrier functions.
		if fn.Pragma&ir.Nowritebarrier != 0 && fn.WBPos.IsKnown() {
			base.ErrorfAt(fn.WBPos, "write barrier prohibited")
		}
	}

	// Perform a BFS of the call graph from all
	// go:nowritebarrierrec functions.
	enqueue := func(src, target *ir.Func, pos src.XPos) {
		if target.Pragma&ir.Yeswritebarrierrec != 0 {
			// Don't flow into this function.
			return
		}
		if _, ok := funcs[target]; ok {
			// Already found a path to target.
			return
		}

		// Record the path.
		funcs[target] = nowritebarrierrecCall{target: src, lineno: pos}
		q.PushRight(target.Nname)
	}
	for !q.Empty() {
		fn := q.PopLeft().Func

		// Check fn.
		if fn.WBPos.IsKnown() {
			var err strings.Builder
			call := funcs[fn]
			for call.target != nil {
				fmt.Fprintf(&err, "\n\t%v: called by %v", base.FmtPos(call.lineno), call.target.Nname)
				call = funcs[call.target]
			}
			base.ErrorfAt(fn.WBPos, "write barrier prohibited by caller; %v%s", fn.Nname, err.String())
			continue
		}

		// Enqueue fn's calls.
		for _, callee := range c.extraCalls[fn] {
			enqueue(fn, callee.target, callee.lineno)
		}
		if fn.NWBRCalls == nil {
			continue
		}
		for _, callee := range *fn.NWBRCalls {
			target := symToFunc[callee.Sym]
			if target != nil {
				enqueue(fn, target, callee.Pos)
			}
		}
	}
}
