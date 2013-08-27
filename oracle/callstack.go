// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// Callstack displays an arbitrary path from a root of the callgraph
// to the function at the current position.
//
// The information may be misleading in a context-insensitive
// analysis. e.g. the call path X->Y->Z might be infeasible if Y never
// calls Z when it is called from X.  TODO(adonovan): think about UI.
//
// TODO(adonovan): permit user to specify a starting point other than
// the analysis root.
//
func callstack(o *oracle) (queryResult, error) {
	pkg := o.prog.Package(o.queryPkgInfo.Pkg)
	if pkg == nil {
		return nil, o.errorf(o.queryPath[0], "no SSA package")
	}

	if !ssa.HasEnclosingFunction(pkg, o.queryPath) {
		return nil, o.errorf(o.queryPath[0], "this position is not inside a function")
	}

	buildSSA(o)

	target := ssa.EnclosingFunction(pkg, o.queryPath)
	if target == nil {
		return nil, o.errorf(o.queryPath[0],
			"no SSA function built for this location (dead code?)")
	}

	// Run the pointer analysis and build the complete call graph.
	callgraph := make(pointer.CallGraph)
	o.config.Call = callgraph.AddEdge
	root := ptrAnalysis(o)

	return &callstackResult{
		target:    target,
		root:      root,
		callgraph: callgraph,
	}, nil
}

type callstackResult struct {
	target    *ssa.Function
	root      pointer.CallGraphNode
	callgraph pointer.CallGraph

	seen map[pointer.CallGraphNode]bool // used by display
}

func (r *callstackResult) search(o *oracle, cgn pointer.CallGraphNode) bool {
	if !r.seen[cgn] {
		r.seen[cgn] = true
		if cgn.Func() == r.target {
			o.printf(o, "Found a call path from root to %s", r.target)
			o.printf(r.target, "%s", r.target)
			return true
		}
		for callee, site := range r.callgraph[cgn] {
			if r.search(o, callee) {
				o.printf(site, "%s from %s", site.Description(), cgn.Func())
				return true
			}
		}
	}
	return false
}

func (r *callstackResult) display(o *oracle) {
	// Show only an arbitrary path from a root to the current function.
	// We use depth-first search.

	r.seen = make(map[pointer.CallGraphNode]bool)

	for toplevel := range r.callgraph[r.root] {
		if r.search(o, toplevel) {
			return
		}
	}
	o.printf(r.target, "%s is unreachable in this analysis scope", r.target)
}
