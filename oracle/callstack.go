// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/token"

	"code.google.com/p/go.tools/oracle/json"
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

	seen := make(map[pointer.CallGraphNode]bool)
	var callstack []pointer.CallSite

	// Use depth-first search to find an arbitrary path from a
	// root to the target function.
	var search func(cgn pointer.CallGraphNode) bool
	search = func(cgn pointer.CallGraphNode) bool {
		if !seen[cgn] {
			seen[cgn] = true
			if cgn.Func() == target {
				return true
			}
			for callee, site := range callgraph[cgn] {
				if search(callee) {
					callstack = append(callstack, site)
					return true
				}
			}
		}
		return false
	}

	for toplevel := range callgraph[root] {
		if search(toplevel) {
			break
		}
	}

	return &callstackResult{
		target:    target,
		callstack: callstack,
	}, nil
}

type callstackResult struct {
	target    *ssa.Function
	callstack []pointer.CallSite
}

func (r *callstackResult) display(printf printfFunc) {
	if r.callstack != nil {
		printf(false, "Found a call path from root to %s", r.target)
		printf(r.target, "%s", r.target)
		for _, site := range r.callstack {
			printf(site, "%s from %s", site.Description(), site.Caller().Func())
		}
	} else {
		printf(r.target, "%s is unreachable in this analysis scope", r.target)
	}
}

func (r *callstackResult) toJSON(res *json.Result, fset *token.FileSet) {
	var callers []json.Caller
	for _, site := range r.callstack {
		callers = append(callers, json.Caller{
			Pos:    site.Caller().Func().Prog.Fset.Position(site.Pos()).String(),
			Caller: site.Caller().Func().String(),
			Desc:   site.Description(),
		})
	}
	res.Callstack = &json.CallStack{
		Pos:     r.target.Prog.Fset.Position(r.target.Pos()).String(),
		Target:  r.target.String(),
		Callers: callers,
	}
}
