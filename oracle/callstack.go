// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/token"

	"code.google.com/p/go.tools/call"
	"code.google.com/p/go.tools/oracle/serial"
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
func callstack(o *Oracle, qpos *QueryPos) (queryResult, error) {
	pkg := o.prog.Package(qpos.info.Pkg)
	if pkg == nil {
		return nil, fmt.Errorf("no SSA package")
	}

	if !ssa.HasEnclosingFunction(pkg, qpos.path) {
		return nil, fmt.Errorf("this position is not inside a function")
	}

	buildSSA(o)

	target := ssa.EnclosingFunction(pkg, qpos.path)
	if target == nil {
		return nil, fmt.Errorf("no SSA function built for this location (dead code?)")
	}

	// Run the pointer analysis and build the complete call graph.
	o.ptaConfig.BuildCallGraph = true
	callgraph := ptrAnalysis(o).CallGraph

	// Search for an arbitrary path from a root to the target function.
	isEnd := func(n call.GraphNode) bool { return n.Func() == target }
	callpath := call.PathSearch(callgraph.Root(), isEnd)
	if callpath != nil {
		callpath = callpath[1:] // remove synthetic edge from <root>
	}

	return &callstackResult{
		qpos:     qpos,
		target:   target,
		callpath: callpath,
	}, nil
}

type callstackResult struct {
	qpos     *QueryPos
	target   *ssa.Function
	callpath []call.Edge
}

func (r *callstackResult) display(printf printfFunc) {
	if r.callpath != nil {
		printf(r.qpos, "Found a call path from root to %s", r.target)
		printf(r.target, "%s", r.target)
		for i := len(r.callpath) - 1; i >= 0; i-- {
			edge := r.callpath[i]
			printf(edge.Site, "%s from %s", edge.Site.Common().Description(), edge.Caller.Func())
		}
	} else {
		printf(r.target, "%s is unreachable in this analysis scope", r.target)
	}
}

func (r *callstackResult) toSerial(res *serial.Result, fset *token.FileSet) {
	var callers []serial.Caller
	for i := len(r.callpath) - 1; i >= 0; i-- { // (innermost first)
		edge := r.callpath[i]
		callers = append(callers, serial.Caller{
			Pos:    fset.Position(edge.Site.Pos()).String(),
			Caller: edge.Caller.Func().String(),
			Desc:   edge.Site.Common().Description(),
		})
	}
	res.Callstack = &serial.CallStack{
		Pos:     fset.Position(r.target.Pos()).String(),
		Target:  r.target.String(),
		Callers: callers,
	}
}
