// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/token"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/oracle/serial"
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
func callstack(q *Query) error {
	fset := token.NewFileSet()
	lconf := loader.Config{Fset: fset, Build: q.Build}

	if err := setPTAScope(&lconf, q.Scope); err != nil {
		return err
	}

	// Load/parse/type-check the program.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}

	qpos, err := parseQueryPos(lprog, q.Pos, false)
	if err != nil {
		return err
	}

	prog := ssautil.CreateProgram(lprog, 0)

	ptaConfig, err := setupPTA(prog, lprog, q.PTALog, q.Reflection)
	if err != nil {
		return err
	}

	pkg := prog.Package(qpos.info.Pkg)
	if pkg == nil {
		return fmt.Errorf("no SSA package")
	}

	if !ssa.HasEnclosingFunction(pkg, qpos.path) {
		return fmt.Errorf("this position is not inside a function")
	}

	// Defer SSA construction till after errors are reported.
	prog.Build()

	target := ssa.EnclosingFunction(pkg, qpos.path)
	if target == nil {
		return fmt.Errorf("no SSA function built for this location (dead code?)")
	}

	// Run the pointer analysis and build the complete call graph.
	ptaConfig.BuildCallGraph = true
	cg := ptrAnalysis(ptaConfig).CallGraph
	cg.DeleteSyntheticNodes()

	// Search for an arbitrary path from a root to the target function.
	isEnd := func(n *callgraph.Node) bool { return n.Func == target }
	callpath := callgraph.PathSearch(cg.Root, isEnd)
	if callpath != nil {
		callpath = callpath[1:] // remove synthetic edge from <root>
	}

	q.Fset = fset
	q.result = &callstackResult{
		qpos:     qpos,
		target:   target,
		callpath: callpath,
	}
	return nil
}

type callstackResult struct {
	qpos     *queryPos
	target   *ssa.Function
	callpath []*callgraph.Edge
}

func (r *callstackResult) display(printf printfFunc) {
	if r.callpath != nil {
		printf(r.qpos, "Found a call path from root to %s", r.target)
		printf(r.target, "%s", r.target)
		for i := len(r.callpath) - 1; i >= 0; i-- {
			edge := r.callpath[i]
			printf(edge, "%s from %s", edge.Description(), edge.Caller.Func)
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
			Pos:    fset.Position(edge.Pos()).String(),
			Caller: edge.Caller.Func.String(),
			Desc:   edge.Description(),
		})
	}
	res.Callstack = &serial.CallStack{
		Pos:     fset.Position(r.target.Pos()).String(),
		Target:  r.target.String(),
		Callers: callers,
	}
}
