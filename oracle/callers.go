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

// Callers reports the possible callers of the function
// immediately enclosing the specified source location.
//
// TODO(adonovan): if a caller is a wrapper, show the caller's caller.
//
func callers(o *Oracle, qpos *QueryPos) (queryResult, error) {
	pkg := o.prog.Package(qpos.info.Pkg)
	if pkg == nil {
		return nil, o.errorf(qpos.path[0], "no SSA package")
	}
	if !ssa.HasEnclosingFunction(pkg, qpos.path) {
		return nil, o.errorf(qpos.path[0], "this position is not inside a function")
	}

	buildSSA(o)

	target := ssa.EnclosingFunction(pkg, qpos.path)
	if target == nil {
		return nil, o.errorf(qpos.path[0], "no SSA function built for this location (dead code?)")
	}

	// Run the pointer analysis, recording each
	// call found to originate from target.
	var calls []pointer.CallSite
	o.config.Call = func(site pointer.CallSite, callee pointer.CallGraphNode) {
		if callee.Func() == target {
			calls = append(calls, site)
		}
	}
	// TODO(adonovan): sort calls, to ensure test determinism.

	root := ptrAnalysis(o)

	return &callersResult{
		target: target,
		root:   root,
		calls:  calls,
	}, nil
}

type callersResult struct {
	target *ssa.Function
	root   pointer.CallGraphNode
	calls  []pointer.CallSite
}

func (r *callersResult) display(printf printfFunc) {
	if r.calls == nil {
		printf(r.target, "%s is not reachable in this program.", r.target)
	} else {
		printf(r.target, "%s is called from these %d sites:", r.target, len(r.calls))
		for _, site := range r.calls {
			if site.Caller() == r.root {
				printf(r.target, "the root of the call graph")
			} else {
				printf(site, "\t%s from %s", site.Description(), site.Caller().Func())
			}
		}
	}
}

func (r *callersResult) toJSON(res *json.Result, fset *token.FileSet) {
	var callers []json.Caller
	for _, site := range r.calls {
		var c json.Caller
		c.Caller = site.Caller().Func().String()
		if site.Caller() == r.root {
			c.Desc = "synthetic call"
		} else {
			c.Pos = fset.Position(site.Pos()).String()
			c.Desc = site.Description()
		}
		callers = append(callers, c)
	}
	res.Callers = callers
}
