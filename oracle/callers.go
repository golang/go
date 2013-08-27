package oracle

import (
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// Callers reports the possible callers of the function
// immediately enclosing the specified source location.
//
// TODO(adonovan): if a caller is a wrapper, show the caller's caller.
//
func callers(o *oracle) (queryResult, error) {
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
		return nil, o.errorf(o.queryPath[0], "no SSA function built for this location (dead code?)")
	}

	// Run the pointer analysis, recording each
	// call found to originate from target.
	var calls []callersCall
	o.config.Call = func(site pointer.CallSite, caller, callee pointer.CallGraphNode) {
		if callee.Func() == target {
			calls = append(calls, callersCall{site, caller})
		}
	}
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
	calls  []callersCall
}

type callersCall struct {
	site   pointer.CallSite
	caller pointer.CallGraphNode
}

func (r *callersResult) display(o *oracle) {
	if r.calls == nil {
		o.printf(r.target, "%s is not reachable in this program.", r.target)
	} else {
		o.printf(r.target, "%s is called from these %d sites:", r.target, len(r.calls))
		// TODO(adonovan): sort, to ensure test determinism.
		for _, call := range r.calls {
			if call.caller == r.root {
				o.printf(r.target, "the root of the call graph")
			} else {
				o.printf(call.site, "\t%s from %s",
					call.site.Description(), call.caller.Func())
			}
		}
	}

}
