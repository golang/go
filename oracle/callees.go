package oracle

import (
	"go/ast"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/pointer"
)

// Callees reports the possible callees of the function call site
// identified by the specified source location.
//
// TODO(adonovan): if a callee is a wrapper, show the callee's callee.
//
func callees(o *oracle) (queryResult, error) {
	// Determine the enclosing call for the specified position.
	var call *ast.CallExpr
	for _, n := range o.queryPath {
		if call, _ = n.(*ast.CallExpr); call != nil {
			break
		}
	}
	if call == nil {
		return nil, o.errorf(o.queryPath[0], "there is no function call here")
	}
	// TODO(adonovan): issue an error if the call is "too far
	// away" from the current selection, as this most likely is
	// not what the user intended.

	// Reject type conversions.
	if o.queryPkgInfo.IsType(call.Fun) {
		return nil, o.errorf(call, "this is a type conversion, not a function call")
	}

	// Reject calls to built-ins.
	if b, ok := o.queryPkgInfo.TypeOf(call.Fun).(*types.Builtin); ok {
		return nil, o.errorf(call, "this is a call to the built-in '%s' operator", b.Name())
	}

	buildSSA(o)

	// Compute the subgraph of the callgraph for callsite(s)
	// arising from 'call'.  There may be more than one if its
	// enclosing function was treated context-sensitively.
	// (Or zero if it was in dead code.)
	//
	// The presence of a key indicates this call site is
	// interesting even if the value is nil.
	querySites := make(map[pointer.CallSite][]pointer.CallGraphNode)
	o.config.CallSite = func(site pointer.CallSite) {
		if site.Pos() == call.Lparen {
			// Not a no-op!  Ensures key is
			// present even if value is nil:
			querySites[site] = querySites[site]
		}
	}
	o.config.Call = func(site pointer.CallSite, caller, callee pointer.CallGraphNode) {
		if targets, ok := querySites[site]; ok {
			querySites[site] = append(targets, callee)
		}
	}
	ptrAnalysis(o)

	return &calleesResult{
		call:       call,
		querySites: querySites,
	}, nil
}

type calleesResult struct {
	call       *ast.CallExpr
	querySites map[pointer.CallSite][]pointer.CallGraphNode
}

func (r *calleesResult) display(o *oracle) {
	// Print the set of discovered call edges.
	if len(r.querySites) == 0 {
		// e.g. it appears within "if false {...}" or within a dead function.
		o.printf(r.call.Lparen, "this call site is unreachable in this analysis")
	}

	// TODO(adonovan): sort, to ensure test determinism.
	// TODO(adonovan): compute union of callees across all contexts.
	for site, callees := range r.querySites {
		if callees == nil {
			// dynamic call on a provably nil func/interface
			o.printf(site, "%s on nil value", site.Description())
			continue
		}

		// TODO(adonovan): sort, to ensure test determinism.
		o.printf(site, "this %s dispatches to:", site.Description())
		for _, callee := range callees {
			o.printf(callee.Func(), "\t%s", callee.Func())
		}
	}
}
