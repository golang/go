// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"go/ast"
	"go/token"
	"sort"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/json"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// Callees reports the possible callees of the function call site
// identified by the specified source location.
//
// TODO(adonovan): if a callee is a wrapper, show the callee's callee.
//
func callees(o *Oracle, qpos *QueryPos) (queryResult, error) {
	// Determine the enclosing call for the specified position.
	var call *ast.CallExpr
	for _, n := range qpos.path {
		if call, _ = n.(*ast.CallExpr); call != nil {
			break
		}
	}
	if call == nil {
		return nil, o.errorf(qpos.path[0], "there is no function call here")
	}
	// TODO(adonovan): issue an error if the call is "too far
	// away" from the current selection, as this most likely is
	// not what the user intended.

	// Reject type conversions.
	if qpos.info.IsType(call.Fun) {
		return nil, o.errorf(call, "this is a type conversion, not a function call")
	}

	// Reject calls to built-ins.
	if id, ok := unparen(call.Fun).(*ast.Ident); ok {
		if b, ok := qpos.info.ObjectOf(id).(*types.Builtin); ok {
			return nil, o.errorf(call, "this is a call to the built-in '%s' operator", b.Name())
		}
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
	var arbitrarySite pointer.CallSite
	o.config.CallSite = func(site pointer.CallSite) {
		if site.Pos() == call.Lparen {
			// Not a no-op!  Ensures key is
			// present even if value is nil:
			querySites[site] = querySites[site]
			arbitrarySite = site
		}
	}
	o.config.Call = func(site pointer.CallSite, callee pointer.CallGraphNode) {
		if targets, ok := querySites[site]; ok {
			querySites[site] = append(targets, callee)
		}
	}
	ptrAnalysis(o)

	if arbitrarySite == nil {
		return nil, o.errorf(call.Lparen, "this call site is unreachable in this analysis")
	}

	// Compute union of callees across all contexts.
	funcsMap := make(map[*ssa.Function]bool)
	for _, callees := range querySites {
		for _, callee := range callees {
			funcsMap[callee.Func()] = true
		}
	}
	funcs := make([]*ssa.Function, 0, len(funcsMap))
	for f := range funcsMap {
		funcs = append(funcs, f)
	}
	sort.Sort(byFuncPos(funcs))

	return &calleesResult{
		site:  arbitrarySite,
		funcs: funcs,
	}, nil
}

type calleesResult struct {
	site  pointer.CallSite
	funcs []*ssa.Function
}

func (r *calleesResult) display(printf printfFunc) {
	if len(r.funcs) == 0 {
		// dynamic call on a provably nil func/interface
		printf(r.site, "%s on nil value", r.site.Description())
	} else {
		printf(r.site, "this %s dispatches to:", r.site.Description())
		for _, callee := range r.funcs {
			printf(callee, "\t%s", callee)
		}
	}
}

func (r *calleesResult) toJSON(res *json.Result, fset *token.FileSet) {
	j := &json.Callees{
		Pos:  fset.Position(r.site.Pos()).String(),
		Desc: r.site.Description(),
	}
	for _, callee := range r.funcs {
		j.Callees = append(j.Callees, &json.CalleesItem{
			Name: callee.String(),
			Pos:  fset.Position(callee.Pos()).String(),
		})
	}
	res.Callees = j
}

type byFuncPos []*ssa.Function

func (a byFuncPos) Len() int           { return len(a) }
func (a byFuncPos) Less(i, j int) bool { return a[i].Pos() < a[j].Pos() }
func (a byFuncPos) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
