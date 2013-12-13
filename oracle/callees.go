// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/oracle/serial"
	"code.google.com/p/go.tools/ssa"
)

// Callees reports the possible callees of the function call site
// identified by the specified source location.
//
// TODO(adonovan): if a callee is a wrapper, show the callee's callee.
//
func callees(o *Oracle, qpos *QueryPos) (queryResult, error) {
	pkg := o.prog.Package(qpos.info.Pkg)
	if pkg == nil {
		return nil, fmt.Errorf("no SSA package")
	}

	// Determine the enclosing call for the specified position.
	var e *ast.CallExpr
	for _, n := range qpos.path {
		if e, _ = n.(*ast.CallExpr); e != nil {
			break
		}
	}
	if e == nil {
		return nil, fmt.Errorf("there is no function call here")
	}
	// TODO(adonovan): issue an error if the call is "too far
	// away" from the current selection, as this most likely is
	// not what the user intended.

	// Reject type conversions.
	if qpos.info.IsType(e.Fun) {
		return nil, fmt.Errorf("this is a type conversion, not a function call")
	}

	// Reject calls to built-ins.
	if id, ok := unparen(e.Fun).(*ast.Ident); ok {
		if b, ok := qpos.info.ObjectOf(id).(*types.Builtin); ok {
			return nil, fmt.Errorf("this is a call to the built-in '%s' operator", b.Name())
		}
	}

	buildSSA(o)

	// Ascertain calling function and call site.
	callerFn := ssa.EnclosingFunction(pkg, qpos.path)
	if callerFn == nil {
		return nil, fmt.Errorf("no SSA function built for this location (dead code?)")
	}

	// Find the call site.
	site, err := findCallSite(callerFn, e.Lparen)
	if err != nil {
		return nil, err
	}

	funcs, err := findCallees(o, site)
	if err != nil {
		return nil, err
	}

	return &calleesResult{
		site:  site,
		funcs: funcs,
	}, nil
}

func findCallSite(fn *ssa.Function, lparen token.Pos) (ssa.CallInstruction, error) {
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if site, ok := instr.(ssa.CallInstruction); ok && instr.Pos() == lparen {
				return site, nil
			}
		}
	}
	return nil, fmt.Errorf("this call site is unreachable in this analysis")
}

func findCallees(o *Oracle, site ssa.CallInstruction) ([]*ssa.Function, error) {
	// Avoid running the pointer analysis for static calls.
	if callee := site.Common().StaticCallee(); callee != nil {
		switch callee.String() {
		case "runtime.SetFinalizer", "(reflect.Value).Call":
			// The PTA treats calls to these intrinsics as dynamic.
			// TODO(adonovan): avoid reliance on PTA internals.

		default:
			return []*ssa.Function{callee}, nil // singleton
		}
	}

	// Dynamic call: use pointer analysis.
	o.ptaConfig.BuildCallGraph = true
	callgraph := ptrAnalysis(o).CallGraph

	// Find all call edges from the site.
	calleesMap := make(map[*ssa.Function]bool)
	var foundCGNode bool
	for _, n := range callgraph.Nodes() {
		if n.Func() == site.Parent() {
			foundCGNode = true
			for _, edge := range n.Edges() {
				if edge.Site == site {
					calleesMap[edge.Callee.Func()] = true
				}
			}
		}
	}
	if !foundCGNode {
		return nil, fmt.Errorf("this call site is unreachable in this analysis")
	}

	// Discard context, de-duplicate and sort.
	funcs := make([]*ssa.Function, 0, len(calleesMap))
	for f := range calleesMap {
		funcs = append(funcs, f)
	}
	sort.Sort(byFuncPos(funcs))
	return funcs, nil
}

type calleesResult struct {
	site  ssa.CallInstruction
	funcs []*ssa.Function
}

func (r *calleesResult) display(printf printfFunc) {
	if len(r.funcs) == 0 {
		// dynamic call on a provably nil func/interface
		printf(r.site, "%s on nil value", r.site.Common().Description())
	} else {
		printf(r.site, "this %s dispatches to:", r.site.Common().Description())
		for _, callee := range r.funcs {
			printf(callee, "\t%s", callee)
		}
	}
}

func (r *calleesResult) toSerial(res *serial.Result, fset *token.FileSet) {
	j := &serial.Callees{
		Pos:  fset.Position(r.site.Pos()).String(),
		Desc: r.site.Common().Description(),
	}
	for _, callee := range r.funcs {
		j.Callees = append(j.Callees, &serial.CalleesItem{
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
