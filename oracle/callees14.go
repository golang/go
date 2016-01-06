// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package oracle

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/go/types"
	"golang.org/x/tools/oracle/serial"
)

// Callees reports the possible callees of the function call site
// identified by the specified source location.
func callees(q *Query) error {
	lconf := loader.Config{Build: q.Build}

	if err := setPTAScope(&lconf, q.Scope); err != nil {
		return err
	}

	// Load/parse/type-check the program.
	lprog, err := lconf.Load()
	if err != nil {
		return err
	}
	q.Fset = lprog.Fset

	qpos, err := parseQueryPos(lprog, q.Pos, true) // needs exact pos
	if err != nil {
		return err
	}

	// Determine the enclosing call for the specified position.
	var e *ast.CallExpr
	for _, n := range qpos.path {
		if e, _ = n.(*ast.CallExpr); e != nil {
			break
		}
	}
	if e == nil {
		return fmt.Errorf("there is no function call here")
	}
	// TODO(adonovan): issue an error if the call is "too far
	// away" from the current selection, as this most likely is
	// not what the user intended.

	// Reject type conversions.
	if qpos.info.Types[e.Fun].IsType() {
		return fmt.Errorf("this is a type conversion, not a function call")
	}

	// Deal with obviously static calls before constructing SSA form.
	// Some static calls may yet require SSA construction,
	// e.g.  f := func(){}; f().
	switch funexpr := unparen(e.Fun).(type) {
	case *ast.Ident:
		switch obj := qpos.info.Uses[funexpr].(type) {
		case *types.Builtin:
			// Reject calls to built-ins.
			return fmt.Errorf("this is a call to the built-in '%s' operator", obj.Name())
		case *types.Func:
			// This is a static function call
			q.result = &calleesTypesResult{
				site:   e,
				callee: obj,
			}
			return nil
		}
	case *ast.SelectorExpr:
		sel := qpos.info.Selections[funexpr]
		if sel == nil {
			// qualified identifier.
			// May refer to top level function variable
			// or to top level function.
			callee := qpos.info.Uses[funexpr.Sel]
			if obj, ok := callee.(*types.Func); ok {
				q.result = &calleesTypesResult{
					site:   e,
					callee: obj,
				}
				return nil
			}
		} else if sel.Kind() == types.MethodVal {
			recvtype := sel.Recv()
			if !types.IsInterface(recvtype) {
				// static method call
				q.result = &calleesTypesResult{
					site:   e,
					callee: sel.Obj().(*types.Func),
				}
				return nil
			}
		}
	}

	prog := ssautil.CreateProgram(lprog, ssa.GlobalDebug)

	ptaConfig, err := setupPTA(prog, lprog, q.PTALog, q.Reflection)
	if err != nil {
		return err
	}

	pkg := prog.Package(qpos.info.Pkg)
	if pkg == nil {
		return fmt.Errorf("no SSA package")
	}

	// Defer SSA construction till after errors are reported.
	prog.Build()

	// Ascertain calling function and call site.
	callerFn := ssa.EnclosingFunction(pkg, qpos.path)
	if callerFn == nil {
		return fmt.Errorf("no SSA function built for this location (dead code?)")
	}

	// Find the call site.
	site, err := findCallSite(callerFn, e)
	if err != nil {
		return err
	}

	funcs, err := findCallees(ptaConfig, site)
	if err != nil {
		return err
	}

	q.result = &calleesSSAResult{
		site:  site,
		funcs: funcs,
	}
	return nil
}

func findCallSite(fn *ssa.Function, call *ast.CallExpr) (ssa.CallInstruction, error) {
	instr, _ := fn.ValueForExpr(call)
	callInstr, _ := instr.(ssa.CallInstruction)
	if instr == nil {
		return nil, fmt.Errorf("this call site is unreachable in this analysis")
	}
	return callInstr, nil
}

func findCallees(conf *pointer.Config, site ssa.CallInstruction) ([]*ssa.Function, error) {
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
	conf.BuildCallGraph = true
	cg := ptrAnalysis(conf).CallGraph
	cg.DeleteSyntheticNodes()

	// Find all call edges from the site.
	n := cg.Nodes[site.Parent()]
	if n == nil {
		return nil, fmt.Errorf("this call site is unreachable in this analysis")
	}
	calleesMap := make(map[*ssa.Function]bool)
	for _, edge := range n.Out {
		if edge.Site == site {
			calleesMap[edge.Callee.Func] = true
		}
	}

	// De-duplicate and sort.
	funcs := make([]*ssa.Function, 0, len(calleesMap))
	for f := range calleesMap {
		funcs = append(funcs, f)
	}
	sort.Sort(byFuncPos(funcs))
	return funcs, nil
}

type calleesSSAResult struct {
	site  ssa.CallInstruction
	funcs []*ssa.Function
}

type calleesTypesResult struct {
	site   *ast.CallExpr
	callee *types.Func
}

func (r *calleesSSAResult) display(printf printfFunc) {
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

func (r *calleesSSAResult) toSerial(res *serial.Result, fset *token.FileSet) {
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

func (r *calleesTypesResult) display(printf printfFunc) {
	printf(r.site, "this static function call dispatches to:")
	printf(r.callee, "\t%s", r.callee.FullName())
}

func (r *calleesTypesResult) toSerial(res *serial.Result, fset *token.FileSet) {
	j := &serial.Callees{
		Pos:  fset.Position(r.site.Pos()).String(),
		Desc: "static function call",
	}
	j.Callees = []*serial.CalleesItem{
		&serial.CalleesItem{
			Name: r.callee.FullName(),
			Pos:  fset.Position(r.callee.Pos()).String(),
		},
	}
	res.Callees = j
}

// NB: byFuncPos is not deterministic across packages since it depends on load order.
// Use lessPos if the tests need it.
type byFuncPos []*ssa.Function

func (a byFuncPos) Len() int           { return len(a) }
func (a byFuncPos) Less(i, j int) bool { return a[i].Pos() < a[j].Pos() }
func (a byFuncPos) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
