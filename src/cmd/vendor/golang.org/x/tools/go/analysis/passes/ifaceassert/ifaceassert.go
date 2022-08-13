// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ifaceassert defines an Analyzer that flags
// impossible interface-interface type assertions.
package ifaceassert

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `detect impossible interface-to-interface type assertions

This checker flags type assertions v.(T) and corresponding type-switch cases
in which the static type V of v is an interface that cannot possibly implement
the target interface T. This occurs when V and T contain methods with the same
name but different signatures. Example:

	var v interface {
		Read()
	}
	_ = v.(io.Reader)

The Read method in v has a different signature than the Read method in
io.Reader, so this assertion cannot succeed.
`

var Analyzer = &analysis.Analyzer{
	Name:     "ifaceassert",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// assertableTo checks whether interface v can be asserted into t. It returns
// nil on success, or the first conflicting method on failure.
func assertableTo(v, t types.Type) *types.Func {
	if t == nil || v == nil {
		// not assertable to, but there is no missing method
		return nil
	}
	// ensure that v and t are interfaces
	V, _ := v.Underlying().(*types.Interface)
	T, _ := t.Underlying().(*types.Interface)
	if V == nil || T == nil {
		return nil
	}

	// Mitigations for interface comparisons and generics.
	// TODO(https://github.com/golang/go/issues/50658): Support more precise conclusion.
	if isParameterized(V) || isParameterized(T) {
		return nil
	}
	if f, wrongType := types.MissingMethod(V, T, false); wrongType {
		return f
	}
	return nil
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.TypeAssertExpr)(nil),
		(*ast.TypeSwitchStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		var (
			assert  *ast.TypeAssertExpr // v.(T) expression
			targets []ast.Expr          // interfaces T in v.(T)
		)
		switch n := n.(type) {
		case *ast.TypeAssertExpr:
			// take care of v.(type) in *ast.TypeSwitchStmt
			if n.Type == nil {
				return
			}
			assert = n
			targets = append(targets, n.Type)
		case *ast.TypeSwitchStmt:
			// retrieve type assertion from type switch's 'assign' field
			switch t := n.Assign.(type) {
			case *ast.ExprStmt:
				assert = t.X.(*ast.TypeAssertExpr)
			case *ast.AssignStmt:
				assert = t.Rhs[0].(*ast.TypeAssertExpr)
			}
			// gather target types from case clauses
			for _, c := range n.Body.List {
				targets = append(targets, c.(*ast.CaseClause).List...)
			}
		}
		V := pass.TypesInfo.TypeOf(assert.X)
		for _, target := range targets {
			T := pass.TypesInfo.TypeOf(target)
			if f := assertableTo(V, T); f != nil {
				pass.Reportf(
					target.Pos(),
					"impossible type assertion: no type can implement both %v and %v (conflicting types for %v method)",
					V, T, f.Name(),
				)
			}
		}
	})
	return nil, nil
}
