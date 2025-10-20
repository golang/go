// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ifaceassert

import (
	_ "embed"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/typeparams"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "ifaceassert",
	Doc:      analysisinternal.MustExtractDoc(doc, "ifaceassert"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/ifaceassert",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// assertableTo checks whether interface v can be asserted into t. It returns
// nil on success, or the first conflicting method on failure.
func assertableTo(free *typeparams.Free, v, t types.Type) *types.Func {
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
	if free.Has(V) || free.Has(T) {
		return nil
	}
	if f, wrongType := types.MissingMethod(V, T, false); wrongType {
		return f
	}
	return nil
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.TypeAssertExpr)(nil),
		(*ast.TypeSwitchStmt)(nil),
	}
	var free typeparams.Free
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
			if f := assertableTo(&free, V, T); f != nil {
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
