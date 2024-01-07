// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nilfunc defines an Analyzer that checks for useless
// comparisons against nil.
package nilfunc

import (
	_ "embed"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "nilfunc",
	Doc:      analysisutil.MustExtractDoc(doc, "nilfunc"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/nilfunc",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.BinaryExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		e := n.(*ast.BinaryExpr)

		// Only want == or != comparisons.
		if e.Op != token.EQL && e.Op != token.NEQ {
			return
		}

		// Only want comparisons with a nil identifier on one side.
		var e2 ast.Expr
		switch {
		case pass.TypesInfo.Types[e.X].IsNil():
			e2 = e.Y
		case pass.TypesInfo.Types[e.Y].IsNil():
			e2 = e.X
		default:
			return
		}

		// Only want identifiers or selector expressions.
		var obj types.Object
		switch v := e2.(type) {
		case *ast.Ident:
			obj = pass.TypesInfo.Uses[v]
		case *ast.SelectorExpr:
			obj = pass.TypesInfo.Uses[v.Sel]
		case *ast.IndexExpr, *ast.IndexListExpr:
			// Check generic functions such as "f[T1,T2]".
			x, _, _, _ := typeparams.UnpackIndexExpr(v)
			if id, ok := x.(*ast.Ident); ok {
				obj = pass.TypesInfo.Uses[id]
			}
		default:
			return
		}

		// Only want functions.
		if _, ok := obj.(*types.Func); !ok {
			return
		}

		pass.ReportRangef(e, "comparison of function %v %v nil is always %v", obj.Name(), e.Op, e.Op == token.NEQ)
	})
	return nil, nil
}
