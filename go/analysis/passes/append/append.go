// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package append defines an Analyzer that detects
// if there is only one variable in append.
package append

import (
	_ "embed"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "append",
	Doc:      analysisutil.MustExtractDoc(doc, "append"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/append",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		if ident, ok := call.Fun.(*ast.Ident); ok && ident.Name == "append" {
			if _, ok := pass.TypesInfo.Uses[ident].(*types.Builtin); ok {
				if len(call.Args) == 1 {
					pass.ReportRangef(call, "append with no values")
				}
			}
		}
	})

	return nil, nil
}
