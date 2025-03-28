// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package waitgroup defines an Analyzer that detects simple misuses
// of sync.WaitGroup.
package waitgroup

import (
	_ "embed"
	"go/ast"
	"reflect"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysisinternal"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "waitgroup",
	Doc:      analysisutil.MustExtractDoc(doc, "waitgroup"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/waitgroup",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
	if !analysisinternal.Imports(pass.Pkg, "sync") {
		return nil, nil // doesn't directly import sync
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}

	inspect.WithStack(nodeFilter, func(n ast.Node, push bool, stack []ast.Node) (proceed bool) {
		if push {
			call := n.(*ast.CallExpr)
			obj := typeutil.Callee(pass.TypesInfo, call)
			if analysisinternal.IsMethodNamed(obj, "sync", "WaitGroup", "Add") &&
				hasSuffix(stack, wantSuffix) &&
				backindex(stack, 1) == backindex(stack, 2).(*ast.BlockStmt).List[0] { // ExprStmt must be Block's first stmt

				pass.Reportf(call.Lparen, "WaitGroup.Add called from inside new goroutine")
			}
		}
		return true
	})

	return nil, nil
}

//	go func() {
//	   wg.Add(1)
//	   ...
//	}()
var wantSuffix = []ast.Node{
	(*ast.GoStmt)(nil),
	(*ast.CallExpr)(nil),
	(*ast.FuncLit)(nil),
	(*ast.BlockStmt)(nil),
	(*ast.ExprStmt)(nil),
	(*ast.CallExpr)(nil),
}

// hasSuffix reports whether stack has the matching suffix,
// considering only node types.
func hasSuffix(stack, suffix []ast.Node) bool {
	// TODO(adonovan): the inspector could implement this for us.
	if len(stack) < len(suffix) {
		return false
	}
	for i := range len(suffix) {
		if reflect.TypeOf(backindex(stack, i)) != reflect.TypeOf(backindex(suffix, i)) {
			return false
		}
	}
	return true
}

// backindex is like [slices.Index] but from the back of the slice.
func backindex[T any](slice []T, i int) T {
	return slice[len(slice)-1-i]
}
