// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package infertypeargs defines an analyzer that checks for explicit function
// arguments that could be inferred.
package infertypeargs

import (
	"go/token"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check for unnecessary type arguments in call expressions

Explicit type arguments may be omitted from call expressions if they can be
inferred from function arguments, or from other type arguments:

	func f[T any](T) {}
	
	func _() {
		f[string]("foo") // string could be inferred
	}
`

var Analyzer = &analysis.Analyzer{
	Name:     "infertypeargs",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

// TODO(rfindley): remove this thin wrapper around the infertypeargs refactoring,
// and eliminate the infertypeargs analyzer.
//
// Previous iterations used the analysis framework for computing refactorings,
// which proved inefficient.
func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	for _, diag := range DiagnoseInferableTypeArgs(pass.Fset, inspect, token.NoPos, token.NoPos, pass.Pkg, pass.TypesInfo) {
		pass.Report(diag)
	}
	return nil, nil
}
