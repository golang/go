// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The errorsas package defines an Analyzer that checks that the second argument to
// errors.As is a pointer to a type implementing error.
package errorsas

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `report passing non-pointer or non-error values to errors.As

The errorsas analysis reports calls to errors.As where the type
of the second argument is not a pointer to a type implementing error.`

var Analyzer = &analysis.Analyzer{
	Name:     "errorsas",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	switch pass.Pkg.Path() {
	case "errors", "errors_test":
		// These packages know how to use their own APIs.
		// Sometimes they are testing what happens to incorrect programs.
		return nil, nil
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		fn := typeutil.StaticCallee(pass.TypesInfo, call)
		if fn == nil {
			return // not a static call
		}
		if len(call.Args) < 2 {
			return // not enough arguments, e.g. called with return values of another function
		}
		if fn.FullName() == "errors.As" && !pointerToInterfaceOrError(pass, call.Args[1]) {
			pass.ReportRangef(call, "second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type")
		}
	})
	return nil, nil
}

var errorType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

// pointerToInterfaceOrError reports whether the type of e is a pointer to an interface or a type implementing error,
// or is the empty interface.
func pointerToInterfaceOrError(pass *analysis.Pass, e ast.Expr) bool {
	t := pass.TypesInfo.Types[e].Type
	if it, ok := t.Underlying().(*types.Interface); ok && it.NumMethods() == 0 {
		return true
	}
	pt, ok := t.Underlying().(*types.Pointer)
	if !ok {
		return false
	}
	_, ok = pt.Elem().Underlying().(*types.Interface)
	return ok || types.Implements(pt.Elem(), errorType)
}
