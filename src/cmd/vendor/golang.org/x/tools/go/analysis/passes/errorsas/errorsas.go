// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The errorsas package defines an Analyzer that checks that the second argument to
// errors.As is a pointer to a type implementing error.
package errorsas

import (
	"errors"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `report passing non-pointer or non-error values to errors.As

The errorsas analysis reports calls to errors.As where the type
of the second argument is not a pointer to a type implementing error.`

var Analyzer = &analysis.Analyzer{
	Name:     "errorsas",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/errorsas",
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

	if !analysisutil.Imports(pass.Pkg, "errors") {
		return nil, nil // doesn't directly import errors
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		fn := typeutil.StaticCallee(pass.TypesInfo, call)
		if !analysisutil.IsFunctionNamed(fn, "errors", "As") {
			return
		}
		if len(call.Args) < 2 {
			return // not enough arguments, e.g. called with return values of another function
		}
		if err := checkAsTarget(pass, call.Args[1]); err != nil {
			pass.ReportRangef(call, "%v", err)
		}
	})
	return nil, nil
}

var errorType = types.Universe.Lookup("error").Type()

// pointerToInterfaceOrError reports whether the type of e is a pointer to an interface or a type implementing error,
// or is the empty interface.

// checkAsTarget reports an error if the second argument to errors.As is invalid.
func checkAsTarget(pass *analysis.Pass, e ast.Expr) error {
	t := pass.TypesInfo.Types[e].Type
	if it, ok := t.Underlying().(*types.Interface); ok && it.NumMethods() == 0 {
		// A target of interface{} is always allowed, since it often indicates
		// a value forwarded from another source.
		return nil
	}
	pt, ok := t.Underlying().(*types.Pointer)
	if !ok {
		return errors.New("second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type")
	}
	if pt.Elem() == errorType {
		return errors.New("second argument to errors.As should not be *error")
	}
	_, ok = pt.Elem().Underlying().(*types.Interface)
	if ok || types.Implements(pt.Elem(), errorType.Underlying().(*types.Interface)) {
		return nil
	}
	return errors.New("second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type")
}
