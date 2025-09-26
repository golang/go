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
	typeindexanalyzer "golang.org/x/tools/internal/analysisinternal/typeindex"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

const Doc = `report passing non-pointer or non-error values to errors.As

The errorsas analyzer reports calls to errors.As where the type
of the second argument is not a pointer to a type implementing error.`

var Analyzer = &analysis.Analyzer{
	Name:     "errorsas",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/errorsas",
	Requires: []*analysis.Analyzer{typeindexanalyzer.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
	switch pass.Pkg.Path() {
	case "errors", "errors_test":
		// These packages know how to use their own APIs.
		// Sometimes they are testing what happens to incorrect programs.
		return nil, nil
	}

	var (
		index = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info  = pass.TypesInfo
	)

	for curCall := range index.Calls(index.Object("errors", "As")) {
		call := curCall.Node().(*ast.CallExpr)
		if len(call.Args) < 2 {
			continue // spread call: errors.As(pair())
		}

		// Check for incorrect arguments.
		if err := checkAsTarget(info, call.Args[1]); err != nil {
			pass.ReportRangef(call, "%v", err)
			continue
		}
	}
	return nil, nil
}

// checkAsTarget reports an error if the second argument to errors.As is invalid.
func checkAsTarget(info *types.Info, e ast.Expr) error {
	t := info.Types[e].Type
	if types.Identical(t.Underlying(), anyType) {
		// A target of any is always allowed, since it often indicates
		// a value forwarded from another source.
		return nil
	}
	pt, ok := t.Underlying().(*types.Pointer)
	if !ok {
		return errors.New("second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type")
	}
	if types.Identical(pt.Elem(), errorType) {
		return errors.New("second argument to errors.As should not be *error")
	}
	if !types.IsInterface(pt.Elem()) && !types.AssignableTo(pt.Elem(), errorType) {
		return errors.New("second argument to errors.As must be a non-nil pointer to either a type that implements error, or to any interface type")
	}
	return nil
}

var (
	anyType   = types.Universe.Lookup("any").Type()
	errorType = types.Universe.Lookup("error").Type()
)
