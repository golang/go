// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package deepequalerrors defines an Analyzer that checks for the use
// of reflect.DeepEqual with error values.
package deepequalerrors

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `check for calls of reflect.DeepEqual on error values

The deepequalerrors checker looks for calls of the form:

    reflect.DeepEqual(err1, err2)

where err1 and err2 are errors. Using reflect.DeepEqual to compare
errors is discouraged.`

var Analyzer = &analysis.Analyzer{
	Name:     "deepequalerrors",
	Doc:      Doc,
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
		fn, ok := typeutil.Callee(pass.TypesInfo, call).(*types.Func)
		if !ok {
			return
		}
		if fn.FullName() == "reflect.DeepEqual" && hasError(pass, call.Args[0]) && hasError(pass, call.Args[1]) {
			pass.Reportf(call.Pos(), "avoid using reflect.DeepEqual with errors")
		}
	})
	return nil, nil
}

var errorType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

// hasError reports whether the type of e contains the type error.
// See containsError, below, for the meaning of "contains".
func hasError(pass *analysis.Pass, e ast.Expr) bool {
	tv, ok := pass.TypesInfo.Types[e]
	if !ok { // no type info, assume good
		return false
	}
	return containsError(tv.Type)
}

// Report whether any type that typ could store and that could be compared is the
// error type. This includes typ itself, as well as the types of struct field, slice
// and array elements, map keys and elements, and pointers. It does not include
// channel types (incomparable), arg and result types of a Signature (not stored), or
// methods of a named or interface type (not stored).
func containsError(typ types.Type) bool {
	// Track types being processed, to avoid infinite recursion.
	// Using types as keys here is OK because we are checking for the identical pointer, not
	// type identity. See analysis/passes/printf/types.go.
	inProgress := make(map[types.Type]bool)

	var check func(t types.Type) bool
	check = func(t types.Type) bool {
		if t == errorType {
			return true
		}
		if inProgress[t] {
			return false
		}
		inProgress[t] = true
		switch t := t.(type) {
		case *types.Pointer:
			return check(t.Elem())
		case *types.Slice:
			return check(t.Elem())
		case *types.Array:
			return check(t.Elem())
		case *types.Map:
			return check(t.Key()) || check(t.Elem())
		case *types.Struct:
			for i := 0; i < t.NumFields(); i++ {
				if check(t.Field(i).Type()) {
					return true
				}
			}
		case *types.Named:
			return check(t.Underlying())

		// We list the remaining valid type kinds for completeness.
		case *types.Basic:
		case *types.Chan: // channels store values, but they are not comparable
		case *types.Signature:
		case *types.Tuple: // tuples are only part of signatures
		case *types.Interface:
		}
		return false
	}

	return check(typ)
}
