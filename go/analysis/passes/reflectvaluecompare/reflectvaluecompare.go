// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reflectvaluecompare defines an Analyzer that checks for accidentally
// using == or reflect.DeepEqual to compare reflect.Value values.
// See issues 43993 and 18871.
package reflectvaluecompare

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `check for comparing reflect.Value values with == or reflect.DeepEqual

The reflectvaluecompare checker looks for expressions of the form:

    v1 == v2
    v1 != v2
    reflect.DeepEqual(v1, v2)

where v1 or v2 are reflect.Values. Comparing reflect.Values directly
is almost certainly not correct, as it compares the reflect package's
internal representation, not the underlying value.
Likely what is intended is:

    v1.Interface() == v2.Interface()
    v1.Interface() != v2.Interface()
    reflect.DeepEqual(v1.Interface(), v2.Interface())
`

var Analyzer = &analysis.Analyzer{
	Name:     "reflectvaluecompare",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.BinaryExpr)(nil),
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		switch n := n.(type) {
		case *ast.BinaryExpr:
			if n.Op != token.EQL && n.Op != token.NEQ {
				return
			}
			if isReflectValue(pass, n.X) || isReflectValue(pass, n.Y) {
				if n.Op == token.EQL {
					pass.ReportRangef(n, "avoid using == with reflect.Value")
				} else {
					pass.ReportRangef(n, "avoid using != with reflect.Value")
				}
			}
		case *ast.CallExpr:
			fn, ok := typeutil.Callee(pass.TypesInfo, n).(*types.Func)
			if !ok {
				return
			}
			if fn.FullName() == "reflect.DeepEqual" && (isReflectValue(pass, n.Args[0]) || isReflectValue(pass, n.Args[1])) {
				pass.ReportRangef(n, "avoid using reflect.DeepEqual with reflect.Value")
			}
		}
	})
	return nil, nil
}

// isReflectValue reports whether the type of e is reflect.Value.
func isReflectValue(pass *analysis.Pass, e ast.Expr) bool {
	tv, ok := pass.TypesInfo.Types[e]
	if !ok { // no type info, something else is wrong
		return false
	}
	// See if the type is reflect.Value
	named, ok := tv.Type.(*types.Named)
	if !ok {
		return false
	}
	if obj := named.Obj(); obj == nil || obj.Pkg() == nil || obj.Pkg().Path() != "reflect" || obj.Name() != "Value" {
		return false
	}
	if _, ok := e.(*ast.CompositeLit); ok {
		// This is reflect.Value{}. Don't treat that as an error.
		// Users should probably use x.IsValid() rather than x == reflect.Value{}, but the latter isn't wrong.
		return false
	}
	return true
}
