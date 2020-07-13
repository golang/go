// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package unusedparams defines an analyzer that checks for unused
// parameters of functions.
package unusedparams

import (
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check for unused parameters of functions

The unusedparams analyzer checks functions to see if there are
any parameters that are not being used.

To reduce false positives it ignores:
- methods
- parameters that do not have a name or are underscored
- functions in test files
- functions with empty bodies or those with just a return stmt`

var Analyzer = &analysis.Analyzer{
	Name:     "unusedparams",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

type paramData struct {
	field  *ast.Field
	ident  *ast.Ident
	typObj types.Object
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.FuncDecl)(nil),
		(*ast.FuncLit)(nil),
	}

	inspect.Preorder(nodeFilter, func(n ast.Node) {
		var fieldList *ast.FieldList
		var body *ast.BlockStmt

		// Get the fieldList and body from the function node.
		switch f := n.(type) {
		case *ast.FuncDecl:
			fieldList, body = f.Type.Params, f.Body
			// TODO(golang/go#36602): add better handling for methods, if we enable methods
			// we will get false positives if a struct is potentially implementing
			// an interface.
			if f.Recv != nil {
				return
			}
			// Ignore functions in _test.go files to reduce false positives.
			if file := pass.Fset.File(n.Pos()); file != nil && strings.HasSuffix(file.Name(), "_test.go") {
				return
			}
		case *ast.FuncLit:
			fieldList, body = f.Type.Params, f.Body
		}
		// If there are no arguments or the function is empty, then return.
		if fieldList.NumFields() == 0 || len(body.List) == 0 {
			return
		}

		switch expr := body.List[0].(type) {
		case *ast.ReturnStmt:
			// Ignore functions that only contain a return statement to reduce false positives.
			return
		case *ast.ExprStmt:
			callExpr, ok := expr.X.(*ast.CallExpr)
			if !ok || len(body.List) > 1 {
				break
			}
			// Ignore functions that only contain a panic statement to reduce false positives.
			if fun, ok := callExpr.Fun.(*ast.Ident); ok && fun.Name == "panic" {
				return
			}
		}

		// Get the useful data from each field.
		params := make(map[string]*paramData)
		unused := make(map[*paramData]bool)
		for _, f := range fieldList.List {
			for _, i := range f.Names {
				if i.Name == "_" {
					continue
				}
				params[i.Name] = &paramData{
					field:  f,
					ident:  i,
					typObj: pass.TypesInfo.ObjectOf(i),
				}
				unused[params[i.Name]] = true
			}
		}

		// Traverse through the body of the function and
		// check to see which parameters are unused.
		ast.Inspect(body, func(node ast.Node) bool {
			n, ok := node.(*ast.Ident)
			if !ok {
				return true
			}
			param, ok := params[n.Name]
			if !ok {
				return false
			}
			if nObj := pass.TypesInfo.ObjectOf(n); nObj != param.typObj {
				return false
			}
			delete(unused, param)
			return false
		})

		// Create the reports for the unused parameters.
		for u := range unused {
			start, end := u.field.Pos(), u.field.End()
			if len(u.field.Names) > 1 {
				start, end = u.ident.Pos(), u.ident.End()
			}
			// TODO(golang/go#36602): Add suggested fixes to automatically
			// remove the unused parameter. To start, just remove it from the
			// function declaration. Later, remove it from every use of this
			// function.
			pass.Report(analysis.Diagnostic{
				Pos:     start,
				End:     end,
				Message: fmt.Sprintf("potentially unused parameter: '%s'", u.ident.Name),
			})
		}
	})
	return nil, nil
}
