// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package infertypeargs

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}

	inspect.Preorder(nodeFilter, func(node ast.Node) {
		call := node.(*ast.CallExpr)
		x, lbrack, indices, rbrack := typeparams.UnpackIndexExpr(call.Fun)
		ident := calledIdent(x)
		if ident == nil || len(indices) == 0 {
			return // no explicit args, nothing to do
		}

		// Confirm that instantiation actually occurred at this ident.
		idata, ok := typeparams.GetInstances(pass.TypesInfo)[ident]
		if !ok {
			return // something went wrong, but fail open
		}
		instance := idata.Type

		// Start removing argument expressions from the right, and check if we can
		// still infer the call expression.
		required := len(indices) // number of type expressions that are required
		for i := len(indices) - 1; i >= 0; i-- {
			var fun ast.Expr
			if i == 0 {
				// No longer an index expression: just use the parameterized operand.
				fun = x
			} else {
				fun = typeparams.PackIndexExpr(x, lbrack, indices[:i], indices[i-1].End())
			}
			newCall := &ast.CallExpr{
				Fun:      fun,
				Lparen:   call.Lparen,
				Args:     call.Args,
				Ellipsis: call.Ellipsis,
				Rparen:   call.Rparen,
			}
			info := new(types.Info)
			typeparams.InitInstanceInfo(info)
			if err := types.CheckExpr(pass.Fset, pass.Pkg, call.Pos(), newCall, info); err != nil {
				// Most likely inference failed.
				break
			}
			newIData := typeparams.GetInstances(info)[ident]
			newInstance := newIData.Type
			if !types.Identical(instance, newInstance) {
				// The inferred result type does not match the original result type, so
				// this simplification is not valid.
				break
			}
			required = i
		}
		if required < len(indices) {
			var start, end token.Pos
			var edit analysis.TextEdit
			if required == 0 {
				start, end = lbrack, rbrack+1 // erase the entire index
				edit = analysis.TextEdit{Pos: start, End: end}
			} else {
				start = indices[required].Pos()
				end = rbrack
				//  erase from end of last arg to include last comma & white-spaces
				edit = analysis.TextEdit{Pos: indices[required-1].End(), End: end}
			}
			pass.Report(analysis.Diagnostic{
				Pos:     start,
				End:     end,
				Message: "unnecessary type arguments",
				SuggestedFixes: []analysis.SuggestedFix{{
					Message:   "simplify type arguments",
					TextEdits: []analysis.TextEdit{edit},
				}},
			})
		}
	})

	return nil, nil
}

func calledIdent(x ast.Expr) *ast.Ident {
	switch x := x.(type) {
	case *ast.Ident:
		return x
	case *ast.SelectorExpr:
		return x.Sel
	}
	return nil
}
