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
		ident, ix := instanceData(call)
		if ix == nil || len(ix.Indices) == 0 {
			return // no explicit args, nothing to do
		}

		// Confirm that instantiation actually occurred at this ident.
		_, instance := typeparams.GetInstance(pass.TypesInfo, ident)
		if instance == nil {
			return // something went wrong, but fail open
		}

		// Start removing argument expressions from the right, and check if we can
		// still infer the call expression.
		required := len(ix.Indices) // number of type expressions that are required
		for i := len(ix.Indices) - 1; i >= 0; i-- {
			var fun ast.Expr
			if i == 0 {
				// No longer an index expression: just use the parameterized operand.
				fun = ix.X
			} else {
				fun = typeparams.PackIndexExpr(ix.X, ix.Lbrack, ix.Indices[:i], ix.Indices[i-1].End())
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
			_, newInstance := typeparams.GetInstance(info, ident)
			if !types.Identical(instance, newInstance) {
				// The inferred result type does not match the original result type, so
				// this simplification is not valid.
				break
			}
			required = i
		}
		if required < len(ix.Indices) {
			var start, end token.Pos
			if required == 0 {
				start, end = ix.Lbrack, ix.Rbrack+1 // erase the entire index
			} else {
				start = ix.Indices[required-1].End()
				end = ix.Rbrack
			}
			pass.Report(analysis.Diagnostic{
				Pos:     start,
				End:     end,
				Message: "unnecessary type arguments",
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: "simplify type arguments",
					TextEdits: []analysis.TextEdit{{
						Pos: start,
						End: end,
					}},
				}},
			})
		}
	})

	return nil, nil
}

// instanceData returns the instantiated identifier and index data.
func instanceData(call *ast.CallExpr) (*ast.Ident, *typeparams.IndexExprData) {
	ix := typeparams.GetIndexExprData(call.Fun)
	if ix == nil {
		return nil, nil
	}
	var id *ast.Ident
	switch x := ix.X.(type) {
	case *ast.SelectorExpr:
		id = x.Sel
	case *ast.Ident:
		id = x
	default:
		return nil, nil
	}
	return id, ix
}
