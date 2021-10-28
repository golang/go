// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sortslice defines an Analyzer that checks for calls
// to sort.Slice that do not use a slice type as first argument.
package sortslice

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `check the argument type of sort.Slice

sort.Slice requires an argument of a slice type. Check that
the interface{} value passed to sort.Slice is actually a slice.`

var Analyzer = &analysis.Analyzer{
	Name:     "sortslice",
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
		fn, _ := typeutil.Callee(pass.TypesInfo, call).(*types.Func)
		if fn == nil {
			return
		}

		fnName := fn.FullName()
		if fnName != "sort.Slice" && fnName != "sort.SliceStable" && fnName != "sort.SliceIsSorted" {
			return
		}

		arg := call.Args[0]
		typ := pass.TypesInfo.Types[arg].Type
		switch typ.Underlying().(type) {
		case *types.Slice, *types.Interface:
			return
		}

		var fixes []analysis.SuggestedFix
		switch v := typ.Underlying().(type) {
		case *types.Array:
			var buf bytes.Buffer
			format.Node(&buf, pass.Fset, &ast.SliceExpr{
				X:      arg,
				Slice3: false,
				Lbrack: arg.End() + 1,
				Rbrack: arg.End() + 3,
			})
			fixes = append(fixes, analysis.SuggestedFix{
				Message: "Get a slice of the full array",
				TextEdits: []analysis.TextEdit{{
					Pos:     arg.Pos(),
					End:     arg.End(),
					NewText: buf.Bytes(),
				}},
			})
		case *types.Pointer:
			_, ok := v.Elem().Underlying().(*types.Slice)
			if !ok {
				break
			}
			var buf bytes.Buffer
			format.Node(&buf, pass.Fset, &ast.StarExpr{
				X: arg,
			})
			fixes = append(fixes, analysis.SuggestedFix{
				Message: "Dereference the pointer to the slice",
				TextEdits: []analysis.TextEdit{{
					Pos:     arg.Pos(),
					End:     arg.End(),
					NewText: buf.Bytes(),
				}},
			})
		case *types.Signature:
			if v.Params().Len() != 0 || v.Results().Len() != 1 {
				break
			}
			if _, ok := v.Results().At(0).Type().Underlying().(*types.Slice); !ok {
				break
			}
			var buf bytes.Buffer
			format.Node(&buf, pass.Fset, &ast.CallExpr{
				Fun: arg,
			})
			fixes = append(fixes, analysis.SuggestedFix{
				Message: "Call the function",
				TextEdits: []analysis.TextEdit{{
					Pos:     arg.Pos(),
					End:     arg.End(),
					NewText: buf.Bytes(),
				}},
			})
		}

		pass.Report(analysis.Diagnostic{
			Pos:            call.Pos(),
			End:            call.End(),
			Message:        fmt.Sprintf("%s's argument must be a slice; is called with %s", fnName, typ.String()),
			SuggestedFixes: fixes,
		})
	})
	return nil, nil
}
