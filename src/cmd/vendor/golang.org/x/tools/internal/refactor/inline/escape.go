// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
)

// escape implements a simple "address-taken" escape analysis. It
// calls f for each local variable that appears on the left side of an
// assignment (escapes=false) or has its address taken (escapes=true).
// The initialization of a variable by its declaration does not count
// as an assignment.
func escape(info *types.Info, root ast.Node, f func(v *types.Var, escapes bool)) {

	// lvalue is called for each address-taken expression or LHS of assignment.
	// Supported forms are: x, (x), x[i], x.f, *x, T{}.
	var lvalue func(e ast.Expr, escapes bool)
	lvalue = func(e ast.Expr, escapes bool) {
		switch e := e.(type) {
		case *ast.Ident:
			if v, ok := info.Uses[e].(*types.Var); ok {
				if !isPkgLevel(v) {
					f(v, escapes)
				}
			}
		case *ast.ParenExpr:
			lvalue(e.X, escapes)
		case *ast.IndexExpr:
			// TODO(adonovan): support generics without assuming e.X has a core type.
			// Consider:
			//
			// func Index[T interface{ [3]int | []int }](t T, i int) *int {
			//     return &t[i]
			// }
			//
			// We must traverse the normal terms and check
			// whether any of them is an array.
			//
			// We assume TypeOf returns non-nil.
			if _, ok := info.TypeOf(e.X).Underlying().(*types.Array); ok {
				lvalue(e.X, escapes) // &a[i] on array
			}
		case *ast.SelectorExpr:
			// We assume TypeOf returns non-nil.
			if _, ok := info.TypeOf(e.X).Underlying().(*types.Struct); ok {
				lvalue(e.X, escapes) // &s.f on struct
			}
		case *ast.StarExpr:
			// *ptr indirects an existing pointer
		case *ast.CompositeLit:
			// &T{...} creates a new variable
		default:
			panic(fmt.Sprintf("&x on %T", e)) // unreachable in well-typed code
		}
	}

	// Search function body for operations &x, x.f(), x++, and x = y
	// where x is a parameter. Each of these treats x as an address.
	ast.Inspect(root, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.UnaryExpr:
			if n.Op == token.AND {
				lvalue(n.X, true) // &x
			}

		case *ast.CallExpr:
			// implicit &x in method call x.f(),
			// where x has type T and method is (*T).f
			if sel, ok := n.Fun.(*ast.SelectorExpr); ok {
				if seln, ok := info.Selections[sel]; ok &&
					seln.Kind() == types.MethodVal &&
					isPointer(seln.Obj().Type().Underlying().(*types.Signature).Recv().Type()) {
					tArg, indirect := effectiveReceiver(seln)
					if !indirect && !isPointer(tArg) {
						lvalue(sel.X, true) // &x.f
					}
				}
			}

		case *ast.AssignStmt:
			for _, lhs := range n.Lhs {
				if id, ok := lhs.(*ast.Ident); ok &&
					info.Defs[id] != nil &&
					n.Tok == token.DEFINE {
					// declaration: doesn't count
				} else {
					lvalue(lhs, false)
				}
			}

		case *ast.IncDecStmt:
			lvalue(n.X, false)
		}
		return true
	})
}
