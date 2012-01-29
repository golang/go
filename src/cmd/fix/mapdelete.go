// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast"

func init() {
	register(mapdeleteFix)
}

var mapdeleteFix = fix{
	"mapdelete",
	"2011-10-18",
	mapdelete,
	`Use delete(m, k) instead of m[k] = 0, false.

http://codereview.appspot.com/5272045
`,
}

func mapdelete(f *ast.File) bool {
	fixed := false
	walk(f, func(n interface{}) {
		stmt, ok := n.(*ast.Stmt)
		if !ok {
			return
		}
		as, ok := (*stmt).(*ast.AssignStmt)
		if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 2 {
			return
		}
		ix, ok := as.Lhs[0].(*ast.IndexExpr)
		if !ok {
			return
		}
		if !isTopName(as.Rhs[1], "false") {
			warn(as.Pos(), "two-element map assignment with non-false second value")
			return
		}
		if !canDrop(as.Rhs[0]) {
			warn(as.Pos(), "two-element map assignment with non-trivial first value")
			return
		}
		*stmt = &ast.ExprStmt{
			X: &ast.CallExpr{
				Fun: &ast.Ident{
					NamePos: as.Pos(),
					Name:    "delete",
				},
				Args: []ast.Expr{ix.X, ix.Index},
			},
		}
		fixed = true
	})
	return fixed
}

// canDrop reports whether it is safe to drop the
// evaluation of n from the program.
// It is very conservative.
func canDrop(n ast.Expr) bool {
	switch n := n.(type) {
	case *ast.Ident, *ast.BasicLit:
		return true
	case *ast.ParenExpr:
		return canDrop(n.X)
	case *ast.SelectorExpr:
		return canDrop(n.X)
	case *ast.CompositeLit:
		if !canDrop(n.Type) {
			return false
		}
		for _, e := range n.Elts {
			if !canDrop(e) {
				return false
			}
		}
		return true
	case *ast.StarExpr:
		// Dropping *x is questionable,
		// but we have to be able to drop (*T)(nil).
		return canDrop(n.X)
	case *ast.ArrayType, *ast.ChanType, *ast.FuncType, *ast.InterfaceType, *ast.MapType, *ast.StructType:
		return true
	}
	return false
}
