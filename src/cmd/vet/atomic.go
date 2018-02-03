// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
	"go/types"
)

func init() {
	register("atomic",
		"check for common mistaken usages of the sync/atomic package",
		checkAtomicAssignment,
		assignStmt)
}

// checkAtomicAssignment walks the assignment statement checking for common
// mistaken usage of atomic package, such as: x = atomic.AddUint64(&x, 1)
func checkAtomicAssignment(f *File, node ast.Node) {
	n := node.(*ast.AssignStmt)
	if len(n.Lhs) != len(n.Rhs) {
		return
	}
	if len(n.Lhs) == 1 && n.Tok == token.DEFINE {
		return
	}

	for i, right := range n.Rhs {
		call, ok := right.(*ast.CallExpr)
		if !ok {
			continue
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			continue
		}
		pkgIdent, _ := sel.X.(*ast.Ident)
		pkgName, ok := f.pkg.uses[pkgIdent].(*types.PkgName)
		if !ok || pkgName.Imported().Path() != "sync/atomic" {
			continue
		}

		switch sel.Sel.Name {
		case "AddInt32", "AddInt64", "AddUint32", "AddUint64", "AddUintptr":
			f.checkAtomicAddAssignment(n.Lhs[i], call)
		}
	}
}

// checkAtomicAddAssignment walks the atomic.Add* method calls checking for assigning the return value
// to the same variable being used in the operation
func (f *File) checkAtomicAddAssignment(left ast.Expr, call *ast.CallExpr) {
	if len(call.Args) != 2 {
		return
	}
	arg := call.Args[0]
	broken := false

	if uarg, ok := arg.(*ast.UnaryExpr); ok && uarg.Op == token.AND {
		broken = f.gofmt(left) == f.gofmt(uarg.X)
	} else if star, ok := left.(*ast.StarExpr); ok {
		broken = f.gofmt(star.X) == f.gofmt(arg)
	}

	if broken {
		f.Bad(left.Pos(), "direct assignment to atomic value")
	}
}
