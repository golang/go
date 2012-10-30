// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(osopenFix)
}

var osopenFix = fix{
	"osopen",
	"2011-04-04",
	osopen,
	`Adapt os.Open calls to new, easier API and rename O_CREAT O_CREATE.

http://codereview.appspot.com/4357052
`,
}

func osopen(f *ast.File) bool {
	if !imports(f, "os") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		// Rename O_CREAT to O_CREATE.
		if expr, ok := n.(ast.Expr); ok && isPkgDot(expr, "os", "O_CREAT") {
			expr.(*ast.SelectorExpr).Sel.Name = "O_CREATE"
			fixed = true
			return
		}

		// Fix up calls to Open.
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 3 {
			return
		}
		if !isPkgDot(call.Fun, "os", "Open") {
			return
		}
		sel := call.Fun.(*ast.SelectorExpr)
		args := call.Args
		// os.Open(a, os.O_RDONLY, c) -> os.Open(a)
		if isPkgDot(args[1], "os", "O_RDONLY") || isPkgDot(args[1], "syscall", "O_RDONLY") {
			call.Args = call.Args[0:1]
			fixed = true
			return
		}
		// os.Open(a, createlike_flags, c) -> os.Create(a, c)
		if isCreateFlag(args[1]) {
			sel.Sel.Name = "Create"
			if !isSimplePerm(args[2]) {
				warn(sel.Pos(), "rewrote os.Open to os.Create with permission not 0666")
			}
			call.Args = args[0:1]
			fixed = true
			return
		}
		// Fallback: os.Open(a, b, c) -> os.OpenFile(a, b, c)
		sel.Sel.Name = "OpenFile"
		fixed = true
	})
	return fixed
}

func isCreateFlag(flag ast.Expr) bool {
	foundCreate := false
	foundTrunc := false
	// OR'ing of flags: is O_CREATE on?  + or | would be fine; we just look for os.O_CREATE
	// and don't worry about the actual operator.
	p := flag.Pos()
	for {
		lhs := flag
		expr, isBinary := flag.(*ast.BinaryExpr)
		if isBinary {
			lhs = expr.Y
		}
		sel, ok := lhs.(*ast.SelectorExpr)
		if !ok || !isTopName(sel.X, "os") {
			return false
		}
		switch sel.Sel.Name {
		case "O_CREATE":
			foundCreate = true
		case "O_TRUNC":
			foundTrunc = true
		case "O_RDONLY", "O_WRONLY", "O_RDWR":
			// okay
		default:
			// Unexpected flag, like O_APPEND or O_EXCL.
			// Be conservative and do not rewrite.
			return false
		}
		if !isBinary {
			break
		}
		flag = expr.X
	}
	if !foundCreate {
		return false
	}
	if !foundTrunc {
		warn(p, "rewrote os.Open with O_CREATE but not O_TRUNC to os.Create")
	}
	return foundCreate
}

func isSimplePerm(perm ast.Expr) bool {
	basicLit, ok := perm.(*ast.BasicLit)
	if !ok {
		return false
	}
	switch basicLit.Value {
	case "0666":
		return true
	}
	return false
}
