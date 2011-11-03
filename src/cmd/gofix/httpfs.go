// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
)

func init() {
	register(httpFileSystemFix)
}

var httpFileSystemFix = fix{
	"httpfs",
	"2011-06-27",
	httpfs,
	`Adapt http FileServer to take a FileSystem.

http://codereview.appspot.com/4629047  http FileSystem interface
`,
}

func httpfs(f *ast.File) bool {
	if !imports(f, "http") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok || !isPkgDot(call.Fun, "http", "FileServer") {
			return
		}
		if len(call.Args) != 2 {
			return
		}
		dir, prefix := call.Args[0], call.Args[1]
		call.Args = []ast.Expr{&ast.CallExpr{
			Fun:  &ast.SelectorExpr{ast.NewIdent("http"), ast.NewIdent("Dir")},
			Args: []ast.Expr{dir},
		}}
		wrapInStripHandler := true
		if prefixLit, ok := prefix.(*ast.BasicLit); ok {
			if prefixLit.Kind == token.STRING && (prefixLit.Value == `"/"` || prefixLit.Value == `""`) {
				wrapInStripHandler = false
			}
		}
		if wrapInStripHandler {
			call.Fun.(*ast.SelectorExpr).Sel = ast.NewIdent("StripPrefix")
			call.Args = []ast.Expr{
				prefix,
				&ast.CallExpr{
					Fun:  &ast.SelectorExpr{ast.NewIdent("http"), ast.NewIdent("FileServer")},
					Args: call.Args,
				},
			}
		}
		fixed = true
	})
	return fixed
}
