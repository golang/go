// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
)

func init() {
	register(procattrFix)
}

var procattrFix = fix{
	"procattr",
	"2011-03-15",
	procattr,
	`Adapt calls to os.StartProcess to use new ProcAttr type.

http://codereview.appspot.com/4253052
`,
}

func procattr(f *ast.File) bool {
	if !imports(f, "os") && !imports(f, "syscall") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 5 {
			return
		}
		var pkg string
		if isPkgDot(call.Fun, "os", "StartProcess") {
			pkg = "os"
		} else if isPkgDot(call.Fun, "syscall", "StartProcess") {
			pkg = "syscall"
		} else {
			return
		}
		// os.StartProcess(a, b, c, d, e) -> os.StartProcess(a, b, &os.ProcAttr{Env: c, Dir: d, Files: e})
		lit := &ast.CompositeLit{Type: ast.NewIdent(pkg + ".ProcAttr")}
		env, dir, files := call.Args[2], call.Args[3], call.Args[4]
		if !isName(env, "nil") && !isCall(env, "os", "Environ") {
			lit.Elts = append(lit.Elts, &ast.KeyValueExpr{Key: ast.NewIdent("Env"), Value: env})
		}
		if !isEmptyString(dir) {
			lit.Elts = append(lit.Elts, &ast.KeyValueExpr{Key: ast.NewIdent("Dir"), Value: dir})
		}
		if !isName(files, "nil") {
			lit.Elts = append(lit.Elts, &ast.KeyValueExpr{Key: ast.NewIdent("Files"), Value: files})
		}
		call.Args[2] = &ast.UnaryExpr{Op: token.AND, X: lit}
		call.Args = call.Args[:3]
		fixed = true
	})
	return fixed
}
