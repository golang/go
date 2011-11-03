// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(filepathFix)
}

var filepathFix = fix{
	"filepath",
	"2011-06-26",
	filepathFunc,
	`Adapt code from filepath.[List]SeparatorString to string(filepath.[List]Separator).

http://codereview.appspot.com/4527090
`,
}

func filepathFunc(f *ast.File) (fixed bool) {
	if !imports(f, "path/filepath") {
		return
	}

	walk(f, func(n interface{}) {
		e, ok := n.(*ast.Expr)
		if !ok {
			return
		}

		var ident string
		switch {
		case isPkgDot(*e, "filepath", "SeparatorString"):
			ident = "filepath.Separator"
		case isPkgDot(*e, "filepath", "ListSeparatorString"):
			ident = "filepath.ListSeparator"
		default:
			return
		}

		// string(filepath.[List]Separator)
		*e = &ast.CallExpr{
			Fun:  ast.NewIdent("string"),
			Args: []ast.Expr{ast.NewIdent(ident)},
		}

		fixed = true
	})

	return
}
