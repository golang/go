// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(ioCopyNFix)
}

var ioCopyNFix = fix{
	"iocopyn",
	"2011-09-30",
	ioCopyN,
	`Rename io.Copyn to io.CopyN.

http://codereview.appspot.com/5157045
`,
}

func ioCopyN(f *ast.File) bool {
	if !imports(f, "io") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		if expr, ok := n.(ast.Expr); ok {
			if isPkgDot(expr, "io", "Copyn") {
				expr.(*ast.SelectorExpr).Sel.Name = "CopyN"
				fixed = true
				return
			}
		}
	})
	return fixed
}
