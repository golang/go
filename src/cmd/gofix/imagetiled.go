// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(imagetiledFix)
}

var imagetiledFix = fix{
	"imagetiled",
	"2012-01-10",
	imagetiled,
	`Rename image.Tiled to image.Repeated.

http://codereview.appspot.com/5530062
`,
}

func imagetiled(f *ast.File) bool {
	if !imports(f, "image") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)
		if !ok || !isTopName(s.X, "image") || s.Sel.String() != "Tiled" {
			return
		}
		s.Sel = &ast.Ident{Name: "Repeated"}
		fixed = true
	})
	return fixed
}
