// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "go/ast"

func init() {
	register(url2Fix)
}

var url2Fix = fix{
	"url2",
	"2012-02-16",
	url2,
	`Rename some functions in net/url.

http://codereview.appspot.com/5671061
`,
}

func url2(f *ast.File) bool {
	if !imports(f, "net/url") {
		return false
	}

	fixed := false

	walk(f, func(n interface{}) {
		// Rename functions and methods.
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if !isTopName(sel.X, "url") {
			return
		}
		if sel.Sel.Name == "ParseWithReference" {
			sel.Sel.Name = "ParseWithFragment"
			fixed = true
		}
	})

	return fixed
}
