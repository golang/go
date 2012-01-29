// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(sortsliceFix)
}

var sortsliceFix = fix{
	"sortslice",
	"2011-06-26",
	sortslice,
	`Adapt code from sort.[Float64|Int|String]Array to  sort.[Float64|Int|String]Slice.
		
http://codereview.appspot.com/4602054
http://codereview.appspot.com/4639041
`,
}

func sortslice(f *ast.File) (fixed bool) {
	if !imports(f, "sort") {
		return
	}

	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)
		if !ok || !isTopName(s.X, "sort") {
			return
		}

		switch s.Sel.String() {
		case "Float64Array":
			s.Sel.Name = "Float64Slice"
		case "IntArray":
			s.Sel.Name = "IntSlice"
		case "StringArray":
			s.Sel.Name = "StringSlice"
		default:
			return
		}

		fixed = true
	})

	return
}
