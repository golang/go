// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(sorthelpersFix)
}

var sorthelpersFix = fix{
	"sorthelpers",
	"2011-07-08",
	sorthelpers,
	`Adapt code from sort.Sort[Ints|Float64s|Strings] to sort.[Ints|Float64s|Strings].
`,
}

func sorthelpers(f *ast.File) (fixed bool) {
	if !imports(f, "sort") {
		return
	}

	walk(f, func(n interface{}) {
		s, ok := n.(*ast.SelectorExpr)
		if !ok || !isTopName(s.X, "sort") {
			return
		}

		switch s.Sel.String() {
		case "SortFloat64s":
			s.Sel.Name = "Float64s"
		case "SortInts":
			s.Sel.Name = "Ints"
		case "SortStrings":
			s.Sel.Name = "Strings"
		default:
			return
		}

		fixed = true
	})

	return
}
