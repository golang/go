// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
)

func init() {
	register(stringssplitFix)
}

var stringssplitFix = fix{
	"stringssplit",
	"2011-06-28",
	stringssplit,
	`Restore strings.Split to its original meaning and add strings.SplitN. Bytes too.

http://codereview.appspot.com/4661051
`,
}

func stringssplit(f *ast.File) bool {
	if !imports(f, "bytes") && !imports(f, "strings") {
		return false
	}

	fixed := false
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		// func Split(s, sep string, n int) []string
		// func SplitAfter(s, sep string, n int) []string
		if !ok || len(call.Args) != 3 {
			return
		}
		// Is this our function?
		switch {
		case isPkgDot(call.Fun, "bytes", "Split"):
		case isPkgDot(call.Fun, "bytes", "SplitAfter"):
		case isPkgDot(call.Fun, "strings", "Split"):
		case isPkgDot(call.Fun, "strings", "SplitAfter"):
		default:
			return
		}

		sel := call.Fun.(*ast.SelectorExpr)
		args := call.Args
		fixed = true // We're committed.

		// Is the last argument -1? If so, drop the arg.
		// (Actually we just look for a negative integer literal.)
		// Otherwise, Split->SplitN and keep the arg.
		final := args[2]
		if unary, ok := final.(*ast.UnaryExpr); ok && unary.Op == token.SUB {
			if lit, ok := unary.X.(*ast.BasicLit); ok {
				// Is it an integer? If so, it's a negative integer and that's what we're after.
				if lit.Kind == token.INT {
					// drop the last arg.
					call.Args = args[0:2]
					return
				}
			}
		}

		// If not, rename and keep the argument list.
		sel.Sel.Name += "N"
	})
	return fixed
}
