// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
)

func init() {
	register(oswaitFix)
}

var oswaitFix = fix{
	"oswait",
	"2012-02-20",
	oswait,
	`Delete options from os.Wait. If the option is the literal 0, rewrite the call.

http://codereview.appspot.com/5688046
`,
}

func oswait(f *ast.File) bool {
	if !imports(f, "os") {
		return false
	}

	fixed := false

	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return
		}
		if !isPkgDot(call.Fun, "os", "Wait") {
			return
		}
		args := call.Args
		const warning = "call to Process.Wait must be fixed manually"
		if len(args) != 1 {
			// Shouldn't happen, but check.
			warn(call.Pos(), warning)
			return
		}
		if basicLit, ok := args[0].(*ast.BasicLit); !ok || basicLit.Value != "0" {
			warn(call.Pos(), warning)
			return
		}
		call.Args = nil
		fixed = true
	})

	return fixed
}
