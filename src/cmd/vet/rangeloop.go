// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
This file contains the code to check range loop variables bound inside function
literals that are deferred or launched in new goroutines. We only check
instances where the defer or go statement is the last statement in the loop
body, as otherwise we would need whole program analysis.

For example:

	for i, v := range s {
		go func() {
			println(i, v) // not what you might expect
		}()
	}

See: https://golang.org/doc/go_faq.html#closures_and_goroutines
*/

package main

import "go/ast"

func init() {
	register("rangeloops",
		"check that range loop variables are used correctly",
		checkRangeLoop,
		rangeStmt)
}

// checkRangeLoop walks the body of the provided range statement, checking if
// its index or value variables are used unsafely inside goroutines or deferred
// function literals.
func checkRangeLoop(f *File, node ast.Node) {
	n := node.(*ast.RangeStmt)
	key, _ := n.Key.(*ast.Ident)
	val, _ := n.Value.(*ast.Ident)
	if key == nil && val == nil {
		return
	}
	sl := n.Body.List
	if len(sl) == 0 {
		return
	}
	var last *ast.CallExpr
	switch s := sl[len(sl)-1].(type) {
	case *ast.GoStmt:
		last = s.Call
	case *ast.DeferStmt:
		last = s.Call
	default:
		return
	}
	lit, ok := last.Fun.(*ast.FuncLit)
	if !ok {
		return
	}
	ast.Inspect(lit.Body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok || id.Obj == nil {
			return true
		}
		if f.pkg.types[id].Type == nil {
			// Not referring to a variable
			return true
		}
		if key != nil && id.Obj == key.Obj || val != nil && id.Obj == val.Obj {
			f.Bad(id.Pos(), "range variable", id.Name, "captured by func literal")
		}
		return true
	})
}
