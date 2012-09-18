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

See: http://golang.org/doc/go_faq.html#closures_and_goroutines
*/

package main

import "go/ast"

// checkRangeLoop walks the body of the provided range statement, checking if
// its index or value variables are used unsafely inside goroutines or deferred
// function literals.
func checkRangeLoop(f *File, n *ast.RangeStmt) {
	if !*vetRangeLoops && !*vetAll {
		return
	}
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
		if n, ok := n.(*ast.Ident); ok && n.Obj != nil && (n.Obj == key.Obj || n.Obj == val.Obj) {
			f.Warn(n.Pos(), "range variable", n.Name, "enclosed by function")
		}
		return true
	})
}

func BadRangeLoopsUsedInTests() {
	var s []int
	for i, v := range s {
		go func() {
			println(i) // ERROR "range variable i enclosed by function"
			println(v) // ERROR "range variable v enclosed by function"
		}()
	}
	for i, v := range s {
		defer func() {
			println(i) // ERROR "range variable i enclosed by function"
			println(v) // ERROR "range variable v enclosed by function"
		}()
	}
	for i := range s {
		go func() {
			println(i) // ERROR "range variable i enclosed by function"
		}()
	}
	for _, v := range s {
		go func() {
			println(v) // ERROR "range variable v enclosed by function"
		}()
	}
	for i, v := range s {
		go func() {
			println(i, v)
		}()
		println("unfortunately, we don't catch the error above because of this statement")
	}
	for i, v := range s {
		go func(i, v int) {
			println(i, v)
		}(i, v)
	}
	for i, v := range s {
		i, v := i, v
		go func() {
			println(i, v)
		}()
	}
}
