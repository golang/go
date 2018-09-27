// +build ignore

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
		"check that loop variables are used correctly",
		checkLoop,
		rangeStmt, forStmt)
}

// checkLoop walks the body of the provided loop statement, checking whether
// its index or value variables are used unsafely inside goroutines or deferred
// function literals.
func checkLoop(f *File, node ast.Node) {
	// Find the variables updated by the loop statement.
	var vars []*ast.Ident
	addVar := func(expr ast.Expr) {
		if id, ok := expr.(*ast.Ident); ok {
			vars = append(vars, id)
		}
	}
	var body *ast.BlockStmt
	switch n := node.(type) {
	case *ast.RangeStmt:
		body = n.Body
		addVar(n.Key)
		addVar(n.Value)
	case *ast.ForStmt:
		body = n.Body
		switch post := n.Post.(type) {
		case *ast.AssignStmt:
			// e.g. for p = head; p != nil; p = p.next
			for _, lhs := range post.Lhs {
				addVar(lhs)
			}
		case *ast.IncDecStmt:
			// e.g. for i := 0; i < n; i++
			addVar(post.X)
		}
	}
	if vars == nil {
		return
	}

	// Inspect a go or defer statement
	// if it's the last one in the loop body.
	// (We give up if there are following statements,
	// because it's hard to prove go isn't followed by wait,
	// or defer by return.)
	if len(body.List) == 0 {
		return
	}
	var last *ast.CallExpr
	switch s := body.List[len(body.List)-1].(type) {
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
			// Not referring to a variable (e.g. struct field name)
			return true
		}
		for _, v := range vars {
			if v.Obj == id.Obj {
				f.Badf(id.Pos(), "loop variable %s captured by func literal",
					id.Name)
			}
		}
		return true
	})
}
