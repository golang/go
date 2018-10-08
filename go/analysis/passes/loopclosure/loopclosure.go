// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loopclosure

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

// TODO(adonovan): also report an error for the following structure,
// which is often used to ensure that deferred calls do not accumulate
// in a loop:
//
//	for i, x := range c {
//		func() {
//			...reference to i or x...
//		}()
//	}

var Analyzer = &analysis.Analyzer{
	Name: "loopclosure",
	Doc: `check references to loop variables from within nested functions

This analyzer checks for references to loop variables from within a
function literal inside the loop body. It checks only instances where
the function literal is called in a defer or go statement that is the
last statement in the loop body, as otherwise we would need whole
program analysis.

For example:

	for i, v := range s {
		go func() {
			println(i, v) // not what you might expect
		}()
	}

See: https://golang.org/doc/go_faq.html#closures_and_goroutines`,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.RangeStmt)(nil),
		(*ast.ForStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		// Find the variables updated by the loop statement.
		var vars []*ast.Ident
		addVar := func(expr ast.Expr) {
			if id, ok := expr.(*ast.Ident); ok {
				vars = append(vars, id)
			}
		}
		var body *ast.BlockStmt
		switch n := n.(type) {
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
			if pass.TypesInfo.Types[id].Type == nil {
				// Not referring to a variable (e.g. struct field name)
				return true
			}
			for _, v := range vars {
				if v.Obj == id.Obj {
					pass.Reportf(id.Pos(), "loop variable %s captured by func literal",
						id.Name)
				}
			}
			return true
		})
	})
	return nil, nil
}
