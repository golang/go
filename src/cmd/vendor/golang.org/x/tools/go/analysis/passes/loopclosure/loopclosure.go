// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loopclosure defines an Analyzer that checks for references to
// enclosing loop variables from within nested functions.
package loopclosure

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

const Doc = `check references to loop variables from within nested functions

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

See: https://golang.org/doc/go_faq.html#closures_and_goroutines`

var Analyzer = &analysis.Analyzer{
	Name:     "loopclosure",
	Doc:      Doc,
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
		// The function invoked in the last return statement.
		var fun ast.Expr
		switch s := body.List[len(body.List)-1].(type) {
		case *ast.GoStmt:
			fun = s.Call.Fun
		case *ast.DeferStmt:
			fun = s.Call.Fun
		case *ast.ExprStmt: // check for errgroup.Group.Go()
			if call, ok := s.X.(*ast.CallExpr); ok {
				fun = goInvokes(pass.TypesInfo, call)
			}
		}
		lit, ok := fun.(*ast.FuncLit)
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
					pass.ReportRangef(id, "loop variable %s captured by func literal",
						id.Name)
				}
			}
			return true
		})
	})
	return nil, nil
}

// goInvokes returns a function expression that would be called asynchronously
// (but not awaited) in another goroutine as a consequence of the call.
// For example, given the g.Go call below, it returns the function literal expression.
//
//	import "sync/errgroup"
//	var g errgroup.Group
//	g.Go(func() error { ... })
//
// Currently only "golang.org/x/sync/errgroup.Group()" is considered.
func goInvokes(info *types.Info, call *ast.CallExpr) ast.Expr {
	f := typeutil.StaticCallee(info, call)
	// Note: Currently only supports: golang.org/x/sync/errgroup.Go.
	if f == nil || f.Name() != "Go" {
		return nil
	}
	recv := f.Type().(*types.Signature).Recv()
	if recv == nil {
		return nil
	}
	rtype, ok := recv.Type().(*types.Pointer)
	if !ok {
		return nil
	}
	named, ok := rtype.Elem().(*types.Named)
	if !ok {
		return nil
	}
	if named.Obj().Name() != "Group" {
		return nil
	}
	pkg := f.Pkg()
	if pkg == nil {
		return nil
	}
	if pkg.Path() != "golang.org/x/sync/errgroup" {
		return nil
	}
	return call.Args[0]
}
