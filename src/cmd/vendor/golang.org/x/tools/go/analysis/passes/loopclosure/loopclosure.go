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

This analyzer checks for references to loop variables from within a function
literal inside the loop body. It checks for patterns where access to a loop
variable is known to escape the current loop iteration:
 1. a call to go or defer at the end of the loop body
 2. a call to golang.org/x/sync/errgroup.Group.Go at the end of the loop body
 3. a call testing.T.Run where the subtest body invokes t.Parallel()

In the case of (1) and (2), the analyzer only considers references in the last
statement of the loop body as it is not deep enough to understand the effects
of subsequent statements which might render the reference benign.

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
		var vars []types.Object
		addVar := func(expr ast.Expr) {
			if id, _ := expr.(*ast.Ident); id != nil {
				if obj := pass.TypesInfo.ObjectOf(id); obj != nil {
					vars = append(vars, obj)
				}
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

		// Inspect statements to find function literals that may be run outside of
		// the current loop iteration.
		//
		// For go, defer, and errgroup.Group.Go, we ignore all but the last
		// statement, because it's hard to prove go isn't followed by wait, or
		// defer by return.
		//
		// We consider every t.Run statement in the loop body, because there is
		// no such commonly used mechanism for synchronizing parallel subtests.
		// It is of course theoretically possible to synchronize parallel subtests,
		// though such a pattern is likely to be exceedingly rare as it would be
		// fighting against the test runner.
		lastStmt := len(body.List) - 1
		for i, s := range body.List {
			var stmts []ast.Stmt // statements that must be checked for escaping references
			switch s := s.(type) {
			case *ast.GoStmt:
				if i == lastStmt {
					stmts = litStmts(s.Call.Fun)
				}

			case *ast.DeferStmt:
				if i == lastStmt {
					stmts = litStmts(s.Call.Fun)
				}

			case *ast.ExprStmt: // check for errgroup.Group.Go and testing.T.Run (with T.Parallel)
				if call, ok := s.X.(*ast.CallExpr); ok {
					if i == lastStmt {
						stmts = litStmts(goInvoke(pass.TypesInfo, call))
					}
					if stmts == nil {
						stmts = parallelSubtest(pass.TypesInfo, call)
					}
				}
			}

			for _, stmt := range stmts {
				ast.Inspect(stmt, func(n ast.Node) bool {
					id, ok := n.(*ast.Ident)
					if !ok {
						return true
					}
					obj := pass.TypesInfo.Uses[id]
					if obj == nil {
						return true
					}
					for _, v := range vars {
						if v == obj {
							pass.ReportRangef(id, "loop variable %s captured by func literal", id.Name)
						}
					}
					return true
				})
			}
		}
	})
	return nil, nil
}

// litStmts returns all statements from the function body of a function
// literal.
//
// If fun is not a function literal, it returns nil.
func litStmts(fun ast.Expr) []ast.Stmt {
	lit, _ := fun.(*ast.FuncLit)
	if lit == nil {
		return nil
	}
	return lit.Body.List
}

// goInvoke returns a function expression that would be called asynchronously
// (but not awaited) in another goroutine as a consequence of the call.
// For example, given the g.Go call below, it returns the function literal expression.
//
//	import "sync/errgroup"
//	var g errgroup.Group
//	g.Go(func() error { ... })
//
// Currently only "golang.org/x/sync/errgroup.Group()" is considered.
func goInvoke(info *types.Info, call *ast.CallExpr) ast.Expr {
	if !isMethodCall(info, call, "golang.org/x/sync/errgroup", "Group", "Go") {
		return nil
	}
	return call.Args[0]
}

// parallelSubtest returns statements that can be easily proven to execute
// concurrently via the go test runner, as t.Run has been invoked with a
// function literal that calls t.Parallel.
//
// In practice, users rely on the fact that statements before the call to
// t.Parallel are synchronous. For example by declaring test := test inside the
// function literal, but before the call to t.Parallel.
//
// Therefore, we only flag references in statements that are obviously
// dominated by a call to t.Parallel. As a simple heuristic, we only consider
// statements following the final labeled statement in the function body, to
// avoid scenarios where a jump would cause either the call to t.Parallel or
// the problematic reference to be skipped.
//
//	import "testing"
//
//	func TestFoo(t *testing.T) {
//		tests := []int{0, 1, 2}
//		for i, test := range tests {
//			t.Run("subtest", func(t *testing.T) {
//				println(i, test) // OK
//		 		t.Parallel()
//				println(i, test) // Not OK
//			})
//		}
//	}
func parallelSubtest(info *types.Info, call *ast.CallExpr) []ast.Stmt {
	if !isMethodCall(info, call, "testing", "T", "Run") {
		return nil
	}

	lit, _ := call.Args[1].(*ast.FuncLit)
	if lit == nil {
		return nil
	}

	// Capture the *testing.T object for the first argument to the function
	// literal.
	if len(lit.Type.Params.List[0].Names) == 0 {
		return nil
	}

	tObj := info.Defs[lit.Type.Params.List[0].Names[0]]
	if tObj == nil {
		return nil
	}

	// Match statements that occur after a call to t.Parallel following the final
	// labeled statement in the function body.
	//
	// We iterate over lit.Body.List to have a simple, fast and "frequent enough"
	// dominance relationship for t.Parallel(): lit.Body.List[i] dominates
	// lit.Body.List[j] for i < j unless there is a jump.
	var stmts []ast.Stmt
	afterParallel := false
	for _, stmt := range lit.Body.List {
		stmt, labeled := unlabel(stmt)
		if labeled {
			// Reset: naively we don't know if a jump could have caused the
			// previously considered statements to be skipped.
			stmts = nil
			afterParallel = false
		}

		if afterParallel {
			stmts = append(stmts, stmt)
			continue
		}

		// Check if stmt is a call to t.Parallel(), for the correct t.
		exprStmt, ok := stmt.(*ast.ExprStmt)
		if !ok {
			continue
		}
		expr := exprStmt.X
		if isMethodCall(info, expr, "testing", "T", "Parallel") {
			call, _ := expr.(*ast.CallExpr)
			if call == nil {
				continue
			}
			x, _ := call.Fun.(*ast.SelectorExpr)
			if x == nil {
				continue
			}
			id, _ := x.X.(*ast.Ident)
			if id == nil {
				continue
			}
			if info.Uses[id] == tObj {
				afterParallel = true
			}
		}
	}

	return stmts
}

// unlabel returns the inner statement for the possibly labeled statement stmt,
// stripping any (possibly nested) *ast.LabeledStmt wrapper.
//
// The second result reports whether stmt was an *ast.LabeledStmt.
func unlabel(stmt ast.Stmt) (ast.Stmt, bool) {
	labeled := false
	for {
		labelStmt, ok := stmt.(*ast.LabeledStmt)
		if !ok {
			return stmt, labeled
		}
		labeled = true
		stmt = labelStmt.Stmt
	}
}

// isMethodCall reports whether expr is a method call of
// <pkgPath>.<typeName>.<method>.
func isMethodCall(info *types.Info, expr ast.Expr, pkgPath, typeName, method string) bool {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return false
	}

	// Check that we are calling a method <method>
	f := typeutil.StaticCallee(info, call)
	if f == nil || f.Name() != method {
		return false
	}
	recv := f.Type().(*types.Signature).Recv()
	if recv == nil {
		return false
	}

	// Check that the receiver is a <pkgPath>.<typeName> or
	// *<pkgPath>.<typeName>.
	rtype := recv.Type()
	if ptr, ok := recv.Type().(*types.Pointer); ok {
		rtype = ptr.Elem()
	}
	named, ok := rtype.(*types.Named)
	if !ok {
		return false
	}
	if named.Obj().Name() != typeName {
		return false
	}
	pkg := f.Pkg()
	if pkg == nil {
		return false
	}
	if pkg.Path() != pkgPath {
		return false
	}

	return true
}
