// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testinggoroutine

import (
	_ "embed"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/typesinternal"
)

//go:embed doc.go
var doc string

var reportSubtest bool

func init() {
	Analyzer.Flags.BoolVar(&reportSubtest, "subtest", false, "whether to check if t.Run subtest is terminated correctly; experimental")
}

var Analyzer = &analysis.Analyzer{
	Name:     "testinggoroutine",
	Doc:      analysisutil.MustExtractDoc(doc, "testinggoroutine"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/testinggoroutine",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	if !analysisinternal.Imports(pass.Pkg, "testing") {
		return nil, nil
	}

	toDecl := localFunctionDecls(pass.TypesInfo, pass.Files)

	// asyncs maps nodes whose statements will be executed concurrently
	// with respect to some test function, to the call sites where they
	// are invoked asynchronously. There may be multiple such call sites
	// for e.g. test helpers.
	asyncs := make(map[ast.Node][]*asyncCall)
	var regions []ast.Node
	addCall := func(c *asyncCall) {
		if c != nil {
			r := c.region
			if asyncs[r] == nil {
				regions = append(regions, r)
			}
			asyncs[r] = append(asyncs[r], c)
		}
	}

	// Collect all of the go callee() and t.Run(name, callee) extents.
	inspect.Nodes([]ast.Node{
		(*ast.FuncDecl)(nil),
		(*ast.GoStmt)(nil),
		(*ast.CallExpr)(nil),
	}, func(node ast.Node, push bool) bool {
		if !push {
			return false
		}
		switch node := node.(type) {
		case *ast.FuncDecl:
			return hasBenchmarkOrTestParams(node)

		case *ast.GoStmt:
			c := goAsyncCall(pass.TypesInfo, node, toDecl)
			addCall(c)

		case *ast.CallExpr:
			c := tRunAsyncCall(pass.TypesInfo, node)
			addCall(c)
		}
		return true
	})

	// Check for t.Forbidden() calls within each region r that is a
	// callee in some go r() or a t.Run("name", r).
	//
	// Also considers a special case when r is a go t.Forbidden() call.
	for _, region := range regions {
		ast.Inspect(region, func(n ast.Node) bool {
			if n == region {
				return true // always descend into the region itself.
			} else if asyncs[n] != nil {
				return false // will be visited by another region.
			}

			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			x, sel, fn := forbiddenMethod(pass.TypesInfo, call)
			if x == nil {
				return true
			}

			for _, e := range asyncs[region] {
				if !withinScope(e.scope, x) {
					forbidden := formatMethod(sel, fn) // e.g. "(*testing.T).Forbidden

					var context string
					var where analysis.Range = e.async // Put the report at the go fun() or t.Run(name, fun).
					if _, local := e.fun.(*ast.FuncLit); local {
						where = call // Put the report at the t.Forbidden() call.
					} else if id, ok := e.fun.(*ast.Ident); ok {
						context = fmt.Sprintf(" (%s calls %s)", id.Name, forbidden)
					}
					if _, ok := e.async.(*ast.GoStmt); ok {
						pass.ReportRangef(where, "call to %s from a non-test goroutine%s", forbidden, context)
					} else if reportSubtest {
						pass.ReportRangef(where, "call to %s on %s defined outside of the subtest%s", forbidden, x.Name(), context)
					}
				}
			}
			return true
		})
	}

	return nil, nil
}

func hasBenchmarkOrTestParams(fnDecl *ast.FuncDecl) bool {
	// Check that the function's arguments include "*testing.T" or "*testing.B".
	params := fnDecl.Type.Params.List

	for _, param := range params {
		if _, ok := typeIsTestingDotTOrB(param.Type); ok {
			return true
		}
	}

	return false
}

func typeIsTestingDotTOrB(expr ast.Expr) (string, bool) {
	starExpr, ok := expr.(*ast.StarExpr)
	if !ok {
		return "", false
	}
	selExpr, ok := starExpr.X.(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	varPkg := selExpr.X.(*ast.Ident)
	if varPkg.Name != "testing" {
		return "", false
	}

	varTypeName := selExpr.Sel.Name
	ok = varTypeName == "B" || varTypeName == "T"
	return varTypeName, ok
}

// asyncCall describes a region of code that needs to be checked for
// t.Forbidden() calls as it is started asynchronously from an async
// node go fun() or t.Run(name, fun).
type asyncCall struct {
	region ast.Node // region of code to check for t.Forbidden() calls.
	async  ast.Node // *ast.GoStmt or *ast.CallExpr (for t.Run)
	scope  ast.Node // Report t.Forbidden() if t is not declared within scope.
	fun    ast.Expr // fun in go fun() or t.Run(name, fun)
}

// withinScope returns true if x.Pos() is in [scope.Pos(), scope.End()].
func withinScope(scope ast.Node, x *types.Var) bool {
	if scope != nil {
		return x.Pos() != token.NoPos && scope.Pos() <= x.Pos() && x.Pos() <= scope.End()
	}
	return false
}

// goAsyncCall returns the extent of a call from a go fun() statement.
func goAsyncCall(info *types.Info, goStmt *ast.GoStmt, toDecl func(*types.Func) *ast.FuncDecl) *asyncCall {
	call := goStmt.Call

	fun := ast.Unparen(call.Fun)
	if id := typesinternal.UsedIdent(info, fun); id != nil {
		if lit := funcLitInScope(id); lit != nil {
			return &asyncCall{region: lit, async: goStmt, scope: nil, fun: fun}
		}
	}

	if fn := typeutil.StaticCallee(info, call); fn != nil { // static call or method in the package?
		if decl := toDecl(fn); decl != nil {
			return &asyncCall{region: decl, async: goStmt, scope: nil, fun: fun}
		}
	}

	// Check go statement for go t.Forbidden() or go func(){t.Forbidden()}().
	return &asyncCall{region: goStmt, async: goStmt, scope: nil, fun: fun}
}

// tRunAsyncCall returns the extent of a call from a t.Run("name", fun) expression.
func tRunAsyncCall(info *types.Info, call *ast.CallExpr) *asyncCall {
	if len(call.Args) != 2 {
		return nil
	}
	run := typeutil.Callee(info, call)
	if run, ok := run.(*types.Func); !ok || !isMethodNamed(run, "testing", "Run") {
		return nil
	}

	fun := ast.Unparen(call.Args[1])
	if lit, ok := fun.(*ast.FuncLit); ok { // function lit?
		return &asyncCall{region: lit, async: call, scope: lit, fun: fun}
	}

	if id := typesinternal.UsedIdent(info, fun); id != nil {
		if lit := funcLitInScope(id); lit != nil { // function lit in variable?
			return &asyncCall{region: lit, async: call, scope: lit, fun: fun}
		}
	}

	// Check within t.Run(name, fun) for calls to t.Forbidden,
	// e.g. t.Run(name, func(t *testing.T){ t.Forbidden() })
	return &asyncCall{region: call, async: call, scope: fun, fun: fun}
}

var forbidden = []string{
	"FailNow",
	"Fatal",
	"Fatalf",
	"Skip",
	"Skipf",
	"SkipNow",
}

// forbiddenMethod decomposes a call x.m() into (x, x.m, m) where
// x is a variable, x.m is a selection, and m is the static callee m.
// Returns (nil, nil, nil) if call is not of this form.
func forbiddenMethod(info *types.Info, call *ast.CallExpr) (*types.Var, *types.Selection, *types.Func) {
	// Compare to typeutil.StaticCallee.
	fun := ast.Unparen(call.Fun)
	selExpr, ok := fun.(*ast.SelectorExpr)
	if !ok {
		return nil, nil, nil
	}
	sel := info.Selections[selExpr]
	if sel == nil {
		return nil, nil, nil
	}

	var x *types.Var
	if id, ok := ast.Unparen(selExpr.X).(*ast.Ident); ok {
		x, _ = info.Uses[id].(*types.Var)
	}
	if x == nil {
		return nil, nil, nil
	}

	fn, _ := sel.Obj().(*types.Func)
	if fn == nil || !isMethodNamed(fn, "testing", forbidden...) {
		return nil, nil, nil
	}
	return x, sel, fn
}

func formatMethod(sel *types.Selection, fn *types.Func) string {
	var ptr string
	rtype := sel.Recv()
	if p, ok := types.Unalias(rtype).(*types.Pointer); ok {
		ptr = "*"
		rtype = p.Elem()
	}
	return fmt.Sprintf("(%s%s).%s", ptr, rtype.String(), fn.Name())
}
