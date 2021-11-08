// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testinggoroutine

import (
	"go/ast"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `report calls to (*testing.T).Fatal from goroutines started by a test.

Functions that abruptly terminate a test, such as the Fatal, Fatalf, FailNow, and
Skip{,f,Now} methods of *testing.T, must be called from the test goroutine itself.
This checker detects calls to these functions that occur within a goroutine
started by the test. For example:

func TestFoo(t *testing.T) {
    go func() {
        t.Fatal("oops") // error: (*T).Fatal called from non-test goroutine
    }()
}
`

var Analyzer = &analysis.Analyzer{
	Name:     "testinggoroutine",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

var forbidden = map[string]bool{
	"FailNow": true,
	"Fatal":   true,
	"Fatalf":  true,
	"Skip":    true,
	"Skipf":   true,
	"SkipNow": true,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	if !analysisutil.Imports(pass.Pkg, "testing") {
		return nil, nil
	}

	// Filter out anything that isn't a function declaration.
	onlyFuncs := []ast.Node{
		(*ast.FuncDecl)(nil),
	}

	inspect.Nodes(onlyFuncs, func(node ast.Node, push bool) bool {
		fnDecl, ok := node.(*ast.FuncDecl)
		if !ok {
			return false
		}

		if !hasBenchmarkOrTestParams(fnDecl) {
			return false
		}

		// Now traverse the benchmark/test's body and check that none of the
		// forbidden methods are invoked in the goroutines within the body.
		ast.Inspect(fnDecl, func(n ast.Node) bool {
			goStmt, ok := n.(*ast.GoStmt)
			if !ok {
				return true
			}

			checkGoStmt(pass, goStmt)

			// No need to further traverse the GoStmt since right
			// above we manually traversed it in the ast.Inspect(goStmt, ...)
			return false
		})

		return false
	})

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

// goStmtFunc returns the ast.Node of a call expression
// that was invoked as a go statement. Currently, only
// function literals declared in the same function, and
// static calls within the same package are supported.
func goStmtFun(goStmt *ast.GoStmt) ast.Node {
	switch goStmt.Call.Fun.(type) {
	case *ast.Ident:
		id := goStmt.Call.Fun.(*ast.Ident)
		// TODO(cuonglm): improve this once golang/go#48141 resolved.
		if id.Obj == nil {
			break
		}
		if funDecl, ok := id.Obj.Decl.(ast.Node); ok {
			return funDecl
		}
	case *ast.FuncLit:
		return goStmt.Call.Fun
	}
	return goStmt.Call
}

// checkGoStmt traverses the goroutine and checks for the
// use of the forbidden *testing.(B, T) methods.
func checkGoStmt(pass *analysis.Pass, goStmt *ast.GoStmt) {
	fn := goStmtFun(goStmt)
	// Otherwise examine the goroutine to check for the forbidden methods.
	ast.Inspect(fn, func(n ast.Node) bool {
		selExpr, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}

		_, bad := forbidden[selExpr.Sel.Name]
		if !bad {
			return true
		}

		// Now filter out false positives by the import-path/type.
		ident, ok := selExpr.X.(*ast.Ident)
		if !ok {
			return true
		}
		if ident.Obj == nil || ident.Obj.Decl == nil {
			return true
		}
		field, ok := ident.Obj.Decl.(*ast.Field)
		if !ok {
			return true
		}
		if typeName, ok := typeIsTestingDotTOrB(field.Type); ok {
			var fnRange analysis.Range = goStmt
			if _, ok := fn.(*ast.FuncLit); ok {
				fnRange = selExpr
			}
			pass.ReportRangef(fnRange, "call to (*%s).%s from a non-test goroutine", typeName, selExpr.Sel)
		}
		return true
	})
}
