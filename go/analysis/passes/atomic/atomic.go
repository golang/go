// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
)

var Analyzer = &analysis.Analyzer{
	Name: "atomic",
	Doc: `check for common mistakes using the sync/atomic package

The atomic checker looks for assignment statements of the form:

	x = atomic.AddUint64(&x, 1)

which are not atomic.`,
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	RunDespiteErrors: true,
	Run:              run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.AssignStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(node ast.Node) {
		n := node.(*ast.AssignStmt)
		if len(n.Lhs) != len(n.Rhs) {
			return
		}
		if len(n.Lhs) == 1 && n.Tok == token.DEFINE {
			return
		}

		for i, right := range n.Rhs {
			call, ok := right.(*ast.CallExpr)
			if !ok {
				continue
			}
			sel, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				continue
			}
			pkgIdent, _ := sel.X.(*ast.Ident)
			pkgName, ok := pass.TypesInfo.Uses[pkgIdent].(*types.PkgName)
			if !ok || pkgName.Imported().Path() != "sync/atomic" {
				continue
			}

			switch sel.Sel.Name {
			case "AddInt32", "AddInt64", "AddUint32", "AddUint64", "AddUintptr":
				checkAtomicAddAssignment(pass, n.Lhs[i], call)
			}
		}
	})
	return nil, nil
}

// checkAtomicAddAssignment walks the atomic.Add* method calls checking
// for assigning the return value to the same variable being used in the
// operation
func checkAtomicAddAssignment(pass *analysis.Pass, left ast.Expr, call *ast.CallExpr) {
	if len(call.Args) != 2 {
		return
	}
	arg := call.Args[0]
	broken := false

	gofmt := func(e ast.Expr) string { return analysisutil.Format(pass.Fset, e) }

	if uarg, ok := arg.(*ast.UnaryExpr); ok && uarg.Op == token.AND {
		broken = gofmt(left) == gofmt(uarg.X)
	} else if star, ok := left.(*ast.StarExpr); ok {
		broken = gofmt(star.X) == gofmt(arg)
	}

	if broken {
		pass.Reportf(left.Pos(), "direct assignment to atomic value")
	}
}
