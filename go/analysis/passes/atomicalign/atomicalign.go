// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package atomicalign defines an Analyzer that checks for non-64-bit-aligned
// arguments to sync/atomic functions. On non-32-bit platforms, those functions
// panic if their argument variables are not 64-bit aligned. It is therefore
// the caller's responsibility to arrange for 64-bit alignment of such variables.
// See https://golang.org/pkg/sync/atomic/#pkg-note-BUG
package atomicalign

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = "check for non-64-bits-aligned arguments to sync/atomic functions"

var Analyzer = &analysis.Analyzer{
	Name:     "atomicalign",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/atomicalign",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	if 8*pass.TypesSizes.Sizeof(types.Typ[types.Uintptr]) == 64 {
		return nil, nil // 64-bit platform
	}
	if !analysisutil.Imports(pass.Pkg, "sync/atomic") {
		return nil, nil // doesn't directly import sync/atomic
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}

	inspect.Preorder(nodeFilter, func(node ast.Node) {
		call := node.(*ast.CallExpr)
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return
		}
		pkgIdent, ok := sel.X.(*ast.Ident)
		if !ok {
			return
		}
		pkgName, ok := pass.TypesInfo.Uses[pkgIdent].(*types.PkgName)
		if !ok || pkgName.Imported().Path() != "sync/atomic" {
			return
		}

		switch sel.Sel.Name {
		case "AddInt64", "AddUint64",
			"LoadInt64", "LoadUint64",
			"StoreInt64", "StoreUint64",
			"SwapInt64", "SwapUint64",
			"CompareAndSwapInt64", "CompareAndSwapUint64":

			// For all the listed functions, the expression to check is always the first function argument.
			check64BitAlignment(pass, sel.Sel.Name, call.Args[0])
		}
	})

	return nil, nil
}

func check64BitAlignment(pass *analysis.Pass, funcName string, arg ast.Expr) {
	// Checks the argument is made of the address operator (&) applied to
	// to a struct field (as opposed to a variable as the first word of
	// uint64 and int64 variables can be relied upon to be 64-bit aligned.
	unary, ok := arg.(*ast.UnaryExpr)
	if !ok || unary.Op != token.AND {
		return
	}

	// Retrieve the types.Struct in order to get the offset of the
	// atomically accessed field.
	sel, ok := unary.X.(*ast.SelectorExpr)
	if !ok {
		return
	}
	tvar, ok := pass.TypesInfo.Selections[sel].Obj().(*types.Var)
	if !ok || !tvar.IsField() {
		return
	}

	stype, ok := pass.TypesInfo.Types[sel.X].Type.Underlying().(*types.Struct)
	if !ok {
		return
	}

	var offset int64
	var fields []*types.Var
	for i := 0; i < stype.NumFields(); i++ {
		f := stype.Field(i)
		fields = append(fields, f)
		if f == tvar {
			// We're done, this is the field we were looking for,
			// no need to fill the fields slice further.
			offset = pass.TypesSizes.Offsetsof(fields)[i]
			break
		}
	}
	if offset&7 == 0 {
		return // 64-bit aligned
	}

	pass.ReportRangef(arg, "address of non 64-bit aligned field .%s passed to atomic.%s", tvar.Name(), funcName)
}
