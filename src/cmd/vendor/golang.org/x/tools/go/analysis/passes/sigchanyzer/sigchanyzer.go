// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sigchanyzer defines an Analyzer that detects
// misuse of unbuffered signal as argument to signal.Notify.
package sigchanyzer

import (
	"bytes"
	"slices"

	_ "embed"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
)

//go:embed doc.go
var doc string

// Analyzer describes sigchanyzer analysis function detector.
var Analyzer = &analysis.Analyzer{
	Name:     "sigchanyzer",
	Doc:      analysisutil.MustExtractDoc(doc, "sigchanyzer"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/sigchanyzer",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
	if !analysisinternal.Imports(pass.Pkg, "os/signal") {
		return nil, nil // doesn't directly import signal
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		call := n.(*ast.CallExpr)
		if !isSignalNotify(pass.TypesInfo, call) {
			return
		}
		var chanDecl *ast.CallExpr
		switch arg := call.Args[0].(type) {
		case *ast.Ident:
			if decl, ok := findDecl(arg).(*ast.CallExpr); ok {
				chanDecl = decl
			}
		case *ast.CallExpr:
			// Only signal.Notify(make(chan os.Signal), os.Interrupt) is safe,
			// conservatively treat others as not safe, see golang/go#45043
			if isBuiltinMake(pass.TypesInfo, arg) {
				return
			}
			chanDecl = arg
		}
		if chanDecl == nil || len(chanDecl.Args) != 1 {
			return
		}

		// Make a copy of the channel's declaration to avoid
		// mutating the AST. See https://golang.org/issue/46129.
		chanDeclCopy := &ast.CallExpr{}
		*chanDeclCopy = *chanDecl
		chanDeclCopy.Args = slices.Clone(chanDecl.Args)
		chanDeclCopy.Args = append(chanDeclCopy.Args, &ast.BasicLit{
			Kind:  token.INT,
			Value: "1",
		})

		var buf bytes.Buffer
		if err := format.Node(&buf, token.NewFileSet(), chanDeclCopy); err != nil {
			return
		}
		pass.Report(analysis.Diagnostic{
			Pos:     call.Pos(),
			End:     call.End(),
			Message: "misuse of unbuffered os.Signal channel as argument to signal.Notify",
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: "Change to buffer channel",
				TextEdits: []analysis.TextEdit{{
					Pos:     chanDecl.Pos(),
					End:     chanDecl.End(),
					NewText: buf.Bytes(),
				}},
			}},
		})
	})
	return nil, nil
}

func isSignalNotify(info *types.Info, call *ast.CallExpr) bool {
	check := func(id *ast.Ident) bool {
		obj := info.ObjectOf(id)
		return obj.Name() == "Notify" && obj.Pkg().Path() == "os/signal"
	}
	switch fun := call.Fun.(type) {
	case *ast.SelectorExpr:
		return check(fun.Sel)
	case *ast.Ident:
		if fun, ok := findDecl(fun).(*ast.SelectorExpr); ok {
			return check(fun.Sel)
		}
		return false
	default:
		return false
	}
}

func findDecl(arg *ast.Ident) ast.Node {
	if arg.Obj == nil {
		return nil
	}
	switch as := arg.Obj.Decl.(type) {
	case *ast.AssignStmt:
		if len(as.Lhs) != len(as.Rhs) {
			return nil
		}
		for i, lhs := range as.Lhs {
			lid, ok := lhs.(*ast.Ident)
			if !ok {
				continue
			}
			if lid.Obj == arg.Obj {
				return as.Rhs[i]
			}
		}
	case *ast.ValueSpec:
		if len(as.Names) != len(as.Values) {
			return nil
		}
		for i, name := range as.Names {
			if name.Obj == arg.Obj {
				return as.Values[i]
			}
		}
	}
	return nil
}

func isBuiltinMake(info *types.Info, call *ast.CallExpr) bool {
	typVal := info.Types[call.Fun]
	if !typVal.IsBuiltin() {
		return false
	}
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		return info.ObjectOf(fun).Name() == "make"
	default:
		return false
	}
}
