// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package simplifyrange defines an Analyzer that simplifies range statements.
// https://golang.org/cmd/gofmt/#hdr-The_simplify_command
// https://github.com/golang/go/blob/master/src/cmd/gofmt/simplify.go
package simplifyrange

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `check for range statement simplifications

A range of the form:
	for x, _ = range v {...}
will be simplified to:
	for x = range v {...}

A range of the form:
	for _ = range v {...}
will be simplified to:
	for range v {...}

This is one of the simplifications that "gofmt -s" applies.`

var Analyzer = &analysis.Analyzer{
	Name:     "simplifyrange",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.RangeStmt)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		var copy *ast.RangeStmt
		if stmt, ok := n.(*ast.RangeStmt); ok {
			x := *stmt
			copy = &x
		}
		if copy == nil {
			return
		}
		end := newlineIndex(pass.Fset, copy)

		// Range statements of the form: for i, _ := range x {}
		var old ast.Expr
		if isBlank(copy.Value) {
			old = copy.Value
			copy.Value = nil
		}
		// Range statements of the form: for _ := range x {}
		if isBlank(copy.Key) && copy.Value == nil {
			old = copy.Key
			copy.Key = nil
		}
		// Return early if neither if condition is met.
		if old == nil {
			return
		}
		pass.Report(analysis.Diagnostic{
			Pos:            old.Pos(),
			End:            old.End(),
			Message:        "simplify range expression",
			SuggestedFixes: suggestedFixes(pass.Fset, copy, end),
		})
	})
	return nil, nil
}

func suggestedFixes(fset *token.FileSet, rng *ast.RangeStmt, end token.Pos) []analysis.SuggestedFix {
	var b bytes.Buffer
	printer.Fprint(&b, fset, rng)
	stmt := b.Bytes()
	index := bytes.Index(stmt, []byte("\n"))
	// If there is a new line character, then don't replace the body.
	if index != -1 {
		stmt = stmt[:index]
	}
	return []analysis.SuggestedFix{{
		Message: "Remove empty value",
		TextEdits: []analysis.TextEdit{{
			Pos:     rng.Pos(),
			End:     end,
			NewText: stmt[:index],
		}},
	}}
}

func newlineIndex(fset *token.FileSet, rng *ast.RangeStmt) token.Pos {
	var b bytes.Buffer
	printer.Fprint(&b, fset, rng)
	contents := b.Bytes()
	index := bytes.Index(contents, []byte("\n"))
	if index == -1 {
		return rng.End()
	}
	return rng.Pos() + token.Pos(index)
}

func isBlank(x ast.Expr) bool {
	ident, ok := x.(*ast.Ident)
	return ok && ident.Name == "_"
}
