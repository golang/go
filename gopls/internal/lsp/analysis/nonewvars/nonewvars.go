// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nonewvars defines an Analyzer that applies suggested fixes
// to errors of the type "no new variables on left side of :=".
package nonewvars

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/token"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/analysisinternal"
)

const Doc = `suggested fixes for "no new vars on left side of :="

This checker provides suggested fixes for type errors of the
type "no new vars on left side of :=". For example:
	z := 1
	z := 2
will turn into
	z := 1
	z = 2
`

var Analyzer = &analysis.Analyzer{
	Name:             string(analysisinternal.NoNewVars),
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{inspect.Analyzer},
	Run:              run,
	RunDespiteErrors: true,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	errors := analysisinternal.GetTypeErrors(pass)

	nodeFilter := []ast.Node{(*ast.AssignStmt)(nil)}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		assignStmt, _ := n.(*ast.AssignStmt)
		// We only care about ":=".
		if assignStmt.Tok != token.DEFINE {
			return
		}

		var file *ast.File
		for _, f := range pass.Files {
			if f.Pos() <= assignStmt.Pos() && assignStmt.Pos() < f.End() {
				file = f
				break
			}
		}
		if file == nil {
			return
		}

		for _, err := range errors {
			if !FixesError(err.Msg) {
				continue
			}
			if assignStmt.Pos() > err.Pos || err.Pos >= assignStmt.End() {
				continue
			}
			var buf bytes.Buffer
			if err := format.Node(&buf, pass.Fset, file); err != nil {
				continue
			}
			pass.Report(analysis.Diagnostic{
				Pos:     err.Pos,
				End:     analysisinternal.TypeErrorEndPos(pass.Fset, buf.Bytes(), err.Pos),
				Message: err.Msg,
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: "Change ':=' to '='",
					TextEdits: []analysis.TextEdit{{
						Pos: err.Pos,
						End: err.Pos + 1,
					}},
				}},
			})
		}
	})
	return nil, nil
}

func FixesError(msg string) bool {
	return msg == "no new variables on left side of :="
}
