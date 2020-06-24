// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package undeclaredname defines an Analyzer that applies suggested fixes
// to errors of the type "undeclared name: %s".
package undeclaredname

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/analysisinternal"
)

const Doc = `suggested fixes for "undeclared name: <>"

This checker provides suggested fixes for type errors of the
type "undeclared name: <>". It will insert a new statement:
"<> := ".`

var Analyzer = &analysis.Analyzer{
	Name:             string(analysisinternal.UndeclaredName),
	Doc:              Doc,
	Requires:         []*analysis.Analyzer{},
	Run:              run,
	RunDespiteErrors: true,
}

const undeclaredNamePrefix = "undeclared name: "

func run(pass *analysis.Pass) (interface{}, error) {
	for _, err := range analysisinternal.GetTypeErrors(pass) {
		if !FixesError(err.Msg) {
			continue
		}
		name := strings.TrimPrefix(err.Msg, undeclaredNamePrefix)
		var file *ast.File
		for _, f := range pass.Files {
			if f.Pos() <= err.Pos && err.Pos < f.End() {
				file = f
				break
			}
		}
		if file == nil {
			continue
		}

		// Get the path for the relevant range.
		path, _ := astutil.PathEnclosingInterval(file, err.Pos, err.Pos)
		if len(path) < 2 {
			continue
		}
		ident, ok := path[0].(*ast.Ident)
		if !ok || ident.Name != name {
			continue
		}
		// Skip selector expressions because it might be too complex
		// to try and provide a suggested fix for fields and methods.
		if _, ok := path[1].(*ast.SelectorExpr); ok {
			continue
		}
		// TODO(golang.org/issue/34644): in a follow up handle call expressions
		// with suggested fix to create function
		if _, ok := path[1].(*ast.CallExpr); ok {
			continue
		}

		// Get the place to insert the new statement.
		insertBeforeStmt := analysisinternal.StmtToInsertVarBefore(path)
		if insertBeforeStmt == nil {
			continue
		}

		var buf bytes.Buffer
		if err := printer.Fprint(&buf, pass.Fset, file); err != nil {
			continue
		}
		old := buf.Bytes()
		insertBefore := pass.Fset.Position(insertBeforeStmt.Pos()).Offset

		// Get the indent to add on the line after the new statement.
		// Since this will have a parse error, we can not use format.Source().
		contentBeforeStmt, indent := old[:insertBefore], "\n"
		if nl := bytes.LastIndex(contentBeforeStmt, []byte("\n")); nl != -1 {
			indent = string(contentBeforeStmt[nl:])
		}
		// Create the new local variable statement.
		newStmt := fmt.Sprintf("%s := %s", ident.Name, indent)

		pass.Report(analysis.Diagnostic{
			Pos:     err.Pos,
			End:     analysisinternal.TypeErrorEndPos(pass.Fset, old, err.Pos),
			Message: err.Msg,
			SuggestedFixes: []analysis.SuggestedFix{{
				Message: fmt.Sprintf("Create variable \"%s\"", ident.Name),
				TextEdits: []analysis.TextEdit{{
					Pos:     insertBeforeStmt.Pos(),
					End:     insertBeforeStmt.Pos(),
					NewText: []byte(newStmt),
				}},
			}},
		})
	}
	return nil, nil
}

func FixesError(msg string) bool {
	return strings.HasPrefix(msg, undeclaredNamePrefix)
}
