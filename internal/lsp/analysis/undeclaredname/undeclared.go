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
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/analysisinternal"
	"golang.org/x/tools/internal/span"
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
		// TODO(golang.org/issue/34644): Handle call expressions with suggested
		// fixes to create a function.
		if _, ok := path[1].(*ast.CallExpr); ok {
			continue
		}
		tok := pass.Fset.File(file.Pos())
		if tok == nil {
			continue
		}
		offset := pass.Fset.Position(err.Pos).Offset
		end := tok.Pos(offset + len(name))
		pass.Report(analysis.Diagnostic{
			Pos:     err.Pos,
			End:     end,
			Message: err.Msg,
		})
	}
	return nil, nil
}

func SuggestedFix(fset *token.FileSet, rng span.Range, content []byte, file *ast.File, _ *types.Package, _ *types.Info) (*analysis.SuggestedFix, error) {
	pos := rng.Start // don't use the end
	path, _ := astutil.PathEnclosingInterval(file, pos, pos)
	if len(path) < 2 {
		return nil, fmt.Errorf("")
	}
	ident, ok := path[0].(*ast.Ident)
	if !ok {
		return nil, fmt.Errorf("")
	}
	// Get the place to insert the new statement.
	insertBeforeStmt := analysisinternal.StmtToInsertVarBefore(path)
	if insertBeforeStmt == nil {
		return nil, fmt.Errorf("")
	}

	insertBefore := fset.Position(insertBeforeStmt.Pos()).Offset

	// Get the indent to add on the line after the new statement.
	// Since this will have a parse error, we can not use format.Source().
	contentBeforeStmt, indent := content[:insertBefore], "\n"
	if nl := bytes.LastIndex(contentBeforeStmt, []byte("\n")); nl != -1 {
		indent = string(contentBeforeStmt[nl:])
	}
	// Create the new local variable statement.
	newStmt := fmt.Sprintf("%s := %s", ident.Name, indent)
	return &analysis.SuggestedFix{
		Message: fmt.Sprintf("Create variable \"%s\"", ident.Name),
		TextEdits: []analysis.TextEdit{{
			Pos:     insertBeforeStmt.Pos(),
			End:     insertBeforeStmt.Pos(),
			NewText: []byte(newStmt),
		}},
	}, nil
}

func FixesError(msg string) bool {
	return strings.HasPrefix(msg, undeclaredNamePrefix)
}
