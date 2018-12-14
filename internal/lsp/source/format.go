// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package source provides core features for use by Go editors and tools.
package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/format"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/imports"
)

// Format formats a file with a given range.
func Format(ctx context.Context, f File, rng Range) ([]TextEdit, error) {
	fAST, err := f.GetAST()
	if err != nil {
		return nil, err
	}
	path, exact := astutil.PathEnclosingInterval(fAST, rng.Start, rng.End)
	if !exact || len(path) == 0 {
		return nil, fmt.Errorf("no exact AST node matching the specified range")
	}
	node := path[0]
	// format.Node can fail when the AST contains a bad expression or
	// statement. For now, we preemptively check for one.
	// TODO(rstambler): This should really return an error from format.Node.
	var isBad bool
	ast.Inspect(node, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.BadDecl, *ast.BadExpr, *ast.BadStmt:
			isBad = true
			return false
		default:
			return true
		}
	})
	if isBad {
		return nil, fmt.Errorf("unable to format file due to a badly formatted AST")
	}
	// format.Node changes slightly from one release to another, so the version
	// of Go used to build the LSP server will determine how it formats code.
	// This should be acceptable for all users, who likely be prompted to rebuild
	// the LSP server on each Go release.
	fset, err := f.GetFileSet()
	if err != nil {
		return nil, err
	}
	buf := &bytes.Buffer{}
	if err := format.Node(buf, fset, node); err != nil {
		return nil, err
	}
	return computeTextEdits(rng, buf.String()), nil
}

// Imports formats a file using the goimports tool.
func Imports(ctx context.Context, filename string, content []byte, rng Range) ([]TextEdit, error) {
	content, err := imports.Process(filename, content, nil)
	if err != nil {
		return nil, err
	}
	return computeTextEdits(rng, string(content)), nil
}

// TODO(rstambler): Compute text edits instead of replacing whole file.
func computeTextEdits(rng Range, content string) []TextEdit {
	return []TextEdit{
		{
			Range:   rng,
			NewText: content,
		},
	}
}
