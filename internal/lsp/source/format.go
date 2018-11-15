// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/format"

	"golang.org/x/tools/go/ast/astutil"
)

// Format formats a document with a given range.
func Format(ctx context.Context, f *File, rng Range) ([]TextEdit, error) {
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
	buf := &bytes.Buffer{}
	if err := format.Node(buf, f.view.Config.Fset, node); err != nil {
		return nil, err
	}
	// TODO(rstambler): Compute text edits instead of replacing whole file.
	return []TextEdit{
		{
			Range:   rng,
			NewText: buf.String(),
		},
	}, nil
}
