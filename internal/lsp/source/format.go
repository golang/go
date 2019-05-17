// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package source provides core features for use by Go editors and tools.
package source

import (
	"bytes"
	"context"
	"fmt"
	"go/format"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/imports"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/span"
)

// Format formats a file with a given range.
func Format(ctx context.Context, f GoFile, rng span.Range) ([]TextEdit, error) {
	file := f.GetAST(ctx)
	if file == nil {
		return nil, fmt.Errorf("no AST for %s", f.URI())
	}
	pkg := f.GetPackage(ctx)
	if hasParseErrors(pkg.GetErrors()) {
		return nil, fmt.Errorf("%s has parse errors, not formatting", f.URI())
	}
	path, exact := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if !exact || len(path) == 0 {
		return nil, fmt.Errorf("no exact AST node matching the specified range")
	}
	node := path[0]
	// format.Node changes slightly from one release to another, so the version
	// of Go used to build the LSP server will determine how it formats code.
	// This should be acceptable for all users, who likely be prompted to rebuild
	// the LSP server on each Go release.
	fset := f.FileSet()
	buf := &bytes.Buffer{}
	if err := format.Node(buf, fset, node); err != nil {
		return nil, err
	}
	return computeTextEdits(ctx, f, buf.String()), nil
}

func hasParseErrors(errors []packages.Error) bool {
	for _, err := range errors {
		if err.Kind == packages.ParseError {
			return true
		}
	}
	return false
}

// Imports formats a file using the goimports tool.
func Imports(ctx context.Context, f GoFile, rng span.Range) ([]TextEdit, error) {
	tok := f.GetToken(ctx)
	if tok == nil {
		return nil, fmt.Errorf("no token file for %s", f.URI())
	}
	formatted, err := imports.Process(tok.Name(), f.GetContent(ctx), nil)
	if err != nil {
		return nil, err
	}
	return computeTextEdits(ctx, f, string(formatted)), nil
}

func computeTextEdits(ctx context.Context, file File, formatted string) (edits []TextEdit) {
	u := diff.SplitLines(string(file.GetContent(ctx)))
	f := diff.SplitLines(formatted)
	return DiffToEdits(file.URI(), diff.Operations(u, f))
}
