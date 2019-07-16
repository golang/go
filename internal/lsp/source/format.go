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
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/trace"
	"golang.org/x/tools/internal/span"
)

// Format formats a file with a given range.
func Format(ctx context.Context, f GoFile, rng span.Range) ([]TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.Format")
	defer done()

	file, err := f.GetAST(ctx, ParseFull)
	if file == nil {
		return nil, err
	}
	pkg := f.GetPackage(ctx)
	if hasListErrors(pkg.GetErrors()) || hasParseErrors(pkg.GetErrors()) {
		return nil, fmt.Errorf("%s has parse errors, not formatting", f.URI())
	}
	path, exact := astutil.PathEnclosingInterval(file, rng.Start, rng.End)
	if !exact || len(path) == 0 {
		return nil, fmt.Errorf("no exact AST node matching the specified range")
	}
	node := path[0]

	fset := f.FileSet()
	buf := &bytes.Buffer{}

	// format.Node changes slightly from one release to another, so the version
	// of Go used to build the LSP server will determine how it formats code.
	// This should be acceptable for all users, who likely be prompted to rebuild
	// the LSP server on each Go release.
	if err := format.Node(buf, fset, node); err != nil {
		return nil, err
	}
	return computeTextEdits(ctx, f, buf.String()), nil
}

// Imports formats a file using the goimports tool.
func Imports(ctx context.Context, view View, f GoFile, rng span.Range) ([]TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.Imports")
	defer done()
	data, _, err := f.Handle(ctx).Read(ctx)
	if err != nil {
		return nil, err
	}
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, fmt.Errorf("no package for file %s", f.URI())
	}
	if hasListErrors(pkg.GetErrors()) {
		return nil, fmt.Errorf("%s has list errors, not running goimports", f.URI())
	}

	if resolver, ok := view.ProcessEnv(ctx).GetResolver().(*imports.ModuleResolver); ok && resolver.Initialized {
		// TODO(suzmue): only reset this state when necessary (eg when the go.mod files of this
		// module or modules with replace directive changes).
		resolver.Initialized = false
		resolver.Main = nil
		resolver.ModsByModPath = nil
		resolver.ModsByDir = nil
		resolver.ModCachePkgs = nil
	}
	options := &imports.Options{
		Env: view.ProcessEnv(ctx),
		// Defaults.
		AllErrors:  true,
		Comments:   true,
		Fragment:   true,
		FormatOnly: false,
		TabIndent:  true,
		TabWidth:   8,
	}
	formatted, err := imports.Process(f.URI().Filename(), data, options)
	if err != nil {
		return nil, err
	}
	return computeTextEdits(ctx, f, string(formatted)), nil
}

func hasParseErrors(errors []packages.Error) bool {
	for _, err := range errors {
		if err.Kind == packages.ParseError {
			return true
		}
	}
	return false
}

func hasListErrors(errors []packages.Error) bool {
	for _, err := range errors {
		if err.Kind == packages.ListError {
			return true
		}
	}
	return false
}

func computeTextEdits(ctx context.Context, file File, formatted string) (edits []TextEdit) {
	ctx, done := trace.StartSpan(ctx, "source.computeTextEdits")
	defer done()
	data, _, err := file.Handle(ctx).Read(ctx)
	if err != nil {
		log.Error(ctx, "Cannot compute text edits", err)
		return nil
	}
	u := diff.SplitLines(string(data))
	f := diff.SplitLines(formatted)
	return DiffToEdits(file.URI(), diff.Operations(u, f))
}
