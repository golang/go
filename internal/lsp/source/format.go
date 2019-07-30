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
		// Even if this package has list or parse errors, this file may not
		// have any parse errors and can still be formatted. Using format.Node
		// on an ast with errors may result in code being added or removed.
		// Attempt to format the source of this file instead.
		formatted, err := formatSource(ctx, f)
		if err != nil {
			return nil, err
		}
		return computeTextEdits(ctx, f, string(formatted)), nil
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

func formatSource(ctx context.Context, file File) ([]byte, error) {
	ctx, done := trace.StartSpan(ctx, "source.formatSource")
	defer done()
	data, _, err := file.Handle(ctx).Read(ctx)
	if err != nil {
		return nil, err
	}
	return format.Source(data)
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

	options := &imports.Options{
		// Defaults.
		AllErrors:  true,
		Comments:   true,
		Fragment:   true,
		FormatOnly: false,
		TabIndent:  true,
		TabWidth:   8,
	}
	var formatted []byte
	importFn := func(opts *imports.Options) error {
		formatted, err = imports.Process(f.URI().Filename(), data, opts)
		return err
	}
	err = view.RunProcessEnvFunc(ctx, importFn, options)
	if err != nil {
		return nil, err
	}

	return computeTextEdits(ctx, f, string(formatted)), nil
}

type ImportFix struct {
	Fix   *imports.ImportFix
	Edits []TextEdit
}

// AllImportsFixes formats f for each possible fix to the imports.
// In addition to returning the result of applying all edits,
// it returns a list of fixes that could be applied to the file, with the
// corresponding TextEdits that would be needed to apply that fix.
func AllImportsFixes(ctx context.Context, view View, f GoFile, rng span.Range) (edits []TextEdit, editsPerFix []*ImportFix, err error) {
	ctx, done := trace.StartSpan(ctx, "source.AllImportsFixes")
	defer done()
	data, _, err := f.Handle(ctx).Read(ctx)
	if err != nil {
		return nil, nil, err
	}
	pkg := f.GetPackage(ctx)
	if pkg == nil || pkg.IsIllTyped() {
		return nil, nil, fmt.Errorf("no package for file %s", f.URI())
	}
	if hasListErrors(pkg.GetErrors()) {
		return nil, nil, fmt.Errorf("%s has list errors, not running goimports", f.URI())
	}

	options := &imports.Options{
		// Defaults.
		AllErrors:  true,
		Comments:   true,
		Fragment:   true,
		FormatOnly: false,
		TabIndent:  true,
		TabWidth:   8,
	}
	importFn := func(opts *imports.Options) error {
		fixes, err := imports.FixImports(f.URI().Filename(), data, opts)
		if err != nil {
			return err
		}
		// Apply all of the import fixes to the file.
		formatted, err := imports.ApplyFixes(fixes, f.URI().Filename(), data, options)
		if err != nil {
			return err
		}
		edits = computeTextEdits(ctx, f, string(formatted))
		// Add the edits for each fix to the result.
		editsPerFix = make([]*ImportFix, len(fixes))
		for i, fix := range fixes {
			formatted, err := imports.ApplyFixes([]*imports.ImportFix{fix}, f.URI().Filename(), data, options)
			if err != nil {
				return err
			}
			editsPerFix[i] = &ImportFix{
				Fix:   fix,
				Edits: computeTextEdits(ctx, f, string(formatted)),
			}
		}
		return err
	}
	err = view.RunProcessEnvFunc(ctx, importFn, options)
	if err != nil {
		return nil, nil, err
	}

	return edits, editsPerFix, nil
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
