// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package source provides core features for use by Go editors and tools.
package source

import (
	"bytes"
	"context"
	"go/ast"
	"go/format"
	"go/token"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

// Format formats a file with a given range.
func Format(ctx context.Context, view View, f File) ([]protocol.TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.Format")
	defer done()

	gof, ok := f.(GoFile)
	if !ok {
		return nil, errors.Errorf("formatting is not supported for non-Go files")
	}
	pkg, err := gof.GetPackage(ctx)
	if err != nil {
		return nil, err
	}
	var file *ast.File
	for _, ph := range pkg.GetHandles() {
		if ph.File().Identity().URI == f.URI() {
			file, err = ph.Cached(ctx)
		}
	}
	if file == nil {
		return nil, err
	}
	if hasListErrors(pkg.GetErrors()) || hasParseErrors(pkg, f.URI()) {
		// Even if this package has list or parse errors, this file may not
		// have any parse errors and can still be formatted. Using format.Node
		// on an ast with errors may result in code being added or removed.
		// Attempt to format the source of this file instead.
		formatted, err := formatSource(ctx, f)
		if err != nil {
			return nil, err
		}
		return computeTextEdits(ctx, view.Session().Cache().FileSet(), f, string(formatted))
	}

	fset := view.Session().Cache().FileSet()
	buf := &bytes.Buffer{}

	// format.Node changes slightly from one release to another, so the version
	// of Go used to build the LSP server will determine how it formats code.
	// This should be acceptable for all users, who likely be prompted to rebuild
	// the LSP server on each Go release.
	if err := format.Node(buf, fset, file); err != nil {
		return nil, err
	}
	return computeTextEdits(ctx, view.Session().Cache().FileSet(), f, buf.String())
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
func Imports(ctx context.Context, view View, f GoFile, rng span.Range) ([]protocol.TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.Imports")
	defer done()
	data, _, err := f.Handle(ctx).Read(ctx)
	if err != nil {
		return nil, err
	}
	pkg, err := f.GetPackage(ctx)
	if err != nil {
		return nil, err
	}
	if hasListErrors(pkg.GetErrors()) {
		return nil, errors.Errorf("%s has list errors, not running goimports", f.URI())
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
	return computeTextEdits(ctx, view.Session().Cache().FileSet(), f, string(formatted))
}

type ImportFix struct {
	Fix   *imports.ImportFix
	Edits []protocol.TextEdit
}

// AllImportsFixes formats f for each possible fix to the imports.
// In addition to returning the result of applying all edits,
// it returns a list of fixes that could be applied to the file, with the
// corresponding TextEdits that would be needed to apply that fix.
func AllImportsFixes(ctx context.Context, view View, f File) (edits []protocol.TextEdit, editsPerFix []*ImportFix, err error) {
	ctx, done := trace.StartSpan(ctx, "source.AllImportsFixes")
	defer done()

	gof, ok := f.(GoFile)
	if !ok {
		return nil, nil, errors.Errorf("no imports fixes for non-Go files: %v", err)
	}

	data, _, err := f.Handle(ctx).Read(ctx)
	if err != nil {
		return nil, nil, err
	}
	pkg, err := gof.GetPackage(ctx)
	if err != nil {
		return nil, nil, err
	}
	if hasListErrors(pkg.GetErrors()) {
		return nil, nil, errors.Errorf("%s has list errors, not running goimports", f.URI())
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
		edits, err = computeTextEdits(ctx, view.Session().Cache().FileSet(), f, string(formatted))
		if err != nil {
			return err
		}
		// Add the edits for each fix to the result.
		editsPerFix = make([]*ImportFix, len(fixes))
		for i, fix := range fixes {
			formatted, err := imports.ApplyFixes([]*imports.ImportFix{fix}, f.URI().Filename(), data, options)
			if err != nil {
				return err
			}
			edits, err := computeTextEdits(ctx, view.Session().Cache().FileSet(), f, string(formatted))
			if err != nil {
				return err
			}
			editsPerFix[i] = &ImportFix{
				Fix:   fix,
				Edits: edits,
			}
		}
		return nil
	}
	err = view.RunProcessEnvFunc(ctx, importFn, options)
	if err != nil {
		return nil, nil, err
	}

	return edits, editsPerFix, nil
}

// AllImportsFixes formats f for each possible fix to the imports.
// In addition to returning the result of applying all edits,
// it returns a list of fixes that could be applied to the file, with the
// corresponding TextEdits that would be needed to apply that fix.
func CandidateImports(ctx context.Context, view View, filename string) (pkgs []imports.ImportFix, err error) {
	ctx, done := trace.StartSpan(ctx, "source.CandidateImports")
	defer done()

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
		pkgs, err = imports.GetAllCandidates(filename, opts)
		return err
	}
	err = view.RunProcessEnvFunc(ctx, importFn, options)
	if err != nil {
		return nil, err
	}

	return pkgs, nil
}

// hasParseErrors returns true if the given file has parse errors.
func hasParseErrors(pkg Package, uri span.URI) bool {
	for _, err := range pkg.GetErrors() {
		spn := packagesErrorSpan(err)
		if spn.URI() == uri && err.Kind == packages.ParseError {
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

func computeTextEdits(ctx context.Context, fset *token.FileSet, f File, formatted string) ([]protocol.TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.computeTextEdits")
	defer done()

	data, _, err := f.Handle(ctx).Read(ctx)
	if err != nil {
		return nil, err
	}
	edits := diff.ComputeEdits(f.URI(), string(data), formatted)
	m := protocol.NewColumnMapper(f.URI(), f.URI().Filename(), fset, nil, data)

	return ToProtocolEdits(m, edits)

}

func ToProtocolEdits(m *protocol.ColumnMapper, edits []diff.TextEdit) ([]protocol.TextEdit, error) {
	if edits == nil {
		return nil, nil
	}
	result := make([]protocol.TextEdit, len(edits))
	for i, edit := range edits {
		rng, err := m.Range(edit.Span)
		if err != nil {
			return nil, err
		}
		result[i] = protocol.TextEdit{
			Range:   rng,
			NewText: edit.NewText,
		}
	}
	return result, nil
}

func FromProtocolEdits(m *protocol.ColumnMapper, edits []protocol.TextEdit) ([]diff.TextEdit, error) {
	if edits == nil {
		return nil, nil
	}
	result := make([]diff.TextEdit, len(edits))
	for i, edit := range edits {
		spn, err := m.RangeSpan(edit.Range)
		if err != nil {
			return nil, err
		}
		result[i] = diff.TextEdit{
			Span:    spn,
			NewText: edit.NewText,
		}
	}
	return result, nil
}
