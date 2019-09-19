// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package source provides core features for use by Go editors and tools.
package source

import (
	"bytes"
	"context"
	"go/format"

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
	cphs, err := gof.CheckPackageHandles(ctx)
	if err != nil {
		return nil, err
	}
	cph := NarrowestCheckPackageHandle(cphs)
	pkg, err := cph.Check(ctx)
	if err != nil {
		return nil, err
	}
	ph, err := pkg.File(f.URI())
	if err != nil {
		return nil, err
	}
	// Be extra careful that the file's ParseMode is correct,
	// otherwise we might replace the user's code with a trimmed AST.
	if ph.Mode() != ParseFull {
		return nil, errors.Errorf("%s was parsed in the incorrect mode", ph.File().Identity().URI)
	}
	file, m, _, err := ph.Parse(ctx)
	if err != nil {
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
		return computeTextEdits(ctx, ph.File(), m, string(formatted))
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
	return computeTextEdits(ctx, ph.File(), m, buf.String())
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

	cphs, err := f.CheckPackageHandles(ctx)
	if err != nil {
		return nil, err
	}
	cph := NarrowestCheckPackageHandle(cphs)
	pkg, err := cph.Check(ctx)
	if err != nil {
		return nil, err
	}
	if hasListErrors(pkg.GetErrors()) {
		return nil, errors.Errorf("%s has list errors, not running goimports", f.URI())
	}
	ph, err := pkg.File(f.URI())
	if err != nil {
		return nil, err
	}
	// Be extra careful that the file's ParseMode is correct,
	// otherwise we might replace the user's code with a trimmed AST.
	if ph.Mode() != ParseFull {
		return nil, errors.Errorf("%s was parsed in the incorrect mode", ph.File().Identity().URI)
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
		data, _, err := ph.File().Read(ctx)
		if err != nil {
			return err
		}
		formatted, err = imports.Process(ph.File().Identity().URI.Filename(), data, opts)
		return err
	}
	err = view.RunProcessEnvFunc(ctx, importFn, options)
	if err != nil {
		return nil, err
	}
	_, m, _, err := ph.Parse(ctx)
	if err != nil {
		return nil, err
	}
	return computeTextEdits(ctx, ph.File(), m, string(formatted))
}

type ImportFix struct {
	Fix   *imports.ImportFix
	Edits []protocol.TextEdit
}

// AllImportsFixes formats f for each possible fix to the imports.
// In addition to returning the result of applying all edits,
// it returns a list of fixes that could be applied to the file, with the
// corresponding TextEdits that would be needed to apply that fix.
func AllImportsFixes(ctx context.Context, view View, f GoFile) (edits []protocol.TextEdit, editsPerFix []*ImportFix, err error) {
	ctx, done := trace.StartSpan(ctx, "source.AllImportsFixes")
	defer done()

	cphs, err := f.CheckPackageHandles(ctx)
	if err != nil {
		return nil, nil, err
	}
	cph := NarrowestCheckPackageHandle(cphs)
	pkg, err := cph.Check(ctx)
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
		var ph ParseGoHandle
		for _, h := range pkg.Files() {
			if h.File().Identity().URI == f.URI() {
				ph = h
			}
		}
		if ph == nil {
			return errors.Errorf("no ParseGoHandle for %s", f.URI())
		}
		data, _, err := ph.File().Read(ctx)
		if err != nil {
			return err
		}
		fixes, err := imports.FixImports(f.URI().Filename(), data, opts)
		if err != nil {
			return err
		}
		// Apply all of the import fixes to the file.
		formatted, err := imports.ApplyFixes(fixes, f.URI().Filename(), data, options)
		if err != nil {
			return err
		}
		_, m, _, err := ph.Parse(ctx)
		if err != nil {
			return err
		}
		edits, err = computeTextEdits(ctx, ph.File(), m, string(formatted))
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
			edits, err := computeTextEdits(ctx, ph.File(), m, string(formatted))
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

func computeTextEdits(ctx context.Context, fh FileHandle, m *protocol.ColumnMapper, formatted string) ([]protocol.TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.computeTextEdits")
	defer done()

	data, _, err := fh.Read(ctx)
	if err != nil {
		return nil, err
	}
	edits := diff.ComputeEdits(fh.Identity().URI, string(data), formatted)
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
