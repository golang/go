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
	"go/parser"
	"go/token"

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

	snapshot, cphs, err := view.CheckPackageHandles(ctx, f)
	if err != nil {
		return nil, err
	}
	cph, err := NarrowestCheckPackageHandle(cphs)
	if err != nil {
		return nil, err
	}
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
	if hasListErrors(pkg) || hasParseErrors(pkg, f.URI()) {
		// Even if this package has list or parse errors, this file may not
		// have any parse errors and can still be formatted. Using format.Node
		// on an ast with errors may result in code being added or removed.
		// Attempt to format the source of this file instead.
		formatted, err := formatSource(ctx, snapshot, f)
		if err != nil {
			return nil, err
		}
		return computeTextEdits(ctx, view, ph.File(), m, string(formatted))
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
	return computeTextEdits(ctx, view, ph.File(), m, buf.String())
}

func formatSource(ctx context.Context, s Snapshot, f File) ([]byte, error) {
	ctx, done := trace.StartSpan(ctx, "source.formatSource")
	defer done()

	data, _, err := s.Handle(ctx, f).Read(ctx)
	if err != nil {
		return nil, err
	}
	return format.Source(data)
}

type ImportFix struct {
	Fix   *imports.ImportFix
	Edits []protocol.TextEdit
}

// AllImportsFixes formats f for each possible fix to the imports.
// In addition to returning the result of applying all edits,
// it returns a list of fixes that could be applied to the file, with the
// corresponding TextEdits that would be needed to apply that fix.
func AllImportsFixes(ctx context.Context, view View, f File) (allFixEdits []protocol.TextEdit, editsPerFix []*ImportFix, err error) {
	ctx, done := trace.StartSpan(ctx, "source.AllImportsFixes")
	defer done()

	_, cphs, err := view.CheckPackageHandles(ctx, f)
	if err != nil {
		return nil, nil, err
	}
	cph, err := NarrowestCheckPackageHandle(cphs)
	if err != nil {
		return nil, nil, err
	}
	pkg, err := cph.Check(ctx)
	if err != nil {
		return nil, nil, err
	}
	if hasListErrors(pkg) {
		return nil, nil, errors.Errorf("%s has list errors, not running goimports", f.URI())
	}
	var ph ParseGoHandle
	for _, h := range pkg.Files() {
		if h.File().Identity().URI == f.URI() {
			ph = h
		}
	}
	if ph == nil {
		return nil, nil, errors.Errorf("no ParseGoHandle for %s", f.URI())
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
	err = view.RunProcessEnvFunc(ctx, func(opts *imports.Options) error {
		allFixEdits, editsPerFix, err = computeImportEdits(ctx, view, ph, opts)
		return err
	}, options)
	if err != nil {
		return nil, nil, err
	}

	return allFixEdits, editsPerFix, nil
}

// computeImportEdits computes a set of edits that perform one or all of the
// necessary import fixes.
func computeImportEdits(ctx context.Context, view View, ph ParseGoHandle, options *imports.Options) (allFixEdits []protocol.TextEdit, editsPerFix []*ImportFix, err error) {
	filename := ph.File().Identity().URI.Filename()

	// Build up basic information about the original file.
	origData, _, err := ph.File().Read(ctx)
	if err != nil {
		return nil, nil, err
	}
	origAST, origMapper, _, err := ph.Parse(ctx)
	if err != nil {
		return nil, nil, err
	}
	origImports, origImportOffset := trimToImports(view.Session().Cache().FileSet(), origAST, origData)

	computeFixEdits := func(fixes []*imports.ImportFix) ([]protocol.TextEdit, error) {
		// Apply the fixes and re-parse the file so that we can locate the
		// new imports.
		fixedData, err := imports.ApplyFixes(fixes, filename, origData, options)
		if err != nil {
			return nil, err
		}
		fixedFset := token.NewFileSet()
		fixedAST, err := parser.ParseFile(fixedFset, filename, fixedData, parser.ImportsOnly)
		if err != nil {
			return nil, err
		}
		fixedImports, fixedImportsOffset := trimToImports(fixedFset, fixedAST, fixedData)

		// Prepare the diff. If both sides had import statements, we can diff
		// just those sections against each other, then shift the resulting
		// edits to the right lines in the original file.
		left, right := origImports, fixedImports
		converter := span.NewContentConverter(filename, origImports)
		offset := origImportOffset

		// If one side or the other has no imports, we won't know where to
		// anchor the diffs. Instead, use the beginning of the file, up to its
		// first non-imports decl. We know the imports code will insert
		// somewhere before that.
		if origImportOffset == 0 || fixedImportsOffset == 0 {
			left = trimToFirstNonImport(view.Session().Cache().FileSet(), origAST, origData)
			// We need the whole AST here, not just the ImportsOnly AST we parsed above.
			fixedAST, err = parser.ParseFile(fixedFset, filename, fixedData, 0)
			if err != nil {
				return nil, err
			}
			right = trimToFirstNonImport(fixedFset, fixedAST, fixedData)
			// We're now working with a prefix of the original file, so we can
			// use the original converter, and there is no offset on the edits.
			converter = origMapper.Converter
			offset = 0
		}

		// Perform the diff and adjust the results for the trimming, if any.
		edits := view.Options().ComputeEdits(ph.File().Identity().URI, string(left), string(right))
		for i := range edits {
			s, err := edits[i].Span.WithPosition(converter)
			if err != nil {
				return nil, err
			}
			start := span.NewPoint(s.Start().Line()+offset, s.Start().Column(), -1)
			end := span.NewPoint(s.End().Line()+offset, s.End().Column(), -1)
			edits[i].Span = span.New(s.URI(), start, end)
		}
		return ToProtocolEdits(origMapper, edits)
	}

	allFixes, err := imports.FixImports(filename, origData, options)
	if err != nil {
		return nil, nil, err
	}

	allFixEdits, err = computeFixEdits(allFixes)
	if err != nil {
		return nil, nil, err
	}

	// Apply all of the import fixes to the file.
	// Add the edits for each fix to the result.
	for _, fix := range allFixes {
		edits, err := computeFixEdits([]*imports.ImportFix{fix})
		if err != nil {
			return nil, nil, err
		}
		editsPerFix = append(editsPerFix, &ImportFix{
			Fix:   fix,
			Edits: edits,
		})
	}
	return allFixEdits, editsPerFix, nil
}

// trimToImports returns a section of the source file that covers all of the
// import declarations, and the line offset into the file that section starts at.
func trimToImports(fset *token.FileSet, f *ast.File, src []byte) ([]byte, int) {
	var firstImport, lastImport ast.Decl
	for _, decl := range f.Decls {
		if gen, ok := decl.(*ast.GenDecl); ok && gen.Tok == token.IMPORT {
			if firstImport == nil {
				firstImport = decl
			}
			lastImport = decl
		}
	}

	if firstImport == nil {
		return nil, 0
	}
	start := firstImport.Pos()
	end := fset.File(f.Pos()).LineStart(fset.Position(lastImport.End()).Line + 1)
	startLineOffset := fset.Position(start).Line - 1 // lines are 1-indexed.
	return src[fset.Position(firstImport.Pos()).Offset:fset.Position(end).Offset], startLineOffset
}

// trimToFirstNonImport returns src from the beginning to the first non-import
// declaration, or the end of the file if there is no such decl.
func trimToFirstNonImport(fset *token.FileSet, f *ast.File, src []byte) []byte {
	var firstDecl ast.Decl
	for _, decl := range f.Decls {
		if gen, ok := decl.(*ast.GenDecl); ok && gen.Tok == token.IMPORT {
			continue
		}
		firstDecl = decl
		break
	}

	end := f.End()
	if firstDecl != nil {
		end = fset.File(f.Pos()).LineStart(fset.Position(firstDecl.Pos()).Line - 1)
	}
	return src[0:fset.Position(end).Offset]
}

// CandidateImports returns every import that could be added to filename.
func CandidateImports(ctx context.Context, view View, filename string) ([]imports.ImportFix, error) {
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

	var imps []imports.ImportFix
	importFn := func(opts *imports.Options) error {
		var err error
		imps, err = imports.GetAllCandidates(filename, opts)
		return err
	}
	err := view.RunProcessEnvFunc(ctx, importFn, options)
	return imps, err
}

// PackageExports returns all the packages named pkg that could be imported by
// filename, and their exports.
func PackageExports(ctx context.Context, view View, pkg, filename string) ([]imports.PackageExport, error) {
	ctx, done := trace.StartSpan(ctx, "source.PackageExports")
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

	var pkgs []imports.PackageExport
	importFn := func(opts *imports.Options) error {
		var err error
		pkgs, err = imports.GetPackageExports(pkg, filename, opts)
		return err
	}
	err := view.RunProcessEnvFunc(ctx, importFn, options)
	return pkgs, err
}

// hasParseErrors returns true if the given file has parse errors.
func hasParseErrors(pkg Package, uri span.URI) bool {
	for _, err := range pkg.GetErrors() {
		if err.URI == uri && err.Kind == ParseError {
			return true
		}
	}
	return false
}

func hasListErrors(pkg Package) bool {
	for _, err := range pkg.GetErrors() {
		if err.Kind == ListError {
			return true
		}
	}
	return false
}

func computeTextEdits(ctx context.Context, view View, fh FileHandle, m *protocol.ColumnMapper, formatted string) ([]protocol.TextEdit, error) {
	ctx, done := trace.StartSpan(ctx, "source.computeTextEdits")
	defer done()

	data, _, err := fh.Read(ctx)
	if err != nil {
		return nil, err
	}
	edits := view.Options().ComputeEdits(fh.Identity().URI, string(data), formatted)
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
