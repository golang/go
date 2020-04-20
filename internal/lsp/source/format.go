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
	"go/scanner"
	"go/token"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// Format formats a file with a given range.
func Format(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.TextEdit, error) {
	ctx, done := event.Start(ctx, "source.Format")
	defer done()

	pgh := snapshot.View().Session().Cache().ParseGoHandle(fh, ParseFull)
	file, _, m, parseErrors, err := pgh.Parse(ctx)
	if err != nil {
		return nil, err
	}
	// Even if this file has parse errors, it might still be possible to format it.
	// Using format.Node on an AST with errors may result in code being modified.
	// Attempt to format the source of this file instead.
	if parseErrors != nil {
		formatted, err := formatSource(ctx, fh)
		if err != nil {
			return nil, err
		}
		return computeTextEdits(ctx, snapshot.View(), pgh.File(), m, string(formatted))
	}

	fset := snapshot.View().Session().Cache().FileSet()
	buf := &bytes.Buffer{}

	// format.Node changes slightly from one release to another, so the version
	// of Go used to build the LSP server will determine how it formats code.
	// This should be acceptable for all users, who likely be prompted to rebuild
	// the LSP server on each Go release.
	if err := format.Node(buf, fset, file); err != nil {
		return nil, err
	}
	return computeTextEdits(ctx, snapshot.View(), pgh.File(), m, buf.String())
}

func formatSource(ctx context.Context, fh FileHandle) ([]byte, error) {
	ctx, done := event.Start(ctx, "source.formatSource")
	defer done()

	data, _, err := fh.Read(ctx)
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
func AllImportsFixes(ctx context.Context, snapshot Snapshot, fh FileHandle) (allFixEdits []protocol.TextEdit, editsPerFix []*ImportFix, err error) {
	ctx, done := event.Start(ctx, "source.AllImportsFixes")
	defer done()

	pgh := snapshot.View().Session().Cache().ParseGoHandle(fh, ParseFull)
	if err := snapshot.View().RunProcessEnvFunc(ctx, func(opts *imports.Options) error {
		allFixEdits, editsPerFix, err = computeImportEdits(ctx, snapshot.View(), pgh, opts)
		return err
	}); err != nil {
		return nil, nil, errors.Errorf("computing fix edits: %v", err)
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
	origAST, _, origMapper, _, err := ph.Parse(ctx)
	if err != nil {
		return nil, nil, err
	}

	allFixes, err := imports.FixImports(filename, origData, options)
	if err != nil {
		return nil, nil, err
	}

	origImports, origImportOffset := trimToImports(view.Session().Cache().FileSet(), origAST, origData)
	allFixEdits, err = computeFixEdits(view, ph, options, origData, origAST, origMapper, origImports, origImportOffset, allFixes)
	if err != nil {
		return nil, nil, err
	}

	// Apply all of the import fixes to the file.
	// Add the edits for each fix to the result.
	for _, fix := range allFixes {
		edits, err := computeFixEdits(view, ph, options, origData, origAST, origMapper, origImports, origImportOffset, []*imports.ImportFix{fix})
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

func computeOneImportFixEdits(ctx context.Context, view View, ph ParseGoHandle, fix *imports.ImportFix) ([]protocol.TextEdit, error) {
	origData, _, err := ph.File().Read(ctx)
	if err != nil {
		return nil, err
	}
	origAST, _, origMapper, _, err := ph.Parse(ctx)
	if err != nil {
		return nil, err
	}
	origImports, origImportOffset := trimToImports(view.Session().Cache().FileSet(), origAST, origData)

	options := &imports.Options{
		// Defaults.
		AllErrors:  true,
		Comments:   true,
		Fragment:   true,
		FormatOnly: false,
		TabIndent:  true,
		TabWidth:   8,
	}
	return computeFixEdits(view, ph, options, origData, origAST, origMapper, origImports, origImportOffset, []*imports.ImportFix{fix})
}

func computeFixEdits(view View, ph ParseGoHandle, options *imports.Options, origData []byte, origAST *ast.File, origMapper *protocol.ColumnMapper, origImports []byte, origImportOffset int, fixes []*imports.ImportFix) ([]protocol.TextEdit, error) {
	filename := ph.File().Identity().URI.Filename()
	// Apply the fixes and re-parse the file so that we can locate the
	// new imports.
	fixedData, err := imports.ApplyFixes(fixes, filename, origData, options, parser.ImportsOnly)
	fixedData = append(fixedData, '\n') // ApplyFixes comes out missing the newline, go figure.
	if err != nil {
		return nil, err
	}
	fixedFset := token.NewFileSet()
	fixedAST, err := parser.ParseFile(fixedFset, filename, fixedData, parser.ImportsOnly)
	// Any error here prevents us from computing the edits.
	if err != nil {
		return nil, err
	}
	fixedImports, fixedImportsOffset := trimToImports(fixedFset, fixedAST, fixedData)

	// Prepare the diff. If both sides had import statements, we can diff
	// just those sections against each other, then shift the resulting
	// edits to the right lines in the original file.
	left, right := origImports, fixedImports

	// If there is no diff, return early, as there's no need to compute edits.
	// imports.ApplyFixes also formats the file, and this way we avoid
	// unnecessary formatting, which may cause further issues if we can't
	// find an import block on which to anchor the diffs.
	if len(left) == 0 && len(right) == 0 {
		return nil, nil
	}

	converter := span.NewContentConverter(filename, origImports)
	offset := origImportOffset

	// If one side or the other has no imports, we won't know where to
	// anchor the diffs. Instead, use the beginning of the file, up to its
	// first non-imports decl. We know the imports code will insert
	// somewhere before that.
	if origImportOffset == 0 || fixedImportsOffset == 0 {
		left, _ = trimToFirstNonImport(view.Session().Cache().FileSet(), origAST, origData, nil)
		fixedData, err = imports.ApplyFixes(fixes, filename, origData, options, 0)
		if err != nil {
			return nil, err
		}
		// We need the whole file here, not just the ImportsOnly versions we made above.
		fixedAST, err = parser.ParseFile(fixedFset, filename, fixedData, 0)
		if fixedAST == nil {
			return nil, err
		}
		var ok bool
		right, ok = trimToFirstNonImport(fixedFset, fixedAST, fixedData, err)
		if !ok {
			return nil, errors.Errorf("error %v detected in the import block", err)
		}
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
	tok := fset.File(f.Pos())
	start := firstImport.Pos()
	end := lastImport.End()
	// The parser will happily feed us nonsense. See golang/go#36610.
	tokStart, tokEnd := token.Pos(tok.Base()), token.Pos(tok.Base()+tok.Size())
	if start < tokStart || start > tokEnd || end < tokStart || end > tokEnd {
		return nil, 0
	}
	if nextLine := fset.Position(end).Line + 1; tok.LineCount() >= nextLine {
		end = fset.File(f.Pos()).LineStart(nextLine)
	}
	if start > end {
		return nil, 0
	}

	startLineOffset := fset.Position(start).Line - 1 // lines are 1-indexed.
	return src[fset.Position(start).Offset:fset.Position(end).Offset], startLineOffset
}

// trimToFirstNonImport returns src from the beginning to the first non-import
// declaration, or the end of the file if there is no such decl.
func trimToFirstNonImport(fset *token.FileSet, f *ast.File, src []byte, err error) ([]byte, bool) {
	var firstDecl ast.Decl
	for _, decl := range f.Decls {
		if gen, ok := decl.(*ast.GenDecl); ok && gen.Tok == token.IMPORT {
			continue
		}
		firstDecl = decl
		break
	}
	tok := fset.File(f.Pos())
	if tok == nil {
		return nil, false
	}
	end := f.End()
	if firstDecl != nil {
		if firstDeclLine := fset.Position(firstDecl.Pos()).Line; firstDeclLine > 1 {
			end = tok.LineStart(firstDeclLine - 1)
		}
	}
	// Any errors in the file must be after the part of the file that we care about.
	switch err := err.(type) {
	case *scanner.Error:
		pos := tok.Pos(err.Pos.Offset)
		if pos <= end {
			return nil, false
		}
	case scanner.ErrorList:
		if err.Len() > 0 {
			pos := tok.Pos(err[0].Pos.Offset)
			if pos <= end {
				return nil, false
			}
		}
	}
	return src[0:fset.Position(end).Offset], true
}

func computeTextEdits(ctx context.Context, view View, fh FileHandle, m *protocol.ColumnMapper, formatted string) ([]protocol.TextEdit, error) {
	ctx, done := event.Start(ctx, "source.computeTextEdits")
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
