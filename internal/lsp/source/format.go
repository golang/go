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
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

// Format formats a file with a given range.
func Format(ctx context.Context, snapshot Snapshot, fh FileHandle) ([]protocol.TextEdit, error) {
	ctx, done := event.Start(ctx, "source.Format")
	defer done()

	pgh := snapshot.View().Session().Cache().ParseGoHandle(ctx, fh, ParseFull)
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
	_, done := event.Start(ctx, "source.formatSource")
	defer done()

	data, err := fh.Read()
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

	pgh := snapshot.View().Session().Cache().ParseGoHandle(ctx, fh, ParseFull)
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
	filename := ph.File().URI().Filename()

	// Build up basic information about the original file.
	origData, err := ph.File().Read()
	if err != nil {
		return nil, nil, err
	}
	_, _, origMapper, _, err := ph.Parse(ctx)
	if err != nil {
		return nil, nil, err
	}

	allFixes, err := imports.FixImports(filename, origData, options)
	if err != nil {
		return nil, nil, err
	}

	allFixEdits, err = computeFixEdits(view, ph, options, origData, origMapper, allFixes)
	if err != nil {
		return nil, nil, err
	}

	// Apply all of the import fixes to the file.
	// Add the edits for each fix to the result.
	for _, fix := range allFixes {
		edits, err := computeFixEdits(view, ph, options, origData, origMapper, []*imports.ImportFix{fix})
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
	origData, err := ph.File().Read()
	if err != nil {
		return nil, err
	}
	_, _, origMapper, _, err := ph.Parse(ctx) // ph.Parse returns values never used
	if err != nil {
		return nil, err
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
	return computeFixEdits(view, ph, options, origData, origMapper, []*imports.ImportFix{fix})
}

func computeFixEdits(view View, ph ParseGoHandle, options *imports.Options, origData []byte, origMapper *protocol.ColumnMapper, fixes []*imports.ImportFix) ([]protocol.TextEdit, error) {
	// trim the original data to match fixedData
	left := importPrefix(origData)
	extra := !strings.Contains(left, "\n") // one line may have more than imports
	if extra {
		left = string(origData)
	}
	if len(left) > 0 && left[len(left)-1] != '\n' {
		left += "\n"
	}
	// Apply the fixes and re-parse the file so that we can locate the
	// new imports.
	flags := parser.ImportsOnly
	if extra {
		// used all of origData above, use all of it here too
		flags = 0
	}
	fixedData, err := imports.ApplyFixes(fixes, "", origData, options, flags)
	if err != nil {
		return nil, err
	}
	if fixedData == nil || fixedData[len(fixedData)-1] != '\n' {
		fixedData = append(fixedData, '\n') // ApplyFixes may miss the newline, go figure.
	}
	uri := ph.File().URI()
	edits := view.Options().ComputeEdits(uri, left, string(fixedData))
	return ToProtocolEdits(origMapper, edits)
}

// return the prefix of the src through the last imports, or if there are
// no imports, through the package statement (and a subsequent comment group)
func importPrefix(src []byte) string {
	fset := token.NewFileSet()
	// do as little parsing as possible
	f, err := parser.ParseFile(fset, "", src, parser.ImportsOnly|parser.ParseComments)
	if err != nil { // This can happen if 'package' is misspelled
		return ""
	}
	myStart := fset.File(f.Pos()).Base() // 1, but the generality costs little
	pkgEnd := int(f.Name.NamePos) + len(f.Name.Name)
	var importEnd int
	for _, d := range f.Decls {
		if x, ok := d.(*ast.GenDecl); ok && x.Tok == token.IMPORT {
			e := int(d.End()) - myStart
			if e > importEnd {
				importEnd = e
			}
		}
	}
	if importEnd == 0 {
		importEnd = pkgEnd
	}
	for _, c := range f.Comments {
		if int(c.End()) > importEnd {
			importEnd = int(c.End())
		}
	}
	return string(src[:importEnd])
}

func computeTextEdits(ctx context.Context, view View, fh FileHandle, m *protocol.ColumnMapper, formatted string) ([]protocol.TextEdit, error) {
	_, done := event.Start(ctx, "source.computeTextEdits")
	defer done()

	data, err := fh.Read()
	if err != nil {
		return nil, err
	}
	edits := view.Options().ComputeEdits(fh.URI(), string(data), formatted)
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
