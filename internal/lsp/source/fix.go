// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/analysis/fillstruct"
	"golang.org/x/tools/internal/lsp/analysis/undeclaredname"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type (
	// SuggestedFixFunc is a function used to get the suggested fixes for a given
	// gopls command, some of which are provided by go/analysis.Analyzers. Some of
	// the analyzers in internal/lsp/analysis are not efficient enough to include
	// suggested fixes with their diagnostics, so we have to compute them
	// separately. Such analyzers should provide a function with a signature of
	// SuggestedFixFunc.
	SuggestedFixFunc  func(ctx context.Context, snapshot Snapshot, fh VersionedFileHandle, pRng protocol.Range) (*analysis.SuggestedFix, error)
	singleFileFixFunc func(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error)
)

const (
	FillStruct      = "fill_struct"
	UndeclaredName  = "undeclared_name"
	ExtractVariable = "extract_variable"
	ExtractFunction = "extract_function"
	ExtractMethod   = "extract_method"
)

// suggestedFixes maps a suggested fix command id to its handler.
var suggestedFixes = map[string]SuggestedFixFunc{
	FillStruct:      singleFile(fillstruct.SuggestedFix),
	UndeclaredName:  singleFile(undeclaredname.SuggestedFix),
	ExtractVariable: singleFile(extractVariable),
	ExtractFunction: singleFile(extractFunction),
	ExtractMethod:   singleFile(extractMethod),
}

// singleFile calls analyzers that expect inputs for a single file
func singleFile(sf singleFileFixFunc) SuggestedFixFunc {
	return func(ctx context.Context, snapshot Snapshot, fh VersionedFileHandle, pRng protocol.Range) (*analysis.SuggestedFix, error) {
		fset, rng, src, file, pkg, info, err := getAllSuggestedFixInputs(ctx, snapshot, fh, pRng)
		if err != nil {
			return nil, err
		}
		return sf(fset, rng, src, file, pkg, info)
	}
}

func SuggestedFixFromCommand(cmd protocol.Command, kind protocol.CodeActionKind) SuggestedFix {
	return SuggestedFix{
		Title:      cmd.Title,
		Command:    &cmd,
		ActionKind: kind,
	}
}

// ApplyFix applies the command's suggested fix to the given file and
// range, returning the resulting edits.
func ApplyFix(ctx context.Context, fix string, snapshot Snapshot, fh VersionedFileHandle, pRng protocol.Range) ([]protocol.TextDocumentEdit, error) {
	handler, ok := suggestedFixes[fix]
	if !ok {
		return nil, fmt.Errorf("no suggested fix function for %s", fix)
	}
	suggestion, err := handler(ctx, snapshot, fh, pRng)
	if err != nil {
		return nil, err
	}
	if suggestion == nil {
		return nil, nil
	}
	fset := snapshot.FileSet()
	editsPerFile := map[span.URI]*protocol.TextDocumentEdit{}
	for _, edit := range suggestion.TextEdits {
		spn, err := span.NewRange(fset, edit.Pos, edit.End).Span()
		if err != nil {
			return nil, err
		}
		fh, err := snapshot.GetVersionedFile(ctx, spn.URI())
		if err != nil {
			return nil, err
		}
		te, ok := editsPerFile[spn.URI()]
		if !ok {
			te = &protocol.TextDocumentEdit{
				TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
					Version: fh.Version(),
					TextDocumentIdentifier: protocol.TextDocumentIdentifier{
						URI: protocol.URIFromSpanURI(fh.URI()),
					},
				},
			}
			editsPerFile[spn.URI()] = te
		}
		_, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
		if err != nil {
			return nil, err
		}
		rng, err := pgf.Mapper.Range(spn)
		if err != nil {
			return nil, err
		}
		te.Edits = append(te.Edits, protocol.TextEdit{
			Range:   rng,
			NewText: string(edit.NewText),
		})
	}
	var edits []protocol.TextDocumentEdit
	for _, edit := range editsPerFile {
		edits = append(edits, *edit)
	}
	return edits, nil
}

// getAllSuggestedFixInputs is a helper function to collect all possible needed
// inputs for an AppliesFunc or SuggestedFixFunc.
func getAllSuggestedFixInputs(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) (*token.FileSet, span.Range, []byte, *ast.File, *types.Package, *types.Info, error) {
	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, errors.Errorf("getting file for Identifier: %w", err)
	}
	rng, err := pgf.Mapper.RangeToSpanRange(pRng)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, err
	}
	return snapshot.FileSet(), rng, pgf.Src, pgf.File, pkg.GetTypes(), pkg.GetTypesInfo(), nil
}
