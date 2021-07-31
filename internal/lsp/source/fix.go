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

// SuggestedFixFunc is a function used to get the suggested fixes for a given
// gopls command, some of which are provided by go/analysis.Analyzers. Some of
// the analyzers in internal/lsp/analysis are not efficient enough to include
// suggested fixes with their diagnostics, so we have to compute them
// separately. Such analyzers should provide a function with a signature of
// SuggestedFixFunc.
type SuggestedFixFunc func(fset *token.FileSet, rng span.Range, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error)

const (
	FillStruct      = "fill_struct"
	UndeclaredName  = "undeclared_name"
	ExtractVariable = "extract_variable"
	ExtractFunction = "extract_function"
	ExtractMethod   = "extract_method"
)

// suggestedFixes maps a suggested fix command id to its handler.
var suggestedFixes = map[string]SuggestedFixFunc{
	FillStruct:      fillstruct.SuggestedFix,
	UndeclaredName:  undeclaredname.SuggestedFix,
	ExtractVariable: extractVariable,
	ExtractFunction: extractFunction,
	ExtractMethod:   extractMethod,
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
	fset, rng, src, file, m, pkg, info, err := getAllSuggestedFixInputs(ctx, snapshot, fh, pRng)
	if err != nil {
		return nil, err
	}
	suggestion, err := handler(fset, rng, src, file, pkg, info)
	if err != nil {
		return nil, err
	}
	if suggestion == nil {
		return nil, nil
	}

	var edits []protocol.TextEdit
	for _, edit := range suggestion.TextEdits {
		rng := span.NewRange(fset, edit.Pos, edit.End)
		spn, err := rng.Span()
		if err != nil {
			return nil, err
		}
		clRng, err := m.Range(spn)
		if err != nil {
			return nil, err
		}
		edits = append(edits, protocol.TextEdit{
			Range:   clRng,
			NewText: string(edit.NewText),
		})
	}
	return []protocol.TextDocumentEdit{{
		TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
			Version: fh.Version(),
			TextDocumentIdentifier: protocol.TextDocumentIdentifier{
				URI: protocol.URIFromSpanURI(fh.URI()),
			},
		},
		Edits: edits,
	}}, nil
}

// getAllSuggestedFixInputs is a helper function to collect all possible needed
// inputs for an AppliesFunc or SuggestedFixFunc.
func getAllSuggestedFixInputs(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) (*token.FileSet, span.Range, []byte, *ast.File, *protocol.ColumnMapper, *types.Package, *types.Info, error) {
	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, errors.Errorf("getting file for Identifier: %w", err)
	}
	rng, err := pgf.Mapper.RangeToSpanRange(pRng)
	if err != nil {
		return nil, span.Range{}, nil, nil, nil, nil, nil, err
	}
	return snapshot.FileSet(), rng, pgf.Src, pgf.File, pgf.Mapper, pkg.GetTypes(), pkg.GetTypesInfo(), nil
}
