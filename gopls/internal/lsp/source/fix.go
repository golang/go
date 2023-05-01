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
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/analysis/fillstruct"
	"golang.org/x/tools/gopls/internal/lsp/analysis/undeclaredname"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
)

type (
	// SuggestedFixFunc is a function used to get the suggested fixes for a given
	// gopls command, some of which are provided by go/analysis.Analyzers. Some of
	// the analyzers in internal/lsp/analysis are not efficient enough to include
	// suggested fixes with their diagnostics, so we have to compute them
	// separately. Such analyzers should provide a function with a signature of
	// SuggestedFixFunc.
	//
	// The returned FileSet must map all token.Pos found in the suggested text
	// edits.
	SuggestedFixFunc  func(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) (*token.FileSet, *analysis.SuggestedFix, error)
	singleFileFixFunc func(fset *token.FileSet, start, end token.Pos, src []byte, file *ast.File, pkg *types.Package, info *types.Info) (*analysis.SuggestedFix, error)
)

const (
	FillStruct        = "fill_struct"
	StubMethods       = "stub_methods"
	UndeclaredName    = "undeclared_name"
	ExtractVariable   = "extract_variable"
	ExtractFunction   = "extract_function"
	ExtractMethod     = "extract_method"
	InvertIfCondition = "invert_if_condition"
)

// suggestedFixes maps a suggested fix command id to its handler.
var suggestedFixes = map[string]SuggestedFixFunc{
	FillStruct:        singleFile(fillstruct.SuggestedFix),
	UndeclaredName:    singleFile(undeclaredname.SuggestedFix),
	ExtractVariable:   singleFile(extractVariable),
	ExtractFunction:   singleFile(extractFunction),
	ExtractMethod:     singleFile(extractMethod),
	InvertIfCondition: singleFile(invertIfCondition),
	StubMethods:       stubSuggestedFixFunc,
}

// singleFile calls analyzers that expect inputs for a single file
func singleFile(sf singleFileFixFunc) SuggestedFixFunc {
	return func(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) (*token.FileSet, *analysis.SuggestedFix, error) {
		pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
		if err != nil {
			return nil, nil, err
		}
		start, end, err := pgf.RangePos(pRng)
		if err != nil {
			return nil, nil, err
		}
		fix, err := sf(pkg.FileSet(), start, end, pgf.Src, pgf.File, pkg.GetTypes(), pkg.GetTypesInfo())
		return pkg.FileSet(), fix, err
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
func ApplyFix(ctx context.Context, fix string, snapshot Snapshot, fh FileHandle, pRng protocol.Range) ([]protocol.TextDocumentEdit, error) {
	handler, ok := suggestedFixes[fix]
	if !ok {
		return nil, fmt.Errorf("no suggested fix function for %s", fix)
	}
	fset, suggestion, err := handler(ctx, snapshot, fh, pRng)
	if err != nil {
		return nil, err
	}
	if suggestion == nil {
		return nil, nil
	}
	editsPerFile := map[span.URI]*protocol.TextDocumentEdit{}
	for _, edit := range suggestion.TextEdits {
		tokFile := fset.File(edit.Pos)
		if tokFile == nil {
			return nil, bug.Errorf("no file for edit position")
		}
		end := edit.End
		if !end.IsValid() {
			end = edit.Pos
		}
		fh, err := snapshot.ReadFile(ctx, span.URIFromPath(tokFile.Name()))
		if err != nil {
			return nil, err
		}
		te, ok := editsPerFile[fh.URI()]
		if !ok {
			te = &protocol.TextDocumentEdit{
				TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
					Version: fh.Version(),
					TextDocumentIdentifier: protocol.TextDocumentIdentifier{
						URI: protocol.URIFromSpanURI(fh.URI()),
					},
				},
			}
			editsPerFile[fh.URI()] = te
		}
		content, err := fh.Content()
		if err != nil {
			return nil, err
		}
		m := protocol.NewMapper(fh.URI(), content)
		rng, err := m.PosRange(tokFile, edit.Pos, end)
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
