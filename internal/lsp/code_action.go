// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) codeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	_, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(params.Range)
	if err != nil {
		return nil, err
	}
	var codeActions []protocol.CodeAction
	// Determine what code actions we should take based on the diagnostics.
	if findImportErrors(params.Context.Diagnostics) {
		edits, err := organizeImports(ctx, view, spn)
		if err != nil {
			return nil, err
		}
		if len(edits) > 0 {
			// TODO(rstambler): Handle params.Context.Only when VSCode-Go uses a
			// version of vscode-languageclient that fixes
			// https://github.com/Microsoft/vscode-languageserver-node/issues/442.
			codeActions = append(codeActions, protocol.CodeAction{
				Title: "Organize Imports",
				Kind:  protocol.SourceOrganizeImports,
				Edit: &protocol.WorkspaceEdit{
					Changes: &map[string][]protocol.TextEdit{
						string(spn.URI()): edits,
					},
				},
			})
			// Add any quick fixes for each import-related diagnostic that we see.
			fixes, err := quickFixes(spn.URI(), params.Context.Diagnostics, edits)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, fixes...)
		}
	}
	return codeActions, nil
}

// findImports determines if a given diagnostic represents an error that could
// be fixed by organizing imports.
// TODO(rstambler): We need a better way to check this than string matching.
func findImportErrors(diagnostics []protocol.Diagnostic) bool {
	for _, diagnostic := range diagnostics {
		// "undeclared name: X" may be an unresolved import.
		if strings.HasPrefix(diagnostic.Message, "undeclared name: ") {
			return true
		}
		// "could not import: X" may be an invalid import.
		if strings.HasPrefix(diagnostic.Message, "could not import: ") {
			return true
		}
		// "X imported but not used" is an unused import.
		if strings.HasSuffix(diagnostic.Message, " imported but not used") {
			return true
		}
	}
	return false
}

func organizeImports(ctx context.Context, v source.View, s span.Span) ([]protocol.TextEdit, error) {
	f, m, err := newColumnMap(ctx, v, s.URI())
	if err != nil {
		return nil, err
	}
	rng, err := s.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	if rng.Start == rng.End {
		// If we have a single point, assume we want the whole file.
		tok := f.GetToken(ctx)
		if tok == nil {
			return nil, fmt.Errorf("no file information for %s", f.URI())
		}
		rng.End = tok.Pos(tok.Size())
	}
	edits, err := source.Imports(ctx, f, rng)
	if err != nil {
		return nil, err
	}
	return ToProtocolEdits(m, edits)
}

// TODO(rstambler): Separate this into a set of codeActions per diagnostic,
// where each action is the addition or removal of one import.
// This can only be done when https://golang.org/issue/31493 is resolved.
func quickFixes(uri span.URI, diagnostics []protocol.Diagnostic, edits []protocol.TextEdit) ([]protocol.CodeAction, error) {
	return []protocol.CodeAction{
		{
			Title: "Organize All Imports",
			Kind:  protocol.QuickFix,
			Edit: &protocol.WorkspaceEdit{
				Changes: &map[string][]protocol.TextEdit{
					string(uri): edits,
				},
			},
		},
	}, nil
}
