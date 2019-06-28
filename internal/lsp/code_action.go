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

	// The Only field of the context specifies which code actions the client wants.
	// If Only is empty, assume that the client wants all of the possible code actions.
	var wanted map[protocol.CodeActionKind]bool
	if len(params.Context.Only) == 0 {
		wanted = s.supportedCodeActions
	} else {
		wanted = make(map[protocol.CodeActionKind]bool)
		for _, only := range params.Context.Only {
			wanted[only] = s.supportedCodeActions[only]
		}
	}

	uri := span.NewURI(params.TextDocument.URI)
	if len(wanted) == 0 {
		return nil, fmt.Errorf("no supported code action to execute for %s, wanted %v", uri, params.Context.Only)
	}

	view := s.session.ViewOf(uri)
	gof, m, err := getGoFile(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(params.Range)
	if err != nil {
		return nil, err
	}

	var codeActions []protocol.CodeAction

	edits, err := organizeImports(ctx, view, spn)
	if err != nil {
		return nil, err
	}

	// If the user wants to see quickfixes.
	if wanted[protocol.QuickFix] {
		// First, add the quick fixes reported by go/analysis.
		// TODO: Enable this when this actually works. For now, it's needless work.
		if s.wantSuggestedFixes {
			qf, err := quickFixes(ctx, view, gof)
			if err != nil {
				view.Session().Logger().Errorf(ctx, "quick fixes failed for %s: %v", uri, err)
			}
			codeActions = append(codeActions, qf...)
		}

		// If we also have diagnostics for missing imports, we can associate them with quick fixes.
		if findImportErrors(params.Context.Diagnostics) {
			// TODO(rstambler): Separate this into a set of codeActions per diagnostic,
			// where each action is the addition or removal of one import.
			// This can only be done when https://golang.org/issue/31493 is resolved.
			codeActions = append(codeActions, protocol.CodeAction{
				Title: "Organize All Imports", // clarify that all imports will change
				Kind:  protocol.QuickFix,
				Edit: &protocol.WorkspaceEdit{
					Changes: &map[string][]protocol.TextEdit{
						string(uri): edits,
					},
				},
			})
		}
	}

	// Add the results of import organization as source.OrganizeImports.
	if wanted[protocol.SourceOrganizeImports] {
		codeActions = append(codeActions, protocol.CodeAction{
			Title: "Organize Imports",
			Kind:  protocol.SourceOrganizeImports,
			Edit: &protocol.WorkspaceEdit{
				Changes: &map[string][]protocol.TextEdit{
					string(spn.URI()): edits,
				},
			},
		})
	}

	return codeActions, nil
}

func organizeImports(ctx context.Context, view source.View, s span.Span) ([]protocol.TextEdit, error) {
	f, m, rng, err := spanToRange(ctx, view, s)
	if err != nil {
		return nil, err
	}
	edits, err := source.Imports(ctx, view, f, rng)
	if err != nil {
		return nil, err
	}
	return ToProtocolEdits(m, edits)
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

func quickFixes(ctx context.Context, view source.View, gof source.GoFile) ([]protocol.CodeAction, error) {
	var codeActions []protocol.CodeAction

	// TODO: This is technically racy because the diagnostics provided by the code action
	// may not be the same as the ones that gopls is aware of.
	// We need to figure out some way to solve this problem.
	diags := gof.GetPackage(ctx).GetDiagnostics()
	for _, diag := range diags {
		pdiag, err := toProtocolDiagnostic(ctx, view, diag)
		if err != nil {
			return nil, err
		}
		for _, ca := range diag.SuggestedFixes {
			_, m, err := getGoFile(ctx, view, diag.URI())
			if err != nil {
				return nil, err
			}
			edits, err := ToProtocolEdits(m, ca.Edits)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, protocol.CodeAction{
				Title: ca.Title,
				Kind:  protocol.QuickFix, // TODO(matloob): Be more accurate about these?
				Edit: &protocol.WorkspaceEdit{
					Changes: &map[string][]protocol.TextEdit{
						string(diag.URI()): edits,
					},
				},
				Diagnostics: []protocol.Diagnostic{pdiag},
			})
		}
	}
	return codeActions, nil
}
