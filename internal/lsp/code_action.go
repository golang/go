// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"strings"

	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/lsp/telemetry/log"
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

	edits, editsPerFix, err := organizeImports(ctx, view, spn)
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
				log.Error(ctx, "quick fixes failed", err, telemetry.File.Of(uri))
			}
			codeActions = append(codeActions, qf...)
		}

		// If we also have diagnostics for missing imports, we can associate them with quick fixes.
		if findImportErrors(params.Context.Diagnostics) {
			// Separate this into a set of codeActions per diagnostic, where
			// each action is the addition, removal, or renaming of one import.
			for _, importFix := range editsPerFix {
				// Get the diagnostics this fix would affect.
				if fixDiagnostics := importDiagnostics(importFix.fix, params.Context.Diagnostics); len(fixDiagnostics) > 0 {
					codeActions = append(codeActions, protocol.CodeAction{
						Title: importFixTitle(importFix.fix),
						Kind:  protocol.QuickFix,
						Edit: &protocol.WorkspaceEdit{
							Changes: &map[string][]protocol.TextEdit{
								string(uri): importFix.edits,
							},
						},
						Diagnostics: fixDiagnostics,
					})
				}
			}
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

type protocolImportFix struct {
	fix   *imports.ImportFix
	edits []protocol.TextEdit
}

func organizeImports(ctx context.Context, view source.View, s span.Span) ([]protocol.TextEdit, []*protocolImportFix, error) {
	f, m, rng, err := spanToRange(ctx, view, s)
	if err != nil {
		return nil, nil, err
	}
	edits, editsPerFix, err := source.AllImportsFixes(ctx, view, f, rng)
	if err != nil {
		return nil, nil, err
	}
	// Convert all source edits to protocol edits.
	pEdits, err := ToProtocolEdits(m, edits)
	if err != nil {
		return nil, nil, err
	}

	pEditsPerFix := make([]*protocolImportFix, len(editsPerFix))
	for i, fix := range editsPerFix {
		pEdits, err := ToProtocolEdits(m, fix.Edits)
		if err != nil {
			return nil, nil, err
		}
		pEditsPerFix[i] = &protocolImportFix{
			fix:   fix.Fix,
			edits: pEdits,
		}
	}
	return pEdits, pEditsPerFix, nil
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
		// "X imported but not used as Y" is an unused import.
		if strings.Contains(diagnostic.Message, " imported but not used") {
			return true
		}
	}
	return false
}

func importFixTitle(fix *imports.ImportFix) string {
	var str string
	switch fix.FixType {
	case imports.AddImport:
		str = fmt.Sprintf("Add import: %s %q", fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
	case imports.DeleteImport:
		str = fmt.Sprintf("Delete import: %s %q", fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
	case imports.SetImportName:
		str = fmt.Sprintf("Rename import: %s %q", fix.StmtInfo.Name, fix.StmtInfo.ImportPath)
	}
	return str
}

func importDiagnostics(fix *imports.ImportFix, diagnostics []protocol.Diagnostic) (results []protocol.Diagnostic) {
	for _, diagnostic := range diagnostics {
		switch {
		// "undeclared name: X" may be an unresolved import.
		case strings.HasPrefix(diagnostic.Message, "undeclared name: "):
			ident := strings.TrimPrefix(diagnostic.Message, "undeclared name: ")
			if ident == fix.IdentName {
				results = append(results, diagnostic)
			}
		// "could not import: X" may be an invalid import.
		case strings.HasPrefix(diagnostic.Message, "could not import: "):
			ident := strings.TrimPrefix(diagnostic.Message, "could not import: ")
			if ident == fix.IdentName {
				results = append(results, diagnostic)
			}
		// "X imported but not used" is an unused import.
		// "X imported but not used as Y" is an unused import.
		case strings.Contains(diagnostic.Message, " imported but not used"):
			idx := strings.Index(diagnostic.Message, " imported but not used")
			importPath := diagnostic.Message[:idx]
			if importPath == fmt.Sprintf("%q", fix.StmtInfo.ImportPath) {
				results = append(results, diagnostic)
			}
		}

	}

	return results
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
