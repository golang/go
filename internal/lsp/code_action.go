// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

func (s *Server) codeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.session.ViewOf(uri)
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}

	snapshot := view.Snapshot()

	// Determine the supported actions for this file kind.
	fileKind := snapshot.Handle(ctx, f).Identity().Kind
	supportedCodeActions, ok := view.Options().SupportedCodeActions[fileKind]
	if !ok {
		return nil, fmt.Errorf("no supported code actions for %v file kind", fileKind)
	}

	// The Only field of the context specifies which code actions the client wants.
	// If Only is empty, assume that the client wants all of the possible code actions.
	var wanted map[protocol.CodeActionKind]bool
	if len(params.Context.Only) == 0 {
		wanted = supportedCodeActions
	} else {
		wanted = make(map[protocol.CodeActionKind]bool)
		for _, only := range params.Context.Only {
			wanted[only] = supportedCodeActions[only]
		}
	}
	if len(wanted) == 0 {
		return nil, errors.Errorf("no supported code action to execute for %s, wanted %v", uri, params.Context.Only)
	}

	var codeActions []protocol.CodeAction
	switch fileKind {
	case source.Mod:
		if !wanted[protocol.SourceOrganizeImports] {
			return nil, nil
		}
		codeActions = append(codeActions, protocol.CodeAction{
			Title: "Tidy",
			Kind:  protocol.SourceOrganizeImports,
			Command: &protocol.Command{
				Title:   "Tidy",
				Command: "tidy",
				Arguments: []interface{}{
					f.URI(),
				},
			},
		})
	case source.Go:
		edits, editsPerFix, err := source.AllImportsFixes(ctx, view, f)
		if err != nil {
			return nil, err
		}
		if diagnostics := params.Context.Diagnostics; wanted[protocol.QuickFix] && len(diagnostics) > 0 {
			// First, add the quick fixes reported by go/analysis.
			qf, err := quickFixes(ctx, snapshot, f, diagnostics)
			if err != nil {
				log.Error(ctx, "quick fixes failed", err, telemetry.File.Of(uri))
			}
			codeActions = append(codeActions, qf...)

			// If we also have diagnostics for missing imports, we can associate them with quick fixes.
			if findImportErrors(diagnostics) {
				// Separate this into a set of codeActions per diagnostic, where
				// each action is the addition, removal, or renaming of one import.
				for _, importFix := range editsPerFix {
					// Get the diagnostics this fix would affect.
					if fixDiagnostics := importDiagnostics(importFix.Fix, diagnostics); len(fixDiagnostics) > 0 {
						codeActions = append(codeActions, protocol.CodeAction{
							Title: importFixTitle(importFix.Fix),
							Kind:  protocol.QuickFix,
							Edit: &protocol.WorkspaceEdit{
								Changes: &map[string][]protocol.TextEdit{
									string(uri): importFix.Edits,
								},
							},
							Diagnostics: fixDiagnostics,
						})
					}
				}
			}
		}
		if wanted[protocol.SourceOrganizeImports] && len(edits) > 0 {
			codeActions = append(codeActions, protocol.CodeAction{
				Title: "Organize Imports",
				Kind:  protocol.SourceOrganizeImports,
				Edit: &protocol.WorkspaceEdit{
					Changes: &map[string][]protocol.TextEdit{
						string(uri): edits,
					},
				},
			})
		}
	default:
		// Unsupported file kind for a code action.
		return nil, nil
	}
	return codeActions, nil
}

func (s *Server) getSupportedCodeActions() []protocol.CodeActionKind {
	allCodeActionKinds := make(map[protocol.CodeActionKind]struct{})
	for _, kinds := range s.session.Options().SupportedCodeActions {
		for kind := range kinds {
			allCodeActionKinds[kind] = struct{}{}
		}
	}
	var result []protocol.CodeActionKind
	for kind := range allCodeActionKinds {
		result = append(result, kind)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i] < result[j]
	})
	return result
}

type protocolImportFix struct {
	fix   *imports.ImportFix
	edits []protocol.TextEdit
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

func quickFixes(ctx context.Context, s source.Snapshot, f source.File, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	var codeActions []protocol.CodeAction
	cphs, err := s.CheckPackageHandles(ctx, f)
	if err != nil {
		return nil, err
	}
	// We get the package that source.Diagnostics would've used. This is hack.
	// TODO(golang/go#32443): The correct solution will be to cache diagnostics per-file per-snapshot.
	cph, err := source.WidestCheckPackageHandle(cphs)
	if err != nil {
		return nil, err
	}
	for _, diag := range diagnostics {
		srcErr, err := s.FindAnalysisError(ctx, cph.ID(), diag)
		if err != nil {
			continue
		}
		for _, fix := range srcErr.SuggestedFixes {
			edits := make(map[string][]protocol.TextEdit)
			for uri, e := range fix.Edits {
				edits[protocol.NewURI(uri)] = e
			}
			codeActions = append(codeActions, protocol.CodeAction{
				Title:       fix.Title,
				Kind:        protocol.QuickFix,
				Diagnostics: []protocol.Diagnostic{diag},
				Edit: &protocol.WorkspaceEdit{
					Changes: &edits,
				},
			})
		}
	}
	return codeActions, nil
}
