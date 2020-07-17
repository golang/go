// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) codeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	snapshot, fh, ok, err := s.beginFileRequest(ctx, params.TextDocument.URI, source.UnknownKind)
	if !ok {
		return nil, err
	}
	uri := fh.URI()

	// Determine the supported actions for this file kind.
	supportedCodeActions, ok := snapshot.View().Options().SupportedCodeActions[fh.Kind()]
	if !ok {
		return nil, fmt.Errorf("no supported code actions for %v file kind", fh.Kind())
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
		return nil, fmt.Errorf("no supported code action to execute for %s, wanted %v", uri, params.Context.Only)
	}

	var codeActions []protocol.CodeAction
	switch fh.Kind() {
	case source.Mod:
		if diagnostics := params.Context.Diagnostics; len(diagnostics) > 0 {
			modFixes, err := mod.SuggestedFixes(ctx, snapshot, diagnostics)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, modFixes...)
		}
		if wanted[protocol.SourceOrganizeImports] {
			codeActions = append(codeActions, protocol.CodeAction{
				Title: "Tidy",
				Kind:  protocol.SourceOrganizeImports,
				Command: &protocol.Command{
					Title:     "Tidy",
					Command:   "tidy",
					Arguments: []interface{}{fh.URI()},
				},
			})
		}
	case source.Go:
		// Don't suggest fixes for generated files, since they are generally
		// not useful and some editors may apply them automatically on save.
		if source.IsGenerated(ctx, snapshot, uri) {
			return nil, nil
		}
		diagnostics := params.Context.Diagnostics

		// First, process any missing imports and pair them with the
		// diagnostics they fix.
		if wantQuickFixes := wanted[protocol.QuickFix] && len(diagnostics) > 0; wantQuickFixes || wanted[protocol.SourceOrganizeImports] {
			importEdits, importEditsPerFix, err := source.AllImportsFixes(ctx, snapshot, fh)
			if err != nil {
				event.Error(ctx, "imports fixes", err, tag.File.Of(fh.URI().Filename()))
			}
			// Separate this into a set of codeActions per diagnostic, where
			// each action is the addition, removal, or renaming of one import.
			if wantQuickFixes {
				for _, importFix := range importEditsPerFix {
					fixes := importDiagnostics(importFix.Fix, diagnostics)
					if len(fixes) == 0 {
						continue
					}
					codeActions = append(codeActions, protocol.CodeAction{
						Title: importFixTitle(importFix.Fix),
						Kind:  protocol.QuickFix,
						Edit: protocol.WorkspaceEdit{
							DocumentChanges: documentChanges(fh, importFix.Edits),
						},
						Diagnostics: fixes,
					})
				}
			}
			// Send all of the import edits as one code action if the file is
			// being organized.
			if wanted[protocol.SourceOrganizeImports] && len(importEdits) > 0 {
				codeActions = append(codeActions, protocol.CodeAction{
					Title: "Organize Imports",
					Kind:  protocol.SourceOrganizeImports,
					Edit: protocol.WorkspaceEdit{
						DocumentChanges: documentChanges(fh, importEdits),
					},
				})
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		phs, err := snapshot.PackageHandles(ctx, fh)
		if err != nil {
			return nil, err
		}
		ph, err := source.WidestPackageHandle(phs)
		if err != nil {
			return nil, err
		}
		if (wanted[protocol.QuickFix] || wanted[protocol.SourceFixAll]) && len(diagnostics) > 0 {
			analysisQuickFixes, highConfidenceEdits, err := analysisFixes(ctx, snapshot, ph, diagnostics)
			if err != nil {
				return nil, err
			}
			if wanted[protocol.QuickFix] {
				// Add the quick fixes reported by go/analysis.
				codeActions = append(codeActions, analysisQuickFixes...)

				// If there are any diagnostics relating to the go.mod file,
				// add their corresponding quick fixes.
				moduleQuickFixes, err := mod.SuggestedFixes(ctx, snapshot, diagnostics)
				if err != nil {
					// Not a fatal error.
					event.Error(ctx, "module suggested fixes failed", err, tag.Directory.Of(snapshot.View().Folder()))
				}
				codeActions = append(codeActions, moduleQuickFixes...)
			}
			if wanted[protocol.SourceFixAll] && len(highConfidenceEdits) > 0 {
				codeActions = append(codeActions, protocol.CodeAction{
					Title: "Simplifications",
					Kind:  protocol.SourceFixAll,
					Edit: protocol.WorkspaceEdit{
						DocumentChanges: highConfidenceEdits,
					},
				})
			}
		}
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		// Add any suggestions that do not necessarily fix any diagnostics.
		if wanted[protocol.RefactorRewrite] {
			fixes, err := convenienceFixes(ctx, snapshot, ph, uri, params.Range)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, fixes...)
		}
		if wanted[protocol.RefactorExtract] {
			fixes, err := extractionFixes(ctx, snapshot, ph, uri, params.Range)
			if err != nil {
				return nil, err
			}
			codeActions = append(codeActions, fixes...)
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

func analysisFixes(ctx context.Context, snapshot source.Snapshot, ph source.PackageHandle, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, []protocol.TextDocumentEdit, error) {
	if len(diagnostics) == 0 {
		return nil, nil, nil
	}

	var codeActions []protocol.CodeAction
	var sourceFixAllEdits []protocol.TextDocumentEdit

	for _, diag := range diagnostics {
		srcErr, analyzer, ok := findSourceError(ctx, snapshot, ph.ID(), diag)
		if !ok {
			continue
		}
		for _, fix := range srcErr.SuggestedFixes {
			action := protocol.CodeAction{
				Title:       fix.Title,
				Kind:        protocol.QuickFix,
				Diagnostics: []protocol.Diagnostic{diag},
				Edit:        protocol.WorkspaceEdit{},
			}
			for uri, edits := range fix.Edits {
				fh, err := snapshot.GetFile(ctx, uri)
				if err != nil {
					return nil, nil, err
				}
				docChanges := documentChanges(fh, edits)
				if analyzer.HighConfidence {
					sourceFixAllEdits = append(sourceFixAllEdits, docChanges...)
				}
				action.Edit.DocumentChanges = append(action.Edit.DocumentChanges, docChanges...)
			}
			codeActions = append(codeActions, action)
		}
	}
	return codeActions, sourceFixAllEdits, nil
}

func findSourceError(ctx context.Context, snapshot source.Snapshot, pkgID string, diag protocol.Diagnostic) (*source.Error, source.Analyzer, bool) {
	var analyzer *source.Analyzer

	// If the source is "compiler", we expect a type error analyzer.
	if diag.Source == "compiler" {
		for _, a := range snapshot.View().Options().TypeErrorAnalyzers {
			if a.FixesError(diag.Message) {
				analyzer = &a
				break
			}
		}
	} else {
		// This code assumes that the analyzer name is the Source of the diagnostic.
		// If this ever changes, this will need to be addressed.
		if a, ok := snapshot.View().Options().DefaultAnalyzers[diag.Source]; ok {
			analyzer = &a
		}
	}
	if analyzer == nil || !analyzer.Enabled(snapshot) {
		return nil, source.Analyzer{}, false
	}
	analysisErrors, err := snapshot.Analyze(ctx, pkgID, analyzer.Analyzer)
	if err != nil {
		return nil, source.Analyzer{}, false
	}
	for _, err := range analysisErrors {
		if err.Message != diag.Message {
			continue
		}
		if protocol.CompareRange(err.Range, diag.Range) != 0 {
			continue
		}
		if err.Category != analyzer.Analyzer.Name {
			continue
		}
		// The error matches.
		return err, *analyzer, true
	}
	return nil, source.Analyzer{}, false
}

func convenienceFixes(ctx context.Context, snapshot source.Snapshot, ph source.PackageHandle, uri span.URI, rng protocol.Range) ([]protocol.CodeAction, error) {
	var analyzers []*analysis.Analyzer
	for _, a := range snapshot.View().Options().ConvenienceAnalyzers {
		if !a.Enabled(snapshot) {
			continue
		}
		analyzers = append(analyzers, a.Analyzer)
	}
	diagnostics, err := snapshot.Analyze(ctx, ph.ID(), analyzers...)
	if err != nil {
		return nil, err
	}
	var codeActions []protocol.CodeAction
	for _, d := range diagnostics {
		// For now, only show diagnostics for matching lines. Maybe we should
		// alter this behavior in the future, depending on the user experience.
		if d.URI != uri {
			continue
		}
		if d.Range.Start.Line != rng.Start.Line {
			continue
		}
		for _, fix := range d.SuggestedFixes {
			action := protocol.CodeAction{
				Title: fix.Title,
				Kind:  protocol.RefactorRewrite,
				Edit:  protocol.WorkspaceEdit{},
			}
			for uri, edits := range fix.Edits {
				fh, err := snapshot.GetFile(ctx, uri)
				if err != nil {
					return nil, err
				}
				docChanges := documentChanges(fh, edits)
				action.Edit.DocumentChanges = append(action.Edit.DocumentChanges, docChanges...)
			}
			codeActions = append(codeActions, action)
		}
	}
	return codeActions, nil
}

func extractionFixes(ctx context.Context, snapshot source.Snapshot, ph source.PackageHandle, uri span.URI, rng protocol.Range) ([]protocol.CodeAction, error) {
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, nil
	}
	edits, err := source.ExtractVariable(ctx, snapshot, fh, rng)
	if err != nil {
		return nil, err
	}
	if len(edits) == 0 {
		return nil, nil
	}
	return []protocol.CodeAction{
		{
			Title: "Extract to variable",
			Kind:  protocol.RefactorExtract,
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: documentChanges(fh, edits),
			},
		},
	}, nil
}

func documentChanges(fh source.FileHandle, edits []protocol.TextEdit) []protocol.TextDocumentEdit {
	return []protocol.TextDocumentEdit{
		{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version: fh.Version(),
				TextDocumentIdentifier: protocol.TextDocumentIdentifier{
					URI: protocol.URIFromSpanURI(fh.URI()),
				},
			},
			Edits: edits,
		},
	}
}
